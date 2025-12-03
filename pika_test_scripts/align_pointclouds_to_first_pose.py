#!/usr/bin/env python
"""
将指定 episode 中每帧的点云(来自深度图)根据末端执行器位姿对齐到第一帧位姿坐标系，
可视化所有帧叠加后夹爪区域是否重合，从而验证位姿估计稳定性。

假设：
1. 深度相机为定向 pinhole，相机深度数据集 camera/depth/pikaDepthCamera_[r|l] 中每个元素是文件名或编码深度(npz/png)。
2. 有对应彩色图 camera/color/pikaDepthCamera_[r|l] 可用于投影着色，如无法读取则使用单色。
3. 位姿 localization/pose/pika_[l|r] 为末端执行器在世界坐标下的 (x,y,z,roll,pitch,yaw) (弧度)。
4. extrinsic: camera/depthExtrinsic/pikaDepthCamera_[r|l] 为相机到世界的4x4矩阵（世界_T_cam）。

流程：
- 读取第一帧左右 EE 位姿与对应点云；将深度点投影到相机坐标再变换到世界坐标。
- 对后续帧：同样生成点云并转换到世界坐标，然后用 (世界_T_EE_first)^{-1} * 世界_T_EE_i 将点云再变换，
  使得各帧 EE 坐标系对齐于第一帧。
- 仅保留靠近夹爪中心盒区域的点(例如以第一帧 EE 原点附近的立方体筛选)以减少冗余。
- 累积点，限制最大点数随机下采样。
- 使用 Open3D 生成简单 3D 视图并保存为 PNG；同时输出所有点的投影 2D 图。

输出：
  out_dir/episodeX_aligned_points.png (俯视投影) + episodeX_aligned_points3d.ply

注意：若深度无法直接解码（现数据集为 bytes 文件名），尝试从 fallback 目录读取并解析为16位或浮点深度；
仅示例实现，具体深度格式需根据真实数据调整。

用法：
  python pika_test_scripts/align_pointclouds_to_first_pose.py --episode pika_raw_data/episode0/data.hdf5 --out-dir pika-videos/align --max-frames 50
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import h5py
import cv2
try:
    import open3d as o3d
except ImportError:
    o3d = None

def read_depth_element(ds, idx, fallback_dir: Path|None):
    x = ds[idx]
    if isinstance(x,(bytes,bytearray,memoryview)):
        name = bytes(x).decode('utf-8','ignore').strip('\x00')
    else:
        name = str(x)
    # 尝试常见扩展
    candidates = [name]
    if '.' not in name:
        candidates += [name+ext for ext in ['.png','.npy','.npz','.jpg','.jpeg']]
    for cand in candidates:
        p = fallback_dir / cand if fallback_dir else None
        if p and p.is_file():
            if p.suffix in ['.png','.jpg','.jpeg']:
                img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if img is not None:
                    return img
            if p.suffix == '.npy':
                return np.load(p)
            if p.suffix == '.npz':
                return np.load(p)['arr_0']
    try:
        arr = np.array(x)
        if arr.ndim>=2:
            return arr
    except Exception:
        pass
    return None

def depth_to_points(depth: np.ndarray, intrinsic: np.ndarray, max_points=50000, return_pixels: bool=False):
    # 假设 depth 单位为米或毫米 (若为毫米则转换)
    d = depth.astype(np.float32)
    # if np.nanmax(d) > 20:  # 粗判若>20视为毫米
    #     d = d / 1000.0
    h,w = d.shape[:2]
    fx = intrinsic[0,0]; fy = intrinsic[1,1]; cx = intrinsic[0,2]; cy = intrinsic[1,2]
    ys, xs = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    z = d
    mask = (z>0) & (z<2.0)  # 只保留近距离 2m 内
    xs = xs[mask]; ys = ys[mask]; z = z[mask]
    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    pts = np.stack([x,y,z], axis=1)
    if pts.shape[0] > max_points:
        sel = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[sel]
        xs = xs[sel]; ys = ys[sel]
    if return_pixels:
        return pts, ys.astype(np.int32), xs.astype(np.int32)
    return pts

def transform_points(pts: np.ndarray, T: np.ndarray):
    R = T[:3,:3]; t = T[:3,3]
    return (R @ pts.T).T + t

def pose_to_T(pose6: np.ndarray):
    x,y,z,roll,pitch,yaw = pose6
    sr,cr = np.sin(roll), np.cos(roll)
    sp,cp = np.sin(pitch), np.cos(pitch)
    sy,cy = np.sin(yaw), np.cos(yaw)
    Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3,:3] = R
    T[:3,3] = [x,y,z]
    return T

def main():
    ap = argparse.ArgumentParser(description='对齐点云到第一帧末端位姿并叠加可视化')
    ap.add_argument('--episode', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--max-frames', type=int, default=100)
    ap.add_argument('--side', type=str, default='r', choices=['r','l'], help='选择使用哪一侧深度相机生成点云')
    ap.add_argument('--max-points', type=int, default=200000)
    ap.add_argument('--num-frames', type=int, default=20, help='均匀采样的帧数')
    ap.add_argument('--points-per-frame', type=int, default=5000, help='每帧可视化的点数')
    ap.add_argument('--crop', action='store_true', help='启用后改为仅保留深度<=0.5m 的点 (原cube参数忽略)')
    args = ap.parse_args()

    h5_path = args.episode
    if h5_path.is_dir():
        h5_path = h5_path / 'data.hdf5'
    assert h5_path.is_file(), f'h5 not found: {h5_path}'

    with h5py.File(h5_path,'r') as f:
        depth_ds = f[f'camera/depth/pikaDepthCamera_{args.side}']
        intrinsic = f[f'camera/depthIntrinsic/pikaDepthCamera_{args.side}'][...]
        extrinsic = f[f'camera/depthExtrinsic/pikaDepthCamera_{args.side}'][...]
        poses = np.array(f[f'localization/pose/pika_{args.side}'])
        if np.mean(np.abs(poses[:,3:])) > 6.5:
            poses[:,3:] = np.radians(poses[:,3:])
        n = min(len(depth_ds), poses.shape[0], args.max_frames)
        T_ee_first = pose_to_T(poses[0])
        T_ee_first_inv = np.linalg.inv(T_ee_first)
        all_pts = []
        color_list_bgr = []
        color_list_rgb = []
        # 均匀采样帧索引
        sel_idx = np.unique(np.linspace(0, n-1, num=min(args.num_frames, n), dtype=int))
        # sel_idx = np.array([200])
        frames_used = int(len(sel_idx))
        print(f'[INFO] 处理 {frames_used} 帧: indices={sel_idx.tolist()}')
        for j, i in enumerate(sel_idx):
            rel = depth_ds[i]
            rel_path = rel.decode('utf-8','ignore') if isinstance(rel,(bytes,bytearray)) else str(rel)
            png_path = h5_path.parent / rel_path
            if not png_path.is_file():
                continue
            depth_raw = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
            if depth_raw is None or depth_raw.dtype != np.uint16:
                continue
            depth = depth_raw.astype(np.float32) / 1000.0
            # 生成较多点后在帧内再采样，保留像素索引
            pts_cam, ys_pix, xs_pix = depth_to_points(depth, intrinsic, max_points=1_000_000, return_pixels=True)
            # debug一下pts_cam的xyz范围
            print(f'[DEBUG] pts_cam xyz range: x={pts_cam[:,0].min()}..{pts_cam[:,0].max()} y={pts_cam[:,1].min()}..{pts_cam[:,1].max()} z={pts_cam[:,2].min()}..{pts_cam[:,2].max()}')
            if pts_cam.size == 0:
                continue
            # 新裁剪: 若启用 --crop 保留 z<=0.5m
            if args.crop:
                msk = pts_cam[:,2] <= 0.5
                pts_cam = pts_cam[msk]
                xs_pix = xs_pix[msk]
                ys_pix = ys_pix[msk]
                if pts_cam.size == 0:
                    continue
            pts_world = transform_points(pts_cam, extrinsic)
            pts_aligned = (T_ee_first_inv[:3,:3] @ pts_world.T).T + T_ee_first_inv[:3,3]
            # 移除旧 cube 裁剪逻辑
            # 每帧固定采样
            cnt = min(args.points_per_frame, pts_aligned.shape[0])
            if cnt <= 0:
                continue
            sel = np.random.choice(pts_aligned.shape[0], cnt, replace=False)
            pts_sel = pts_aligned[sel]
            all_pts.append(pts_sel)
            # 读取对应彩色图用于着色
            color_img = None
            try:
                color_ds = f[f'camera/color/pikaDepthCamera_{args.side}']
                color_rel = color_ds[i]
                color_path = h5_path.parent / (color_rel.decode('utf-8','ignore') if isinstance(color_rel,(bytes,bytearray)) else str(color_rel))
                if color_path.is_file():
                    color_img = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
            except Exception:
                color_img = None
            if color_img is not None:
                dh, dw = depth_raw.shape[:2]
                ch, cw = color_img.shape[:2]
                if (ch, cw) != (dh, dw):
                    color_img = cv2.resize(color_img, (dw, dh), interpolation=cv2.INTER_NEAREST)
                xs_s = np.clip(xs_pix[sel], 0, color_img.shape[1]-1)
                ys_s = np.clip(ys_pix[sel], 0, color_img.shape[0]-1)
                bgr = color_img[ys_s, xs_s, :].astype(np.uint8)
                rgb = bgr[:, ::-1]
            else:
                # 为该帧生成唯一颜色（fallback）
                hue = int(180 * j / max(1, frames_used-1)) if frames_used > 1 else 0
                hsv = np.zeros((cnt,1,3), dtype=np.uint8)
                hsv[...,0] = hue; hsv[...,1] = 200; hsv[...,2] = 255
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(-1,3)
                rgb = bgr[:, ::-1]
            color_list_bgr.append(bgr)
            color_list_rgb.append(rgb)
        if not all_pts:
            print('[WARN] 无有效点')
            return
        pts_all = np.concatenate(all_pts, axis=0)
        colors_bgr = np.concatenate(color_list_bgr, axis=0)
        colors_rgb = np.concatenate(color_list_rgb, axis=0)
        if pts_all.shape[0] > args.max_points:
            sel = np.random.choice(pts_all.shape[0], args.max_points, replace=False)
            pts_all = pts_all[sel]
            colors_bgr = colors_bgr[sel]
            colors_rgb = colors_rgb[sel]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # 俯视投影 (XY)
    xy = pts_all[:, :2]
    # 归一化到图像
    minv = xy.min(axis=0); maxv = xy.max(axis=0)
    span = np.maximum(maxv - minv, 1e-6)
    IMG=800
    pts2 = ((xy - minv) / span * (IMG-1)).astype(int)
    canvas = np.ones((IMG, IMG, 3), dtype=np.uint8)*255
    for (px,py), c in zip(pts2, colors_bgr):
        cv2.circle(canvas, (int(px), int(py)), 1, (int(c[0]), int(c[1]), int(c[2])), -1)
    cv2.putText(canvas,f'Aligned pointcloud side={args.side} frames={frames_used} depth<=0.5m filter={args.crop }',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2,cv2.LINE_AA)
    out_img = args.out_dir / f'{h5_path.parent.name}_aligned_points.png'
    cv2.imwrite(str(out_img), canvas)

    # 保存 PLY
    out_ply = args.out_dir / f'{h5_path.parent.name}_aligned_points3d.ply'
    if o3d is not None:
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts_all))
        pc.colors = o3d.utility.Vector3dVector(colors_rgb.astype(np.float64)/255.0)
        o3d.visualization.draw_geometries([pc])
        o3d.io.write_point_cloud(str(out_ply), pc)
    else:
        # 简单 ASCII PLY
        with open(out_ply,'w') as fw:
            fw.write('ply\nformat ascii 1.0\n')
            fw.write(f'element vertex {pts_all.shape[0]}\n')
            fw.write('property float x\nproperty float y\nproperty float z\nend_header\n')
            for p in pts_all:
                fw.write(f'{p[0]} {p[1]} {p[2]}\n')
    print('[OK] 输出', out_img, out_ply)

if __name__ == '__main__':
    main()

    """
    运行实例：
    python pika_test_scripts/align_pointclouds_to_first_pose.py --episode pika_raw_data/episode0/data.hdf5 --out-dir pika-videos/align --side r --num-frames 1 --points-per-frame 10000
    """