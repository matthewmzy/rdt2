#!/usr/bin/env python
"""
综合可视化一个 episode 的全部关键信息并生成视频：
布局（已修正左右，物理左手侧画面在左列）：
  第一排：左彩色(depth)  |  右彩色(depth)
  第二排：左鱼眼          |  右鱼眼
  第三排：双夹爪 6D 位姿 (3D 视角+世界坐标系) | 抓手开合距离/角度曲线（带时间游标）
注意：彩色(depth)相机命名中 camera_r 为左、camera_l 为右，反了。鱼眼是正确的。
新增: 支持在每帧 RGB(depth) 图上投影未来若干帧 (horizon) 的末端执行器锚点轨迹。锚点定义为当前及未来帧末端坐标加上其局部坐标系下 (0,0,anchor_offset_z) 的点位。
"""
from __future__ import annotations
import os
import math
import argparse
from pathlib import Path
import numpy as np
import h5py
import cv2
import matplotlib
matplotlib.use('Agg')  # 使用无界面后端避免 Tk/PIL resize 错误
matplotlib.rcParams['toolbar'] = 'None'
import matplotlib.pyplot as plt

# ----------------- 通用辅助 -----------------

def to_bytes(x) -> bytes:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return bytes(x)
    try:
        return x.tobytes()
    except Exception:
        try:
            return bytes(np.array(x).tobytes())
        except Exception:
            return b""


def decode_bgr(img_bytes: bytes):
    if not img_bytes:
        return None
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def _find_file_by_name(dirpath: Path, name: str):
    s = name.strip().strip('\x00')
    if not s:
        return None
    p = dirpath / s
    if p.is_file():
        return p
    base = s
    if '.' not in Path(s).name:
        for ext in ('.jpg', '.jpeg', '.png'):
            p2 = dirpath / (base + ext)
            if p2.is_file():
                return p2
    stem = Path(s).stem
    candidates = list(dirpath.glob(stem + '.*'))
    if candidates:
        return sorted(candidates)[0]
    return None


def read_frame(ds, idx, fallback_dir: Path | None):
    x = ds[idx]
    img = decode_bgr(to_bytes(x))
    if img is not None:
        return img
    # 展开像素数组
    try:
        arr = np.array(x)
        if arr.ndim == 3 and arr.dtype == np.uint8:
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        if arr.ndim == 2 and arr.dtype == np.uint8:
            return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    except Exception:
        pass
    # 文件名回退
    if fallback_dir is not None:
        if isinstance(x, (bytes, bytearray, memoryview)):
            try:
                name = bytes(x).decode('utf-8', 'ignore')
            except Exception:
                name = ''
        else:
            name = str(x)
        p = _find_file_by_name(fallback_dir, name)
        if p and p.is_file():
            img2 = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img2 is not None and img2.size > 0:
                return img2
    return None


def estimate_fps(timestamps: np.ndarray) -> float:
    if timestamps is None or len(timestamps) < 2:
        return 30.0
    dt = np.diff(timestamps)
    dt = dt[dt > 1e-6]
    if len(dt) == 0:
        return 30.0
    fps = float(np.clip(1.0 / np.median(dt), 1.0, 240.0))
    return fps

# ----------------- 位姿与绘制 -----------------

def is_degree(rpy: np.ndarray) -> bool:
    # 简单判定：绝对值若大部分 > 2*pi 则认为是度
    if rpy.size == 0:
        return False
    return np.mean(np.abs(rpy)) > 6.5  # > ~ 2*pi


def rpy_to_rot(rpy):
    roll, pitch, yaw = rpy
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    # ZYX convention: R = Rz * Ry * Rx
    Rz = np.array([[cy, -sy, 0],[sy, cy, 0],[0,0,1]])
    Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
    Rx = np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
    R = Rz @ Ry @ Rx
    return R


def pose_to_mat(pose6: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3,3] = pose6[:3]
    T[:3,:3] = rpy_to_rot(pose6[3:])
    return T


def project_points(pts3: np.ndarray, view_rot: np.ndarray, view_trans: np.ndarray, scale=300, img_size=(400,400)):
    # 简单正射 + 透视近似：应用视角旋转和平移，然后忽略深度缩放
    P = (view_rot @ (pts3.T - view_trans.reshape(3,1))).T
    # 取 X,Y 作为平面
    x = P[:,0]; y = P[:,1]
    cx, cy = img_size[0]//2, img_size[1]//2
    pts2 = np.stack([cx + x*scale, cy - y*scale], axis=1)
    return pts2.astype(int)


def draw_pose_canvas(pose_l: np.ndarray, pose_r: np.ndarray, canvas_size=(500,500)):
    # 3D 立体感：采用固定视角旋转 + 简单透视缩放
    def euler_to_rot(yaw_deg, pitch_deg):
        yaw = math.radians(yaw_deg)
        pitch = math.radians(pitch_deg)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        Rz = np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
        Ry = np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
        return Ry @ Rz
    view_R = euler_to_rot(40,30)
    view_T = np.array([0.0,0.0,0.0])
    canvas = np.ones((canvas_size[1], canvas_size[0],3), dtype=np.uint8)*255
    # 世界坐标系
    world_axes = np.array([[0,0,0],[0.06,0,0],[0,0.06,0],[0,0,0.06]])
    pts_world = project_points(world_axes, view_R, view_T, scale=750, img_size=canvas_size)
    o_w = pts_world[0]
    cv2.putText(canvas,'World',(o_w[0]+5,o_w[1]-5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    cv2.line(canvas, tuple(o_w), tuple(pts_world[1]), (0,0,255),2)
    cv2.putText(canvas,'X',(pts_world[1][0]+3, pts_world[1][1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,0,255),1,cv2.LINE_AA)
    cv2.line(canvas, tuple(o_w), tuple(pts_world[2]), (0,255,0),2)
    cv2.putText(canvas,'Y',(pts_world[2][0]+3, pts_world[2][1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,128,0),1,cv2.LINE_AA)
    cv2.line(canvas, tuple(o_w), tuple(pts_world[3]), (255,0,0),2)
    cv2.putText(canvas,'Z',(pts_world[3][0]+3, pts_world[3][1]),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1,cv2.LINE_AA)
    # 设备颜色方案
    axes_colors_left = [(255,0,0),(0,255,0),(0,0,255)]  # X,Y,Z
    axes_colors_right = [(255,0,0),(0,255,0),(0,0,255)] # X,Y,Z
    def draw_ee(pose, colors, label):
        p = pose[:3]
        rpy = pose[3:]
        R = rpy_to_rot(rpy)
        axis_len = 0.05
        axes = np.array([[0,0,0],[axis_len,0,0],[0,axis_len,0],[0,0,axis_len]])
        world_axes = (R @ axes.T).T + p  # 旋转后平移
        pts2 = project_points(world_axes, view_R, view_T, scale=750, img_size=canvas_size)
        o = pts2[0]
        cv2.circle(canvas, tuple(o), 5, (0,0,0), -1)
        for k in range(3):
            cv2.line(canvas, tuple(o), tuple(pts2[k+1]), colors[k], 3)
        cv2.putText(canvas,label,(o[0]+5,o[1]+5),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
    draw_ee(pose_l, axes_colors_left, 'Left EE')
    draw_ee(pose_r, axes_colors_right, 'Right EE')
    # 网格
    for g in range(0, canvas_size[0], 50):
        cv2.line(canvas,(g,0),(g,canvas_size[1]-1),(230,230,230),1)
    for g in range(0, canvas_size[1], 50):
        cv2.line(canvas,(0,g),(canvas_size[0]-1,g),(230,230,230),1)
    cv2.putText(canvas,'End Effector 6D Pose (3D View)',(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,0),2,cv2.LINE_AA)
    return canvas

# ----------------- 曲线预渲染 -----------------

def render_time_series(dist_l, dist_r, angle_l, angle_r, timestamps):
    t = timestamps - timestamps[0]
    fig, axes = plt.subplots(2,1, figsize=(6,4), dpi=100)
    axes[0].plot(t, dist_l, label='dist_l')
    axes[0].plot(t, dist_r, label='dist_r')
    axes[0].set_title('Gripper Distance')
    axes[0].legend(loc='upper right', fontsize=8)
    axes[0].set_xlabel('t (s)')
    axes[1].plot(t, angle_l, label='angle_l')
    axes[1].plot(t, angle_r, label='angle_r')
    axes[1].set_title('Gripper Angle')
    axes[1].legend(loc='upper right', fontsize=8)
    axes[1].set_xlabel('t (s)')
    fig.tight_layout()
    fig.canvas.draw()
    # 兼容不同后端: 优先 rgb, 回退 argb
    try:
        buf = fig.canvas.tostring_rgb()
        w, h = fig.canvas.get_width_height()
        img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
    except Exception:
        buf = fig.canvas.tostring_argb()
        w, h = fig.canvas.get_width_height()
        argb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
        # 转 BGR
        rgb = argb[:,:,1:4]
        img = rgb
    plt.close(fig)
    return img


def add_cursor_to_plot(plot_img: np.ndarray, progress: float, left_margin_frac: float = 0.08, right_margin_frac: float = 0.02) -> np.ndarray:
    """在曲线图上添加蓝色时间游标。
    left_margin_frac/right_margin_frac: 以图像宽度的比例设定左右边距，游标仅在中间可视曲线区域内移动。
    """
    img = plot_img.copy()
    h, w = img.shape[:2]
    progress = float(np.clip(progress, 0.0, 1.0))
    # 计算有效绘制区间
    lm = max(0, min(0.49, float(left_margin_frac)))
    rm = max(0, min(0.49, float(right_margin_frac)))
    x0 = int(round(lm * w))
    x1 = int(round((1.0 - rm) * w)) - 1
    if x1 <= x0:
        x0, x1 = 0, w - 1
    span = max(1, x1 - x0)
    x = x0 + int(round(progress * span))
    x = int(np.clip(x, x0, x1))
    cv2.line(img, (x, 0), (x, h - 1), (255, 0, 0), 2)
    return img

# ----------------- 主逻辑 -----------------

def compose_frame(layout_imgs: dict, info: dict) -> np.ndarray:
    # layout_imgs keys: depth_left, depth_right, fisheye_left, fisheye_right, pose, plot
    # 规范化尺寸与列宽
    cells = [img for img in layout_imgs.values() if img is not None]
    if not cells:
        return None
    target_w = max(img.shape[1] for img in cells)
    def pad_to_w(img):
        h,w = img.shape[:2]
        if w == target_w:
            return img
        pad = np.ones((h, target_w - w, 3), dtype=np.uint8)*255
        return np.hstack([img, pad])
    # 行内高度统一（通过缩放第二张到第一张高度）
    def unify_row(a,b):
        if a is None and b is None:
            return np.zeros((100, target_w*2,3), dtype=np.uint8)
        if a is None:
            a = np.ones_like(b)*255
        if b is None:
            b = np.ones_like(a)*255
        ha, wa = a.shape[:2]; hb, wb = b.shape[:2]
        if ha != hb:
            b = cv2.resize(b,(wb,ha),interpolation=cv2.INTER_AREA)
        a = pad_to_w(a); b = pad_to_w(b)
        return np.hstack([a,b])
    row1 = unify_row(layout_imgs['depth_left'], layout_imgs['depth_right'])
    row2 = unify_row(layout_imgs['fisheye_left'], layout_imgs['fisheye_right'])
    row3 = unify_row(layout_imgs['pose'], layout_imgs['plot'])
    out = np.vstack([row1,row2,row3])
    txt = f"frame={info['frame']} / {info['total']}  t={info['time']:.2f}s  dist_l={info['dist_l']:.3f} dist_r={info['dist_r']:.3f} angle_l={info['angle_l']:.2f} angle_r={info['angle_r']:.2f}" 
    cv2.putText(out, txt, (10, out.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),1,cv2.LINE_AA)
    return out

# ---- 投影 ----

def _project_world_points(world_pts: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, mode: str):
    """mode: 'world_T_cam' or 'cam_T_world'"""
    if world_pts.size == 0:
        return np.zeros((0,2)), np.zeros((0,), dtype=bool)
    R = extrinsic[:3,:3]; t = extrinsic[:3,3]
    if mode == 'world_T_cam':
        cam = (R.T @ (world_pts - t).T).T
    else:  # cam_T_world
        cam = (R @ world_pts.T).T + t
    fx = intrinsic[0,0]; fy = intrinsic[1,1]; cx = intrinsic[0,2]; cy = intrinsic[1,2]
    z = cam[:,2]
    u = fx * cam[:,0] / np.clip(z,1e-6,None) + cx
    v = fy * cam[:,1] / np.clip(z,1e-6,None) + cy
    uv = np.stack([u,v], axis=1)
    mask = (z > 1e-6)
    return uv, mask

def draw_coordinate_axes(img: np.ndarray, intrinsic: np.ndarray, extrinsic: np.ndarray, axis_length: float = 0.03):
    """
    在图像上同时绘制【夹爪坐标系】和【相机坐标系】，方便对比。
    
    夹爪坐标系（Gripper Frame）- 使用 cam_T_gripper 变换：
    - 原点用白色圆点标记
    - X轴红色, Y轴绿色, Z轴蓝色
    - 标签: gX, gY, gZ
    
    相机坐标系（Camera Frame）- 直接在相机系下定义：
    - 原点用黄色圆点标记
    - X轴橙色, Y轴青色, Z轴品红色
    - 标签: cX, cY, cZ
    
    Args:
        img: BGR图像
        intrinsic: 相机内参矩阵 3x3
        extrinsic: cam_T_gripper 外参矩阵 4x4（将夹爪系点变换到相机系）
        axis_length: 坐标轴长度（米），默认0.03m
    """
    h, w = img.shape[:2]
    fx, fy = intrinsic[0,0], intrinsic[1,1]
    cx, cy = intrinsic[0,2], intrinsic[1,2]
    
    def in_bounds(pt):
        return 0 <= pt[0] < w and 0 <= pt[1] < h
    
    def clip_to_bounds(pt):
        return (max(0, min(w-1, pt[0])), max(0, min(h-1, pt[1])))
    
    def project_cam_point(pt_cam):
        """将相机坐标系下的点投影到像素"""
        if pt_cam[2] <= 0:
            return None
        u = int(round(fx * pt_cam[0] / pt_cam[2] + cx))
        v = int(round(fy * pt_cam[1] / pt_cam[2] + cy))
        return (u, v)
    
    line_thickness = 3

    # ========== 1. 绘制相机坐标系（Camera Frame）==========
    # 相机坐标系：X向右，Y向下，Z向前（深度方向）
    # 在相机前方 offset 处绘制，放在画面左侧
    # cam_offset = 0.12  # 相机前方的距离
    # cam_origin_cam = np.array([-0.06, 0.0, cam_offset])  # 放在画面左侧
    # cam_x_end = cam_origin_cam + np.array([axis_length, 0, 0])
    # cam_y_end = cam_origin_cam + np.array([0, axis_length, 0])
    # cam_z_end = cam_origin_cam + np.array([0, 0, axis_length])
    
    # cam_o_pix = project_cam_point(cam_origin_cam)
    # cam_x_pix = project_cam_point(cam_x_end)
    # cam_y_pix = project_cam_point(cam_y_end)
    # cam_z_pix = project_cam_point(cam_z_end)
    
    # 
    # # 相机坐标系使用不同的颜色：橙色X，青色Y，品红Z
    # if cam_o_pix and cam_x_pix:
    #     cv2.line(img, clip_to_bounds(cam_o_pix), clip_to_bounds(cam_x_pix), (0, 165, 255), line_thickness)  # 橙色
    #     if in_bounds(cam_x_pix):
    #         cv2.putText(img, 'cX', cam_x_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2, cv2.LINE_AA)
    # if cam_o_pix and cam_y_pix:
    #     cv2.line(img, clip_to_bounds(cam_o_pix), clip_to_bounds(cam_y_pix), (255, 255, 0), line_thickness)  # 青色
    #     if in_bounds(cam_y_pix):
    #         cv2.putText(img, 'cY', cam_y_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
    # if cam_o_pix and cam_z_pix:
    #     cv2.line(img, clip_to_bounds(cam_o_pix), clip_to_bounds(cam_z_pix), (255, 0, 255), line_thickness)  # 品红
    #     if in_bounds(cam_z_pix):
    #         cv2.putText(img, 'cZ', cam_z_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2, cv2.LINE_AA)
    # if cam_o_pix and in_bounds(cam_o_pix):
    #     cv2.circle(img, cam_o_pix, 5, (0, 255, 255), -1)  # 黄色原点
    
    # ========== 2. 绘制夹爪坐标系（Gripper Frame）==========
    # 夹爪坐标系下的点，需要用 cam_T_gripper 变换到相机系
    gripper_offset = 0.1  # 沿夹爪X轴偏移
    side_offset = 0    #0.06  # 沿夹爪Y轴偏移
    # 放在画面右侧（夹爪Y轴负方向）
    gripper_origin_gripper = np.array([gripper_offset, -side_offset, 0.0, 1.0])
    gripper_x_end_gripper = np.array([gripper_offset + axis_length, -side_offset, 0.0, 1.0])
    gripper_y_end_gripper = np.array([gripper_offset, -side_offset + axis_length, 0.0, 1.0])
    gripper_z_end_gripper = np.array([gripper_offset, -side_offset, axis_length, 1.0])
    
    # 变换到相机坐标系
    gripper_origin_cam = (extrinsic @ gripper_origin_gripper)[:3]
    gripper_x_end_cam = (extrinsic @ gripper_x_end_gripper)[:3]
    gripper_y_end_cam = (extrinsic @ gripper_y_end_gripper)[:3]
    gripper_z_end_cam = (extrinsic @ gripper_z_end_gripper)[:3]
    
    gripper_o_pix = project_cam_point(gripper_origin_cam)
    gripper_x_pix = project_cam_point(gripper_x_end_cam)
    gripper_y_pix = project_cam_point(gripper_y_end_cam)
    gripper_z_pix = project_cam_point(gripper_z_end_cam)
    
    # 夹爪坐标系使用标准RGB颜色
    if gripper_o_pix and gripper_x_pix:
        cv2.line(img, clip_to_bounds(gripper_o_pix), clip_to_bounds(gripper_x_pix), (0, 0, 255), line_thickness)  # 红色
        if in_bounds(gripper_x_pix):
            cv2.putText(img, 'gX', gripper_x_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)
    if gripper_o_pix and gripper_y_pix:
        cv2.line(img, clip_to_bounds(gripper_o_pix), clip_to_bounds(gripper_y_pix), (0, 255, 0), line_thickness)  # 绿色
        if in_bounds(gripper_y_pix):
            cv2.putText(img, 'gY', gripper_y_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
    if gripper_o_pix and gripper_z_pix:
        cv2.line(img, clip_to_bounds(gripper_o_pix), clip_to_bounds(gripper_z_pix), (255, 0, 0), line_thickness)  # 蓝色
        if in_bounds(gripper_z_pix):
            cv2.putText(img, 'gZ', gripper_z_pix, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2, cv2.LINE_AA)
    if gripper_o_pix and in_bounds(gripper_o_pix):
        cv2.circle(img, gripper_o_pix, 5, (255, 255, 255), -1)  # 白色原点

def compute_ego_centric_trajectory(pose_seq, current_idx, horizon, anchor_offset):
    """
    计算未来帧的末端锚点在当前夹爪坐标系下的3D位置序列。
    
    anchor_offset: 锚点相对于末端坐标系原点的偏移，格式为 (x, y, z)
                   根据你的设置，相机朝向夹爪X轴，所以偏移应在X轴方向
    """
    # 1. 获取当前帧的世界位姿 T_world_current
    world_T_cur = pose_to_mat(pose_seq[current_idx])
    cur_T_world = np.linalg.inv(world_T_cur) # 逆矩阵：世界 -> 当前夹爪
    
    # 锚点在其自身局部坐标系的定义
    # 齐次坐标 (x, y, z, 1)
    anchor_local = np.array([anchor_offset[0], anchor_offset[1], anchor_offset[2], 1.0]) 
    
    points_in_current_frame = []
    
    # 2. 遍历未来帧
    end_idx = min(len(pose_seq), current_idx + horizon + 1)
    for j in range(current_idx, end_idx):
        # 获取未来帧的世界位姿 T_world_future
        world_T_fut = pose_to_mat(pose_seq[j])
        
        # 3. 核心魔法：计算相对变换
        # 未来相对于当前 = (当前逆) * (未来)
        # 这一步把“世界坐标”消掉了，只剩下“相对运动”
        cur_T_fut = cur_T_world @ world_T_fut
        
        # 4. 将锚点从“未来夹爪系”转到“当前夹爪系”
        # 逻辑：锚点跟着未来夹爪走，所以它是 world_T_fut * anchor_local
        # 但我们要看它在当前视角的哪里，所以左乘 cur_T_world
        # 公式化简后就是： cur_T_fut * anchor_local
        p_in_cur = cur_T_fut @ anchor_local
        
        points_in_current_frame.append(p_in_cur[:3]) # 取 (x,y,z)
        
    return np.array(points_in_current_frame)

def process_episode_full(h5_path: Path, out_path: Path, use_fisheye=False, cursor_margin_left: float = 0.08, cursor_margin_right: float = 0.02,
                         future_horizon: int = 30, anchor_offset: tuple = (0.10, 0.0, 0.0), draw_anchors: bool = True):
    """
    anchor_offset: 锚点在夹爪局部坐标系的偏移 (x, y, z)。
                   相机光轴朝向夹爪X轴，所以默认在X轴方向偏移0.1m。
    """
    # 增强：同时读取 depth 与 fisheye
    with h5py.File(h5_path,'r') as f:
        # 使用切片 [:] 代替 np.array() 以兼容 numpy 2.x 与 h5py
        ts = f['timestamp'][:] if 'timestamp' in f else None
        fps = estimate_fps(ts)
        # 相机键
        cam_color = 'camera/color'
        # 彩色(depth)左右需交换
        depth_left_key = 'pikaDepthCamera_r'
        depth_right_key = 'pikaDepthCamera_l'
        # 鱼眼保持原始命名
        fisheye_left_key = 'pikaFisheyeCamera_l'
        fisheye_right_key = 'pikaFisheyeCamera_r'
        ds_depth_l = f[f'{cam_color}/{depth_left_key}'] if f'{cam_color}/{depth_left_key}' in f else None
        ds_depth_r = f[f'{cam_color}/{depth_right_key}'] if f'{cam_color}/{depth_right_key}' in f else None
        ds_fish_l = f[f'{cam_color}/{fisheye_left_key}'] if f'{cam_color}/{fisheye_left_key}' in f else None
        ds_fish_r = f[f'{cam_color}/{fisheye_right_key}'] if f'{cam_color}/{fisheye_right_key}' in f else None
        # 帧数取存在数据集最小
        lens = [len(ds) for ds in [ds_depth_l, ds_depth_r, ds_fish_l, ds_fish_r] if ds is not None]
        if not lens:
            return None
        n = min(lens)
        # 抓手与位姿 - 使用切片 [:] 代替 np.array() 以兼容 numpy 2.x
        dist_l = f['gripper/encoderDistance/pika_l'][:]
        dist_r = f['gripper/encoderDistance/pika_r'][:]
        ang_l = f['gripper/encoderAngle/pika_l'][:]
        ang_r = f['gripper/encoderAngle/pika_r'][:]
        pose_l = f['localization/pose/pika_l'][:]
        pose_r = f['localization/pose/pika_r'][:]
        n = min(n, pose_l.shape[0], pose_r.shape[0], dist_l.shape[0], dist_r.shape[0])
        if is_degree(pose_l[:,3:]) or is_degree(pose_r[:,3:]):
            pose_l[:,3:] = np.radians(pose_l[:,3:])
            pose_r[:,3:] = np.radians(pose_r[:,3:])
        print(f"Processing episode: {h5_path}, frames={n}, fps={fps:.2f}")
        print(f"[DEBUG] dist_l : {dist_l.shape}, dist_r : {dist_r.shape}, ang_l : {ang_l.shape}, ang_r : {ang_r.shape}, pose_l : {pose_l.shape}, pose_r : {pose_r.shape}")
        plot_img_full = render_time_series(dist_l[:n], dist_r[:n], ang_l[:n], ang_r[:n], ts[:n])
        # 回退目录
        fb_depth_l = h5_path.parent / 'camera' / 'color' / depth_left_key
        fb_depth_r = h5_path.parent / 'camera' / 'color' / depth_right_key
        fb_fish_l = h5_path.parent / 'camera' / 'color' / fisheye_left_key
        fb_fish_r = h5_path.parent / 'camera' / 'color' / fisheye_right_key
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = None
        # 读取相机内外参 (使用 color 相机键，因为绘制在RGB图像上)
        intrinsic_l = f.get(f'camera/colorIntrinsic/pikaDepthCamera_l')
        intrinsic_r = f.get(f'camera/colorIntrinsic/pikaDepthCamera_l')
        extrinsic_l = f.get(f'camera/colorExtrinsic/pikaDepthCamera_r')
        extrinsic_r = f.get(f'camera/colorExtrinsic/pikaDepthCamera_l')
        K_l = intrinsic_l[...].astype(float) if intrinsic_l is not None else None
        K_r = intrinsic_r[...].astype(float) if intrinsic_r is not None else None
        # 注意：存储的外参是 gripper_T_cam，需要求逆得到 cam_T_gripper
        T_l_raw = extrinsic_l[...].astype(float) if extrinsic_l is not None else None
        T_r_raw = extrinsic_r[...].astype(float) if extrinsic_r is not None else None
        T_l = np.linalg.inv(T_l_raw) if T_l_raw is not None else None  # cam_T_gripper
        T_r = np.linalg.inv(T_r_raw) if T_r_raw is not None else None  # cam_T_gripper
        for i in range(n):
            depth_l = read_frame(ds_depth_l, i, fb_depth_l) if ds_depth_l is not None else None
            depth_r = read_frame(ds_depth_r, i, fb_depth_r) if ds_depth_r is not None else None
            fish_l = read_frame(ds_fish_l, i, fb_fish_l) if ds_fish_l is not None else None
            fish_r = read_frame(ds_fish_r, i, fb_fish_r) if ds_fish_r is not None else None
            if depth_l is None or depth_r is None:
                continue
            # 高度对齐
            def align_pair(a,b):
                if a is None or b is None:
                    return a,b
                ha,wa = a.shape[:2]; hb,wb = b.shape[:2]
                if ha != hb:
                    b = cv2.resize(b,(int(round(wb*(ha/hb))),ha),interpolation=cv2.INTER_AREA)
                return a,b
            depth_l, depth_r = align_pair(depth_l, depth_r)
            if fish_l is not None and fish_r is not None:
                fish_l, fish_r = align_pair(fish_l, fish_r)
            # 在合成前绘制未来锚点轨迹
            if draw_anchors and K_l is not None and T_l is not None:
                # 1. 计算轨迹点（都在当前夹爪坐标系下）
                traj_points_3d = compute_ego_centric_trajectory(pose_l, i, future_horizon, anchor_offset)
                
                # 2. 投影到像素 (Project)
                # 关键点：这里的 T_l 应该是 "Camera_from_Gripper" 的外参
                # 如果你的 T_l 是 cam_T_gripper (把点从夹爪系转到相机系)，直接用 _project_world_points 的 cam_T_world 模式即可
                # 因为此时我们的 "World" 就是 "Current Gripper Frame"
                
                # 注意：这里不再需要 choose_extrinsic_mode 去猜了，因为坐标系已经锁定在局部了
                # 我们假设 T_l 是把点从 Pose系 搬运到 Camera系 的矩阵
                
                uv_l, mask_l = _project_world_points(traj_points_3d, K_l, T_l, mode='cam_T_world')
                pix_l = np.round(uv_l).astype(int)

                # 3. 绘制
                if pix_l.shape[0] > 0:
                    # 画起始点（当前帧末端），应该是固定不动的
                    x0, y0 = pix_l[0]
                    # 只有在画面内才画
                    if 0 <= x0 < depth_l.shape[1] and 0 <= y0 < depth_l.shape[0]:
                        cv2.circle(depth_l, (x0, y0), 6, (0, 0, 255), -1) # 红色圆点
                        
                    # 画折线
                    for k in range(1, pix_l.shape[0]):
                        x_curr, y_curr = pix_l[k]
                        x_prev, y_prev = pix_l[k-1]
                        # 简单的边界检查
                        h_img, w_img = depth_l.shape[:2]
                        if 0<=x_curr<w_img and 0<=y_curr<h_img and 0<=x_prev<w_img and 0<=y_prev<h_img:
                            # 颜色随时间渐变，以此表示“未来”
                            color = (0, int(255 * (k/len(pix_l))), 255) # BGR: 红色渐变到橙黄
                            cv2.line(depth_l, (x_prev, y_prev), (x_curr, y_curr), color, 2)
            
            # 绘制左侧相机的夹爪坐标系坐标轴
            if K_l is not None and T_l is not None:
                draw_coordinate_axes(depth_l, K_l, T_l, axis_length=0.05)
            
            # 在合成前绘制未来锚点轨迹 - 右侧相机
            if draw_anchors and K_r is not None and T_r is not None:
                # 1. 计算轨迹点（都在当前夹爪坐标系下）- 使用右手位姿
                traj_points_3d_r = compute_ego_centric_trajectory(pose_r, i, future_horizon, anchor_offset)
                
                # 2. 投影到像素
                uv_r, mask_r = _project_world_points(traj_points_3d_r, K_r, T_r, mode='cam_T_world')
                pix_r = np.round(uv_r).astype(int)

                # 3. 绘制
                if pix_r.shape[0] > 0:
                    # 画起始点（当前帧末端），应该是固定不动的
                    x0, y0 = pix_r[0]
                    # 只有在画面内才画
                    if 0 <= x0 < depth_r.shape[1] and 0 <= y0 < depth_r.shape[0]:
                        cv2.circle(depth_r, (x0, y0), 6, (0, 0, 255), -1) # 红色圆点
                        
                    # 画折线
                    for k in range(1, pix_r.shape[0]):
                        x_curr, y_curr = pix_r[k]
                        x_prev, y_prev = pix_r[k-1]
                        # 简单的边界检查
                        h_img, w_img = depth_r.shape[:2]
                        if 0<=x_curr<w_img and 0<=y_curr<h_img and 0<=x_prev<w_img and 0<=y_prev<h_img:
                            # 颜色随时间渐变
                            color = (0, int(255 * (k/len(pix_r))), 255) # BGR: 红色渐变到橙黄
                            cv2.line(depth_r, (x_prev, y_prev), (x_curr, y_curr), color, 2)
            
            # 绘制右侧相机的夹爪坐标系坐标轴
            if K_r is not None and T_r is not None:
                draw_coordinate_axes(depth_r, K_r, T_r, axis_length=0.05)
            
            # 后续布局
            pose_canvas = draw_pose_canvas(pose_l[i], pose_r[i])
            progress = i / max(1,(n-1))
            plot_cursor = add_cursor_to_plot(plot_img_full, progress, left_margin_frac=cursor_margin_left, right_margin_frac=cursor_margin_right)
            layout = dict(depth_left=depth_l, depth_right=depth_r, fisheye_left=fish_l, fisheye_right=fish_r, pose=pose_canvas, plot=plot_cursor)
            info = dict(frame=i,total=n,time=float(ts[i]-ts[0]),dist_l=float(dist_l[i]),dist_r=float(dist_r[i]),angle_l=float(ang_l[i]),angle_r=float(ang_r[i]))
            frame_img = compose_frame(layout, info)
            if frame_img is None:
                continue
            if writer is None:
                h,w = frame_img.shape[:2]
                writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w,h))
            writer.write(frame_img)
    if writer is not None:
        writer.release()
    return out_path if out_path.is_file() else None

# ----------------- 批量 -----------------

def find_h5_list(episodes_glob: str) -> list[Path]:
    paths = [Path(p) for p in sorted(Path('.').glob(episodes_glob))]
    out = []
    for p in paths:
        if p.is_dir():
            h = p / 'data.hdf5'
            if h.is_file():
                out.append(h)
        elif p.is_file() and p.name.endswith('.hdf5'):
            out.append(p)
    return out

# ----------------- CLI -----------------

def main():
    ap = argparse.ArgumentParser(description='综合 episode 可视化并导出视频')
    ap.add_argument('--episode', type=Path, help='单个 data.hdf5 路径')
    ap.add_argument('--episodes-glob', type=str, help='批量模式：如 pika_raw_data/episode*/data.hdf5')
    ap.add_argument('--out', type=Path, help='单集输出视频路径')
    ap.add_argument('--out-dir', type=Path, help='批量输出目录')
    ap.add_argument('--use-fisheye', action='store_true', help='使用鱼眼图像')
    ap.add_argument('--cursor-margin-left', type=float, default=0.134, help='时间游标左边距（占plot宽度比例）')
    ap.add_argument('--cursor-margin-right', type=float, default=0.063, help='时间游标右边距（占plot宽度比例）')
    ap.add_argument('--future-horizon', type=int, default=30, help='未来轨迹长度(帧数)')
    ap.add_argument('--anchor-offset', type=float, nargs=3, default=[0.10, 0.0, 0.0], 
                    metavar=('X', 'Y', 'Z'), help='末端局部坐标 (X,Y,Z) 作为锚点偏移，相机朝向夹爪X轴')
    ap.add_argument('--no-anchors', action='store_true', help='关闭未来锚点轨迹投影')
    args = ap.parse_args()

    if args.episode and args.out:
        r = process_episode_full(args.episode, args.out, use_fisheye=args.use_fisheye,
                                 cursor_margin_left=args.cursor_margin_left, cursor_margin_right=args.cursor_margin_right,
                                 future_horizon=args.future_horizon, anchor_offset=tuple(args.anchor_offset), draw_anchors=not args.no_anchors)
        print('[OK]' if r else '[FAIL]', args.out)
        return
    if args.episodes_glob and args.out_dir:
        h5_list = find_h5_list(args.episodes_glob)
        if not h5_list:
            print('[WARN] 未找到匹配 hdf5')
            return
        args.out_dir.mkdir(parents=True, exist_ok=True)
        ok, fail = 0,0
        for h5 in h5_list:
            out_path = args.out_dir / f'{h5.parent.name}_full.mp4'
            try:
                r = process_episode_full(h5, out_path, use_fisheye=args.use_fisheye,
                                         cursor_margin_left=args.cursor_margin_left, cursor_margin_right=args.cursor_margin_right,
                                         future_horizon=args.future_horizon, anchor_offset=tuple(args.anchor_offset), draw_anchors=not args.no_anchors)
                if r:
                    print(f'[OK] {h5} -> {out_path}')
                    ok += 1
                else:
                    print(f'[FAIL] {h5}')
                    fail += 1
            except Exception as e:
                print(f'[ERROR] {h5}: {e}')
                fail += 1
        print(f'完成：成功 {ok} 失败 {fail} 输出目录 {args.out_dir}')
        return
    ap.print_help()

if __name__ == '__main__':
    main()



"""
单个 episode:
python pika_test_scripts/visualize_episode_full.py --episode pika_raw_data/episode0/data.hdf5 --out pika-videos/episode0_full.mp4
批量处理:
python pika_test_scripts/visualize_episode_full.py --episodes-glob 'pika_raw_data/episode*/data.hdf5' --out-dir pika-videos/full

"""