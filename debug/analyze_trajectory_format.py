#!/usr/bin/env python3
"""
Debug 脚本：深入分析轨迹的坐标系

检查 gt_trajectory 存储的是什么：
1. 绝对位姿（全局坐标系）？
2. 相对于第一帧的累积位姿？
3. 还是其他格式？
"""

import os
import sys
import io
import json
import tarfile
from pathlib import Path
from glob import glob

import numpy as np

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rot6d_to_mat(d6):
    """将 6D 旋转表示转换为旋转矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    out = np.stack((b1, b2, b3), axis=-1)
    return out


def pose10d_to_mat(pose9d):
    """9D pose 转 4x4 变换矩阵"""
    pos = pose9d[..., :3]
    rot6d = pose9d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def main():
    # 加载 npz
    output_dir = Path(PROJECT_ROOT / 'inference_outputs_fm_full')
    npz_files = sorted(output_dir.glob('full_episode_*.npz'))
    
    if not npz_files:
        print("No npz files found")
        return
    
    npz_path = npz_files[-1]
    print(f"Loading: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    gt_trajectory = data['gt_trajectory']
    pred_trajectory = data['pred_trajectory']
    frame_indices = data['frame_indices']
    
    print(f"\nGT trajectory shape: {gt_trajectory.shape}")
    print(f"Frame indices: {frame_indices[0]} - {frame_indices[-1]}")
    
    # ================================================================
    # 分析 gt_trajectory 的第一帧
    # ================================================================
    print("\n" + "=" * 70)
    print("分析 gt_trajectory 的前几帧")
    print("=" * 70)
    
    for i in range(min(5, len(gt_trajectory))):
        frame = frame_indices[i]
        pose = gt_trajectory[i]
        
        print(f"\nFrame {frame}:")
        print(f"  右臂 position: [{pose[0]:.6f}, {pose[1]:.6f}, {pose[2]:.6f}]")
        print(f"  右臂 rot6d[:3]: [{pose[3]:.6f}, {pose[4]:.6f}, {pose[5]:.6f}]")
        print(f"  右臂 rot6d[3:]: [{pose[6]:.6f}, {pose[7]:.6f}, {pose[8]:.6f}]")
        print(f"  右臂 gripper:   {pose[9]:.6f}")
        
        # 检查旋转是否接近单位矩阵
        rot6d_r = pose[3:9]
        rotmat_r = rot6d_to_mat(rot6d_r)
        identity_err = np.linalg.norm(rotmat_r - np.eye(3))
        print(f"  右臂 旋转与单位矩阵的误差: {identity_err:.6f}")
    
    # ================================================================
    # 检查相邻帧的位置差异
    # ================================================================
    print("\n" + "=" * 70)
    print("相邻帧的位置变化")
    print("=" * 70)
    
    print("\n右臂 position 变化 (直接减法):")
    for i in range(min(5, len(gt_trajectory) - 1)):
        pos_diff = gt_trajectory[i + 1, :3] - gt_trajectory[i, :3]
        print(f"  Frame {frame_indices[i]} -> {frame_indices[i+1]}: "
              f"[{pos_diff[0]:.6f}, {pos_diff[1]:.6f}, {pos_diff[2]:.6f}]")
    
    # ================================================================
    # 加载 tar 数据，检查 action[0] 的格式
    # ================================================================
    print("\n" + "=" * 70)
    print("从 tar 加载 action[0] 进行对比")
    print("=" * 70)
    
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    
    # 加载 frame 0 的 action
    with tarfile.open(shard_dir / 'shard-000000.tar', 'r') as tar:
        for sample_idx in range(100):
            try:
                meta_file = tar.extractfile(f'{sample_idx}.meta.json')
                meta = json.load(meta_file)
                
                if meta['episode'] == 'episode0' and meta['frame'] == 0:
                    action_file = tar.extractfile(f'{sample_idx}.action.npy')
                    action = np.load(io.BytesIO(action_file.read()))
                    
                    print(f"\nFrame 0 的 action (shape: {action.shape}):")
                    print(f"  action[0] 右臂 pos: [{action[0, 0]:.6f}, {action[0, 1]:.6f}, {action[0, 2]:.6f}]")
                    print(f"  action[0] 右臂 rot6d[:3]: [{action[0, 3]:.6f}, {action[0, 4]:.6f}, {action[0, 5]:.6f}]")
                    print(f"  action[0] 右臂 rot6d[3:]: [{action[0, 6]:.6f}, {action[0, 7]:.6f}, {action[0, 8]:.6f}]")
                    
                    # action[0] 应该等于 gt_trajectory[0]（frame 1 相对于 frame 0）
                    # 但 gt_trajectory[0] 是 frame 1 的绝对位姿
                    # 如果 frame 0 的位姿是单位矩阵，那么应该相等
                    print(f"\n  gt_trajectory[0] 右臂 pos: [{gt_trajectory[0, 0]:.6f}, {gt_trajectory[0, 1]:.6f}, {gt_trajectory[0, 2]:.6f}]")
                    
                    # 比较
                    diff = action[0, :10] - gt_trajectory[0, :10]
                    print(f"\n  action[0] 与 gt_trajectory[0] 的差异:")
                    print(f"    位置差: [{diff[0]:.10f}, {diff[1]:.10f}, {diff[2]:.10f}]")
                    print(f"    最大差: {np.abs(diff).max():.10f}")
                    break
            except Exception:
                continue
    
    # ================================================================
    # 验证：gt_trajectory 是否是相对于 frame 0 的绝对位姿
    # ================================================================
    print("\n" + "=" * 70)
    print("验证：gt_trajectory[i] = T_0 @ action[0]_{0} @ action[0]_{1} @ ... @ action[0]_{i-1}")
    print("即：gt_trajectory[i] 是从第 0 帧累积相对变换得到的")
    print("=" * 70)
    
    # 从 tar 加载 action 数据
    actions = {}
    for shard_path in sorted(shard_dir.glob('shard-*.tar')):
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            sample_indices = set()
            for m in members:
                if m.name.endswith('.meta.json'):
                    sample_indices.add(int(m.name.split('.')[0]))
            
            for sample_idx in sample_indices:
                try:
                    meta_file = tar.extractfile(f'{sample_idx}.meta.json')
                    meta = json.load(meta_file)
                    
                    if meta['episode'] != 'episode0':
                        continue
                    
                    frame_idx = meta['frame']
                    action_file = tar.extractfile(f'{sample_idx}.action.npy')
                    action = np.load(io.BytesIO(action_file.read()))
                    actions[frame_idx] = action
                except Exception:
                    continue
    
    print(f"\n加载了 {len(actions)} 帧的 action")
    
    # 从 frame 0 开始，只用 action[0] 累积
    cumulative_mat_r = np.eye(4, dtype=np.float64)
    cumulative_mat_l = np.eye(4, dtype=np.float64)
    
    print("\n验证前 10 帧:")
    for target_frame in range(1, 11):
        src_frame = target_frame - 1
        if src_frame not in actions:
            print(f"  Frame {src_frame} 不存在")
            continue
        
        # 用 action[0] 更新
        action = actions[src_frame][0]  # (20,)
        delta_mat_r = pose10d_to_mat(action[:9])
        delta_mat_l = pose10d_to_mat(action[10:19])
        
        cumulative_mat_r = cumulative_mat_r @ delta_mat_r
        cumulative_mat_l = cumulative_mat_l @ delta_mat_l
        
        # 提取位置
        pos_r = cumulative_mat_r[:3, 3]
        pos_l = cumulative_mat_l[:3, 3]
        
        # 与 gt_trajectory 比较
        if target_frame in frame_indices:
            idx = list(frame_indices).index(target_frame)
            gt_pos_r = gt_trajectory[idx, :3]
            gt_pos_l = gt_trajectory[idx, 10:13]
            
            err_r = np.linalg.norm(pos_r - gt_pos_r)
            err_l = np.linalg.norm(pos_l - gt_pos_l)
            
            print(f"  Frame {target_frame}: 累积pos_r={pos_r}, gt_pos_r={gt_pos_r}, err={err_r:.10f}")
    
    # ================================================================
    # 核心问题检查：step_size > 1 时的累积
    # ================================================================
    print("\n" + "=" * 70)
    print("核心检查：step_size=12 累积 vs 逐帧 action[0] 累积")
    print("=" * 70)
    
    # 方法A: inference_offline_fm_full.py 的方式
    # 每 12 帧一个 window，用 action[0:12] 累积
    step_size = 12
    action_horizon = 24
    
    traj_A = []
    frames_A = []
    base_mat_r = np.eye(4, dtype=np.float64)
    base_mat_l = np.eye(4, dtype=np.float64)
    
    sorted_frames = sorted(actions.keys())
    max_frame = max(sorted_frames)
    current_frame = 0
    
    while current_frame <= max_frame - action_horizon:
        if current_frame not in actions:
            current_frame += step_size
            continue
        
        action_gt = actions[current_frame]
        
        for t in range(step_size):
            target_frame = current_frame + t + 1
            if target_frame > max_frame:
                break
            
            # 关键：这里用 action_gt[t]，它是 frame (current_frame + t + 1) 相对于 current_frame
            delta_mat_r = pose10d_to_mat(action_gt[t, :9])
            delta_mat_l = pose10d_to_mat(action_gt[t, 10:19])
            
            # T_{target} = T_{base} @ T_{rel}
            # 但 T_{rel} = T_{current}^{-1} @ T_{target}
            # 所以 T_{target} = T_{base} @ T_{rel}
            # 注意：这里 T_{base} 是 current_frame 的绝对位姿
            abs_mat_r = base_mat_r @ delta_mat_r
            abs_mat_l = base_mat_l @ delta_mat_l
            
            traj_A.append({
                'frame': target_frame,
                'pos_r': abs_mat_r[:3, 3].copy(),
                'pos_l': abs_mat_l[:3, 3].copy(),
            })
            frames_A.append(target_frame)
        
        # 更新 base：用 action_gt[step_size-1]
        # 这是 frame (current_frame + step_size) 相对于 current_frame
        gt_rel_mat_r = pose10d_to_mat(action_gt[step_size - 1, :9])
        gt_rel_mat_l = pose10d_to_mat(action_gt[step_size - 1, 10:19])
        
        base_mat_r = base_mat_r @ gt_rel_mat_r
        base_mat_l = base_mat_l @ gt_rel_mat_l
        
        current_frame += step_size
    
    # 方法B: 逐帧 action[0] 累积
    traj_B = []
    frames_B = []
    cum_mat_r = np.eye(4, dtype=np.float64)
    cum_mat_l = np.eye(4, dtype=np.float64)
    
    for frame_idx in sorted_frames:
        if frame_idx not in actions:
            continue
        
        action = actions[frame_idx][0]
        delta_mat_r = pose10d_to_mat(action[:9])
        delta_mat_l = pose10d_to_mat(action[10:19])
        
        cum_mat_r = cum_mat_r @ delta_mat_r
        cum_mat_l = cum_mat_l @ delta_mat_l
        
        target_frame = frame_idx + 1
        traj_B.append({
            'frame': target_frame,
            'pos_r': cum_mat_r[:3, 3].copy(),
            'pos_l': cum_mat_l[:3, 3].copy(),
        })
        frames_B.append(target_frame)
    
    # 比较
    common = set(frames_A) & set(frames_B)
    print(f"方法A (step_size累积) 生成 {len(frames_A)} 帧")
    print(f"方法B (逐帧action[0]) 生成 {len(frames_B)} 帧")
    print(f"共同帧: {len(common)}")
    
    if common:
        common = sorted(common)
        A_dict = {t['frame']: t for t in traj_A}
        B_dict = {t['frame']: t for t in traj_B}
        
        errors_r = []
        errors_l = []
        
        for f in common:
            err_r = np.linalg.norm(A_dict[f]['pos_r'] - B_dict[f]['pos_r'])
            err_l = np.linalg.norm(A_dict[f]['pos_l'] - B_dict[f]['pos_l'])
            errors_r.append(err_r)
            errors_l.append(err_l)
        
        print(f"\n右臂位置误差: max={max(errors_r):.10f}, mean={np.mean(errors_r):.10f}")
        print(f"左臂位置误差: max={max(errors_l):.10f}, mean={np.mean(errors_l):.10f}")
        
        # 打印特定帧的对比
        check_frames = [12, 13, 24, 25]
        print("\n特定帧对比:")
        for f in check_frames:
            if f in common:
                print(f"  Frame {f}:")
                print(f"    方法A: {A_dict[f]['pos_r']}")
                print(f"    方法B: {B_dict[f]['pos_r']}")
                print(f"    误差: {np.linalg.norm(A_dict[f]['pos_r'] - B_dict[f]['pos_r']):.10f}")


if __name__ == '__main__':
    main()
