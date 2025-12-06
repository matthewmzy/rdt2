#!/usr/bin/env python3
"""
Debug 脚本：检验 npz 中保存的轨迹与从 delta 重建的轨迹是否一致

问题描述：
用户说从 npz 中读取的 gt_trajectory 计算相邻两帧的 delta 然后 replay 不一致。

可能的原因：
1. 保存的是绝对位姿，但用户按照相对动作的方式计算 delta
2. 保存的绝对位姿已经累积了旋转，相邻帧的差不是简单的减法
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


def mat_to_rot6d(mat):
    """将旋转矩阵转换为 6D 旋转表示"""
    col0 = mat[..., :, 0]
    col1 = mat[..., :, 1]
    out = np.concatenate((col0, col1), axis=-1)
    return out


def pose10d_to_mat(pose10d):
    """10D pose 转 4x4 变换矩阵"""
    pos = pose10d[..., :3]
    rot6d = pose10d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def mat_to_pose10d(mat, gripper):
    """4x4 变换矩阵转 10D pose"""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    
    gripper = np.atleast_1d(np.asarray(gripper))
    if gripper.ndim == pos.ndim - 1 or gripper.ndim == 0:
        gripper = gripper.reshape(pos.shape[:-1] + (1,))
    
    return np.concatenate([pos, rot6d, gripper], axis=-1)


def compute_delta_from_trajectory_naive(trajectory: np.ndarray) -> np.ndarray:
    """
    错误方式：直接用减法计算相邻帧的 delta
    
    这对于位置可能大致正确，但对于旋转是完全错误的！
    """
    return trajectory[1:] - trajectory[:-1]


def compute_delta_from_trajectory_correct(trajectory: np.ndarray) -> np.ndarray:
    """
    正确方式：计算相对变换 T_rel = T_i^{-1} @ T_{i+1}
    
    Args:
        trajectory: (N, 20) 绝对位姿序列，每帧 [right(10), left(10)]
    Returns:
        deltas: (N-1, 20) 相对动作序列
    """
    deltas = []
    
    for i in range(len(trajectory) - 1):
        # 右臂
        right_curr = trajectory[i, :10]
        right_next = trajectory[i + 1, :10]
        
        mat_curr_r = pose10d_to_mat(right_curr[:9])
        mat_next_r = pose10d_to_mat(right_next[:9])
        
        # T_rel = T_curr^{-1} @ T_next
        mat_rel_r = np.linalg.inv(mat_curr_r) @ mat_next_r
        delta_right = mat_to_pose10d(mat_rel_r, right_next[9])
        
        # 左臂
        left_curr = trajectory[i, 10:]
        left_next = trajectory[i + 1, 10:]
        
        mat_curr_l = pose10d_to_mat(left_curr[:9])
        mat_next_l = pose10d_to_mat(left_next[:9])
        
        mat_rel_l = np.linalg.inv(mat_curr_l) @ mat_next_l
        delta_left = mat_to_pose10d(mat_rel_l, left_next[9])
        
        deltas.append(np.concatenate([delta_right, delta_left], axis=-1))
    
    return np.stack(deltas, axis=0)


def replay_from_deltas(deltas: np.ndarray, start_pose: np.ndarray = None) -> np.ndarray:
    """
    从 delta 序列重建轨迹
    
    Args:
        deltas: (N-1, 20) 相对动作序列
        start_pose: (20,) 起始位姿，默认为原点
    Returns:
        trajectory: (N, 20) 绝对位姿序列
    """
    if start_pose is None:
        # 默认起始位姿为单位矩阵
        start_pose = np.zeros(20, dtype=np.float64)
        start_pose[3:9] = [1, 0, 0, 0, 1, 0]  # identity rotation (first two columns of I)
        start_pose[13:19] = [1, 0, 0, 0, 1, 0]
    
    trajectory = [start_pose]
    
    mat_right = pose10d_to_mat(start_pose[:9])
    mat_left = pose10d_to_mat(start_pose[10:19])
    
    for i in range(len(deltas)):
        delta = deltas[i]
        
        # 右臂
        delta_mat_r = pose10d_to_mat(delta[:9])
        mat_right = mat_right @ delta_mat_r
        pose_right = mat_to_pose10d(mat_right, delta[9])
        
        # 左臂
        delta_mat_l = pose10d_to_mat(delta[10:19])
        mat_left = mat_left @ delta_mat_l
        pose_left = mat_to_pose10d(mat_left, delta[19])
        
        trajectory.append(np.concatenate([pose_right, pose_left], axis=-1))
    
    return np.stack(trajectory, axis=0)


def load_episode_actions(shard_dir: Path, episode_name: str) -> dict:
    """从 shard 加载指定 episode 的所有动作"""
    samples = {}
    
    shard_files = sorted(shard_dir.glob('shard-*.tar'))
    
    for shard_path in shard_files:
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            
            sample_indices = set()
            for m in members:
                if m.name.endswith('.meta.json'):
                    sample_idx = int(m.name.split('.')[0])
                    sample_indices.add(sample_idx)
            
            for sample_idx in sample_indices:
                try:
                    meta_file = tar.extractfile(f'{sample_idx}.meta.json')
                    meta = json.load(meta_file)
                    
                    if meta['episode'] != episode_name:
                        continue
                    
                    frame_idx = meta['frame']
                    
                    action_file = tar.extractfile(f'{sample_idx}.action.npy')
                    action = np.load(io.BytesIO(action_file.read()))
                    
                    samples[frame_idx] = action  # (24, 20)
                except Exception:
                    continue
    
    return samples


def main():
    # 找最新的 npz 文件
    output_dir = Path(PROJECT_ROOT / 'inference_outputs_fm_full')
    npz_files = sorted(output_dir.glob('full_episode_*.npz'))
    
    if not npz_files:
        print("No npz files found in inference_outputs_fm_full/")
        print("Please run inference_offline_fm_full.py first")
        return
    
    npz_path = npz_files[-1]
    print(f"Loading: {npz_path}")
    
    data = np.load(npz_path, allow_pickle=True)
    gt_trajectory = data['gt_trajectory']  # (N, 20)
    pred_trajectory = data['pred_trajectory']  # (N, 20)
    frame_indices = data['frame_indices']
    
    print(f"GT trajectory shape: {gt_trajectory.shape}")
    print(f"Frame indices: {frame_indices[0]} - {frame_indices[-1]}")
    
    # ================================================================
    # 检验1: 从 npz 的 gt_trajectory 计算 delta 并重建
    # ================================================================
    print("\n" + "=" * 70)
    print("检验1: 从 gt_trajectory 计算 delta 并重建")
    print("=" * 70)
    
    # 用正确方式计算 delta
    deltas_correct = compute_delta_from_trajectory_correct(gt_trajectory)
    print(f"计算得到 {len(deltas_correct)} 个 delta")
    
    # 从第一帧重建
    rebuilt_traj = replay_from_deltas(deltas_correct, gt_trajectory[0])
    
    # 比较重建轨迹和原轨迹
    diff = rebuilt_traj - gt_trajectory
    pos_error_right = np.linalg.norm(diff[:, :3], axis=1)
    pos_error_left = np.linalg.norm(diff[:, 10:13], axis=1)
    
    print(f"\n用正确方式 (T_i^{{-1}} @ T_{{i+1}}) 重建:")
    print(f"  右臂位置误差: max={pos_error_right.max():.10f}, mean={pos_error_right.mean():.10f}")
    print(f"  左臂位置误差: max={pos_error_left.max():.10f}, mean={pos_error_left.mean():.10f}")
    
    # 用错误方式（直接减法）计算 delta
    deltas_naive = compute_delta_from_trajectory_naive(gt_trajectory)
    
    # 错误的重建：直接累加
    rebuilt_naive = np.zeros_like(gt_trajectory)
    rebuilt_naive[0] = gt_trajectory[0]
    for i in range(len(deltas_naive)):
        rebuilt_naive[i + 1] = rebuilt_naive[i] + deltas_naive[i]
    
    diff_naive = rebuilt_naive - gt_trajectory
    pos_error_naive_right = np.linalg.norm(diff_naive[:, :3], axis=1)
    pos_error_naive_left = np.linalg.norm(diff_naive[:, 10:13], axis=1)
    
    print(f"\n用错误方式 (直接减法再累加) 重建:")
    print(f"  右臂位置误差: max={pos_error_naive_right.max():.6f}, mean={pos_error_naive_right.mean():.6f}")
    print(f"  左臂位置误差: max={pos_error_naive_left.max():.6f}, mean={pos_error_naive_left.mean():.6f}")
    
    # ================================================================
    # 检验2: 比较 npz 中的 delta 和 tar 中的 action[0]
    # ================================================================
    print("\n" + "=" * 70)
    print("检验2: 比较 npz 中相邻帧的 delta 和 tar 中的 action[0]")
    print("=" * 70)
    
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    actions = load_episode_actions(shard_dir, 'episode0')
    
    if actions:
        # 对比每一帧
        errors = []
        for i in range(len(frame_indices) - 1):
            curr_frame = frame_indices[i]
            next_frame = frame_indices[i + 1]
            
            # 只有连续帧才能比较
            if next_frame != curr_frame + 1:
                continue
            
            if curr_frame not in actions:
                continue
            
            # npz 中计算的 delta
            delta_npz = deltas_correct[i]
            
            # tar 中的 action[0] (下一帧相对于当前帧)
            action_tar = actions[curr_frame][0]  # (20,)
            
            error = np.abs(delta_npz - action_tar).max()
            errors.append((curr_frame, error, delta_npz[:3], action_tar[:3]))
        
        if errors:
            max_err = max(e[1] for e in errors)
            mean_err = np.mean([e[1] for e in errors])
            print(f"比较了 {len(errors)} 对连续帧")
            print(f"最大误差: {max_err:.6f}")
            print(f"平均误差: {mean_err:.6f}")
            
            if max_err > 0.001:
                print("\n前 5 个误差最大的帧:")
                sorted_errors = sorted(errors, key=lambda x: x[1], reverse=True)
                for frame, err, delta_pos, tar_pos in sorted_errors[:5]:
                    print(f"  Frame {frame}: error={err:.6f}")
                    print(f"    npz delta pos: {delta_pos}")
                    print(f"    tar action pos: {tar_pos}")
    
    # ================================================================
    # 检验3: 检查 npz gt_trajectory 的构建过程
    # ================================================================
    print("\n" + "=" * 70)
    print("检验3: 重新从 tar 构建 gt_trajectory 并与 npz 比较")
    print("=" * 70)
    
    # 重新按照 inference_offline_fm_full.py 的逻辑构建
    step_size = 12
    action_horizon = 24
    
    gt_rebuilt_list = []
    gt_rebuilt_frames = []
    
    base_mat_right = np.eye(4, dtype=np.float64)
    base_mat_left = np.eye(4, dtype=np.float64)
    
    sorted_frames = sorted(actions.keys())
    max_frame = max(sorted_frames)
    
    current_frame = 0
    while current_frame <= max_frame - action_horizon:
        if current_frame not in actions:
            current_frame += step_size
            continue
        
        action_gt = actions[current_frame]
        actual_steps = min(step_size, action_horizon)
        
        for t in range(actual_steps):
            target_frame = current_frame + t + 1
            if target_frame > max_frame:
                break
            
            gt_right = action_gt[t, :10]
            gt_left = action_gt[t, 10:]
            
            # 计算绝对位姿
            delta_mat_r = pose10d_to_mat(gt_right[:9])
            delta_mat_l = pose10d_to_mat(gt_left[:9])
            
            gt_mat_right = base_mat_right @ delta_mat_r
            gt_mat_left = base_mat_left @ delta_mat_l
            
            gt_pose_right = mat_to_pose10d(gt_mat_right, gt_right[9])
            gt_pose_left = mat_to_pose10d(gt_mat_left, gt_left[9])
            
            gt_rebuilt_list.append(np.concatenate([gt_pose_right, gt_pose_left]))
            gt_rebuilt_frames.append(target_frame)
        
        # 更新 base
        gt_rel_mat_right = pose10d_to_mat(action_gt[step_size - 1, :9])
        gt_rel_mat_left = pose10d_to_mat(action_gt[step_size - 1, 10:19])
        
        base_mat_right = base_mat_right @ gt_rel_mat_right
        base_mat_left = base_mat_left @ gt_rel_mat_left
        
        current_frame += step_size
    
    gt_rebuilt = np.stack(gt_rebuilt_list, axis=0)
    
    # 找共同帧比较
    npz_frame_to_idx = {f: i for i, f in enumerate(frame_indices)}
    common_frames = [f for f in gt_rebuilt_frames if f in npz_frame_to_idx]
    
    if common_frames:
        rebuilt_idx = [gt_rebuilt_frames.index(f) for f in common_frames]
        npz_idx = [npz_frame_to_idx[f] for f in common_frames]
        
        gt_from_rebuilt = gt_rebuilt[rebuilt_idx]
        gt_from_npz = gt_trajectory[npz_idx]
        
        diff = gt_from_rebuilt - gt_from_npz
        pos_err_r = np.linalg.norm(diff[:, :3], axis=1)
        pos_err_l = np.linalg.norm(diff[:, 10:13], axis=1)
        
        print(f"比较了 {len(common_frames)} 帧")
        print(f"右臂位置误差: max={pos_err_r.max():.10f}, mean={pos_err_r.mean():.10f}")
        print(f"左臂位置误差: max={pos_err_l.max():.10f}, mean={pos_err_l.mean():.10f}")
        
        if pos_err_r.max() > 0.001 or pos_err_l.max() > 0.001:
            print("\n有显著误差！打印前几帧:")
            for i in range(min(5, len(common_frames))):
                frame = common_frames[i]
                print(f"  Frame {frame}:")
                print(f"    重建: {gt_from_rebuilt[i, :3]}")
                print(f"    npz:  {gt_from_npz[i, :3]}")


if __name__ == '__main__':
    main()
