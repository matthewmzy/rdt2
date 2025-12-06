#!/usr/bin/env python3
"""
Debug 脚本：验证 GT 轨迹累积的正确性

比较两种累积方式：
1. inference_offline_fm_full.py 的方式：step_size=12，每次用 action[0:12]
2. 逐帧累积方式：每帧只用 action[0]（下一帧的相对变换）

如果两种方式产生不同的轨迹，说明有 bug。
"""

import os
import sys
import io
import json
import tarfile
from pathlib import Path

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


def action_to_pose_mat(action_9d):
    """9D 动作转 4x4 变换矩阵"""
    pos = action_9d[..., :3]
    rot6d = action_9d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def pose_mat_to_action(mat, gripper):
    """4x4 变换矩阵转 10D 动作"""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    
    gripper = np.atleast_1d(np.asarray(gripper))
    if gripper.ndim == pos.ndim - 1 or gripper.ndim == 0:
        gripper = gripper.reshape(pos.shape[:-1] + (1,))
    
    return np.concatenate([pos, rot6d, gripper], axis=-1)


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


def method1_step_accumulation(actions: dict, step_size: int = 12) -> tuple:
    """
    方法1: 按 step_size 累积 (inference_offline_fm_full.py 的方式)
    
    每次迭代:
    - 从 current_frame 开始，取 action[0:step_size]
    - 累积到绝对轨迹
    - base_pose 用 action[step_size-1] 更新
    """
    frame_indices = sorted(actions.keys())
    max_frame = max(frame_indices)
    action_horizon = 24
    
    trajectory_list = []
    traj_frame_indices = []
    
    base_pose_mat_right = np.eye(4, dtype=np.float64)
    base_pose_mat_left = np.eye(4, dtype=np.float64)
    
    current_frame = 0
    
    while current_frame <= max_frame - action_horizon:
        if current_frame not in actions:
            current_frame += step_size
            continue
        
        action_gt = actions[current_frame]  # (24, 20)
        actual_steps = min(step_size, action_horizon)
        
        for t in range(actual_steps):
            target_frame = current_frame + t + 1
            
            if target_frame > max_frame:
                break
            
            gt_right = action_gt[t, :10]
            gt_left = action_gt[t, 10:]
            
            # 计算绝对位姿
            gt_mat_right = base_pose_mat_right @ action_to_pose_mat(gt_right[:9])
            gt_mat_left = base_pose_mat_left @ action_to_pose_mat(gt_left[:9])
            
            gt_pose_right = pose_mat_to_action(gt_mat_right, gt_right[9])
            gt_pose_left = pose_mat_to_action(gt_mat_left, gt_left[9])
            
            trajectory_list.append(np.concatenate([gt_pose_right, gt_pose_left], axis=-1))
            traj_frame_indices.append(target_frame)
        
        # 更新 base_pose
        if step_size <= action_horizon:
            gt_rel_mat_right = action_to_pose_mat(action_gt[step_size - 1, :9])
            gt_rel_mat_left = action_to_pose_mat(action_gt[step_size - 1, 10:19])
            
            base_pose_mat_right = base_pose_mat_right @ gt_rel_mat_right
            base_pose_mat_left = base_pose_mat_left @ gt_rel_mat_left
        
        current_frame += step_size
    
    if trajectory_list:
        trajectory = np.stack(trajectory_list, axis=0)
    else:
        trajectory = np.array([])
    
    return trajectory, traj_frame_indices


def method2_frame_by_frame(actions: dict) -> tuple:
    """
    方法2: 逐帧累积 (每帧只用 action[0])
    
    这应该是 "正确" 的方式，因为：
    - action[0] from frame i = T_i^{-1} @ T_{i+1}
    - 所以 T_{i+1} = T_i @ action[0]
    """
    frame_indices = sorted(actions.keys())
    
    trajectory_list = []
    traj_frame_indices = []
    
    base_pose_mat_right = np.eye(4, dtype=np.float64)
    base_pose_mat_left = np.eye(4, dtype=np.float64)
    
    for frame_idx in frame_indices:
        action = actions[frame_idx]  # (24, 20)
        
        # 只用 action[0]
        action_right = action[0, :10]
        action_left = action[0, 10:]
        
        # 计算下一帧的绝对位姿
        next_mat_right = base_pose_mat_right @ action_to_pose_mat(action_right[:9])
        next_mat_left = base_pose_mat_left @ action_to_pose_mat(action_left[:9])
        
        next_pose_right = pose_mat_to_action(next_mat_right, action_right[9])
        next_pose_left = pose_mat_to_action(next_mat_left, action_left[9])
        
        trajectory_list.append(np.concatenate([next_pose_right, next_pose_left], axis=-1))
        traj_frame_indices.append(frame_idx + 1)
        
        # 更新 base_pose 为下一帧
        base_pose_mat_right = next_mat_right
        base_pose_mat_left = next_mat_left
    
    if trajectory_list:
        trajectory = np.stack(trajectory_list, axis=0)
    else:
        trajectory = np.array([])
    
    return trajectory, traj_frame_indices


def method3_check_action_consistency(actions: dict) -> None:
    """
    方法3: 检查动作数据的一致性
    
    验证: action[t] from frame i 是否等于 action[0] from frame i+t (理论上应该不相等！)
    
    因为:
    - action[t] from frame i = T_i^{-1} @ T_{i+t+1}  (frame i+t+1 相对于 frame i)
    - action[0] from frame i+t = T_{i+t}^{-1} @ T_{i+t+1}  (frame i+t+1 相对于 frame i+t)
    
    这两个是不同的参考系！只有当 t=0 时才相等。
    """
    frame_indices = sorted(actions.keys())
    
    print("\n" + "=" * 70)
    print("检查动作数据一致性：action[t] from frame i vs action[0] from frame i+t")
    print("=" * 70)
    
    for t in [0, 1, 5, 11]:
        errors = []
        for frame_idx in frame_indices:
            target_frame = frame_idx + t
            if target_frame in actions:
                # action[t] from frame i
                action_t_from_i = actions[frame_idx][t]
                
                # action[0] from frame i+t
                action_0_from_it = actions[target_frame][0]
                
                error = np.abs(action_t_from_i - action_0_from_it).max()
                errors.append(error)
        
        if errors:
            print(f"  t={t}: max_error = {np.max(errors):.6f}, mean_error = {np.mean(errors):.6f}")
            if t == 0:
                assert np.max(errors) < 1e-6, f"action[0] should be consistent! max_error={np.max(errors)}"
                print(f"    ✓ t=0 一致 (action[0] from frame i == action[0] from frame i)")
            else:
                print(f"    ✗ t={t} 不一致，这是预期的行为（不同参考系）")


def main():
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    episode_name = 'episode0'
    step_size = 12
    
    print(f"Loading actions from {shard_dir} for episode '{episode_name}'...")
    actions = load_episode_actions(shard_dir, episode_name)
    print(f"Loaded {len(actions)} frames")
    
    if not actions:
        print("No actions loaded!")
        return
    
    # 方法3: 检查动作一致性
    method3_check_action_consistency(actions)
    
    # 方法1: step_size 累积
    print(f"\n方法1: step_size={step_size} 累积 (inference_offline_fm_full.py 方式)")
    traj1, frames1 = method1_step_accumulation(actions, step_size=step_size)
    print(f"  生成 {len(frames1)} 帧, frame range: {frames1[0]} - {frames1[-1]}")
    
    # 方法2: 逐帧累积
    print(f"\n方法2: 逐帧累积 (每帧只用 action[0])")
    traj2, frames2 = method2_frame_by_frame(actions)
    print(f"  生成 {len(frames2)} 帧, frame range: {frames2[0]} - {frames2[-1]}")
    
    # 比较两种方法
    print("\n" + "=" * 70)
    print("比较两种方法的 GT 轨迹")
    print("=" * 70)
    
    # 找共同的帧
    common_frames = set(frames1) & set(frames2)
    print(f"共同帧数: {len(common_frames)}")
    
    if common_frames:
        common_frames = sorted(common_frames)
        
        # 取方法1和方法2中对应的位置
        idx1 = [frames1.index(f) for f in common_frames]
        idx2 = [frames2.index(f) for f in common_frames]
        
        traj1_common = traj1[idx1]
        traj2_common = traj2[idx2]
        
        # 计算误差
        diff = traj1_common - traj2_common
        
        # Position error (前3维 + 第10-12维)
        pos_error_right = np.linalg.norm(diff[:, :3], axis=1)
        pos_error_left = np.linalg.norm(diff[:, 10:13], axis=1)
        
        print(f"\n右臂位置误差:")
        print(f"  Max: {pos_error_right.max():.6f}")
        print(f"  Mean: {pos_error_right.mean():.6f}")
        print(f"  第一帧: {pos_error_right[0]:.6f}")
        print(f"  最后帧: {pos_error_right[-1]:.6f}")
        
        print(f"\n左臂位置误差:")
        print(f"  Max: {pos_error_left.max():.6f}")
        print(f"  Mean: {pos_error_left.mean():.6f}")
        print(f"  第一帧: {pos_error_left[0]:.6f}")
        print(f"  最后帧: {pos_error_left[-1]:.6f}")
        
        # 打印前几帧的详细对比
        print("\n前 5 个共同帧的对比 (右臂 xyz):")
        for i in range(min(5, len(common_frames))):
            frame = common_frames[i]
            pos1 = traj1_common[i, :3]
            pos2 = traj2_common[i, :3]
            print(f"  Frame {frame}:")
            print(f"    方法1: [{pos1[0]:.6f}, {pos1[1]:.6f}, {pos1[2]:.6f}]")
            print(f"    方法2: [{pos2[0]:.6f}, {pos2[1]:.6f}, {pos2[2]:.6f}]")
            print(f"    差异:  [{pos1[0]-pos2[0]:.6f}, {pos1[1]-pos2[1]:.6f}, {pos1[2]-pos2[2]:.6f}]")
        
        # 检查是否在第一个 step_size 之后开始出现差异
        if len(common_frames) > step_size:
            print(f"\n在 frame {step_size} 附近的对比:")
            for i in range(max(0, step_size-2), min(len(common_frames), step_size+3)):
                if i < len(common_frames):
                    frame = common_frames[i]
                    pos1 = traj1_common[i, :3]
                    pos2 = traj2_common[i, :3]
                    err = np.linalg.norm(pos1 - pos2)
                    print(f"  Frame {frame}: error = {err:.6f}")
        
        # 分析误差出现的模式
        print("\n误差随帧变化:")
        for i in range(0, len(common_frames), step_size):
            if i < len(common_frames):
                frame = common_frames[i]
                err_right = pos_error_right[i]
                err_left = pos_error_left[i]
                print(f"  Frame {frame}: right_err = {err_right:.6f}, left_err = {err_left:.6f}")


if __name__ == '__main__':
    main()
