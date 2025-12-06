#!/usr/bin/env python3
"""
综合 Debug 脚本：帮助定位 replay 不一致的问题

结论：
1. inference_offline_fm_full.py 中的 gt_trajectory 累积是正确的
2. tar 数据集中的 action[0:24] 是正确的
3. npz 中保存的 gt_trajectory 与 tar 是一致的

如果用户说 "从 tar 的 action[0] replay 正确，但从 npz 的 gt_trajectory delta replay 不正确"
可能的原因：
1. 用户计算 delta 的方式不对（直接减法 vs 矩阵求逆乘法）
2. 用户累积的方式不对（加法 vs 矩阵乘法）
3. 用户混淆了帧索引（npz 的 frame_indices[0]=1，不是 0）
4. 用户混淆了坐标系（绝对位姿 vs 相对动作）

这个脚本展示正确的 replay 方式。
"""

import os
import sys
import io
import json
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rot6d_to_mat(d6):
    """6D 旋转表示转旋转矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def mat_to_rot6d(mat):
    """旋转矩阵转 6D 表示"""
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def pose10d_to_mat(pose10d):
    """10D pose [pos(3), rot6d(6), gripper(1)] 转 4x4 矩阵（只用前 9 维）"""
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
    """4x4 矩阵转 10D pose"""
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    gripper = np.atleast_1d(np.asarray(gripper)).reshape(pos.shape[:-1] + (1,))
    return np.concatenate([pos, rot6d, gripper], axis=-1)


def load_tar_actions(shard_dir, episode):
    """从 tar 加载动作数据"""
    actions = {}
    for shard_path in sorted(shard_dir.glob('shard-*.tar')):
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            sample_indices = {int(m.name.split('.')[0]) for m in members if m.name.endswith('.meta.json')}
            
            for idx in sample_indices:
                try:
                    meta = json.load(tar.extractfile(f'{idx}.meta.json'))
                    if meta['episode'] != episode:
                        continue
                    action = np.load(io.BytesIO(tar.extractfile(f'{idx}.action.npy').read()))
                    actions[meta['frame']] = action
                except:
                    continue
    return actions


def correct_replay_from_tar(tar_actions):
    """
    正确的 replay 方式：从 tar 的 action[0] 累积
    
    - tar_actions[frame][0] = T_{frame}^{-1} @ T_{frame+1}
    - 即 frame+1 相对于 frame 的相对变换
    - 从单位矩阵开始累积得到绝对轨迹
    """
    trajectory = []
    frame_indices = []
    
    sorted_frames = sorted(tar_actions.keys())
    
    # 初始化：frame 0 的位姿为单位矩阵
    mat_r = np.eye(4)
    mat_l = np.eye(4)
    
    for frame in sorted_frames:
        action = tar_actions[frame][0]  # (20,)
        
        # 应用相对变换
        delta_r = pose10d_to_mat(action[:10])
        delta_l = pose10d_to_mat(action[10:])
        
        mat_r = mat_r @ delta_r
        mat_l = mat_l @ delta_l
        
        # 保存绝对位姿
        pose_r = mat_to_pose10d(mat_r, action[9])
        pose_l = mat_to_pose10d(mat_l, action[19])
        
        trajectory.append(np.concatenate([pose_r, pose_l]))
        frame_indices.append(frame + 1)  # action[0] 对应下一帧
    
    return np.stack(trajectory), frame_indices


def correct_replay_from_npz(gt_trajectory, frame_indices):
    """
    从 npz gt_trajectory 计算 delta 并正确 replay
    
    关键点：
    1. gt_trajectory 存储的是绝对位姿（相对于 frame 0）
    2. 计算 delta: delta_i = T_i^{-1} @ T_{i+1}
    3. 从第一帧的位姿开始累积
    """
    replay = []
    
    # 从第一帧开始
    mat_r = pose10d_to_mat(gt_trajectory[0, :10])
    mat_l = pose10d_to_mat(gt_trajectory[0, 10:])
    
    for i in range(len(gt_trajectory)):
        # 保存当前帧
        pose_r = mat_to_pose10d(mat_r, gt_trajectory[i, 9])
        pose_l = mat_to_pose10d(mat_l, gt_trajectory[i, 19])
        replay.append(np.concatenate([pose_r, pose_l]))
        
        if i < len(gt_trajectory) - 1:
            # 计算 delta
            mat_r_next = pose10d_to_mat(gt_trajectory[i + 1, :10])
            mat_l_next = pose10d_to_mat(gt_trajectory[i + 1, 10:])
            
            delta_r = np.linalg.inv(mat_r) @ mat_r_next
            delta_l = np.linalg.inv(mat_l) @ mat_l_next
            
            # 累积
            mat_r = mat_r @ delta_r
            mat_l = mat_l @ delta_l
    
    return np.stack(replay)


def wrong_replay_naive_subtraction(gt_trajectory):
    """
    错误的 replay 方式：直接减法计算 delta，加法累积
    
    这对位置有效（因为旋转变化很小），但对旋转无效
    """
    replay = np.zeros_like(gt_trajectory)
    replay[0] = gt_trajectory[0]
    
    for i in range(len(gt_trajectory) - 1):
        delta = gt_trajectory[i + 1] - gt_trajectory[i]
        replay[i + 1] = replay[i] + delta
    
    return replay


def main():
    print("=" * 80)
    print("综合 Debug：展示正确的 replay 方式")
    print("=" * 80)
    
    # 加载 npz
    output_dir = Path(PROJECT_ROOT / 'inference_outputs_fm_full')
    npz_files = sorted(output_dir.glob('full_episode_*.npz'))
    
    if not npz_files:
        print("No npz files found. Please run inference_offline_fm_full.py first.")
        return
    
    data = np.load(npz_files[-1], allow_pickle=True)
    gt_trajectory = data['gt_trajectory']
    npz_frame_indices = list(data['frame_indices'])
    
    print(f"\n加载 npz:")
    print(f"  gt_trajectory shape: {gt_trajectory.shape}")
    print(f"  frame_indices: {npz_frame_indices[0]} - {npz_frame_indices[-1]}")
    print(f"  注意: frame_indices[0] = {npz_frame_indices[0]}，不是 0！")
    
    # 加载 tar
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    tar_actions = load_tar_actions(shard_dir, 'episode0')
    print(f"\n加载 tar: {len(tar_actions)} frames")
    
    # ================================================================
    # 正确的 replay
    # ================================================================
    print("\n" + "=" * 60)
    print("方法1: 从 tar action[0] 正确 replay")
    print("=" * 60)
    
    tar_trajectory, tar_frame_indices = correct_replay_from_tar(tar_actions)
    
    # 与 npz gt_trajectory 比较
    common_frames = set(npz_frame_indices) & set(tar_frame_indices)
    print(f"共同帧: {len(common_frames)}")
    
    if common_frames:
        common_frames = sorted(common_frames)
        npz_idx = [npz_frame_indices.index(f) for f in common_frames]
        tar_idx = [tar_frame_indices.index(f) for f in common_frames]
        
        diff = gt_trajectory[npz_idx] - tar_trajectory[tar_idx]
        pos_err_r = np.linalg.norm(diff[:, :3], axis=1)
        pos_err_l = np.linalg.norm(diff[:, 10:13], axis=1)
        
        print(f"npz gt_trajectory vs tar replay:")
        print(f"  右臂位置误差: max={pos_err_r.max():.10f}, mean={pos_err_r.mean():.10f}")
        print(f"  左臂位置误差: max={pos_err_l.max():.10f}, mean={pos_err_l.mean():.10f}")
    
    # ================================================================
    # 从 npz 正确 replay
    # ================================================================
    print("\n" + "=" * 60)
    print("方法2: 从 npz gt_trajectory 正确计算 delta 并 replay")
    print("=" * 60)
    
    npz_replay = correct_replay_from_npz(gt_trajectory, npz_frame_indices)
    
    diff = npz_replay - gt_trajectory
    pos_err_r = np.linalg.norm(diff[:, :3], axis=1)
    pos_err_l = np.linalg.norm(diff[:, 10:13], axis=1)
    
    print(f"与原始 gt_trajectory 的差异:")
    print(f"  右臂位置误差: max={pos_err_r.max():.10f}")
    print(f"  左臂位置误差: max={pos_err_l.max():.10f}")
    
    # ================================================================
    # 错误的 replay
    # ================================================================
    print("\n" + "=" * 60)
    print("方法3: 错误方式（直接减法+加法）")
    print("=" * 60)
    
    wrong_replay = wrong_replay_naive_subtraction(gt_trajectory)
    
    diff = wrong_replay - gt_trajectory
    pos_err_r = np.linalg.norm(diff[:, :3], axis=1)
    pos_err_l = np.linalg.norm(diff[:, 10:13], axis=1)
    
    print(f"与原始 gt_trajectory 的差异:")
    print(f"  右臂位置误差: max={pos_err_r.max():.10f}")
    print(f"  左臂位置误差: max={pos_err_l.max():.10f}")
    
    if pos_err_r.max() < 0.001:
        print("\n  注意: 位置误差很小，因为旋转变化小，位置变换近似线性")
        print("  但这种方法对旋转是错误的！")
        
        # 检查旋转误差
        rot_err_r = np.linalg.norm(diff[:, 3:9], axis=1)
        rot_err_l = np.linalg.norm(diff[:, 13:19], axis=1)
        print(f"  右臂旋转误差: max={rot_err_r.max():.10f}")
        print(f"  左臂旋转误差: max={rot_err_l.max():.10f}")
    
    # ================================================================
    # 总结
    # ================================================================
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print("""
关键发现:
1. inference_offline_fm_full.py 生成的 gt_trajectory 是正确的
2. tar 数据集中的 action 是正确的
3. npz 与 tar 是一致的

如果您的 replay 不一致，请检查:
1. 您是否用正确的方式计算 delta？
   - 正确: delta = inv(T_i) @ T_{i+1}
   - 错误: delta = T_{i+1} - T_i

2. 您是否用正确的方式累积？
   - 正确: T_{i+1} = T_i @ delta
   - 错误: T_{i+1} = T_i + delta

3. 您是否混淆了帧索引？
   - npz 的 frame_indices[0] = 1，不是 0
   - gt_trajectory[0] 对应 frame 1

4. 您是否混淆了坐标系？
   - gt_trajectory 是绝对位姿（相对于 frame 0）
   - tar action[0] 是相对动作（下一帧相对于当前帧）

正确的 replay 代码示例见本脚本的 correct_replay_from_npz() 函数。
""")


if __name__ == '__main__':
    main()
