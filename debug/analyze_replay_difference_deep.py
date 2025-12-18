#!/usr/bin/env python3
"""
深入分析 tar replay 和 npz replay 不一致的原因

重点检查：
1. 单帧误差可能很小，但累积误差会不会放大？
2. 您实际代码中的 delta 计算方式和我们测试的是否完全一致？
"""

import os
import sys
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rot6d_to_mat(d6):
    """6D 旋转转旋转矩阵"""
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """将 6D 旋转表示转换为欧拉角 (roll, pitch, yaw)"""
    col0 = rot6d[:3]
    col1 = rot6d[3:6]
    col0 = col0 / (np.linalg.norm(col0) + 1e-8)
    col1 = col1 - np.dot(col1, col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)
    rotmat = np.stack([col0, col1, col2], axis=1)
    euler = R.from_matrix(rotmat).as_euler('xyz')
    return euler


def pose10d_to_mat(pose10d):
    """10D pose 转 4x4 变换矩阵"""
    pos = pose10d[:3]
    rot6d = pose10d[3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rotmat
    mat[:3, 3] = pos
    return mat


def mat_to_pose7d(mat):
    """4x4 变换矩阵转 7D (xyz + rpy)"""
    pos = mat[:3, 3]
    rotmat = mat[:3, :3]
    rpy = R.from_matrix(rotmat).as_euler('xyz')
    return np.concatenate([pos, rpy])


def convert_action_to_rpy(action_10d: np.ndarray) -> np.ndarray:
    """将 10D 动作转换为 7D (xyz + rpy + gripper)"""
    pos = action_10d[:3]
    rot6d = action_10d[3:9]
    gripper = action_10d[9:10]
    rpy = rot6d_to_euler(rot6d)
    return np.concatenate([pos, rpy, gripper])


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


def replay_with_tar_delta(tar_actions, num_frames):
    """使用 tar 的 action[0] 进行 replay（这是正确的方式）"""
    # 从单位矩阵开始
    current_mat = np.eye(4, dtype=np.float64)
    
    trajectory = [current_mat.copy()]
    
    for frame in range(num_frames - 1):
        if frame not in tar_actions:
            break
        
        # action[0] 是当前帧到下一帧的相对变换
        delta_10d = tar_actions[frame][0, :10]  # 右臂
        delta_mat = pose10d_to_mat(delta_10d)
        
        # T_next = T_current @ delta
        current_mat = current_mat @ delta_mat
        trajectory.append(current_mat.copy())
    
    return np.array(trajectory)


def replay_with_npz_naive_delta(gt_trajectory):
    """使用简单减法进行 replay（您代码中的方式）"""
    # 模拟您的 get_full_rdt2_data 函数
    # 从单位矩阵开始
    current_pose_rpy = np.zeros(6)  # xyz + rpy
    
    trajectory_rpy = [current_pose_rpy.copy()]
    
    for i in range(len(gt_trajectory)):
        if i == 0:
            # 第一帧：gt_trajectory[0] 本身就是 delta
            delta_rpy = convert_action_to_rpy(gt_trajectory[0, :10])[:6]
        else:
            # 后续帧：简单减法
            pose_curr_rpy = convert_action_to_rpy(gt_trajectory[i-1, :10])[:6]
            pose_next_rpy = convert_action_to_rpy(gt_trajectory[i, :10])[:6]
            delta_rpy = pose_next_rpy - pose_curr_rpy
        
        # 累加（这就是您代码中的问题！）
        current_pose_rpy = current_pose_rpy + delta_rpy
        trajectory_rpy.append(current_pose_rpy.copy())
    
    return np.array(trajectory_rpy)


def replay_with_npz_correct_delta(gt_trajectory):
    """使用矩阵运算进行 replay（正确的方式）"""
    # 从单位矩阵开始
    current_mat = np.eye(4, dtype=np.float64)
    
    trajectory = [current_mat.copy()]
    
    for i in range(len(gt_trajectory)):
        if i == 0:
            # 第一帧：gt_trajectory[0] 是相对于单位矩阵的位姿
            delta_mat = pose10d_to_mat(gt_trajectory[0, :10])
        else:
            # 后续帧：计算相对变换
            mat_prev = pose10d_to_mat(gt_trajectory[i-1, :10])
            mat_curr = pose10d_to_mat(gt_trajectory[i, :10])
            delta_mat = np.linalg.inv(mat_prev) @ mat_curr
        
        # T_next = T_current @ delta
        current_mat = current_mat @ delta_mat
        trajectory.append(current_mat.copy())
    
    return np.array(trajectory)


def main():
    print("=" * 80)
    print("深入分析累积误差")
    print("=" * 80)
    
    # 加载 npz
    npz_path = Path('/home/ubuntu/mzy/RDT2/inference_outputs_fm_full')
    npz_files = sorted(npz_path.glob('full_episode_*.npz'))
    if not npz_files:
        print("No npz files found")
        return
    
    data = np.load(npz_files[-1], allow_pickle=True)
    gt_trajectory = data['gt_trajectory']  # (N, 20) 绝对位姿
    
    print(f"\nnpz gt_trajectory shape: {gt_trajectory.shape}")
    
    # 加载 tar
    shard_dir = Path('/home/ubuntu/mzy/RDT2/rdt2_pika_shards')
    tar_actions = load_tar_actions(shard_dir, 'episode0')
    print(f"tar actions: {len(tar_actions)} frames")
    
    # ================================================================
    # 方法1: 使用 tar 的 action[0] replay
    # ================================================================
    print("\n" + "=" * 60)
    print("方法1: 使用 tar action[0] replay（正确基准）")
    print("=" * 60)
    
    tar_trajectory = replay_with_tar_delta(tar_actions, len(gt_trajectory) + 1)
    print(f"tar replay trajectory shape: {tar_trajectory.shape}")
    
    # ================================================================
    # 方法2: 使用简单减法 replay（您代码中的方式）
    # ================================================================
    print("\n" + "=" * 60)
    print("方法2: 使用简单减法 replay（您代码中的方式）")
    print("=" * 60)
    
    naive_trajectory_rpy = replay_with_npz_naive_delta(gt_trajectory)
    print(f"naive replay trajectory shape: {naive_trajectory_rpy.shape}")
    
    # ================================================================
    # 方法3: 使用矩阵运算 replay（正确的方式）
    # ================================================================
    print("\n" + "=" * 60)
    print("方法3: 使用矩阵运算 replay（正确的方式）")
    print("=" * 60)
    
    correct_trajectory = replay_with_npz_correct_delta(gt_trajectory)
    print(f"correct replay trajectory shape: {correct_trajectory.shape}")
    
    # ================================================================
    # 比较三种方法的累积误差
    # ================================================================
    print("\n" + "=" * 80)
    print("累积误差比较（只看位置，单位：米）")
    print("=" * 80)
    
    # 转换 tar 和 correct 到位置
    tar_pos = tar_trajectory[:, :3, 3]  # (N, 3)
    correct_pos = np.array([mat[:3, 3] for mat in correct_trajectory])  # (N, 3)
    naive_pos = naive_trajectory_rpy[:, :3]  # (N, 3)
    
    print(f"\n{'帧':>6} {'tar_x':>12} {'naive_x':>12} {'correct_x':>12} | {'naive误差':>12} {'correct误差':>12}")
    print("-" * 80)
    
    check_frames = [0, 10, 50, 100, 200, 300, min(400, len(tar_pos)-1)]
    for frame in check_frames:
        if frame >= len(tar_pos) or frame >= len(naive_pos) or frame >= len(correct_pos):
            continue
        
        naive_err = np.linalg.norm(tar_pos[frame] - naive_pos[frame])
        correct_err = np.linalg.norm(tar_pos[frame] - correct_pos[frame])
        
        print(f"{frame:>6} {tar_pos[frame, 0]:>12.6f} {naive_pos[frame, 0]:>12.6f} {correct_pos[frame, 0]:>12.6f} | {naive_err:>12.6f} {correct_err:>12.6f}")
    
    # ================================================================
    # 检查问题可能出在哪里
    # ================================================================
    print("\n" + "=" * 80)
    print("诊断：检查您实际代码中的问题")
    print("=" * 80)
    
    # 读取您的代码
    get_delta_path = Path('/home/ubuntu/mzy/RDT2/whole_body_sim_demo/get_delta_ee.py')
    if get_delta_path.exists():
        with open(get_delta_path) as f:
            content = f.read()
        
        print("\n您的 get_full_rdt2_data() 函数中的 delta 计算:")
        
        # 检查是否使用简单减法
        if 'pred_trajectory[i, 0:10]' in content and 'pred_trajectory[i-1, 0:10]' in content:
            print("  ✓ 找到了相邻帧的计算")
        
        if 'convert_action_to_rpy' in content:
            print("  ✓ 使用了 convert_action_to_rpy 转换")
            
        if '] - convert_action_to_rpy' in content:
            print("  ⚠️ 使用了简单减法计算 delta")
            print("     这可能是导致误差的原因！")
    
    # ================================================================
    # 关键问题
    # ================================================================
    print("\n" + "=" * 80)
    print("⚠️ 关键问题分析")
    print("=" * 80)
    
    print("""
从数值上看，单帧的简单减法和矩阵运算的差异确实很小。
但是，您的代码中有一个更根本的问题：

在 get_full_rdt2_data() 中，您读取的是 **pred_trajectory**（预测轨迹），
而不是 **gt_trajectory**（GT 轨迹）。

如果 pred_trajectory 和 gt_trajectory 不一样，那无论用什么方法计算 delta，
结果都会和 tar 不同！

请检查：
1. npz 文件中是否有 'pred_trajectory' 和 'gt_trajectory' 两个字段？
2. 您使用的是哪一个？
3. 它们是否相同？
""")
    
    # 检查 npz 中的字段
    print("\nnpz 文件中的所有字段:")
    for key in data.files:
        arr = data[key]
        if hasattr(arr, 'shape'):
            print(f"  {key}: shape={arr.shape}")
        else:
            print(f"  {key}: {type(arr)}")
    
    # 检查 pred_trajectory 和 gt_trajectory 是否相同
    if 'pred_trajectory' in data.files:
        pred_trajectory = data['pred_trajectory']
        print(f"\npred_trajectory shape: {pred_trajectory.shape}")
        print(f"gt_trajectory shape: {gt_trajectory.shape}")
        
        diff = np.abs(pred_trajectory - gt_trajectory).max()
        print(f"pred_trajectory 和 gt_trajectory 的最大差异: {diff}")
        
        if diff > 0.01:
            print("\n⚠️ pred_trajectory 和 gt_trajectory 不同！")
            print("这就是您的 replay 和 tar 不一致的原因！")
        else:
            print("\n✓ pred_trajectory 和 gt_trajectory 基本相同")


if __name__ == '__main__':
    main()
