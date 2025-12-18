#!/usr/bin/env python3
"""
Debug 脚本：找出 tar replay 和 npz replay 不一致的根本原因

问题：
- 方法2 (tar): 直接用 action[0] 作为 delta (相对变换)
- 方法3 (npz): 用 pose[i] - pose[i-1] 作为 delta (简单减法)

这两种 delta 的定义完全不同！

tar 中的 action[0] = T_current^{-1} @ T_next (相对变换矩阵)
npz 减法得到的 = pose_next - pose_current (向量差)

正确的 npz delta 计算应该是：
delta = inv(T_current) @ T_next
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
    """4x4 变换矩阵转 7D (xyz + rpy + gripper=0)"""
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


def main():
    print("=" * 80)
    print("分析 tar replay 和 npz replay 不一致的原因")
    print("=" * 80)
    
    # 加载 npz
    npz_path = Path('/home/ubuntu/mzy/RDT2/inference_outputs_fm_full')
    npz_files = sorted(npz_path.glob('full_episode_*.npz'))
    if not npz_files:
        print("No npz files found")
        return
    
    data = np.load(npz_files[-1], allow_pickle=True)
    gt_trajectory = data['gt_trajectory']  # (N, 20) 绝对位姿
    frame_indices = list(data['frame_indices'])
    
    print(f"\nnpz gt_trajectory shape: {gt_trajectory.shape}")
    print(f"frame_indices: {frame_indices[0]} - {frame_indices[-1]}")
    
    # 加载 tar
    shard_dir = Path('/home/ubuntu/mzy/RDT2/rdt2_pika_shards')
    tar_actions = load_tar_actions(shard_dir, 'episode0')
    print(f"tar actions: {len(tar_actions)} frames")
    
    # ================================================================
    # 比较两种 delta 计算方式
    # ================================================================
    print("\n" + "=" * 60)
    print("比较两种 delta 计算方式")
    print("=" * 60)
    
    print("\n方式1 (tar): action[0] = T_current^{-1} @ T_next (相对变换)")
    print("方式2 (npz 错误): pose[i] - pose[i-1] (简单减法)")
    print("方式3 (npz 正确): inv(T_current) @ T_next (矩阵运算)")
    
    # 比较前几帧
    print("\n" + "-" * 60)
    print("前 5 帧的 delta 对比 (右臂)")
    print("-" * 60)
    
    for i in range(5):
        frame_curr = i
        frame_next = i + 1
        
        if frame_curr not in tar_actions:
            continue
        
        # 方式1: tar 的 action[0]
        tar_action = tar_actions[frame_curr][0]  # (20,)
        tar_delta_rpy = convert_action_to_rpy(tar_action[:10])[:6]  # 右臂 (6,)
        
        # 方式2: npz 简单减法 (get_full_rdt2_data 中的方式)
        # gt_trajectory[0] 对应 frame 1，所以要调整索引
        if frame_next in frame_indices and frame_curr in frame_indices:
            idx_curr = frame_indices.index(frame_curr + 1)  # +1 因为 frame_indices 从 1 开始
            idx_next = frame_indices.index(frame_next + 1)
            
            if idx_curr > 0:  # 确保有前一帧
                pose_curr_rpy = convert_action_to_rpy(gt_trajectory[idx_curr - 1, :10])[:6]
                pose_next_rpy = convert_action_to_rpy(gt_trajectory[idx_curr, :10])[:6]
                npz_delta_naive = pose_next_rpy - pose_curr_rpy
            else:
                # 第一帧特殊处理
                npz_delta_naive = convert_action_to_rpy(gt_trajectory[0, :10])[:6]
        
        # 方式3: npz 正确的矩阵运算
        if frame_next in frame_indices:
            idx = frame_indices.index(frame_next)  # frame_next 对应的 gt_trajectory 索引
            if idx > 0:
                # T_current 和 T_next
                pose_curr_10d = gt_trajectory[idx - 1, :10]
                pose_next_10d = gt_trajectory[idx, :10]
                
                mat_curr = pose10d_to_mat(pose_curr_10d)
                mat_next = pose10d_to_mat(pose_next_10d)
                
                # delta = inv(T_curr) @ T_next
                delta_mat = np.linalg.inv(mat_curr) @ mat_next
                npz_delta_correct = mat_to_pose7d(delta_mat)
            else:
                # 第一帧：gt_trajectory[0] 就是 frame 1 相对于 frame 0(单位矩阵) 的位姿
                # 所以它本身就是 delta
                npz_delta_correct = convert_action_to_rpy(gt_trajectory[0, :10])[:6]
        
        print(f"\nFrame {frame_curr} -> {frame_next}:")
        print(f"  tar action[0] (相对变换):    pos={tar_delta_rpy[:3]}, rpy={tar_delta_rpy[3:]}")
        
        if frame_curr == 0:
            print(f"  npz 第一帧 (特殊):           与 tar 相同")
        else:
            print(f"  npz 简单减法 (错误):         pos={npz_delta_naive[:3]}, rpy={npz_delta_naive[3:]}")
            print(f"  npz 矩阵运算 (正确):         pos={npz_delta_correct[:3]}, rpy={npz_delta_correct[3:]}")
            
            # 计算误差
            err_naive = np.linalg.norm(tar_delta_rpy - npz_delta_naive)
            err_correct = np.linalg.norm(tar_delta_rpy - npz_delta_correct)
            print(f"  与 tar 的误差: 简单减法={err_naive:.6f}, 矩阵运算={err_correct:.6f}")
    
    # ================================================================
    # 问题的根本原因
    # ================================================================
    print("\n" + "=" * 80)
    print("问题的根本原因")
    print("=" * 80)
    print("""
您的 get_full_rdt2_data() 函数计算 delta 的方式是**简单减法**:

    delta = pose[i] - pose[i-1]

但 tar 中存储的 action[0] 是**相对变换矩阵**:

    action[0] = T_current^{-1} @ T_next

这两个在数学上是**完全不同的**！

简单减法只在以下情况下近似正确：
1. 旋转非常小（接近单位矩阵）
2. 位姿是在同一个参考系下表示的

但当旋转不是单位矩阵时，相对变换和简单差值是不同的：
- 相对变换: 先旋转到当前帧坐标系，再计算下一帧的相对位置
- 简单差值: 在全局坐标系下直接计算差

您的 whole_body_control.py 中的 IK solver 需要的是**相对变换**，
因为它需要知道末端执行器在**当前坐标系**下要移动多少。
""")
    
    # ================================================================
    # 解决方案
    # ================================================================
    print("\n" + "=" * 80)
    print("解决方案")
    print("=" * 80)
    print("""
修改 get_full_rdt2_data() 函数，使用正确的矩阵运算计算 delta:

def compute_relative_delta(pose_curr_10d, pose_next_10d):
    '''计算正确的相对变换 delta = inv(T_curr) @ T_next'''
    mat_curr = pose10d_to_mat(pose_curr_10d)
    mat_next = pose10d_to_mat(pose_next_10d)
    delta_mat = np.linalg.inv(mat_curr) @ mat_next
    return mat_to_pose7d(delta_mat)  # 转成 xyz + rpy

# 第一帧特殊处理：gt_trajectory[0] 本身就是相对于单位矩阵的位姿
delta_first = convert_action_to_rpy(gt_trajectory[0, :10])[:6]

# 后续帧：使用矩阵运算
for i in range(1, len(gt_trajectory)):
    delta = compute_relative_delta(gt_trajectory[i-1, :10], gt_trajectory[i, :10])
""")


if __name__ == '__main__':
    main()
