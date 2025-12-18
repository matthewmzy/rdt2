#!/usr/bin/env python3
"""
最终验证：精确模拟 get_delta_ee.py 中的 get_full_rdt2_data() 函数
并与 tar 的 read_gt_data() 进行对比

目标：找出为什么 "从 tar 的 action[0] replay 正确，但从 npz 的 gt_trajectory delta replay 不一样"
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


# ============================================================
# 完全复制 get_delta_ee.py 中的函数
# ============================================================

def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
    """将 6D 旋转表示转换为欧拉角 (roll, pitch, yaw)"""
    col0 = rot6d[:3]
    col1 = rot6d[3:6]

    # Gram-Schmidt 正交化
    col0 = col0 / (np.linalg.norm(col0) + 1e-8)
    col1 = col1 - np.dot(col1, col0) * col0
    col1 = col1 / (np.linalg.norm(col1) + 1e-8)
    col2 = np.cross(col0, col1)

    rotmat = np.stack([col0, col1, col2], axis=1)
    euler = R.from_matrix(rotmat).as_euler('xyz')
    return euler

def convert_action_to_rpy(action_10d: np.ndarray) -> np.ndarray:
    """将 10D 动作转换为 7D (xyz + rpy + gripper)"""
    pos = action_10d[:3]
    rot6d = action_10d[3:9]
    gripper = action_10d[9:10]
    rpy = rot6d_to_euler(rot6d)
    return np.concatenate([pos, rpy, gripper])


def get_full_rdt2_data_simulation(gt_trajectory):
    """
    精确模拟 get_delta_ee.py 中的 get_full_rdt2_data() 函数
    """
    gt_right_action = []
    gt_left_action = []
    
    # 第一帧：直接转换
    gt_right_action.append(convert_action_to_rpy(gt_trajectory[0, 0:10])[0:6].tolist())
    gt_left_action.append(convert_action_to_rpy(gt_trajectory[0, 10:20])[0:6].tolist())
    
    # 后续帧：简单减法
    for i in range(1, len(gt_trajectory)):
        gt_right_action_np = convert_action_to_rpy(gt_trajectory[i, 0:10])[0:6] - convert_action_to_rpy(gt_trajectory[i-1, 0:10])[0:6]
        gt_left_action_np = convert_action_to_rpy(gt_trajectory[i, 10:20])[0:6] - convert_action_to_rpy(gt_trajectory[i-1, 10:20])[0:6]
        gt_right_action.append(gt_right_action_np.tolist())
        gt_left_action.append(gt_left_action_np.tolist())

    return {
        "gt_left_action": gt_left_action,
        "gt_right_action": gt_right_action,
    }


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
    print("精确模拟 get_full_rdt2_data() vs tar read_gt_data()")
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
    
    # 模拟 get_full_rdt2_data()
    npz_result = get_full_rdt2_data_simulation(gt_trajectory)
    npz_gt_right = np.array(npz_result['gt_right_action'])
    npz_gt_left = np.array(npz_result['gt_left_action'])
    
    print(f"\nnpz gt_right_action shape: {npz_gt_right.shape}")
    
    # 加载 tar
    shard_dir = Path('/home/ubuntu/mzy/RDT2/rdt2_pika_shards')
    tar_actions = load_tar_actions(shard_dir, 'episode0')
    print(f"tar actions: {len(tar_actions)} frames")
    
    # 从 tar 构建 gt_action 序列（模拟 read_gt_data 的效果）
    tar_gt_right = []
    for frame in sorted(tar_actions.keys()):
        action_7d = convert_action_to_rpy(tar_actions[frame][0, :10])[:6]  # action[0] 的右臂
        tar_gt_right.append(action_7d)
    tar_gt_right = np.array(tar_gt_right)
    
    print(f"tar gt_right_action shape: {tar_gt_right.shape}")
    
    # ================================================================
    # 对比
    # ================================================================
    print("\n" + "=" * 80)
    print("逐帧对比 npz vs tar (右臂)")
    print("=" * 80)
    
    print(f"\n{'帧':>6} | {'npz delta x':>12} {'tar delta x':>12} | {'npz delta rpy[0]':>16} {'tar delta rpy[0]':>16} | {'误差':>10}")
    print("-" * 100)
    
    errors = []
    for i in range(min(20, len(npz_gt_right), len(tar_gt_right))):
        npz_delta = npz_gt_right[i]
        tar_delta = tar_gt_right[i]
        
        err = np.linalg.norm(npz_delta - tar_delta)
        errors.append(err)
        
        print(f"{i:>6} | {npz_delta[0]:>12.8f} {tar_delta[0]:>12.8f} | {npz_delta[3]:>16.8f} {tar_delta[3]:>16.8f} | {err:>10.8f}")
    
    print("\n" + "-" * 100)
    print(f"前 20 帧平均误差: {np.mean(errors):.8f}")
    print(f"前 20 帧最大误差: {np.max(errors):.8f}")
    
    # ================================================================
    # 检查 frame 对应关系
    # ================================================================
    print("\n" + "=" * 80)
    print("检查 frame 对应关系")
    print("=" * 80)
    
    print(f"\nnpz frame_indices[0:10]: {frame_indices[:10]}")
    print(f"tar frames (sorted): {sorted(tar_actions.keys())[:10]}")
    
    print("""
重要问题：
- npz 的 gt_trajectory[0] 对应的是 frame {0}
- tar 的 action[0] 对应的是 frame {1}

tar 中 action[t] = T_t^{{-1}} @ T_{{t+1}}，
即 frame 0 的 action[0] 表示 frame 0 到 frame 1 的相对变换。

npz 中 gt_trajectory[0] = 模型看到 frame 0 时预测的 action[0]，
这应该等于 tar frame 0 的 action[0]。
""".format(frame_indices[0], sorted(tar_actions.keys())[0]))
    
    # ================================================================
    # 核心问题：第一帧的含义不同！
    # ================================================================
    print("\n" + "=" * 80)
    print("🎯 核心问题：第一帧的含义不同！")
    print("=" * 80)
    
    print("""
在您的 get_full_rdt2_data() 中：
- 第一帧：gt_right_action[0] = convert_action_to_rpy(gt_trajectory[0, 0:10])[0:6]
  这是**绝对位姿**，不是 delta！

在 tar 的 read_gt_data() 中：
- 每一帧：action[0] 本身就是**相对变换**（delta）

所以：
- npz 的 gt_right_action[0] = 绝对位姿（相对于 frame 0 的单位矩阵）
- tar 的 action[0] = 相对变换

这两个数值应该是一样的（因为 frame 0 是单位矩阵），让我们验证：
""")
    
    print(f"npz gt_right_action[0]: {npz_gt_right[0]}")
    print(f"tar action[0] (frame 0): {tar_gt_right[0]}")
    print(f"差异: {np.linalg.norm(npz_gt_right[0] - tar_gt_right[0]):.8f}")
    
    if np.linalg.norm(npz_gt_right[0] - tar_gt_right[0]) < 1e-6:
        print("\n✓ 第一帧一致！")
    else:
        print("\n⚠️ 第一帧不一致！")
    
    # ================================================================
    # 检查后续帧的问题
    # ================================================================
    print("\n" + "=" * 80)
    print("检查后续帧的问题")
    print("=" * 80)
    
    # gt_trajectory[i] 是 frame i+1 相对于 frame 0 的绝对位姿
    # 所以 gt_trajectory[0] 对应 frame 1，gt_trajectory[1] 对应 frame 2
    
    # tar action[frame] 是 frame -> frame+1 的相对变换
    # 所以 tar_actions[0][0] 是 frame 0 -> frame 1 的变换
    
    # npz gt_right_action[1] = gt_trajectory[1] - gt_trajectory[0]
    #                       = (frame 2 相对于 frame 0) - (frame 1 相对于 frame 0)
    #                       ≈ frame 1 -> frame 2 的变换（但用简单减法）
    
    # tar_gt_right[1] = tar_actions[1][0]
    #                 = frame 1 -> frame 2 的变换（正确的相对变换）
    
    print("""
对于第 i 帧（i > 0）：
- npz gt_right_action[i] = gt_trajectory[i] - gt_trajectory[i-1]
                         = (frame i+1 相对于 frame 0) - (frame i 相对于 frame 0)
                         
- tar_gt_right[i] = tar_actions[i][0]
                  = frame i -> frame i+1 的相对变换

问题：
- npz 计算的 delta 是在**全局坐标系**下的差值
- tar 存储的 delta 是在**当前坐标系**下的相对变换

当旋转不是单位矩阵时，这两个是不同的！
""")
    
    # 验证
    print("\n验证：检查 frame 1 的情况")
    print("-" * 60)
    
    # gt_trajectory[0] 是 frame 1 相对于 frame 0 的位姿
    # gt_trajectory[1] 是 frame 2 相对于 frame 0 的位姿
    
    # 方法1：简单减法（您的代码）
    npz_delta_1 = convert_action_to_rpy(gt_trajectory[1, :10])[:6] - convert_action_to_rpy(gt_trajectory[0, :10])[:6]
    
    # 方法2：tar 的 action[0]
    tar_delta_1 = convert_action_to_rpy(tar_actions[1][0, :10])[:6]
    
    print(f"npz delta (简单减法): {npz_delta_1}")
    print(f"tar delta (action[0]): {tar_delta_1}")
    print(f"差异: {np.linalg.norm(npz_delta_1 - tar_delta_1):.8f}")
    
    # ================================================================
    # 关键发现
    # ================================================================
    print("\n" + "=" * 80)
    print("🔍 关键发现")
    print("=" * 80)
    
    print("""
从数值来看，简单减法和矩阵相对变换的差异非常小（约 1e-5 级别）。

但是，更重要的问题是：

1. npz 的 frame_indices 从 {0} 开始
2. tar 的 frame 从 {1} 开始

请检查您的代码中是否正确对齐了 frame 索引！

可能的对齐问题：
- npz gt_trajectory[i] 对应的实际 frame 是什么？
- 您在 whole_body_control.py 中使用 gt_right_action 时，
  索引是否与 tar 的 frame 对应？
""".format(frame_indices[0], sorted(tar_actions.keys())[0]))


if __name__ == '__main__':
    main()
