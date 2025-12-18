#!/usr/bin/env python3
"""
最终验证：对比 Method 2 (tar) 和 Method 3 (npz) 的输出

Method 2 (tar): 
  for i in range(447):
    gt_data = read_gt_data(shard_idx=i)
    delta = gt_data["right_action"][0:6]

Method 3 (npz):
  full_rdt2_data = get_full_rdt2_data()
  for i in range(len(full_rdt2_data["gt_right_action"])):
    delta = full_rdt2_data["gt_right_action"][i][0:6]
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


def convert_action_to_rpy(action_10d: np.ndarray) -> np.ndarray:
    """将 10D 动作转换为 7D (xyz + rpy + gripper)"""
    pos = action_10d[:3]
    rot6d = action_10d[3:9]
    gripper = action_10d[9:10]
    rpy = rot6d_to_euler(rot6d)
    return np.concatenate([pos, rpy, gripper])


def simulate_read_gt_data(shard_dir, episode, num_frames):
    """
    模拟 Method 2: read_gt_data(shard_idx=i)
    返回每个 shard_idx 对应的 action[0]
    """
    # 加载所有 tar 文件
    actions_by_frame = {}
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
                    actions_by_frame[meta['frame']] = action
                except:
                    continue
    
    # 按 frame 顺序返回 action[0]
    result = []
    for frame in range(num_frames):
        if frame in actions_by_frame:
            # 模拟 read_gt_data 的返回：action[0] 转成 rpy
            action_0 = actions_by_frame[frame][0]  # (20,)
            right_action = convert_action_to_rpy(action_0[:10])[:6]  # (6,) xyz + rpy
            left_action = convert_action_to_rpy(action_0[10:])[:6]
            result.append({
                'frame': frame,
                'right_action': right_action,
                'left_action': left_action,
            })
    return result


def simulate_get_full_rdt2_data(gt_trajectory):
    """
    模拟 Method 3: get_full_rdt2_data()
    """
    gt_right_action = []
    gt_left_action = []
    
    # 第一帧：直接转换
    gt_right_action.append(convert_action_to_rpy(gt_trajectory[0, 0:10])[:6])
    gt_left_action.append(convert_action_to_rpy(gt_trajectory[0, 10:20])[:6])
    
    # 后续帧：简单减法
    for i in range(1, len(gt_trajectory)):
        gt_right_action_np = convert_action_to_rpy(gt_trajectory[i, 0:10])[:6] - convert_action_to_rpy(gt_trajectory[i-1, 0:10])[:6]
        gt_left_action_np = convert_action_to_rpy(gt_trajectory[i, 10:20])[:6] - convert_action_to_rpy(gt_trajectory[i-1, 10:20])[:6]
        gt_right_action.append(gt_right_action_np)
        gt_left_action.append(gt_left_action_np)

    return {
        "gt_right_action": gt_right_action,
        "gt_left_action": gt_left_action,
    }


def main():
    print("=" * 80)
    print("对比 Method 2 (tar) 和 Method 3 (npz) 的输出")
    print("=" * 80)
    
    # 加载 npz
    npz_path = Path('/home/ubuntu/mzy/RDT2/inference_outputs_fm_full')
    npz_files = sorted(npz_path.glob('full_episode_*.npz'))
    if not npz_files:
        print("No npz files found")
        return
    
    data = np.load(npz_files[-1], allow_pickle=True)
    gt_trajectory = data['gt_trajectory']
    frame_indices = list(data['frame_indices'])
    
    print(f"\nnpz gt_trajectory shape: {gt_trajectory.shape}")
    print(f"npz frame_indices: {frame_indices[:5]} ... {frame_indices[-5:]}")
    
    # 模拟 Method 3
    method3_result = simulate_get_full_rdt2_data(gt_trajectory)
    
    # 加载 tar (模拟 Method 2)
    shard_dir = Path('/home/ubuntu/mzy/RDT2/rdt2_pika_shards')
    method2_result = simulate_read_gt_data(shard_dir, 'episode0', 448)
    
    print(f"\nMethod 2 (tar): {len(method2_result)} frames")
    print(f"Method 3 (npz): {len(method3_result['gt_right_action'])} frames")
    
    # ================================================================
    # 检查对应关系
    # ================================================================
    print("\n" + "=" * 80)
    print("🔍 检查 Method 2 和 Method 3 的对应关系")
    print("=" * 80)
    
    print("""
Method 2 (tar): for i in range(447), read_gt_data(shard_idx=i)
  - shard_idx=0 读取的是 frame=0 的 action[0]
  - shard_idx=1 读取的是 frame=1 的 action[0]
  - ...

Method 3 (npz): for i in range(len(gt_right_action))
  - i=0: gt_trajectory[0] 转成 rpy (对应 frame 1 相对于 frame 0 的位姿)
  - i=1: gt_trajectory[1] - gt_trajectory[0] (对应 frame 2 - frame 1)
  - ...

问题：索引对应关系！
- Method 2 的 i=0 对应 frame 0 -> frame 1 的变换
- Method 3 的 i=0 对应 frame 1 相对于 frame 0 的绝对位姿
- Method 3 的 i=1 对应 frame 2 - frame 1 的差值
""")
    
    # ================================================================
    # 验证第一帧
    # ================================================================
    print("\n" + "=" * 60)
    print("验证第一帧 (i=0)")
    print("=" * 60)
    
    method2_i0 = method2_result[0]['right_action']  # tar frame 0 的 action[0]
    method3_i0 = method3_result['gt_right_action'][0]  # npz gt_trajectory[0]
    
    print(f"\nMethod 2 (tar shard_idx=0): {method2_i0}")
    print(f"Method 3 (npz i=0):         {method3_i0}")
    print(f"差异: {np.linalg.norm(method2_i0 - method3_i0):.8f}")
    
    if np.linalg.norm(method2_i0 - method3_i0) < 1e-6:
        print("✓ 第一帧一致！")
    else:
        print("⚠️ 第一帧不一致！")
    
    # ================================================================
    # 验证后续帧
    # ================================================================
    print("\n" + "=" * 60)
    print("验证后续帧对应关系")
    print("=" * 60)
    
    print("""
关键问题：
- Method 2 的 i=1 应该对应 Method 3 的哪个索引？

tar 的 action[0] 含义：
  - tar frame 0: action[0] = 从 frame 0 到 frame 1 的相对变换
  - tar frame 1: action[0] = 从 frame 1 到 frame 2 的相对变换
  - tar frame N: action[0] = 从 frame N 到 frame N+1 的相对变换

npz 的 gt_trajectory 含义：
  - gt_trajectory[0] = frame 1 相对于 frame 0 的绝对位姿 (= tar frame 0 的 action[0])
  - gt_trajectory[1] = frame 2 相对于 frame 0 的绝对位姿
  - gt_trajectory[N] = frame N+1 相对于 frame 0 的绝对位姿

npz get_full_rdt2_data 计算的 delta：
  - i=0: gt_trajectory[0] 本身 (= frame 1 相对于 frame 0)
  - i=1: gt_trajectory[1] - gt_trajectory[0] ≈ frame 2 - frame 1 ≈ tar frame 1 的 action[0]
  - i=N: gt_trajectory[N] - gt_trajectory[N-1] ≈ tar frame N 的 action[0]

所以：Method 2 的 i 应该等于 Method 3 的 i！
""")
    
    # 验证对应关系
    print("\n对应关系验证：")
    print(f"{'i':>4} | {'Method2 (tar i)':>40} | {'Method3 (npz i)':>40} | {'差异':>12}")
    print("-" * 110)
    
    errors = []
    for i in range(min(10, len(method2_result), len(method3_result['gt_right_action']))):
        m2 = method2_result[i]['right_action']
        m3 = method3_result['gt_right_action'][i]
        err = np.linalg.norm(m2 - m3)
        errors.append(err)
        print(f"{i:>4} | {str(m2[:3]):>40} | {str(m3[:3]):>40} | {err:>12.8f}")
    
    print(f"\n平均误差: {np.mean(errors):.8f}")
    print(f"最大误差: {np.max(errors):.8f}")
    
    # ================================================================
    # 结论
    # ================================================================
    print("\n" + "=" * 80)
    print("🎯 结论")
    print("=" * 80)
    
    if np.max(errors) < 1e-3:
        print("""
✓ Method 2 和 Method 3 的索引对应关系是正确的！
✓ 数值上非常接近，误差在 1e-4 ~ 1e-6 级别

如果您的 replay 结果不同，可能的原因：
1. IK solver 对误差敏感，1e-4 级别的累积误差可能导致可见差异
2. 初始状态不同
3. 其他配置差异（如步长、循环次数等）

建议：检查您的仿真是否完全按照相同的初始条件运行。
""")
    else:
        print("""
⚠️ 发现显著误差！

可能的原因：
1. 索引对应关系错误
2. npz 文件和 tar 文件对应的数据不同
3. 数据处理过程中的差异
""")


if __name__ == '__main__':
    main()
