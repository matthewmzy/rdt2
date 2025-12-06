#!/usr/bin/env python3
"""
Debug 脚本：模拟用户的 replay 方式，找出问题所在

用户说：
1. 从 tar 数据集里每帧的下一帧 delta action replay 是正确的
2. 从 npz 里的 gt_trajectory 计算相邻两帧的 delta 并 replay 不一样

可能的问题：
1. 用户用简单减法计算 delta 而不是 T_i^{-1} @ T_{i+1}
2. 用户累积时用加法而不是矩阵乘法
3. 或者用户 replay 的方式本身就不对

这个脚本模拟各种可能的错误方式，找出哪种会导致不一致。
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
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def mat_to_rot6d(mat):
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def pose9d_to_mat(pose9d):
    pos = pose9d[..., :3]
    rot6d = pose9d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def mat_to_pose9d(mat):
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    return np.concatenate([pos, rot6d], axis=-1)


def load_tar_actions(shard_dir, episode):
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
    print("调试：模拟各种 replay 方式")
    print("=" * 80)
    
    # 加载 npz
    output_dir = Path(PROJECT_ROOT / 'inference_outputs_fm_full')
    npz_files = sorted(output_dir.glob('full_episode_*.npz'))
    if not npz_files:
        print("No npz files found")
        return
    
    data = np.load(npz_files[-1], allow_pickle=True)
    gt_trajectory = data['gt_trajectory']  # (N, 20)
    frame_indices = list(data['frame_indices'])
    
    print(f"GT trajectory shape: {gt_trajectory.shape}")
    print(f"Frame range: {frame_indices[0]} - {frame_indices[-1]}")
    
    # 加载 tar actions
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    tar_actions = load_tar_actions(shard_dir, 'episode0')
    print(f"Loaded {len(tar_actions)} tar actions")
    
    # ================================================================
    # 方法1: 正确的 replay (从 tar action[0])
    # ================================================================
    print("\n" + "-" * 60)
    print("方法1: 从 tar 的 action[0] replay (正确方式)")
    print("-" * 60)
    
    replay1_r = [np.eye(4)]  # 从单位矩阵开始
    replay1_l = [np.eye(4)]
    
    sorted_frames = sorted(tar_actions.keys())
    for frame in sorted_frames:
        action = tar_actions[frame][0]  # (20,)
        delta_r = pose9d_to_mat(action[:9])
        delta_l = pose9d_to_mat(action[10:19])
        
        replay1_r.append(replay1_r[-1] @ delta_r)
        replay1_l.append(replay1_l[-1] @ delta_l)
    
    # replay1_r[i] 对应 frame i 的绝对位姿
    # replay1_r[0] = 单位矩阵 = frame 0
    # replay1_r[1] = frame 1
    
    # ================================================================
    # 方法2: 从 npz gt_trajectory 简单减法计算 delta 再加法累积
    # ================================================================
    print("\n" + "-" * 60)
    print("方法2: 从 npz gt_trajectory 简单减法+加法 replay")
    print("-" * 60)
    
    # 直接减法计算 delta
    deltas_naive = gt_trajectory[1:] - gt_trajectory[:-1]  # (N-1, 20)
    
    # 加法累积
    replay2 = np.zeros_like(gt_trajectory)
    replay2[0] = gt_trajectory[0]  # 从第一帧开始
    for i in range(len(deltas_naive)):
        replay2[i + 1] = replay2[i] + deltas_naive[i]
    
    # 比较
    diff2 = replay2 - gt_trajectory
    print(f"  与原始 gt_trajectory 的差异: max={np.abs(diff2).max():.10f}")
    
    # ================================================================
    # 方法3: 从 npz 正确计算 delta (T_i^{-1} @ T_{i+1}) 再矩阵乘法累积
    # ================================================================
    print("\n" + "-" * 60)
    print("方法3: 从 npz 正确计算 delta 再矩阵乘法 replay")
    print("-" * 60)
    
    replay3 = []
    
    # 初始化
    mat_r = pose9d_to_mat(gt_trajectory[0, :9])
    mat_l = pose9d_to_mat(gt_trajectory[0, 10:19])
    
    for i in range(len(gt_trajectory)):
        # 保存当前帧
        pos_r = mat_r[:3, 3]
        rot6d_r = mat_to_rot6d(mat_r[:3, :3])
        pos_l = mat_l[:3, 3]
        rot6d_l = mat_to_rot6d(mat_l[:3, :3])
        
        pose_r = np.concatenate([pos_r, rot6d_r, [gt_trajectory[i, 9]]])
        pose_l = np.concatenate([pos_l, rot6d_l, [gt_trajectory[i, 19]]])
        replay3.append(np.concatenate([pose_r, pose_l]))
        
        if i < len(gt_trajectory) - 1:
            # 计算正确的 delta
            mat_r_next = pose9d_to_mat(gt_trajectory[i + 1, :9])
            mat_l_next = pose9d_to_mat(gt_trajectory[i + 1, 10:19])
            
            delta_r = np.linalg.inv(mat_r) @ mat_r_next
            delta_l = np.linalg.inv(mat_l) @ mat_l_next
            
            mat_r = mat_r @ delta_r
            mat_l = mat_l @ delta_l
    
    replay3 = np.stack(replay3, axis=0)
    diff3 = replay3 - gt_trajectory
    print(f"  与原始 gt_trajectory 的差异: max={np.abs(diff3).max():.10f}")
    
    # ================================================================
    # 方法4: 从 npz 的 delta 与 tar 的 action[0] 比较
    # ================================================================
    print("\n" + "-" * 60)
    print("方法4: 比较 npz delta 与 tar action[0]")
    print("-" * 60)
    
    errors = []
    for i in range(len(gt_trajectory) - 1):
        frame_curr = frame_indices[i]
        frame_next = frame_indices[i + 1]
        
        # 只比较连续帧
        if frame_next != frame_curr + 1:
            continue
        
        if frame_curr not in tar_actions:
            continue
        
        # 从 npz 计算 delta (正确方式)
        mat_curr_r = pose9d_to_mat(gt_trajectory[i, :9])
        mat_next_r = pose9d_to_mat(gt_trajectory[i + 1, :9])
        delta_npz_r = np.linalg.inv(mat_curr_r) @ mat_next_r
        delta_npz_pos_r = delta_npz_r[:3, 3]
        
        # tar 的 action[0]
        tar_action = tar_actions[frame_curr][0]
        delta_tar_pos_r = tar_action[:3]
        
        error = np.linalg.norm(delta_npz_pos_r - delta_tar_pos_r)
        errors.append((frame_curr, error, delta_npz_pos_r, delta_tar_pos_r))
    
    if errors:
        max_err = max(e[1] for e in errors)
        mean_err = np.mean([e[1] for e in errors])
        print(f"  比较了 {len(errors)} 对")
        print(f"  位置误差: max={max_err:.10f}, mean={mean_err:.10f}")
    
    # ================================================================
    # 方法5: 检查 npz gt_trajectory 的构建
    # 关键问题：gt_trajectory[i] 对应 frame_indices[i]
    # 但 frame_indices[0] 是 1，不是 0！
    # ================================================================
    print("\n" + "-" * 60)
    print("方法5: 检查 frame_indices 与实际帧的对应关系")
    print("-" * 60)
    
    print(f"  frame_indices[0] = {frame_indices[0]}")
    print(f"  frame_indices[-1] = {frame_indices[-1]}")
    
    # gt_trajectory[0] 对应 frame 1，它是 frame 1 相对于 frame 0 的绝对位姿
    # 但在构建时，frame 0 的位姿被设为单位矩阵
    # 所以 gt_trajectory[0] = T_0 @ action[0]_frame0 = I @ action[0]_frame0 = action[0]_frame0
    
    if 0 in tar_actions:
        print(f"\n  tar action[0] from frame 0:")
        print(f"    pos: {tar_actions[0][0, :3]}")
        print(f"  gt_trajectory[0] (frame {frame_indices[0]}):")
        print(f"    pos: {gt_trajectory[0, :3]}")
        print(f"  差异: {np.linalg.norm(tar_actions[0][0, :3] - gt_trajectory[0, :3]):.10f}")
    
    # ================================================================
    # 方法6: 模拟用户可能的错误用法
    # ================================================================
    print("\n" + "-" * 60)
    print("方法6: 模拟可能的错误用法")
    print("-" * 60)
    
    # 错误1: 用户把 gt_trajectory 的相邻帧差当作 action[0]
    # 然后从 frame 0 开始累积，但 gt_trajectory[0] 已经是 frame 1 了
    
    print("\n  错误1: 混淆起始帧")
    print("    gt_trajectory 的 frame_indices 从 1 开始，不是从 0 开始")
    print("    如果用户从 frame 0（单位矩阵）开始累积 gt_trajectory 的差值，")
    print("    得到的轨迹会与原始 gt_trajectory 偏移一帧")
    
    # 错误2: 直接把 gt_trajectory 当作 delta
    print("\n  错误2: 直接把 gt_trajectory 当作 delta")
    print("    gt_trajectory 存储的是绝对位姿（相对于 frame 0），不是相对 delta")
    
    # ================================================================
    # 最终验证：模拟用户的 replay
    # ================================================================
    print("\n" + "=" * 60)
    print("最终验证：检查各种 replay 方式的结果")
    print("=" * 60)
    
    # 假设用户的 replay 代码是这样的：
    # for i in range(len(trajectory) - 1):
    #     delta = trajectory[i+1] - trajectory[i]  # 或者其他方式
    #     current_pose = current_pose + delta  # 或者矩阵乘法
    
    # 如果 gt_trajectory 是正确的，那么任何正确的 replay 都应该能还原
    
    # 让我检查 gt_trajectory 本身是否有问题
    print("\n检查 gt_trajectory 的位置是否单调/合理:")
    pos_range_r = gt_trajectory[:, :3].max(axis=0) - gt_trajectory[:, :3].min(axis=0)
    pos_range_l = gt_trajectory[:, 10:13].max(axis=0) - gt_trajectory[:, 10:13].min(axis=0)
    print(f"  右臂位置范围: {pos_range_r}")
    print(f"  左臂位置范围: {pos_range_l}")
    
    # 检查是否有异常跳变
    pos_diff_r = np.diff(gt_trajectory[:, :3], axis=0)
    pos_diff_l = np.diff(gt_trajectory[:, 10:13], axis=0)
    print(f"\n  右臂相邻帧位置变化:")
    print(f"    最大变化: {np.abs(pos_diff_r).max(axis=0)}")
    print(f"    平均变化: {np.abs(pos_diff_r).mean(axis=0)}")
    
    # 检查 step_size 边界处是否有跳变
    print("\n  检查 step_size=12 边界处的位置变化:")
    step_size = 12
    for i in range(0, len(gt_trajectory) - 1, step_size):
        if i + step_size < len(gt_trajectory):
            # 边界前后的帧
            before_boundary = i + step_size - 1
            after_boundary = i + step_size
            
            diff_at_boundary = np.linalg.norm(
                gt_trajectory[after_boundary, :3] - gt_trajectory[before_boundary, :3]
            )
            
            # 与正常的相邻帧变化比较
            if before_boundary > 0:
                diff_before = np.linalg.norm(
                    gt_trajectory[before_boundary, :3] - gt_trajectory[before_boundary - 1, :3]
                )
                ratio = diff_at_boundary / (diff_before + 1e-10)
                
                if ratio > 2 or ratio < 0.5:  # 如果变化差异很大
                    print(f"    帧 {frame_indices[before_boundary]}->{frame_indices[after_boundary]}: "
                          f"变化={diff_at_boundary:.6f}, 前一步={diff_before:.6f}, ratio={ratio:.2f}")


if __name__ == '__main__':
    main()
