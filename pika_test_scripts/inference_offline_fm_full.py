#!/usr/bin/env python3
"""
RDT2 Flow Matching 完整 Episode 离线推理脚本

该脚本实现了对完整 episode 的滚动预测：
- 每次预测 24 帧的相对动作
- 取前 12 帧执行（累积到轨迹）
- 下一次预测从第 12 帧结束的地方继续，使用该帧的 GT 观测
- 最终得到完整 episode 的预测轨迹并与 GT 对比

用法:
    python pika_test_scripts/inference_offline_fm_full.py \
        --fm-checkpoint outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
        --shard-dir rdt2_pika_shards \
        --episode episode0

关键数据格式说明:
    - action[t] 是第 (current_frame + t + 1) 帧相对于 current_frame 的相对位姿变换
    - action[0] = T_current^{-1} @ T_{current+1}
    - action[11] = T_current^{-1} @ T_{current+12}
    - 相对动作格式: [pos(3), rot6d(6), gripper(1)] × 2 arms = 20D
"""

import os
import sys
import io
import json
import tarfile
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import yaml
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from models.rdt_inferencer import RDTInferencer
from models.normalizer import LinearNormalizer
from data.umi.pose_util import pose10d_to_mat, mat_to_pose10d


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
    """
    将 9D 动作 (相对位姿，不含 gripper) 转换为 4x4 变换矩阵
    
    Args:
        action_9d: (..., 9) = [pos(3), rot6d(6)]
    Returns:
        mat: (..., 4, 4) 变换矩阵
    """
    pos = action_9d[..., :3]
    rot6d = action_9d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=action_9d.dtype)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def pose_mat_to_action(mat, gripper):
    """
    将 4x4 变换矩阵转换回 10D 动作格式
    
    Args:
        mat: (..., 4, 4) 变换矩阵
        gripper: scalar, (...,) 或 (..., 1) gripper 宽度
    Returns:
        action_10d: (..., 10)
    """
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    
    # 确保 gripper 是正确的形状
    gripper = np.atleast_1d(np.asarray(gripper))
    if gripper.ndim == pos.ndim - 1 or gripper.ndim == 0:
        gripper = gripper.reshape(pos.shape[:-1] + (1,))
    
    return np.concatenate([pos, rot6d, gripper], axis=-1)


def accumulate_relative_actions(
    base_pose_mat: np.ndarray,
    relative_actions: np.ndarray,
    num_steps: int = 12
) -> tuple:
    """
    将相对动作累积到绝对位姿
    
    相对动作的定义:
        T_rel[t] = T_base^{-1} @ T_{base+t+1}
    反向计算:
        T_{base+t+1} = T_base @ T_rel[t]
    
    Args:
        base_pose_mat: (4, 4) 基准位姿矩阵
        relative_actions: (T, 10) 相对动作 [pos(3), rot6d(6), gripper(1)]
        num_steps: 要累积的步数
    
    Returns:
        absolute_poses: (num_steps, 10) 绝对位姿
        final_pose_mat: (4, 4) 最后一帧的位姿矩阵
    """
    absolute_poses = []
    
    for t in range(num_steps):
        # 获取相对变换矩阵
        rel_mat = action_to_pose_mat(relative_actions[t, :9])
        
        # 计算绝对位姿: T_abs = T_base @ T_rel
        abs_mat = base_pose_mat @ rel_mat
        
        # 转回 10D 格式
        gripper = relative_actions[t, 9]
        abs_pose = pose_mat_to_action(abs_mat, np.array(gripper))
        absolute_poses.append(abs_pose)
    
    absolute_poses = np.stack(absolute_poses, axis=0)  # (num_steps, 10)
    
    # 返回最后一帧的位姿矩阵作为下一次预测的基准
    final_rel_mat = action_to_pose_mat(relative_actions[num_steps - 1, :9])
    final_pose_mat = base_pose_mat @ final_rel_mat
    
    return absolute_poses, final_pose_mat


def load_episode_samples(shard_dir: Path, episode_name: str) -> dict:
    """
    从所有 shard 文件中加载指定 episode 的所有样本
    
    Returns:
        samples: dict[frame_idx] = {'image': np.ndarray, 'action': np.ndarray, 'meta': dict}
    """
    samples = {}
    
    shard_files = sorted(shard_dir.glob('shard-*.tar'))
    print(f"Searching for episode '{episode_name}' in {len(shard_files)} shards...")
    
    for shard_path in shard_files:
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            
            # 找出该 shard 中的所有样本
            sample_indices = set()
            for m in members:
                if m.name.endswith('.meta.json'):
                    sample_idx = int(m.name.split('.')[0])
                    sample_indices.add(sample_idx)
            
            for sample_idx in sample_indices:
                try:
                    # 读取元数据
                    meta_file = tar.extractfile(f'{sample_idx}.meta.json')
                    meta = json.load(meta_file)
                    
                    if meta['episode'] != episode_name:
                        continue
                    
                    frame_idx = meta['frame']
                    
                    # 读取图像
                    img_file = tar.extractfile(f'{sample_idx}.image.jpg')
                    image = Image.open(io.BytesIO(img_file.read()))
                    image = np.array(image)
                    
                    # 读取动作 (GT)
                    action_file = tar.extractfile(f'{sample_idx}.action.npy')
                    action = np.load(io.BytesIO(action_file.read()))
                    
                    samples[frame_idx] = {
                        'image': image,
                        'action': action,
                        'meta': meta,
                    }
                except Exception as e:
                    continue
    
    print(f"Loaded {len(samples)} samples from episode '{episode_name}'")
    if samples:
        frame_indices = sorted(samples.keys())
        print(f"Frame range: {frame_indices[0]} - {frame_indices[-1]}")
    
    return samples


def run_full_episode_inference(
    inferencer: RDTInferencer,
    samples: dict,
    instruction: str,
    config: dict,
    step_size: int = 12,
    action_horizon: int = 24,
) -> tuple:
    """
    对完整 episode 进行滚动预测
    
    策略:
    1. 从第 0 帧开始，使用 GT 观测预测未来 24 帧的相对动作
    2. 取前 step_size (12) 帧累积到预测轨迹
    3. 同时用 GT 的相对动作累积 GT 轨迹
    4. 移动到第 step_size 帧，使用 GT 累积得到的基准位姿继续预测
    5. 重复直到 episode 结束
    
    关键理解:
    - 相对动作 action[t] = T_current^{-1} @ T_{current+t+1}
    - 即 action[t] 代表第 (current_frame + t + 1) 帧相对于 current_frame 的变换
    - 绝对位姿: T_{current+t+1} = T_current @ action[t]
    
    累积策略:
    - 预测轨迹: 用模型预测的相对动作累积
    - GT 轨迹: 用数据集中的 GT 相对动作累积
    - 基准位姿更新: 使用 GT 累积，确保与 GT 轨迹对齐
    
    Args:
        inferencer: RDT2 推理器
        samples: episode 样本字典
        instruction: 语言指令
        config: 配置
        step_size: 每次执行的步数 (默认 12)
        action_horizon: 预测的动作长度 (默认 24)
    
    Returns:
        pred_trajectory: 预测的轨迹 (N, 20) - 绝对位姿
        gt_trajectory: GT 轨迹 (N, 20) - 绝对位姿
        frame_indices: 对应的帧索引
    """
    frame_indices = sorted(samples.keys())
    
    # 初始化存储
    pred_trajectory_list = []   # 预测轨迹 (绝对位姿)
    gt_trajectory_list = []     # GT 轨迹 (绝对位姿)
    pred_frame_indices = []     # 预测对应的帧索引
    
    # 初始化基准位姿
    # 第 0 帧的位姿作为世界坐标系原点（单位矩阵）
    # 后续帧的位姿相对于第 0 帧
    base_pose_mat_right = np.eye(4, dtype=np.float32)
    base_pose_mat_left = np.eye(4, dtype=np.float32)
    
    # 同时维护预测累积的基准位姿（用于对比误差累积效应）
    pred_base_pose_mat_right = np.eye(4, dtype=np.float32)
    pred_base_pose_mat_left = np.eye(4, dtype=np.float32)
    
    current_frame = 0
    max_frame = max(frame_indices)
    
    print(f"\nRunning inference with step_size={step_size}, action_horizon={action_horizon}")
    print(f"Total frames: {len(frame_indices)}, max_frame: {max_frame}")
    
    iteration = 0
    while current_frame <= max_frame - action_horizon:
        if current_frame not in samples:
            print(f"Warning: Frame {current_frame} not in samples, skipping...")
            current_frame += step_size
            continue
        
        sample = samples[current_frame]
        image = sample['image']
        action_gt = sample['action']  # (24, 20) GT 相对动作
        
        # 准备输入
        h, w = image.shape[:2]
        half_w = w // 2
        img_left = image[:, :half_w]
        img_right = image[:, half_w:]
        
        observations = {
            'images': {
                'left_stereo': img_left,
                'right_stereo': img_right,
            },
            'state': np.zeros(config['common']['state_dim'], dtype=np.float32)
        }
        
        # 推理
        with torch.no_grad():
            action_pred = inferencer.step(observations, instruction)
            action_pred = action_pred.cpu().numpy()  # (24, 20)
        
        # 累积预测和 GT 的相对动作到绝对轨迹
        actual_steps = min(step_size, action_horizon)
        
        for t in range(actual_steps):
            target_frame = current_frame + t + 1  # action[t] 对应 frame current+t+1
            
            if target_frame > max_frame:
                break
            
            # 预测动作 (相对于 current_frame)
            pred_right = action_pred[t, :10]  # (10,)
            pred_left = action_pred[t, 10:]   # (10,)
            
            # GT 动作 (相对于 current_frame)
            gt_right = action_gt[t, :10]
            gt_left = action_gt[t, 10:]
            
            # ============ 计算 GT 绝对位姿 ============
            # T_abs = T_base @ T_rel
            # 这里 base 是 current_frame 的绝对位姿
            gt_mat_right = base_pose_mat_right @ action_to_pose_mat(gt_right[:9])
            gt_mat_left = base_pose_mat_left @ action_to_pose_mat(gt_left[:9])
            
            # ============ 计算预测绝对位姿 ============
            # 同样从 current_frame 的基准位姿累积（使用与 GT 相同的基准）
            pred_mat_right = base_pose_mat_right @ action_to_pose_mat(pred_right[:9])
            pred_mat_left = base_pose_mat_left @ action_to_pose_mat(pred_left[:9])
            
            # 转回 10D 格式
            pred_pose_right = pose_mat_to_action(pred_mat_right, pred_right[9])
            pred_pose_left = pose_mat_to_action(pred_mat_left, pred_left[9])
            gt_pose_right = pose_mat_to_action(gt_mat_right, gt_right[9])
            gt_pose_left = pose_mat_to_action(gt_mat_left, gt_left[9])
            
            # 添加到轨迹列表
            pred_trajectory_list.append(np.concatenate([pred_pose_right, pred_pose_left], axis=-1))
            gt_trajectory_list.append(np.concatenate([gt_pose_right, gt_pose_left], axis=-1))
            pred_frame_indices.append(target_frame)
        
        # 更新基准位姿为第 step_size 帧的 GT 绝对位姿
        # 这样下一次预测的起点与 GT 轨迹对齐
        if step_size <= action_horizon:
            # 使用 GT 的相对动作来更新基准位姿
            gt_rel_mat_right = action_to_pose_mat(action_gt[step_size - 1, :9])
            gt_rel_mat_left = action_to_pose_mat(action_gt[step_size - 1, 10:19])
            
            base_pose_mat_right = base_pose_mat_right @ gt_rel_mat_right
            base_pose_mat_left = base_pose_mat_left @ gt_rel_mat_left
        
        current_frame += step_size
        iteration += 1
        
        if iteration % 10 == 0:
            print(f"  Iteration {iteration}: processed up to frame {current_frame}")
    
    # 合并轨迹
    if pred_trajectory_list:
        pred_trajectory = np.stack(pred_trajectory_list, axis=0)
        gt_trajectory = np.stack(gt_trajectory_list, axis=0)
    else:
        pred_trajectory = np.array([])
        gt_trajectory = np.array([])
    
    print(f"\nInference complete: {len(pred_frame_indices)} frames predicted")
    
    return pred_trajectory, gt_trajectory, pred_frame_indices


def visualize_full_trajectory(
    pred_trajectory: np.ndarray,
    gt_trajectory: np.ndarray,
    frame_indices: list,
    instruction: str,
    episode_name: str,
    output_path: str,
):
    """
    可视化完整轨迹对比
    """
    fig = plt.figure(figsize=(24, 20))
    
    t = np.array(frame_indices)
    
    # =========== 右臂 ===========
    # 1. 右臂 Position
    ax1 = fig.add_subplot(4, 3, 1)
    for i, (dim, label, color) in enumerate([(0, 'x', 'C0'), (1, 'y', 'C1'), (2, 'z', 'C2')]):
        ax1.plot(t, gt_trajectory[:, dim], f'{color}-', label=f'GT {label}', alpha=0.7, linewidth=1)
        ax1.plot(t, pred_trajectory[:, dim], f'{color}--', label=f'Pred {label}', alpha=0.7, linewidth=1)
    ax1.set_xlabel('Frame')
    ax1.set_ylabel('Position')
    ax1.set_title('Right Arm Position (累积)')
    ax1.legend(loc='upper right', fontsize=7)
    ax1.grid(True, alpha=0.3)
    
    # 2. 右臂 Rotation (6D 的前两个分量)
    ax2 = fig.add_subplot(4, 3, 2)
    for i, (dim, label, color) in enumerate([(3, 'r0', 'C0'), (4, 'r1', 'C1')]):
        ax2.plot(t, gt_trajectory[:, dim], f'{color}-', label=f'GT {label}', alpha=0.7, linewidth=1)
        ax2.plot(t, pred_trajectory[:, dim], f'{color}--', label=f'Pred {label}', alpha=0.7, linewidth=1)
    ax2.set_xlabel('Frame')
    ax2.set_ylabel('Rotation 6D')
    ax2.set_title('Right Arm Rotation (前2维)')
    ax2.legend(loc='upper right', fontsize=7)
    ax2.grid(True, alpha=0.3)
    
    # 3. 右臂 Gripper
    ax3 = fig.add_subplot(4, 3, 3)
    ax3.plot(t, gt_trajectory[:, 9], 'C0-', label='GT Gripper', linewidth=2)
    ax3.plot(t, pred_trajectory[:, 9], 'C0--', label='Pred Gripper', linewidth=2)
    ax3.set_xlabel('Frame')
    ax3.set_ylabel('Gripper Width')
    ax3.set_title('Right Arm Gripper')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # =========== 左臂 ===========
    # 4. 左臂 Position
    ax4 = fig.add_subplot(4, 3, 4)
    for i, (dim, label, color) in enumerate([(10, 'x', 'C0'), (11, 'y', 'C1'), (12, 'z', 'C2')]):
        ax4.plot(t, gt_trajectory[:, dim], f'{color}-', label=f'GT {label}', alpha=0.7, linewidth=1)
        ax4.plot(t, pred_trajectory[:, dim], f'{color}--', label=f'Pred {label}', alpha=0.7, linewidth=1)
    ax4.set_xlabel('Frame')
    ax4.set_ylabel('Position')
    ax4.set_title('Left Arm Position (累积)')
    ax4.legend(loc='upper right', fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # 5. 左臂 Rotation
    ax5 = fig.add_subplot(4, 3, 5)
    for i, (dim, label, color) in enumerate([(13, 'r0', 'C0'), (14, 'r1', 'C1')]):
        ax5.plot(t, gt_trajectory[:, dim], f'{color}-', label=f'GT {label}', alpha=0.7, linewidth=1)
        ax5.plot(t, pred_trajectory[:, dim], f'{color}--', label=f'Pred {label}', alpha=0.7, linewidth=1)
    ax5.set_xlabel('Frame')
    ax5.set_ylabel('Rotation 6D')
    ax5.set_title('Left Arm Rotation (前2维)')
    ax5.legend(loc='upper right', fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    # 6. 左臂 Gripper
    ax6 = fig.add_subplot(4, 3, 6)
    ax6.plot(t, gt_trajectory[:, 19], 'C1-', label='GT Gripper', linewidth=2)
    ax6.plot(t, pred_trajectory[:, 19], 'C1--', label='Pred Gripper', linewidth=2)
    ax6.set_xlabel('Frame')
    ax6.set_ylabel('Gripper Width')
    ax6.set_title('Left Arm Gripper')
    ax6.legend(loc='upper right', fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # =========== 3D 轨迹 ===========
    # 7. 右臂 3D 轨迹
    ax7 = fig.add_subplot(4, 3, 7, projection='3d')
    ax7.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], gt_trajectory[:, 2], 
             'b-', label='GT', alpha=0.7, linewidth=1)
    ax7.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], pred_trajectory[:, 2], 
             'r--', label='Pred', alpha=0.7, linewidth=1)
    ax7.set_xlabel('X')
    ax7.set_ylabel('Y')
    ax7.set_zlabel('Z')
    ax7.set_title('Right Arm 3D Trajectory')
    ax7.legend()
    
    # 8. 左臂 3D 轨迹
    ax8 = fig.add_subplot(4, 3, 8, projection='3d')
    ax8.plot(gt_trajectory[:, 10], gt_trajectory[:, 11], gt_trajectory[:, 12], 
             'b-', label='GT', alpha=0.7, linewidth=1)
    ax8.plot(pred_trajectory[:, 10], pred_trajectory[:, 11], pred_trajectory[:, 12], 
             'r--', label='Pred', alpha=0.7, linewidth=1)
    ax8.set_xlabel('X')
    ax8.set_ylabel('Y')
    ax8.set_zlabel('Z')
    ax8.set_title('Left Arm 3D Trajectory')
    ax8.legend()
    
    # =========== Error 分析 ===========
    # 9. Position Error
    ax9 = fig.add_subplot(4, 3, 9)
    pos_error_right = np.linalg.norm(pred_trajectory[:, :3] - gt_trajectory[:, :3], axis=1)
    pos_error_left = np.linalg.norm(pred_trajectory[:, 10:13] - gt_trajectory[:, 10:13], axis=1)
    ax9.plot(t, pos_error_right, 'C0-', label='Right Arm', linewidth=1)
    ax9.plot(t, pos_error_left, 'C1-', label='Left Arm', linewidth=1)
    ax9.set_xlabel('Frame')
    ax9.set_ylabel('Position Error (L2)')
    ax9.set_title('Position Error over Time')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # 10. 累积误差统计
    ax10 = fig.add_subplot(4, 3, 10)
    ax10.axis('off')
    
    # 计算误差统计
    mse_total = np.mean((pred_trajectory - gt_trajectory) ** 2)
    mae_total = np.mean(np.abs(pred_trajectory - gt_trajectory))
    
    pos_mse_r = np.mean((pred_trajectory[:, :3] - gt_trajectory[:, :3]) ** 2)
    pos_mse_l = np.mean((pred_trajectory[:, 10:13] - gt_trajectory[:, 10:13]) ** 2)
    rot_mse_r = np.mean((pred_trajectory[:, 3:9] - gt_trajectory[:, 3:9]) ** 2)
    rot_mse_l = np.mean((pred_trajectory[:, 13:19] - gt_trajectory[:, 13:19]) ** 2)
    grip_mse_r = np.mean((pred_trajectory[:, 9] - gt_trajectory[:, 9]) ** 2)
    grip_mse_l = np.mean((pred_trajectory[:, 19] - gt_trajectory[:, 19]) ** 2)
    
    # 最终位置误差
    final_pos_err_r = np.linalg.norm(pred_trajectory[-1, :3] - gt_trajectory[-1, :3])
    final_pos_err_l = np.linalg.norm(pred_trajectory[-1, 10:13] - gt_trajectory[-1, 10:13])
    
    text = f"""
    Full Episode Inference Results
    ══════════════════════════════════════════════
    Episode: {episode_name}
    Instruction: "{instruction}"
    
    Frames Predicted: {len(frame_indices)}
    Frame Range: {frame_indices[0]} - {frame_indices[-1]}
    
    ── Overall Metrics ──
    Total MSE:           {mse_total:.6f}
    Total MAE:           {mae_total:.6f}
    
    ── Position MSE ──
    Right Arm:           {pos_mse_r:.6f}
    Left Arm:            {pos_mse_l:.6f}
    
    ── Rotation MSE ──
    Right Arm:           {rot_mse_r:.6f}
    Left Arm:            {rot_mse_l:.6f}
    
    ── Gripper MSE ──
    Right:               {grip_mse_r:.6f}
    Left:                {grip_mse_l:.6f}
    
    ── Final Position Error (L2) ──
    Right Arm:           {final_pos_err_r:.6f}
    Left Arm:            {final_pos_err_l:.6f}
    """
    ax10.text(0.05, 0.95, text, transform=ax10.transAxes, fontsize=10,
              verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 11 & 12. XY 平面轨迹
    ax11 = fig.add_subplot(4, 3, 11)
    ax11.plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'b-', label='GT', alpha=0.7)
    ax11.plot(pred_trajectory[:, 0], pred_trajectory[:, 1], 'r--', label='Pred', alpha=0.7)
    ax11.scatter([gt_trajectory[0, 0]], [gt_trajectory[0, 1]], c='g', marker='o', s=100, label='Start')
    ax11.scatter([gt_trajectory[-1, 0]], [gt_trajectory[-1, 1]], c='m', marker='x', s=100, label='End')
    ax11.set_xlabel('X')
    ax11.set_ylabel('Y')
    ax11.set_title('Right Arm XY Trajectory')
    ax11.legend(fontsize=7)
    ax11.grid(True, alpha=0.3)
    ax11.axis('equal')
    
    ax12 = fig.add_subplot(4, 3, 12)
    ax12.plot(gt_trajectory[:, 10], gt_trajectory[:, 11], 'b-', label='GT', alpha=0.7)
    ax12.plot(pred_trajectory[:, 10], pred_trajectory[:, 11], 'r--', label='Pred', alpha=0.7)
    ax12.scatter([gt_trajectory[0, 10]], [gt_trajectory[0, 11]], c='g', marker='o', s=100, label='Start')
    ax12.scatter([gt_trajectory[-1, 10]], [gt_trajectory[-1, 11]], c='m', marker='x', s=100, label='End')
    ax12.set_xlabel('X')
    ax12.set_ylabel('Y')
    ax12.set_title('Left Arm XY Trajectory')
    ax12.legend(fontsize=7)
    ax12.grid(True, alpha=0.3)
    ax12.axis('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to: {output_path}")
    
    return {
        'mse_total': mse_total,
        'mae_total': mae_total,
        'pos_mse_r': pos_mse_r,
        'pos_mse_l': pos_mse_l,
        'final_pos_err_r': final_pos_err_r,
        'final_pos_err_l': final_pos_err_l,
    }


def main():
    parser = argparse.ArgumentParser(description='RDT2 FM Full Episode Inference')
    parser.add_argument('--fm-checkpoint', type=str, required=True,
                        help='Path to FM checkpoint directory')
    parser.add_argument('--vlm-model', type=str, 
                        default='robotics-diffusion-transformer/RDT2-VQ',
                        help='VLM backbone model')
    parser.add_argument('--config-path', type=str,
                        default='configs/rdt/post_train.yaml',
                        help='Path to RDT config')
    parser.add_argument('--normalizer-path', type=str, default='normalizer.pt',
                        help='Path to normalizer')
    parser.add_argument('--shard-dir', type=str, default='rdt2_pika_shards',
                        help='Directory containing shards')
    parser.add_argument('--episode', type=str, default='episode0',
                        help='Episode name to inference')
    parser.add_argument('--step-size', type=int, default=12,
                        help='Number of steps to execute per prediction')
    parser.add_argument('--output-dir', type=str, default='inference_outputs_fm_full',
                        help='Output directory')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    print(f"Loading config from {args.config_path}...")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载指令
    instruction_path = Path(args.shard_dir) / 'instruction.json'
    with open(instruction_path, 'r') as f:
        instructions = json.load(f)
    
    # 获取指令
    instruction = instructions.get('pika_task', "Put the bottle into the tape ring.")
    print(f"Instruction: {instruction}")
    
    # 加载 episode 样本
    shard_dir = Path(args.shard_dir)
    samples = load_episode_samples(shard_dir, args.episode)
    
    if not samples:
        print(f"Error: No samples found for episode '{args.episode}'")
        return
    
    # 初始化推理器
    print(f"\nLoading FM model from {args.fm_checkpoint}...")
    print(f"Loading VLM from {args.vlm_model}...")
    
    inferencer = RDTInferencer(
        config=config,
        pretrained_path=args.fm_checkpoint,
        normalizer_path=args.normalizer_path,
        pretrained_vision_language_model_name_or_path=args.vlm_model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    
    # 运行完整 episode 推理
    pred_trajectory, gt_trajectory, frame_indices = run_full_episode_inference(
        inferencer=inferencer,
        samples=samples,
        instruction=instruction,
        config=config,
        step_size=args.step_size,
    )
    
    if len(pred_trajectory) == 0:
        print("Error: No predictions generated")
        return
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 保存轨迹数据
    npz_path = Path(args.output_dir) / f"full_episode_{args.episode}_{timestamp}.npz"
    np.savez(npz_path,
             pred_trajectory=pred_trajectory,
             gt_trajectory=gt_trajectory,
             frame_indices=frame_indices,
             instruction=instruction,
             step_size=args.step_size)
    print(f"Saved trajectory data to: {npz_path}")
    
    # 可视化
    fig_path = Path(args.output_dir) / f"full_episode_{args.episode}_{timestamp}.png"
    metrics = visualize_full_trajectory(
        pred_trajectory=pred_trajectory,
        gt_trajectory=gt_trajectory,
        frame_indices=frame_indices,
        instruction=instruction,
        episode_name=args.episode,
        output_path=str(fig_path),
    )
    
    # 打印总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Episode: {args.episode}")
    print(f"Frames predicted: {len(frame_indices)}")
    print(f"Step size: {args.step_size}")
    print(f"\nMetrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")


if __name__ == '__main__':
    main()
