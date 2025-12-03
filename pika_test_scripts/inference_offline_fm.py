#!/usr/bin/env python3
"""
RDT2 Flow Matching 离线推理脚本

用于测试 FM action expert 的表现，输出与 inference_offline.py 类似的可视化图

用法:
    python pika_test_scripts/inference_offline_fm.py \
        --fm-checkpoint outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
        --shard-dir rdt2_pika_shards \
        --shard-idx 5 \
        --sample-idx 10

    # 批量测试
    python pika_test_scripts/inference_offline_fm.py \
        --fm-checkpoint outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
        --shard-dir rdt2_pika_shards \
        --batch --num-samples 10
"""

import os
import sys
import io
import json
import tarfile
import argparse
from pathlib import Path
from datetime import datetime

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


def load_sample_from_shard(shard_path: str, sample_idx: int):
    """从 shard 加载样本"""
    with tarfile.open(shard_path, 'r') as tar:
        # 读取图像
        img_file = tar.extractfile(f'{sample_idx}.image.jpg')
        image = Image.open(io.BytesIO(img_file.read()))
        image = np.array(image)
        
        # 读取动作 (ground truth)
        action_file = tar.extractfile(f'{sample_idx}.action.npy')
        action_gt = np.load(io.BytesIO(action_file.read()))
        
        # 读取元数据
        meta_file = tar.extractfile(f'{sample_idx}.meta.json')
        meta = json.load(meta_file)
    
    return image, action_gt, meta


def visualize_and_save(
    image: np.ndarray,
    action_gt: np.ndarray,
    action_pred: np.ndarray,
    instruction: str,
    output_path: str,
    title_suffix: str = ""
):
    """可视化并保存结果"""
    fig = plt.figure(figsize=(20, 16))
    
    # 1. 显示输入图像
    ax_img = fig.add_subplot(3, 2, 1)
    ax_img.imshow(image)
    ax_img.set_title(f'Input Image\n"{instruction}"', fontsize=10)
    ax_img.axis('off')
    
    # 2. Position 对比 (dims 0-2: right arm, 10-12: left arm)
    ax_pos = fig.add_subplot(3, 2, 2)
    t = np.arange(action_gt.shape[0])
    
    # Right arm position
    for i, (dim, label) in enumerate([(0, 'R_x'), (1, 'R_y'), (2, 'R_z')]):
        ax_pos.plot(t, action_gt[:, dim], f'C{i}-', label=f'GT {label}', alpha=0.7)
        ax_pos.plot(t, action_pred[:, dim], f'C{i}--', label=f'Pred {label}', alpha=0.7)
    
    ax_pos.set_xlabel('Timestep')
    ax_pos.set_ylabel('Position')
    ax_pos.set_title('Right Arm Position (dims 0-2)')
    ax_pos.legend(loc='upper right', fontsize=8)
    ax_pos.grid(True, alpha=0.3)
    
    # 3. Left arm position
    ax_pos_l = fig.add_subplot(3, 2, 3)
    for i, (dim, label) in enumerate([(10, 'L_x'), (11, 'L_y'), (12, 'L_z')]):
        ax_pos_l.plot(t, action_gt[:, dim], f'C{i}-', label=f'GT {label}', alpha=0.7)
        ax_pos_l.plot(t, action_pred[:, dim], f'C{i}--', label=f'Pred {label}', alpha=0.7)
    
    ax_pos_l.set_xlabel('Timestep')
    ax_pos_l.set_ylabel('Position')
    ax_pos_l.set_title('Left Arm Position (dims 10-12)')
    ax_pos_l.legend(loc='upper right', fontsize=8)
    ax_pos_l.grid(True, alpha=0.3)
    
    # 4. Rotation 对比 (dims 3-8: right arm rotation 6d, 13-18: left arm)
    ax_rot = fig.add_subplot(3, 2, 4)
    # 只显示前两个 rotation dims
    for i, (dim, label) in enumerate([(3, 'R_r0'), (4, 'R_r1'), (13, 'L_r0'), (14, 'L_r1')]):
        color = f'C{i}'
        ax_rot.plot(t, action_gt[:, dim], f'{color}-', label=f'GT {label}', alpha=0.7)
        ax_rot.plot(t, action_pred[:, dim], f'{color}--', label=f'Pred {label}', alpha=0.7)
    
    ax_rot.set_xlabel('Timestep')
    ax_rot.set_ylabel('Rotation (6D)')
    ax_rot.set_title('Rotation (first 2 dims per arm)')
    ax_rot.legend(loc='upper right', fontsize=8)
    ax_rot.grid(True, alpha=0.3)
    
    # 5. Gripper 对比 (dim 9: right, dim 19: left)
    ax_grip = fig.add_subplot(3, 2, 5)
    ax_grip.plot(t, action_gt[:, 9], 'C0-', label='GT Right Gripper', linewidth=2)
    ax_grip.plot(t, action_pred[:, 9], 'C0--', label='Pred Right Gripper', linewidth=2)
    ax_grip.plot(t, action_gt[:, 19], 'C1-', label='GT Left Gripper', linewidth=2)
    ax_grip.plot(t, action_pred[:, 19], 'C1--', label='Pred Left Gripper', linewidth=2)
    
    ax_grip.set_xlabel('Timestep')
    ax_grip.set_ylabel('Gripper Width')
    ax_grip.set_title('Gripper (dims 9, 19)')
    ax_grip.legend(loc='upper right', fontsize=8)
    ax_grip.grid(True, alpha=0.3)
    
    # 6. Error metrics
    ax_text = fig.add_subplot(3, 2, 6)
    ax_text.axis('off')
    
    # 计算误差
    mse = np.mean((action_gt - action_pred) ** 2)
    mae = np.mean(np.abs(action_gt - action_pred))
    
    # 分维度误差
    pos_mse_r = np.mean((action_gt[:, 0:3] - action_pred[:, 0:3]) ** 2)
    pos_mse_l = np.mean((action_gt[:, 10:13] - action_pred[:, 10:13]) ** 2)
    rot_mse_r = np.mean((action_gt[:, 3:9] - action_pred[:, 3:9]) ** 2)
    rot_mse_l = np.mean((action_gt[:, 13:19] - action_pred[:, 13:19]) ** 2)
    grip_mse_r = np.mean((action_gt[:, 9] - action_pred[:, 9]) ** 2)
    grip_mse_l = np.mean((action_gt[:, 19] - action_pred[:, 19]) ** 2)
    
    text = f"""
    Error Metrics (Flow Matching){title_suffix}
    ══════════════════════════════════════
    Overall MSE:     {mse:.6f}
    Overall MAE:     {mae:.6f}
    
    Position MSE:
      Right arm:     {pos_mse_r:.6f}
      Left arm:      {pos_mse_l:.6f}
    
    Rotation MSE:
      Right arm:     {rot_mse_r:.6f}
      Left arm:      {rot_mse_l:.6f}
    
    Gripper MSE:
      Right:         {grip_mse_r:.6f}
      Left:          {grip_mse_l:.6f}
    
    Action shape: {action_gt.shape}
    """
    ax_text.text(0.1, 0.5, text, transform=ax_text.transAxes, fontsize=11,
                 verticalalignment='center', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")
    return mse, mae


def main():
    parser = argparse.ArgumentParser(description='RDT2 FM Offline Inference')
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
    parser.add_argument('--shard-idx', type=int, default=0,
                        help='Shard index')
    parser.add_argument('--sample-idx', type=int, default=0,
                        help='Sample index within shard')
    parser.add_argument('--output-dir', type=str, default='inference_outputs_fm',
                        help='Output directory')
    parser.add_argument('--batch', action='store_true',
                        help='Batch inference mode')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples for batch mode')
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
    
    # 初始化推理器
    print(f"Loading FM model from {args.fm_checkpoint}...")
    print(f"Loading VLM from {args.vlm_model}...")
    
    inferencer = RDTInferencer(
        config=config,
        pretrained_path=args.fm_checkpoint,
        normalizer_path=args.normalizer_path,
        pretrained_vision_language_model_name_or_path=args.vlm_model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    
    # 获取 shard 文件列表
    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob('shard-*.tar'))
    print(f"Found {len(shard_files)} shards")
    
    if args.batch:
        # 批量推理
        print(f"\n{'='*60}")
        print(f"Batch Inference: {args.num_samples} samples")
        print(f"{'='*60}")
        
        all_mse = []
        all_mae = []
        
        for i in range(args.num_samples):
            shard_idx = i % len(shard_files)
            sample_idx = (i * 7) % 50  # 分散采样
            
            shard_path = shard_files[shard_idx]
            
            try:
                image, action_gt, meta = load_sample_from_shard(str(shard_path), sample_idx)
            except Exception as e:
                print(f"  Skip shard{shard_idx} sample{sample_idx}: {e}")
                continue
            
            # 获取指令
            instruction_key = meta.get('sub_task_instruction_key', 'pika_task')
            instruction = instructions.get(instruction_key, "Put the bottle into the tape ring.")
            
            # 准备输入
            # 图像是拼接的 [left, right]，需要分开
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
                action_pred = action_pred.cpu().numpy()
            
            # 可视化
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(args.output_dir) / f"fm_shard{shard_idx}_sample{sample_idx}_{timestamp}.png"
            
            mse, mae = visualize_and_save(
                image, action_gt, action_pred, instruction, str(output_path),
                title_suffix=f"\nShard {shard_idx}, Sample {sample_idx}"
            )
            
            all_mse.append(mse)
            all_mae.append(mae)
            print(f"  [{i+1}/{args.num_samples}] shard{shard_idx} sample{sample_idx}: MSE={mse:.6f}, MAE={mae:.6f}")
        
        # 汇总
        print(f"\n{'='*60}")
        print(f"Summary ({len(all_mse)} samples)")
        print(f"{'='*60}")
        print(f"  Mean MSE: {np.mean(all_mse):.6f} ± {np.std(all_mse):.6f}")
        print(f"  Mean MAE: {np.mean(all_mae):.6f} ± {np.std(all_mae):.6f}")
        
    else:
        # 单样本推理
        shard_path = shard_files[args.shard_idx]
        print(f"\nLoading sample {args.sample_idx} from {shard_path.name}...")
        
        image, action_gt, meta = load_sample_from_shard(str(shard_path), args.sample_idx)
        
        # 获取指令
        instruction_key = meta.get('sub_task_instruction_key', 'pika_task')
        instruction = instructions.get(instruction_key, "Put the bottle into the tape ring.")
        print(f"Instruction: {instruction}")
        
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
        print("Running inference...")
        with torch.no_grad():
            action_pred = inferencer.step(observations, instruction)
            action_pred = action_pred.cpu().numpy()
        
        print(f"  GT shape: {action_gt.shape}")
        print(f"  Pred shape: {action_pred.shape}")
        
        # 可视化
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(args.output_dir) / f"fm_shard{args.shard_idx}_sample{args.sample_idx}_{timestamp}.png"
        
        mse, mae = visualize_and_save(
            image, action_gt, action_pred, instruction, str(output_path),
            title_suffix=f"\nShard {args.shard_idx}, Sample {args.sample_idx}"
        )
        
        print(f"\nResults:")
        print(f"  MSE: {mse:.6f}")
        print(f"  MAE: {mae:.6f}")


if __name__ == '__main__':
    main()
