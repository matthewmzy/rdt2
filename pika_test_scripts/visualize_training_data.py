#!/usr/bin/env python
"""
可视化训练数据集，检查图片和动作是否正确
"""

import os
import sys
import json
import tarfile
import tempfile
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # 使用非交互后端
import matplotlib.pyplot as plt
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_sample_from_shard(shard_path: str, sample_idx: int = 0):
    """从 shard 中加载一个样本"""
    import io
    
    with tarfile.open(shard_path, 'r') as tar:
        members = tar.getmembers()
        
        # 按 sample key 分组
        sample_files = {}
        for member in members:
            if member.name.startswith('.'):
                continue
            # 文件名格式: 000000.image.jpg, 000000.action.npy, etc.
            parts = member.name.split('.')
            if len(parts) >= 2:
                key = parts[0]
                ext = '.'.join(parts[1:])
                if key not in sample_files:
                    sample_files[key] = {}
                sample_files[key][ext] = member
        
        # 获取指定样本
        sorted_keys = sorted(sample_files.keys())
        if sample_idx >= len(sorted_keys):
            sample_idx = len(sorted_keys) - 1
        
        key = sorted_keys[sample_idx]
        files = sample_files[key]
        
        sample = {'key': key}
        
        # 读取图片
        if 'image.jpg' in files:
            f = tar.extractfile(files['image.jpg'])
            img = Image.open(f)
            sample['image'] = np.array(img)
        
        # 读取动作 - 使用 BytesIO 包装避免 fileno 问题
        if 'action.npy' in files:
            f = tar.extractfile(files['action.npy'])
            data = io.BytesIO(f.read())
            sample['action'] = np.load(data)
        
        # 读取动作 token
        if 'action_token.npy' in files:
            f = tar.extractfile(files['action_token.npy'])
            data = io.BytesIO(f.read())
            sample['action_token'] = np.load(data)
        
        # 读取元数据
        if 'meta.json' in files:
            f = tar.extractfile(files['meta.json'])
            sample['meta'] = json.load(f)
    
    return sample, len(sorted_keys)


def visualize_sample(sample: dict, save_path: str = None):
    """可视化一个样本"""
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 显示图片
    ax1 = fig.add_subplot(2, 2, 1)
    if 'image' in sample:
        img = sample['image']
        ax1.imshow(img)
        ax1.set_title(f"Image: {img.shape}")
        
        # 检查是否是拼接图像 (左右两个相机)
        h, w = img.shape[:2]
        if w == h * 2:  # 宽度是高度的两倍，说明是左右拼接
            ax1.axvline(x=w//2, color='red', linestyle='--', linewidth=2, label='Left | Right')
            ax1.text(w//4, 20, 'LEFT', fontsize=12, color='white', ha='center',
                    bbox=dict(boxstyle='round', facecolor='blue', alpha=0.7))
            ax1.text(3*w//4, 20, 'RIGHT', fontsize=12, color='white', ha='center',
                    bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
    ax1.axis('off')
    
    # 2. 显示动作 token 分布
    ax2 = fig.add_subplot(2, 2, 2)
    if 'action_token' in sample:
        tokens = sample['action_token']
        ax2.bar(range(len(tokens)), tokens, color='steelblue')
        ax2.set_xlabel('Token Index')
        ax2.set_ylabel('Token Value')
        ax2.set_title(f'Action Tokens: shape={tokens.shape}, range=[{tokens.min()}, {tokens.max()}]')
        ax2.set_ylim(0, 1024)
        
        # 标注 token 类型
        ax2.axvspan(0, 17.5, alpha=0.2, color='red', label='Position (18)')
        ax2.axvspan(17.5, 23.5, alpha=0.2, color='green', label='Rotation (6)')
        ax2.axvspan(23.5, 27, alpha=0.2, color='blue', label='Gripper (3)')
        ax2.legend(loc='upper right')
    
    # 3. 显示动作值 - 位置
    ax3 = fig.add_subplot(2, 2, 3)
    if 'action' in sample:
        action = sample['action']  # (24, 20)
        
        # 右臂位置 (0:3)
        ax3.plot(action[:, 0], label='R_x', color='red', linestyle='-')
        ax3.plot(action[:, 1], label='R_y', color='green', linestyle='-')
        ax3.plot(action[:, 2], label='R_z', color='blue', linestyle='-')
        # 左臂位置 (10:13)
        ax3.plot(action[:, 10], label='L_x', color='red', linestyle='--')
        ax3.plot(action[:, 11], label='L_y', color='green', linestyle='--')
        ax3.plot(action[:, 12], label='L_z', color='blue', linestyle='--')
        
        ax3.set_xlabel('Frame')
        ax3.set_ylabel('Position (relative)')
        ax3.set_title(f'Position Actions: shape={action.shape}')
        ax3.legend(ncol=2)
        ax3.grid(True, alpha=0.3)
    
    # 4. 显示动作值 - 夹爪
    ax4 = fig.add_subplot(2, 2, 4)
    if 'action' in sample:
        action = sample['action']
        
        # 右臂夹爪 (9) 和左臂夹爪 (19)
        ax4.plot(action[:, 9], label='Right Gripper', color='orange', linewidth=2)
        ax4.plot(action[:, 19], label='Left Gripper', color='purple', linewidth=2)
        
        ax4.axhline(y=0.088, color='red', linestyle='--', alpha=0.5, label='Official Max (0.088)')
        ax4.axhline(y=0.0, color='blue', linestyle='--', alpha=0.5, label='Official Min (0.0)')
        
        ax4.set_xlabel('Frame')
        ax4.set_ylabel('Gripper Width')
        ax4.set_title('Gripper Actions (should be in [0, 0.088])')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    # 元数据信息
    if 'meta' in sample:
        meta = sample['meta']
        info_text = f"Episode: {meta.get('episode_idx', 'N/A')}\n"
        info_text += f"Frame: {meta.get('frame_idx', 'N/A')}\n"
        info_text += f"Instruction: {meta.get('instruction', 'N/A')[:50]}..."
        fig.text(0.02, 0.02, info_text, fontsize=9, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close(fig)  # 关闭图形，避免内存泄漏
    return fig


def main():
    # 数据集路径
    shard_dir = Path("/home/ubuntu/mzy/RDT2/rdt2_pika_shards")
    output_dir = Path("/home/ubuntu/mzy/RDT2/pika_test_scripts/data_visualization")
    output_dir.mkdir(exist_ok=True)
    
    # 获取所有 shard 文件
    shard_files = sorted(shard_dir.glob("shard-*.tar"))
    print(f"Found {len(shard_files)} shards")
    
    # 从不同的 shard 中抽取样本
    samples_to_show = [
        (0, 0),   # shard 0, sample 0
        (0, 50),  # shard 0, sample 50
        (5, 100), # shard 5, sample 100 (如果存在)
        (10, 200), # shard 10, sample 200 (如果存在)
    ]
    
    for shard_idx, sample_idx in samples_to_show:
        if shard_idx >= len(shard_files):
            print(f"Shard {shard_idx} not found, skipping...")
            continue
        
        shard_path = shard_files[shard_idx]
        print(f"\n{'='*60}")
        print(f"Loading from {shard_path.name}, sample {sample_idx}")
        print('='*60)
        
        try:
            sample, total_samples = load_sample_from_shard(str(shard_path), sample_idx)
            print(f"Total samples in shard: {total_samples}")
            print(f"Sample key: {sample['key']}")
            
            if 'action' in sample:
                action = sample['action']
                print(f"\nAction shape: {action.shape}")
                print(f"Action dtype: {action.dtype}")
                print(f"Position range (right): [{action[:, 0:3].min():.4f}, {action[:, 0:3].max():.4f}]")
                print(f"Position range (left): [{action[:, 10:13].min():.4f}, {action[:, 10:13].max():.4f}]")
                print(f"Gripper range (right): [{action[:, 9].min():.4f}, {action[:, 9].max():.4f}]")
                print(f"Gripper range (left): [{action[:, 19].min():.4f}, {action[:, 19].max():.4f}]")
            
            if 'action_token' in sample:
                tokens = sample['action_token']
                print(f"\nAction token shape: {tokens.shape}")
                print(f"Action token range: [{tokens.min()}, {tokens.max()}]")
                print(f"Action tokens: {tokens}")
                
                # 检查是否全是 0
                if tokens.max() == 0:
                    print("⚠️ WARNING: All tokens are 0! Data not fixed properly!")
                else:
                    print("✓ Tokens look valid (not all zeros)")
            
            if 'image' in sample:
                img = sample['image']
                print(f"\nImage shape: {img.shape}")
                print(f"Image dtype: {img.dtype}")
            
            # 保存可视化
            save_path = output_dir / f"shard{shard_idx}_sample{sample_idx}.png"
            visualize_sample(sample, str(save_path))
            
        except Exception as e:
            print(f"Error loading sample: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n✓ Visualizations saved to {output_dir}")


if __name__ == '__main__':
    main()
