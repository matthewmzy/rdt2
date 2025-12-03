#!/usr/bin/env python3
"""
RDT2 离线推理脚本
用于测试训练好的 LoRA checkpoint，不需要真机设备。
输出预测的动作和观测图像用于 debug。
支持计算与训练一致的 CrossEntropy loss。

重要说明：
    RVQ Tokenizer 是在官方数据上训练的，它期望的数据分布与官方 normalizer 一致。
    如果你的数据使用了自定义 normalizer，推理时也必须使用相同的 normalizer。
    
    推荐做法：
    1. 数据转换时使用官方 normalizer (convert_pika_to_rdt2_official_norm.py)
    2. 推理时也使用官方 normalizer (--normalizer-path normalizer.pt)
"""

import os
import sys
import io
import json
import tarfile
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 添加项目根目录到 path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.normalizer import LinearNormalizer
from vqvae.models.multivqvae import MultiVQVAE
from data.image_corrupt import image_corrupt


def compute_training_loss(
    processor,
    model,
    image_pil: Image.Image,
    instruction: str,
    gt_action_tokens: np.ndarray,  # [27] or [480], VAE codebook indices
    device: str = "cuda",
):
    """
    计算与训练时完全一致的 CrossEntropy loss。
    
    训练时的流程:
    1. 构建 messages，包含 instruction 和 <action> placeholder
    2. 将 <action> placeholder 替换为 GT action token ids
    3. 构建 labels，只在 assistant response 部分计算 loss
    4. 前向传播计算 CrossEntropyLoss
    
    Args:
        processor: Qwen2.5-VL processor
        model: 加载了 LoRA 的模型
        image_pil: PIL Image 输入
        instruction: 任务指令
        gt_action_tokens: GT 动作 token (VAE codebook indices)
        device: 设备
        
    Returns:
        loss: 与训练一致的 loss 值
    """
    # Step 1: 将 GT action tokens 转换为 input_ids (与 train.py 的 collate_fn 一致)
    # action_input_ids = vocab_size - (action_tokens + 1)
    gt_action_tokens_tensor = torch.from_numpy(gt_action_tokens).long()
    action_input_ids = processor.tokenizer.vocab_size - (gt_action_tokens_tensor + 1)
    action_input_ids = action_input_ids.tolist()
    
    # Step 2: 构建 messages (与 train.py 的 collate_fn 一致)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"<|quad_start|>{'<action>' * len(action_input_ids)}<|quad_end|>"}
            ],
        },
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    text = text.strip()
    
    # Step 3: 处理输入
    batch = processor(text=[text], images=[[image_pil]], return_tensors="pt", padding=True)
    
    # Step 4: 将 <action> placeholder 替换为 GT action token ids
    input_ids = batch["input_ids"][0]
    action_token_id = processor.tokenizer.additional_special_tokens_ids[
        processor.tokenizer.additional_special_tokens.index("<action>")
    ]
    action_idx = (input_ids == action_token_id)
    action_positions = action_idx.nonzero(as_tuple=True)[0]
    
    for idx, action_id in zip(action_positions, action_input_ids):
        input_ids[idx] = action_id
    batch["input_ids"][0] = input_ids
    
    # Step 5: 构建 labels (与 train.py 的 collate_fn 一致)
    labels = batch["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    # 找到 assistant marker 的位置，只在 assistant response 后计算 loss
    assistant_marker = "assistant"
    assistant_marker_id = processor.tokenizer.convert_tokens_to_ids(assistant_marker)
    input_ids_list = batch["input_ids"][0].tolist()
    
    try:
        start_index = input_ids_list.index(assistant_marker_id)
    except ValueError:
        start_index = len(input_ids_list)
    
    labels[0, :start_index - 1] = -100
    batch["labels"] = labels
    
    # Step 6: 前向传播计算 loss
    batch = {k: v.to(device) for k, v in batch.items()}
    
    with torch.no_grad():
        outputs = model(**batch)
    
    return outputs.loss.item()


def load_sample_from_webdataset(shard_path: str, sample_idx: int = 0):
    """
    从 WebDataset shard 中加载一个样本
    
    Args:
        shard_path: shard tar 文件路径
        sample_idx: 要加载的样本索引
        
    Returns:
        dict: 包含 image, action, action_token, meta 的字典
    """
    with tarfile.open(shard_path, 'r') as tar:
        # 获取所有文件名
        members = tar.getnames()
        
        # 找到指定索引的文件
        image_name = f"{sample_idx}.image.jpg"
        action_name = f"{sample_idx}.action.npy"
        action_token_name = f"{sample_idx}.action_token.npy"
        meta_name = f"{sample_idx}.meta.json"
        
        sample = {}
        
        # 读取图像
        if image_name in members:
            f = tar.extractfile(image_name)
            image_bytes = f.read()
            image = Image.open(io.BytesIO(image_bytes))
            sample['image'] = np.array(image)
        else:
            raise FileNotFoundError(f"Image {image_name} not found in {shard_path}")
        
        # 读取动作
        if action_name in members:
            f = tar.extractfile(action_name)
            sample['action'] = np.load(io.BytesIO(f.read()))
        
        # 读取动作 token
        if action_token_name in members:
            f = tar.extractfile(action_token_name)
            sample['action_token'] = np.load(io.BytesIO(f.read()))
        
        # 读取元数据
        if meta_name in members:
            f = tar.extractfile(meta_name)
            sample['meta'] = json.load(f)
            
    return sample


def load_models(
    base_model_path: str,
    lora_path: str,
    vae_path: str,
    normalizer_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """
    加载模型、VAE 和 normalizer
    """
    print(f"[INFO] Loading processor from {base_model_path}...")
    processor = AutoProcessor.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        use_fast=True
    )
    
    # 添加 <action> 特殊 token (与 train.py 保持一致)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<action>"]},
        replace_additional_special_tokens=False,
    )
    print(f"[INFO] Added <action> special token to tokenizer")
    
    print(f"[INFO] Loading base model from {base_model_path}...")
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    
    print(f"[INFO] Loading LoRA adapter from {lora_path}...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    model.eval()
    
    print(f"[INFO] Loading VAE from {vae_path}...")
    vae = MultiVQVAE.from_pretrained(vae_path)
    vae = vae.to(device=device, dtype=torch.float32)
    vae.eval()
    vae_device = device  # 保存 VAE 设备
    
    # 计算有效的 action token 长度
    valid_action_id_length = vae.pos_id_len + vae.rot_id_len + vae.grip_id_len
    print(f"[INFO] VAE token length: {valid_action_id_length}")
    
    print(f"[INFO] Loading normalizer from {normalizer_path}...")
    normalizer = LinearNormalizer.load(normalizer_path)
    
    return processor, model, vae, normalizer, vae_device, valid_action_id_length


def predict_action(
    processor,
    model,
    vae,
    normalizer,
    image: np.ndarray,
    instruction: str,
    vae_device: str = "cuda",
    valid_action_id_length: int = 27,  # VAE token length (pos+rot+grip)
    apply_jpeg_compression: bool = True,
):
    """
    执行单次推理
    
    Args:
        processor: Qwen processor
        model: VLA model (with LoRA)
        vae: RVQ VAE decoder
        normalizer: action normalizer
        image: 拼接后的图像 [H, W*2, 3] uint8
        instruction: 任务指令
        apply_jpeg_compression: 是否应用 JPEG 压缩 (与训练保持一致)
        valid_action_id_length: 动作 token 长度
        
    Returns:
        dict: 包含预测动作和中间结果
    """
    device = model.device
    
    # 准备图像
    if apply_jpeg_compression:
        # 应用 JPEG 压缩以匹配训练时的数据增强
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encoded = cv2.imencode('.jpg', image[..., ::-1], encode_param)  # RGB -> BGR for cv2
        image_pil = Image.open(io.BytesIO(encoded.tobytes()))
    else:
        image_pil = Image.fromarray(image)
    
    # 构建消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": instruction}
            ]
        }
    ]
    
    text = processor.apply_chat_template(messages, add_generation_prompt=False)
    # 插入引导 token
    text += "<|im_start|>assistant\n<|quad_start|>"
    
    # 处理输入
    inputs = processor(
        text=[text], 
        images=[[image_pil]], 
        padding=True, 
        return_tensors="pt"
    ).to(model.device)
    
    # 生成
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs, 
            max_new_tokens=(valid_action_id_length + 2),
            do_sample=False,  # 确定性采样
        )
    
    # 提取生成的 action ids
    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    
    # 解析 action ids
    quad_end_id = processor.tokenizer.convert_tokens_to_ids("<|quad_end|>")
    
    action_ids = torch.zeros(valid_action_id_length, dtype=torch.long, device=device)
    
    quad_end_idx = (generated_ids[0] == quad_end_id).nonzero(as_tuple=True)[0]
    
    action_valid = False
    if len(quad_end_idx) > 0:
        start_idx = 0
        end_idx = quad_end_idx[0]
        
        if end_idx - start_idx == valid_action_id_length:
            action_ids = generated_ids[0, start_idx:end_idx]
            action_valid = True
    
    # 解码动作
    action_ids = action_ids.unsqueeze(0)  # [1, 480]
    
    # 映射 token id 到 VAE codebook index
    action_tokens = processor.tokenizer.vocab_size - (action_ids + 1)
    action_tokens = torch.clamp(action_tokens, min=0, max=vae.num_embeddings - 1)
    
    # VAE 解码
    with torch.no_grad():
        action_normalized = vae.decode(action_tokens.to(vae_device))  # [1, 24, 20]
    
    # 反归一化
    action_pred = normalizer['action'].unnormalize(action_normalized)
    
    return {
        'action_pred': action_pred.detach().cpu().numpy(),  # [1, 24, 20]
        'action_ids': action_ids.detach().cpu().numpy(),
        'action_valid': action_valid,
        'generated_ids': generated_ids.detach().cpu().numpy(),
    }


def visualize_and_save(
    image: np.ndarray,
    gt_action: np.ndarray,
    pred_action: np.ndarray,
    instruction: str,
    output_path: str,
    sample_info: str = "",
):
    """
    可视化并保存结果
    
    动作格式 (20维):
    - 右臂: [pos(3), rot6d(6), gripper(1)] = dims 0-9
    - 左臂: [pos(3), rot6d(6), gripper(1)] = dims 10-19
    """
    import matplotlib.pyplot as plt
    
    fig = plt.figure(figsize=(24, 20))
    
    timesteps = np.arange(gt_action.shape[0])
    
    # ==================== Row 1: Image and Instruction ====================
    # 1. 显示观测图像
    ax1 = fig.add_subplot(4, 4, 1)
    ax1.imshow(image)
    ax1.set_title(f"Observation Image\n{sample_info}", fontsize=10)
    ax1.axis('off')
    
    # 2. 显示指令和误差统计
    ax2 = fig.add_subplot(4, 4, 2)
    
    # 计算各部分误差
    pos_mse_r = np.mean((gt_action[:, 0:3] - pred_action[:, 0:3]) ** 2)
    pos_mse_l = np.mean((gt_action[:, 10:13] - pred_action[:, 10:13]) ** 2)
    rot_mse_r = np.mean((gt_action[:, 3:9] - pred_action[:, 3:9]) ** 2)
    rot_mse_l = np.mean((gt_action[:, 13:19] - pred_action[:, 13:19]) ** 2)
    grip_mse_r = np.mean((gt_action[:, 9] - pred_action[:, 9]) ** 2)
    grip_mse_l = np.mean((gt_action[:, 19] - pred_action[:, 19]) ** 2)
    
    stats_text = f"Instruction:\n{instruction}\n\n"
    stats_text += "="*40 + "\n"
    stats_text += "MSE Statistics:\n"
    stats_text += f"  Right Position: {pos_mse_r:.6f}\n"
    stats_text += f"  Left Position:  {pos_mse_l:.6f}\n"
    stats_text += f"  Right Rotation: {rot_mse_r:.6f}\n"
    stats_text += f"  Left Rotation:  {rot_mse_l:.6f}\n"
    stats_text += f"  Right Gripper:  {grip_mse_r:.6f}\n"
    stats_text += f"  Left Gripper:   {grip_mse_l:.6f}\n"
    
    ax2.text(0.05, 0.95, stats_text, fontsize=9, ha='left', va='top', 
             transform=ax2.transAxes, family='monospace')
    ax2.axis('off')
    ax2.set_title("Task Info & Error Statistics")
    
    # ==================== Row 2: Position ====================
    # 3. 右臂位置
    ax3 = fig.add_subplot(4, 4, 3)
    ax3.plot(timesteps, gt_action[:, 0], 'r-', label='GT X', linewidth=2)
    ax3.plot(timesteps, gt_action[:, 1], 'g-', label='GT Y', linewidth=2)
    ax3.plot(timesteps, gt_action[:, 2], 'b-', label='GT Z', linewidth=2)
    ax3.plot(timesteps, pred_action[:, 0], 'r--', label='Pred X', linewidth=2, alpha=0.7)
    ax3.plot(timesteps, pred_action[:, 1], 'g--', label='Pred Y', linewidth=2, alpha=0.7)
    ax3.plot(timesteps, pred_action[:, 2], 'b--', label='Pred Z', linewidth=2, alpha=0.7)
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Position (m)')
    ax3.set_title(f'Right Arm Position (MSE: {pos_mse_r:.6f})')
    ax3.legend(loc='upper right', fontsize=7)
    ax3.grid(True, alpha=0.3)
    
    # 4. 左臂位置
    ax4 = fig.add_subplot(4, 4, 4)
    ax4.plot(timesteps, gt_action[:, 10], 'r-', label='GT X', linewidth=2)
    ax4.plot(timesteps, gt_action[:, 11], 'g-', label='GT Y', linewidth=2)
    ax4.plot(timesteps, gt_action[:, 12], 'b-', label='GT Z', linewidth=2)
    ax4.plot(timesteps, pred_action[:, 10], 'r--', label='Pred X', linewidth=2, alpha=0.7)
    ax4.plot(timesteps, pred_action[:, 11], 'g--', label='Pred Y', linewidth=2, alpha=0.7)
    ax4.plot(timesteps, pred_action[:, 12], 'b--', label='Pred Z', linewidth=2, alpha=0.7)
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Position (m)')
    ax4.set_title(f'Left Arm Position (MSE: {pos_mse_l:.6f})')
    ax4.legend(loc='upper right', fontsize=7)
    ax4.grid(True, alpha=0.3)
    
    # ==================== Row 3: Rotation (6D representation) ====================
    # 5. 右臂旋转 (前3维 of rot6d)
    ax5 = fig.add_subplot(4, 4, 5)
    ax5.plot(timesteps, gt_action[:, 3], 'r-', label='GT r0', linewidth=2)
    ax5.plot(timesteps, gt_action[:, 4], 'g-', label='GT r1', linewidth=2)
    ax5.plot(timesteps, gt_action[:, 5], 'b-', label='GT r2', linewidth=2)
    ax5.plot(timesteps, pred_action[:, 3], 'r--', label='Pred r0', linewidth=2, alpha=0.7)
    ax5.plot(timesteps, pred_action[:, 4], 'g--', label='Pred r1', linewidth=2, alpha=0.7)
    ax5.plot(timesteps, pred_action[:, 5], 'b--', label='Pred r2', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Rot6D (col0)')
    ax5.set_title('Right Arm Rotation (col0)')
    ax5.legend(loc='upper right', fontsize=7)
    ax5.grid(True, alpha=0.3)
    
    # 6. 右臂旋转 (后3维 of rot6d)
    ax6 = fig.add_subplot(4, 4, 6)
    ax6.plot(timesteps, gt_action[:, 6], 'r-', label='GT r3', linewidth=2)
    ax6.plot(timesteps, gt_action[:, 7], 'g-', label='GT r4', linewidth=2)
    ax6.plot(timesteps, gt_action[:, 8], 'b-', label='GT r5', linewidth=2)
    ax6.plot(timesteps, pred_action[:, 6], 'r--', label='Pred r3', linewidth=2, alpha=0.7)
    ax6.plot(timesteps, pred_action[:, 7], 'g--', label='Pred r4', linewidth=2, alpha=0.7)
    ax6.plot(timesteps, pred_action[:, 8], 'b--', label='Pred r5', linewidth=2, alpha=0.7)
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Rot6D (col1)')
    ax6.set_title(f'Right Arm Rotation (col1, MSE: {rot_mse_r:.6f})')
    ax6.legend(loc='upper right', fontsize=7)
    ax6.grid(True, alpha=0.3)
    
    # 7. 左臂旋转 (前3维 of rot6d)
    ax7 = fig.add_subplot(4, 4, 7)
    ax7.plot(timesteps, gt_action[:, 13], 'r-', label='GT r0', linewidth=2)
    ax7.plot(timesteps, gt_action[:, 14], 'g-', label='GT r1', linewidth=2)
    ax7.plot(timesteps, gt_action[:, 15], 'b-', label='GT r2', linewidth=2)
    ax7.plot(timesteps, pred_action[:, 13], 'r--', label='Pred r0', linewidth=2, alpha=0.7)
    ax7.plot(timesteps, pred_action[:, 14], 'g--', label='Pred r1', linewidth=2, alpha=0.7)
    ax7.plot(timesteps, pred_action[:, 15], 'b--', label='Pred r2', linewidth=2, alpha=0.7)
    ax7.set_xlabel('Timestep')
    ax7.set_ylabel('Rot6D (col0)')
    ax7.set_title('Left Arm Rotation (col0)')
    ax7.legend(loc='upper right', fontsize=7)
    ax7.grid(True, alpha=0.3)
    
    # 8. 左臂旋转 (后3维 of rot6d)
    ax8 = fig.add_subplot(4, 4, 8)
    ax8.plot(timesteps, gt_action[:, 16], 'r-', label='GT r3', linewidth=2)
    ax8.plot(timesteps, gt_action[:, 17], 'g-', label='GT r4', linewidth=2)
    ax8.plot(timesteps, gt_action[:, 18], 'b-', label='GT r5', linewidth=2)
    ax8.plot(timesteps, pred_action[:, 16], 'r--', label='Pred r3', linewidth=2, alpha=0.7)
    ax8.plot(timesteps, pred_action[:, 17], 'g--', label='Pred r4', linewidth=2, alpha=0.7)
    ax8.plot(timesteps, pred_action[:, 18], 'b--', label='Pred r5', linewidth=2, alpha=0.7)
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Rot6D (col1)')
    ax8.set_title(f'Left Arm Rotation (col1, MSE: {rot_mse_l:.6f})')
    ax8.legend(loc='upper right', fontsize=7)
    ax8.grid(True, alpha=0.3)
    
    # ==================== Row 4: Gripper ====================
    # 9. 右臂夹爪
    ax9 = fig.add_subplot(4, 4, 9)
    ax9.plot(timesteps, gt_action[:, 9], 'b-', label='GT Gripper', linewidth=2)
    ax9.plot(timesteps, pred_action[:, 9], 'r--', label='Pred Gripper', linewidth=2, alpha=0.7)
    ax9.fill_between(timesteps, gt_action[:, 9], pred_action[:, 9], alpha=0.3, color='gray')
    ax9.set_xlabel('Timestep')
    ax9.set_ylabel('Gripper Width (m)')
    ax9.set_title(f'Right Gripper (MSE: {grip_mse_r:.6f})')
    ax9.legend(loc='upper right', fontsize=8)
    ax9.grid(True, alpha=0.3)
    ax9.set_ylim([-0.01, 0.1])  # gripper range [0, 0.088]
    
    # 10. 左臂夹爪
    ax10 = fig.add_subplot(4, 4, 10)
    ax10.plot(timesteps, gt_action[:, 19], 'b-', label='GT Gripper', linewidth=2)
    ax10.plot(timesteps, pred_action[:, 19], 'r--', label='Pred Gripper', linewidth=2, alpha=0.7)
    ax10.fill_between(timesteps, gt_action[:, 19], pred_action[:, 19], alpha=0.3, color='gray')
    ax10.set_xlabel('Timestep')
    ax10.set_ylabel('Gripper Width (m)')
    ax10.set_title(f'Left Gripper (MSE: {grip_mse_l:.6f})')
    ax10.legend(loc='upper right', fontsize=8)
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim([-0.01, 0.1])  # gripper range [0, 0.088]
    
    # ==================== Row 4 (continued): Error distribution ====================
    # 11. 各维度误差分布 (右臂)
    ax11 = fig.add_subplot(4, 4, 11)
    error_r = np.abs(gt_action[:, :10] - pred_action[:, :10]).mean(axis=0)
    dim_labels_r = ['x', 'y', 'z', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'grip']
    colors_r = ['red', 'green', 'blue'] + ['orange']*6 + ['purple']
    ax11.bar(range(10), error_r, color=colors_r, alpha=0.7)
    ax11.set_xticks(range(10))
    ax11.set_xticklabels(dim_labels_r, fontsize=8)
    ax11.set_ylabel('Mean Absolute Error')
    ax11.set_title('Right Arm Error by Dimension')
    ax11.grid(True, alpha=0.3, axis='y')
    
    # 12. 各维度误差分布 (左臂)
    ax12 = fig.add_subplot(4, 4, 12)
    error_l = np.abs(gt_action[:, 10:20] - pred_action[:, 10:20]).mean(axis=0)
    dim_labels_l = ['x', 'y', 'z', 'r0', 'r1', 'r2', 'r3', 'r4', 'r5', 'grip']
    colors_l = ['red', 'green', 'blue'] + ['orange']*6 + ['purple']
    ax12.bar(range(10), error_l, color=colors_l, alpha=0.7)
    ax12.set_xticks(range(10))
    ax12.set_xticklabels(dim_labels_l, fontsize=8)
    ax12.set_ylabel('Mean Absolute Error')
    ax12.set_title('Left Arm Error by Dimension')
    ax12.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Visualization saved to {output_path}")


def print_action_comparison(gt_action: np.ndarray, pred_action: np.ndarray, loss: float = None):
    """
    打印动作对比
    """
    print("\n" + "="*80)
    print("ACTION COMPARISON (first 5 timesteps)")
    print("="*80)
    
    if loss is not None:
        print(f"\n[Training-consistent CrossEntropy Loss]: {loss:.4f}")
    
    print("\n[Right Arm] Format: [dx, dy, dz, rot6d(6), gripper]")
    print("-"*80)
    for t in range(min(5, gt_action.shape[0])):
        gt_right = gt_action[t, :10]
        pred_right = pred_action[t, :10]
        print(f"  t={t:02d} GT:   pos=[{gt_right[0]:+.4f}, {gt_right[1]:+.4f}, {gt_right[2]:+.4f}] gripper={gt_right[9]:.4f}")
        print(f"       Pred: pos=[{pred_right[0]:+.4f}, {pred_right[1]:+.4f}, {pred_right[2]:+.4f}] gripper={pred_right[9]:.4f}")
    
    print("\n[Left Arm] Format: [dx, dy, dz, rot6d(6), gripper]")
    print("-"*80)
    for t in range(min(5, gt_action.shape[0])):
        gt_left = gt_action[t, 10:20]
        pred_left = pred_action[t, 10:20]
        print(f"  t={t:02d} GT:   pos=[{gt_left[0]:+.4f}, {gt_left[1]:+.4f}, {gt_left[2]:+.4f}] gripper={gt_left[9]:.4f}")
        print(f"       Pred: pos=[{pred_left[0]:+.4f}, {pred_left[1]:+.4f}, {pred_left[2]:+.4f}] gripper={pred_left[9]:.4f}")
    
    # 计算误差
    mse = np.mean((gt_action - pred_action) ** 2)
    pos_mse = np.mean((gt_action[:, [0,1,2,10,11,12]] - pred_action[:, [0,1,2,10,11,12]]) ** 2)
    
    print("\n[Error Metrics]")
    print(f"  Overall MSE: {mse:.6f}")
    print(f"  Position MSE: {pos_mse:.6f}")
    print("="*80 + "\n")


def get_all_samples_in_shard(shard_path: str):
    """
    获取 shard 中的所有样本索引
    """
    with tarfile.open(shard_path, 'r') as tar:
        members = tar.getnames()
        # 找到所有 .image.jpg 文件
        sample_indices = set()
        for name in members:
            if name.endswith('.image.jpg'):
                idx = int(name.split('.')[0])
                sample_indices.add(idx)
        return sorted(list(sample_indices))


def run_batch_inference(
    processor,
    model,
    vae,
    normalizer,
    shard_dir: str,
    instruction: str,
    vae_device: str,
    valid_action_id_length: int,
    num_samples: int = 5,
    apply_jpeg_compression: bool = True,
    output_dir: str = "inference_outputs",
):
    """
    批量推理多个样本并计算 loss
    
    Args:
        num_samples: 要推理的样本数量
    """
    import random
    import matplotlib.pyplot as plt
    
    # 获取所有 shard 文件
    shard_files = sorted([f for f in os.listdir(shard_dir) if f.startswith('shard-') and f.endswith('.tar')])
    print(f"[INFO] Found {len(shard_files)} shard files")
    
    # 收集所有可用样本
    all_samples = []
    for shard_file in shard_files:
        shard_path = os.path.join(shard_dir, shard_file)
        shard_idx = int(shard_file.split('-')[1].split('.')[0])
        sample_indices = get_all_samples_in_shard(shard_path)
        for sample_idx in sample_indices:
            all_samples.append((shard_idx, sample_idx, shard_path))
    
    print(f"[INFO] Total samples available: {len(all_samples)}")
    
    # 随机选择样本
    if num_samples > len(all_samples):
        num_samples = len(all_samples)
    selected_samples = random.sample(all_samples, num_samples)
    
    print(f"\n[INFO] Running inference on {num_samples} samples...")
    
    # 存储结果
    results = []
    losses = []
    
    for i, (shard_idx, sample_idx, shard_path) in enumerate(selected_samples):
        print(f"\n{'='*80}")
        print(f"SAMPLE {i+1}/{num_samples}: Shard {shard_idx}, Sample {sample_idx}")
        print(f"{'='*80}")
        
        # 加载样本
        sample = load_sample_from_webdataset(shard_path, sample_idx)
        
        # 准备图像用于 loss 计算
        image_np = sample['image']
        if apply_jpeg_compression:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
            _, encoded = cv2.imencode('.jpg', image_np[..., ::-1], encode_param)
            image_pil = Image.open(io.BytesIO(encoded.tobytes()))
        else:
            image_pil = Image.fromarray(image_np)
        
        # 计算与训练一致的 loss
        gt_action_tokens = sample['action_token']
        loss = compute_training_loss(
            processor=processor,
            model=model,
            image_pil=image_pil,
            instruction=instruction,
            gt_action_tokens=gt_action_tokens,
            device=model.device,
        )
        losses.append(loss)
        print(f"[LOSS] Training-consistent CrossEntropy Loss: {loss:.4f}")
        
        # 执行推理
        result = predict_action(
            processor=processor,
            model=model,
            vae=vae,
            normalizer=normalizer,
            image=image_np,
            instruction=instruction,
            vae_device=vae_device,
            valid_action_id_length=valid_action_id_length,
            apply_jpeg_compression=apply_jpeg_compression,
        )
        
        gt_action = sample['action']
        pred_action = result['action_pred'][0]
        
        # 计算误差
        mse = np.mean((gt_action - pred_action) ** 2)
        pos_mse = np.mean((gt_action[:, [0,1,2,10,11,12]] - pred_action[:, [0,1,2,10,11,12]]) ** 2)
        
        results.append({
            'shard_idx': shard_idx,
            'sample_idx': sample_idx,
            'loss': loss,
            'mse': mse,
            'pos_mse': pos_mse,
            'action_valid': result['action_valid'],
            'gt_action': gt_action,
            'pred_action': pred_action,
        })
        
        print(f"[MSE] Overall: {mse:.6f}, Position: {pos_mse:.6f}")
        print(f"[VALID] Action valid: {result['action_valid']}")
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("BATCH INFERENCE SUMMARY")
    print("="*80)
    
    avg_loss = np.mean(losses)
    std_loss = np.std(losses)
    avg_mse = np.mean([r['mse'] for r in results])
    avg_pos_mse = np.mean([r['pos_mse'] for r in results])
    valid_rate = np.mean([r['action_valid'] for r in results])
    
    print(f"\n[Statistics over {num_samples} samples]")
    print(f"  CrossEntropy Loss: {avg_loss:.4f} ± {std_loss:.4f}")
    print(f"  (Compare to training final loss: ~1.15)")
    print(f"  Overall MSE: {avg_mse:.6f}")
    print(f"  Position MSE: {avg_pos_mse:.6f}")
    print(f"  Action Valid Rate: {valid_rate*100:.1f}%")
    
    print("\n[Per-sample Results]")
    print("-"*80)
    print(f"{'Sample':<20} {'Loss':<10} {'MSE':<12} {'Pos MSE':<12} {'Valid'}")
    print("-"*80)
    for r in results:
        sample_id = f"shard{r['shard_idx']}_idx{r['sample_idx']}"
        print(f"{sample_id:<20} {r['loss']:<10.4f} {r['mse']:<12.6f} {r['pos_mse']:<12.6f} {r['action_valid']}")
    
    # 绘制 loss 分布图
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss 分布
    ax1 = axes[0]
    ax1.bar(range(len(losses)), losses, color='steelblue', alpha=0.7)
    ax1.axhline(y=avg_loss, color='red', linestyle='--', label=f'Mean: {avg_loss:.4f}')
    ax1.axhline(y=1.15, color='green', linestyle=':', label='Training final: ~1.15')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('CrossEntropy Loss')
    ax1.set_title('Inference Loss (Training-consistent)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # MSE 分布
    ax2 = axes[1]
    ax2.bar(range(len(results)), [r['mse'] for r in results], color='coral', alpha=0.7)
    ax2.axhline(y=avg_mse, color='red', linestyle='--', label=f'Mean: {avg_mse:.6f}')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('MSE')
    ax2.set_title('Action MSE')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Position MSE 分布
    ax3 = axes[2]
    ax3.bar(range(len(results)), [r['pos_mse'] for r in results], color='seagreen', alpha=0.7)
    ax3.axhline(y=avg_pos_mse, color='red', linestyle='--', label=f'Mean: {avg_pos_mse:.6f}')
    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Position MSE')
    ax3.set_title('Position MSE')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, f"batch_inference_summary_{timestamp}.png")
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[INFO] Summary plot saved to {summary_path}")
    
    # 保存结果到 npz
    results_path = os.path.join(output_dir, f"batch_inference_results_{timestamp}.npz")
    np.savez(
        results_path,
        losses=np.array(losses),
        mses=np.array([r['mse'] for r in results]),
        pos_mses=np.array([r['pos_mse'] for r in results]),
        valid_rates=np.array([r['action_valid'] for r in results]),
        avg_loss=avg_loss,
        std_loss=std_loss,
        avg_mse=avg_mse,
        avg_pos_mse=avg_pos_mse,
        sample_info=[(r['shard_idx'], r['sample_idx']) for r in results],
    )
    print(f"[INFO] Results saved to {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RDT2 离线推理脚本")
    
    # 模型路径
    parser.add_argument("--base-model", type=str, 
                        default="robotics-diffusion-transformer/RDT2-VQ",
                        help="Base model path or HuggingFace ID")
    parser.add_argument("--lora-path", type=str, 
                        default="outputs/vqvla-sft-pika-bottle-fisheye-lora/checkpoint-5000",
                        help="LoRA checkpoint path")
    parser.add_argument("--vae-path", type=str,
                        default="robotics-diffusion-transformer/RVQActionTokenizer",
                        help="VAE model path")
    parser.add_argument("--normalizer-path", type=str,
                        default="normalizer.pt",
                        help="Normalizer checkpoint path (use project root normalizer.pt)")
    
    # 数据路径
    parser.add_argument("--shard-dir", type=str,
                        default="rdt2_pika_shards",
                        help="WebDataset shards directory")
    parser.add_argument("--shard-idx", type=int, default=0,
                        help="Shard index to load")
    parser.add_argument("--sample-idx", type=int, default=0,
                        help="Sample index within shard")
    
    # 推理参数
    parser.add_argument("--instruction", type=str, default=None,
                        help="Override instruction (default: from instruction.json)")
    parser.add_argument("--output-dir", type=str, default="inference_outputs",
                        help="Output directory for visualizations")
    parser.add_argument("--no-jpeg-compression", action="store_true",
                        help="Disable JPEG compression")
    
    # 批量推理参数
    parser.add_argument("--batch", action="store_true",
                        help="Run batch inference on multiple random samples")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of samples for batch inference")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置环境
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # 加载指令
    if args.instruction is None:
        instruction_path = os.path.join(args.shard_dir, "instruction.json")
        with open(instruction_path, 'r') as f:
            instructions = json.load(f)
        # 获取第一个指令
        instruction = list(instructions.values())[0]
    else:
        instruction = args.instruction
    print(f"[INFO] Instruction: {instruction}")
    
    # 加载模型
    print("\n" + "="*80)
    print("LOADING MODELS")
    print("="*80)
    processor, model, vae, normalizer, vae_device, valid_action_id_length = load_models(
        base_model_path=args.base_model,
        lora_path=args.lora_path,
        vae_path=args.vae_path,
        normalizer_path=args.normalizer_path,
        device=device,
    )
    
    # 批量推理模式
    if args.batch:
        run_batch_inference(
            processor=processor,
            model=model,
            vae=vae,
            normalizer=normalizer,
            shard_dir=args.shard_dir,
            instruction=instruction,
            vae_device=vae_device,
            valid_action_id_length=valid_action_id_length,
            num_samples=args.num_samples,
            apply_jpeg_compression=not args.no_jpeg_compression,
            output_dir=args.output_dir,
        )
        print("\n[DONE] Batch inference completed!")
        return
    
    # 单样本推理模式
    shard_path = os.path.join(args.shard_dir, f"shard-{args.shard_idx:06d}.tar")
    print(f"\n[INFO] Loading sample from {shard_path}, sample_idx={args.sample_idx}")
    sample = load_sample_from_webdataset(shard_path, args.sample_idx)
    
    print(f"[INFO] Image shape: {sample['image'].shape}")
    print(f"[INFO] Action shape: {sample['action'].shape}")
    print(f"[INFO] Action token shape: {sample['action_token'].shape}")
    print(f"[INFO] Meta: {sample['meta']}")
    
    # 准备图像用于 loss 计算
    image_np = sample['image']
    if not args.no_jpeg_compression:
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encoded = cv2.imencode('.jpg', image_np[..., ::-1], encode_param)
        image_pil = Image.open(io.BytesIO(encoded.tobytes()))
    else:
        image_pil = Image.fromarray(image_np)
    
    # 计算与训练一致的 loss
    print("\n" + "="*80)
    print("COMPUTING TRAINING-CONSISTENT LOSS")
    print("="*80)
    gt_action_tokens = sample['action_token']
    loss = compute_training_loss(
        processor=processor,
        model=model,
        image_pil=image_pil,
        instruction=instruction,
        gt_action_tokens=gt_action_tokens,
        device=device,
    )
    print(f"[LOSS] CrossEntropy Loss (training-consistent): {loss:.4f}")
    print(f"[INFO] Compare to training final loss: ~1.15")
    
    # 执行推理
    print("\n" + "="*80)
    print("RUNNING INFERENCE")
    print("="*80)
    result = predict_action(
        processor=processor,
        model=model,
        vae=vae,
        normalizer=normalizer,
        image=sample['image'],
        instruction=instruction,
        vae_device=vae_device,
        valid_action_id_length=valid_action_id_length,
        apply_jpeg_compression=not args.no_jpeg_compression,
    )
    
    print(f"[INFO] Action valid: {result['action_valid']}")
    print(f"[INFO] Predicted action shape: {result['action_pred'].shape}")
    
    # 打印动作对比
    gt_action = sample['action']  # [24, 20]
    pred_action = result['action_pred'][0]  # [24, 20]
    print_action_comparison(gt_action, pred_action, loss=loss)
    
    # 可视化并保存
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(
        args.output_dir, 
        f"inference_shard{args.shard_idx}_sample{args.sample_idx}_{timestamp}.png"
    )
    visualize_and_save(
        image=sample['image'],
        gt_action=gt_action,
        pred_action=pred_action,
        instruction=instruction,
        output_path=output_path,
        sample_info=f"Shard: {args.shard_idx}, Sample: {args.sample_idx}, Loss: {loss:.4f}",
    )
    
    # 保存原始数据
    np_output_path = os.path.join(
        args.output_dir,
        f"inference_shard{args.shard_idx}_sample{args.sample_idx}_{timestamp}.npz"
    )
    np.savez(
        np_output_path,
        gt_action=gt_action,
        pred_action=pred_action,
        action_valid=result['action_valid'],
        instruction=instruction,
        loss=loss,
    )
    print(f"[INFO] Raw data saved to {np_output_path}")
    
    # 保存观测图像
    obs_output_path = os.path.join(
        args.output_dir,
        f"observation_shard{args.shard_idx}_sample{args.sample_idx}_{timestamp}.jpg"
    )
    Image.fromarray(sample['image']).save(obs_output_path)
    print(f"[INFO] Observation image saved to {obs_output_path}")
    
    print("\n[DONE] Inference completed successfully!")


if __name__ == "__main__":
    main()
