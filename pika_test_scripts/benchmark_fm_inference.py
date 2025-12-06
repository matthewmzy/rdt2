#!/usr/bin/env python3
"""
RDT2 Flow Matching 推理时间基准测试

测试 FM action expert 的推理延迟

用法:
    python pika_test_scripts/benchmark_fm_inference.py \
        --fm-checkpoint outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
        --shard-dir rdt2_pika_shards \
        --num-warmup 5 \
        --num-runs 50
"""

import os
import sys
import io
import json
import tarfile
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from PIL import Image

# 添加项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from models.rdt_inferencer import RDTInferencer


def load_sample_from_shard(shard_path: str, sample_idx: int):
    """从 shard 加载样本"""
    with tarfile.open(shard_path, 'r') as tar:
        # 读取图像
        img_file = tar.extractfile(f'{sample_idx}.image.jpg')
        image = Image.open(io.BytesIO(img_file.read()))
        image = np.array(image)
        
        # 读取元数据
        meta_file = tar.extractfile(f'{sample_idx}.meta.json')
        meta = json.load(meta_file)
    
    return image, meta


def main():
    parser = argparse.ArgumentParser(description='RDT2 FM Inference Benchmark')
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
    parser.add_argument('--num-warmup', type=int, default=5,
                        help='Number of warmup runs (not counted)')
    parser.add_argument('--num-runs', type=int, default=50,
                        help='Number of benchmark runs')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device')
    
    args = parser.parse_args()
    
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
    
    load_start = time.time()
    inferencer = RDTInferencer(
        config=config,
        pretrained_path=args.fm_checkpoint,
        normalizer_path=args.normalizer_path,
        pretrained_vision_language_model_name_or_path=args.vlm_model,
        device=args.device,
        dtype=torch.bfloat16,
    )
    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.2f}s")
    
    # 获取 shard 文件列表
    shard_dir = Path(args.shard_dir)
    shard_files = sorted(shard_dir.glob('shard-*.tar'))
    print(f"Found {len(shard_files)} shards")
    
    # 加载测试样本
    print("\nLoading test samples...")
    test_samples = []
    for i in range(args.num_warmup + args.num_runs):
        shard_idx = i % len(shard_files)
        sample_idx = (i * 7) % 50
        shard_path = shard_files[shard_idx]
        
        try:
            image, meta = load_sample_from_shard(str(shard_path), sample_idx)
            instruction_key = meta.get('sub_task_instruction_key', 'pika_task')
            instruction = instructions.get(instruction_key, "Put the bottle into the tape ring.")
            
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
            
            test_samples.append((observations, instruction))
        except Exception as e:
            print(f"  Skip shard{shard_idx} sample{sample_idx}: {e}")
    
    print(f"Loaded {len(test_samples)} samples")
    
    # Warmup
    print(f"\n{'='*60}")
    print(f"Warmup: {args.num_warmup} runs")
    print(f"{'='*60}")
    
    for i in range(min(args.num_warmup, len(test_samples))):
        observations, instruction = test_samples[i]
        with torch.no_grad():
            _ = inferencer.step(observations, instruction)
        print(f"  Warmup {i+1}/{args.num_warmup} done")
    
    # 同步 CUDA
    if 'cuda' in args.device:
        torch.cuda.synchronize()
    
    # Benchmark
    print(f"\n{'='*60}")
    print(f"Benchmark: {args.num_runs} runs")
    print(f"{'='*60}")
    
    latencies = []
    
    for i in range(args.num_runs):
        sample_idx = (args.num_warmup + i) % len(test_samples)
        observations, instruction = test_samples[sample_idx]
        
        # 同步 CUDA 并计时
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        
        start_time = time.perf_counter()
        
        with torch.no_grad():
            action_pred = inferencer.step(observations, instruction)
        
        if 'cuda' in args.device:
            torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        latency = (end_time - start_time) * 1000  # ms
        latencies.append(latency)
        
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/{args.num_runs}: {latency:.2f} ms")
    
    # 统计结果
    latencies = np.array(latencies)
    
    print(f"\n{'='*60}")
    print(f"Results")
    print(f"{'='*60}")
    print(f"  Number of runs:    {args.num_runs}")
    print(f"  Action shape:      {action_pred.shape}")
    print(f"")
    print(f"  Latency Statistics (ms):")
    print(f"    Mean:            {np.mean(latencies):.2f}")
    print(f"    Std:             {np.std(latencies):.2f}")
    print(f"    Min:             {np.min(latencies):.2f}")
    print(f"    Max:             {np.max(latencies):.2f}")
    print(f"    Median:          {np.median(latencies):.2f}")
    print(f"    P90:             {np.percentile(latencies, 90):.2f}")
    print(f"    P95:             {np.percentile(latencies, 95):.2f}")
    print(f"    P99:             {np.percentile(latencies, 99):.2f}")
    print(f"")
    print(f"  Throughput:")
    print(f"    FPS:             {1000 / np.mean(latencies):.2f}")
    print(f"")
    print(f"  Action Horizon:    24 frames")
    print(f"  Effective Rate:    {24 * 1000 / np.mean(latencies):.2f} actions/s")
    print(f"                     (if using all 24 predicted actions)")
    
    # GPU 信息
    if 'cuda' in args.device:
        gpu_idx = int(args.device.split(':')[1]) if ':' in args.device else 0
        print(f"\n  GPU Info:")
        print(f"    Device:          {torch.cuda.get_device_name(gpu_idx)}")
        print(f"    Memory Allocated: {torch.cuda.memory_allocated(gpu_idx) / 1024**3:.2f} GB")
        print(f"    Memory Cached:   {torch.cuda.memory_reserved(gpu_idx) / 1024**3:.2f} GB")


if __name__ == '__main__':
    main()
