"""
Debug script to inspect training input data in single process mode.
This script loads the dataset and shows detailed information about the data pipeline.
"""

import os
import sys
import json
import socket

import torch
import yaml
import numpy as np
from PIL import Image

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoProcessor
from data.utils import get_instructions_and_blended_train_dataset
from models.normalizer import LinearNormalizer
from vqvae.models.multivqvae import MultiVQVAE


def main():
    # ==================== Configuration ====================
    dataset_config_path = "rdt2_pika_shards/dataset_config.yaml"
    tokenizer_name = "Qwen/Qwen2.5-VL-7B-Instruct"
    vae_name = "robotics-diffusion-transformer/RVQActionTokenizer"
    num_samples_to_inspect = 3
    
    print("=" * 80)
    print("RDT2 Training Data Inspector")
    print("=" * 80)
    
    # ==================== Load Config ====================
    print("\n[1] Loading dataset config...")
    with open(dataset_config_path, "r") as f:
        hostname = socket.gethostname()
        dataset_config_str = f.read().format(hostname=hostname)
        dataset_config = yaml.safe_load(dataset_config_str)
    
    print(f"  Dataset name: {dataset_config['name']}")
    print(f"  Shards dir: {dataset_config['shards_dir']}")
    print(f"  Instruction path: {dataset_config['kwargs']['instruction_path']}")
    print(f"  Normalizer path: {dataset_config['kwargs']['normalizer_path']}")
    
    # ==================== Load Components ====================
    print("\n[2] Loading components...")
    
    # Processor
    print("  Loading processor...")
    processor = AutoProcessor.from_pretrained(tokenizer_name, use_fast=True)
    processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<action>"]},
        replace_additional_special_tokens=False,
    )
    print(f"  Tokenizer vocab_size: {processor.tokenizer.vocab_size}")
    
    # VAE
    print("  Loading VAE...")
    vae = MultiVQVAE.from_pretrained(vae_name)
    vae.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae.to(device, dtype=torch.float32)
    print(f"  VAE pos_id_len: {vae.pos_id_len}, rot_id_len: {vae.rot_id_len}, grip_id_len: {vae.grip_id_len}")
    print(f"  Total action_token length: {vae.pos_id_len + vae.rot_id_len + vae.grip_id_len}")
    
    # Normalizer
    print("  Loading normalizer...")
    normalizer = LinearNormalizer.load(dataset_config["kwargs"]["normalizer_path"])
    print(f"  Normalizer keys: {list(normalizer.params_dict.keys())}")
    
    # Print normalizer parameters
    if "action" in normalizer.params_dict:
        action_params = normalizer.params_dict["action"]
        print(f"\n  [Normalizer 'action' parameters]")
        print(f"    scale shape: {action_params['scale'].shape}")
        print(f"    offset shape: {action_params['offset'].shape}")
        print(f"    scale: {action_params['scale'].numpy().flatten()}")
        print(f"    offset: {action_params['offset'].numpy().flatten()}")
    
    # ==================== Load Dataset ====================
    print("\n[3] Loading dataset...")
    instructions, train_ds = get_instructions_and_blended_train_dataset(dataset_config)
    print(f"  Instructions: {json.dumps(instructions, indent=4, ensure_ascii=False)}")
    
    # ==================== Inspect Samples ====================
    print(f"\n[4] Inspecting {num_samples_to_inspect} samples...")
    
    for sample_idx, example in enumerate(train_ds):
        if sample_idx >= num_samples_to_inspect:
            break
        
        print(f"\n{'=' * 80}")
        print(f"Sample {sample_idx}")
        print("=" * 80)
        
        # Meta info
        meta = example["meta"]
        print(f"\n[Meta]")
        print(f"  Episode: {meta.get('episode', 'N/A')}")
        print(f"  Frame: {meta.get('frame', 'N/A')}")
        print(f"  Instruction key: {meta.get('sub_task_instruction_key', 'N/A')}")
        instruction = instructions.get(meta["sub_task_instruction_key"], "")
        print(f"  Instruction: {instruction}")
        
        # Image info
        image = example["image"]
        print(f"\n[Image]")
        print(f"  Type: {type(image)}")
        if isinstance(image, Image.Image):
            print(f"  Size: {image.size}")
            print(f"  Mode: {image.mode}")
        
        # Action info (raw, before tokenization)
        if "action" in example:
            action = example["action"]
            print(f"\n[Raw Action]")
            print(f"  Shape: {action.shape}")
            print(f"  Dtype: {action.dtype}")
            print(f"  Range: [{action.min():.6f}, {action.max():.6f}]")
            print(f"  First frame (20 dims): {action[0] if len(action.shape) > 1 else action}")
        
        # Action token info
        action_token = example["action_token"]
        action_token_tensor = torch.from_numpy(action_token).to(dtype=torch.long)
        print(f"\n[Action Token] (from WebDataset)")
        print(f"  Shape: {action_token.shape}")
        print(f"  Dtype: {action_token.dtype}")
        print(f"  Range: [{action_token.min()}, {action_token.max()}]")
        print(f"  Values: {action_token.tolist()}")
        
        # Decode action token back to action using VAE
        print(f"\n[VAE Decode: action_token -> normalized_action]")
        with torch.no_grad():
            # action_token 是 flat 的 [27]，需要 reshape 成 [1, 27] for batch
            action_token_batch = action_token_tensor.unsqueeze(0).to(device)
            decoded_action = vae.decode(action_token_batch)  # [1, 24, 20]
            decoded_action = decoded_action.cpu().numpy()[0]  # [24, 20]
        
        print(f"  Decoded shape: {decoded_action.shape}")
        print(f"  Decoded range: [{decoded_action.min():.6f}, {decoded_action.max():.6f}]")
        print(f"  Decoded first frame: {decoded_action[0]}")
        
        # Unnormalize decoded action
        print(f"\n[Unnormalize: normalized_action -> raw_action]")
        decoded_action_tensor = torch.from_numpy(decoded_action).float()
        unnorm_action = normalizer["action"].unnormalize(decoded_action_tensor).numpy()
        
        print(f"  Unnormalized shape: {unnorm_action.shape}")
        print(f"  Unnormalized range: [{unnorm_action.min():.6f}, {unnorm_action.max():.6f}]")
        print(f"  Unnormalized first frame: {unnorm_action[0]}")
        
        # Compare with original action if available
        if "action" in example:
            original_action = example["action"]
            if len(original_action.shape) == 1:
                # 如果原始 action 是 [20]，取 unnorm 的第一帧比较
                diff = np.abs(unnorm_action[0] - original_action)
            else:
                diff = np.abs(unnorm_action - original_action)
            print(f"\n[Comparison: Original vs Decoded-Unnormalized]")
            print(f"  Max absolute difference: {diff.max():.6f}")
            print(f"  Mean absolute difference: {diff.mean():.6f}")
        
        # Token mapping to vocab space
        print(f"\n[Token Mapping: action_token -> vocab_ids]")
        vocab_size = processor.tokenizer.vocab_size
        action_input_ids = vocab_size - (action_token_tensor + 1)
        print(f"  vocab_size: {vocab_size}")
        print(f"  Formula: action_input_ids = vocab_size - (action_token + 1)")
        print(f"  action_input_ids range: [{action_input_ids.min().item()}, {action_input_ids.max().item()}]")
        print(f"  action_input_ids values: {action_input_ids.tolist()}")
        
        # Reverse mapping verification
        recovered = vocab_size - (action_input_ids + 1)
        print(f"  Reverse verification: {torch.allclose(action_token_tensor, recovered)}")
    
    # ==================== Interactive Mode ====================
    print("\n" + "=" * 80)
    print("Entering interactive mode (ipdb)...")
    print("Available variables:")
    print("  - processor: Qwen2.5-VL processor")
    print("  - vae: MultiVQVAE model")
    print("  - normalizer: LinearNormalizer")
    print("  - train_ds: Training dataset")
    print("  - instructions: Instruction dict")
    print("  - example: Last loaded sample")
    print("=" * 80)
    
    from ipdb import set_trace
    set_trace()


if __name__ == "__main__":
    main()
