"""
测试 VAE 对全零输入的重建效果
"""
import torch
import numpy as np
import os, sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from vqvae.models.multivqvae import MultiVQVAE

# 加载 VAE
print("Loading VAE...")
vae = MultiVQVAE.from_pretrained('robotics-diffusion-transformer/RVQActionTokenizer')
vae = vae.to('cuda', dtype=torch.float32)
vae.eval()

print(f"VAE config:")
print(f"  pos_id_len: {vae.pos_id_len}")
print(f"  rot_id_len: {vae.rot_id_len}")
print(f"  grip_id_len: {vae.grip_id_len}")
print(f"  num_embeddings: {vae.num_embeddings}")

# # 加载官方 normalizer
# official_norm = torch.load('normalizer.pt', weights_only=False)
# official_scale = official_norm['action']['scale'].detach().to('cuda')
# official_offset = official_norm['action']['offset'].detach().to('cuda')

print("\n" + "=" * 80)
print("全零输入通过 VAE 的重建效果测试")
print("=" * 80)

# 创建平凡输入
actions = torch.zeros(1, 24, 20, dtype=torch.float32, device='cuda')
actions[:,:,3] = 1.0
actions[:,:,7] = 1.0
actions[:,:,13] = 1.0
actions[:,:,17] = 1.0

print(f"\n输入 shape: {actions.shape}")
print(f"输入 (全零): min={actions.min():.6f}, max={actions.max():.6f}")

# 归一化 (y = x * scale + offset, 当 x=0 时, y = offset)
# actions_norm = actions * official_scale + official_offset
actions_norm = actions

print("\n【平凡动作】")
print(f"  全部:       min={actions_norm.min():.6f}, max={actions_norm.max():.6f}")
print(f"  Position:   {actions_norm[0, 0, :3].cpu().numpy()}")
print(f"  Rot6D:      {actions_norm[0, 0, 3:9].cpu().numpy()}")
print(f"  Gripper R:  {actions_norm[0, 0, 9].item():.6f}")
print(f"  Position L: {actions_norm[0, 0, 10:13].cpu().numpy()}")
print(f"  Rot6D L:    {actions_norm[0, 0, 13:19].cpu().numpy()}")
print(f"  Gripper L:  {actions_norm[0, 0, 19].item():.6f}")

# VAE 编码
with torch.no_grad():
    tokens = vae.encode(actions_norm)
    
print(f"\n【编码后的 tokens】")
print(f"  Shape: {tokens.shape}")
print(f"  Values: {tokens[0].cpu().numpy().tolist()}")
print(f"  Range: [{tokens.min().item()}, {tokens.max().item()}]")

# VAE 解码
with torch.no_grad():
    decoded_norm = vae.decode(tokens)

print(f"\n【解码后】")
print(f"  全部:       min={decoded_norm.min():.6f}, max={decoded_norm.max():.6f}")
print(f"  Position:   {decoded_norm[0, 0, :3].cpu().numpy()}")
print(f"  Rot6D:      {decoded_norm[0, 0, 3:9].cpu().numpy()}")
print(f"  Gripper R:  {decoded_norm[0, 0, 9].item():.6f}")

# 反归一化 (x = (y - offset) / scale)
# decoded = (decoded_norm - official_offset) / official_scale
decoded = decoded_norm


# 输出每个维度的值
print("\n【各维度重建值 (应该接近 0)】")
print("-" * 60)

dim_names = [
    "右 Pos X", "右 Pos Y", "右 Pos Z",
    "右 Rot 0", "右 Rot 1", "右 Rot 2", "右 Rot 3", "右 Rot 4", "右 Rot 5",
    "右 Gripper",
    "左 Pos X", "左 Pos Y", "左 Pos Z",
    "左 Rot 0", "左 Rot 1", "左 Rot 2", "左 Rot 3", "左 Rot 4", "左 Rot 5",
    "左 Gripper",
]

# 取第一个 timestep
decoded_t0 = decoded[0, 0].cpu().numpy()
input_t0 = actions[0, 0].cpu().numpy()

print(f"{'Dim':<4} {'Name':<12} {'Input':>12} {'Output':>12} {'Error':>12}")
print("-" * 60)
for i in range(20):
    error = abs(decoded_t0[i] - input_t0[i])
    print(f"{i:<4} {dim_names[i]:<12} {input_t0[i]:>12.6f} {decoded_t0[i]:>12.6f} {error:>12.6f}")
        

# 计算重建误差
mse = ((decoded - actions) ** 2).mean().item()
print(f"\n【总体 MSE】: {mse:.10f}")

