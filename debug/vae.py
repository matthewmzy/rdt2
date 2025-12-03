import json
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

# 加载官方 normalizer
official_norm = torch.load('normalizer.pt', weights_only=False)
official_scale = official_norm['action']['scale'].detach().to('cuda')
official_offset = official_norm['action']['offset'].detach().to('cuda')

print("=" * 80)
print("官方数据通过 VAE 的重建效果测试")
print("=" * 80)

# 加载官方数据
frames = []
for i in range(1, 6):
    with open(f'pika-videos/test_action/frame{i}.json', 'r') as f:
        frames.append(json.load(f))

# 提取完整的 action (24 timesteps x 20 dims)
# 使用 frame1 的完整数据
frame1 = frames[0]
action_chunk = np.array(frame1)

print(f"\n官方 Action Chunk shape: {action_chunk.shape}")

# 转换为 tensor
actions = torch.from_numpy(action_chunk).float().unsqueeze(0).to('cuda')  # (1, 24, 20)

print("\n【原始数据各维度范围】")
print(f"  Position (0-2):  min={actions[..., :3].min():.6f}, max={actions[..., :3].max():.6f}")
print(f"  Rot6D (3-8):     min={actions[..., 3:9].min():.6f}, max={actions[..., 3:9].max():.6f}")
print(f"  Gripper (9):     min={actions[..., 9].min():.6f}, max={actions[..., 9].max():.6f}")

# 归一化
actions_norm = actions * official_scale + official_offset

print("\n【归一化后各维度范围】")
print(f"  Position (0-2):  min={actions_norm[..., :3].min():.6f}, max={actions_norm[..., :3].max():.6f}")
print(f"  Rot6D (3-8):     min={actions_norm[..., 3:9].min():.6f}, max={actions_norm[..., 3:9].max():.6f}")
print(f"  Gripper (9):     min={actions_norm[..., 9].min():.6f}, max={actions_norm[..., 9].max():.6f}")

# VAE 编码/解码
with torch.no_grad():
    tokens = vae.encode(actions_norm)
    decoded_norm = vae.decode(tokens)

# 反归一化
decoded = (decoded_norm - official_offset) / official_scale

print("\n【重建后各维度范围】")
print(f"  Position (0-2):  min={decoded[..., :3].min():.6f}, max={decoded[..., :3].max():.6f}")
print(f"  Rot6D (3-8):     min={decoded[..., 3:9].min():.6f}, max={decoded[..., 3:9].max():.6f}")
print(f"  Gripper (9):     min={decoded[..., 9].min():.6f}, max={decoded[..., 9].max():.6f}")

# 计算各维度 MSE
print("\n" + "=" * 80)
print("【各维度 MSE】")
print("=" * 80)

dim_names = [
    "右 Pos X", "右 Pos Y", "右 Pos Z",
    "右 Rot 0", "右 Rot 1", "右 Rot 2", "右 Rot 3", "右 Rot 4", "右 Rot 5",
    "右 Grip",
    "左 Pos X", "左 Pos Y", "左 Pos Z",
    "左 Rot 0", "左 Rot 1", "左 Rot 2", "左 Rot 3", "左 Rot 4", "左 Rot 5",
    "左 Grip",
]

print(f"\n{'Dim':<4} {'Name':<10} {'MSE':>15} {'原始范围':>25} {'重建范围':>25}")
print("-" * 85)

total_mse = 0
for i in range(20):
    mse = ((decoded[..., i] - actions[..., i]) ** 2).mean().item()
    total_mse += mse
    orig_min = actions[..., i].min().item()
    orig_max = actions[..., i].max().item()
    recon_min = decoded[..., i].min().item()
    recon_max = decoded[..., i].max().item()
    print(f"{i:<4} {dim_names[i]:<10} {mse:>15.10f} [{orig_min:>10.6f}, {orig_max:>10.6f}] [{recon_min:>10.6f}, {recon_max:>10.6f}]")

print("-" * 85)
print(f"Overall MSE: {total_mse:.10f}")

# 具体样本对比
print("\n" + "=" * 80)
print("【具体样本对比 - Timestep 23 (旋转变化最大)】")
print("=" * 80)

t = 23
orig = actions[0, t].cpu().numpy()
recon = decoded[0, t].cpu().numpy()

print("\n右臂 Rot6D:")
print(f"  原始:  [{orig[3]:.6f}, {orig[4]:.6f}, {orig[5]:.6f}, {orig[6]:.6f}, {orig[7]:.6f}, {orig[8]:.6f}]")
print(f"  重建:  [{recon[3]:.6f}, {recon[4]:.6f}, {recon[5]:.6f}, {recon[6]:.6f}, {recon[7]:.6f}, {recon[8]:.6f}]")
print(f"  误差:  [{abs(orig[3]-recon[3]):.6f}, {abs(orig[4]-recon[4]):.6f}, {abs(orig[5]-recon[5]):.6f}, {abs(orig[6]-recon[6]):.6f}, {abs(orig[7]-recon[7]):.6f}, {abs(orig[8]-recon[8]):.6f}]")

print("\n左臂 Rot6D:")
print(f"  原始:  [{orig[13]:.6f}, {orig[14]:.6f}, {orig[15]:.6f}, {orig[16]:.6f}, {orig[17]:.6f}, {orig[18]:.6f}]")
print(f"  重建:  [{recon[13]:.6f}, {recon[14]:.6f}, {recon[15]:.6f}, {recon[16]:.6f}, {recon[17]:.6f}, {recon[18]:.6f}]")
print(f"  误差:  [{abs(orig[13]-recon[13]):.6f}, {abs(orig[14]-recon[14]):.6f}, {abs(orig[15]-recon[15]):.6f}, {abs(orig[16]-recon[16]):.6f}, {abs(orig[17]-recon[17]):.6f}, {abs(orig[18]-recon[18]):.6f}]")

# 估算旋转角度误差
def estimate_rotation_angle(rot6d):
    trace_approx = rot6d[0] + rot6d[4] + 1.0
    cos_theta = np.clip((trace_approx - 1) / 2, -1, 1)
    return np.degrees(np.arccos(cos_theta))

print("\n旋转角度:")
r_orig_angle = estimate_rotation_angle(orig[3:9])
r_recon_angle = estimate_rotation_angle(recon[3:9])
l_orig_angle = estimate_rotation_angle(orig[13:19])
l_recon_angle = estimate_rotation_angle(recon[13:19])
print(f"  右臂: 原始={r_orig_angle:.4f}°, 重建={r_recon_angle:.4f}°, 误差={abs(r_orig_angle-r_recon_angle):.4f}°")
print(f"  左臂: 原始={l_orig_angle:.4f}°, 重建={l_recon_angle:.4f}°, 误差={abs(l_orig_angle-l_recon_angle):.4f}°")

print("\n" + "=" * 80)
print("【结论】")
print("=" * 80)