#!/usr/bin/env python
"""
将 Pika 数据集转换为 RDT2 WebDataset 格式用于后训练。

输出格式:
    shard-XXXXXX.tar
    ├── 0.image.jpg        # 双目拼接图像 (384, 768, 3) = 左384×384 + 右384×384
    ├── 0.action.npy       # 相对动作序列 (24, 20) float32
    ├── 0.action_token.npy # VQ 编码的 token (27,) int16
    ├── 0.meta.json        # 元数据
    ├── 1.image.jpg
    ├── ...

动作定义（与官方 RDT2 一致）:
    - 动作类型: 相对位姿 (Relative Pose)
    - 参考帧: 当前帧的 TCP 位姿
    - 计算公式: T_rel = inv(T_current) @ T_future
    - 每帧动作 20 维 = 右臂(10) + 左臂(10)
        - 每臂: [相对位移(3) + rotation_6d(6) + gripper_width(1)]
    - action chunk: 24 步 (0.8秒 @ 30Hz)

图像定义:
    - 使用鱼眼相机 (FisheyeCamera) 作为 RGB 输入，与官方 wrist camera 对应
    - 存储约定: camera0=右臂, camera1=左臂
    - 但 preprocess_data_from_umi 会反转顺序后拼接，最终模型输入是 [左, 右]
    - 分辨率: 384×384 per camera -> 拼接后 384×768

Usage:
    # 直接转换（使用项目根目录的官方 normalizer.pt）
    python pika_test_scripts/convert_pika_to_rdt2.py \
        --input-dir pika_raw_data \
        --output-dir rdt2_pika_shards \
        --instruction "Put the bottle into the tape ring, and then take it out with the other hand." \
        --normalizer-path normalizer.pt
        
注意：
    - 必须使用项目根目录下的官方 normalizer.pt
    - RVQ tokenizer 是在官方数据上训练的，必须使用官方 normalizer 确保分布一致
"""
from __future__ import annotations
import os
import io
import sys
import json
import math
import tarfile
import argparse
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import h5py
import cv2
from tqdm import tqdm
import torch

# 添加项目根目录到路径，以便导入 vqvae 和 models
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===================== HDF5 读取工具 =====================

def safe_read_hdf5_dataset(dataset, dtype=np.float64) -> np.ndarray:
    """
    安全读取 HDF5 数据集，处理非标准浮点类型。
    
    某些 HDF5 文件使用非标准浮点类型 (63, 52, 11, 0, 52)，
    h5py 无法自动转换，需要使用 read_direct 方法。
    
    Args:
        dataset: h5py Dataset 对象
        dtype: 目标数据类型
        
    Returns:
        numpy 数组
    """
    try:
        # 首先尝试标准方式读取
        return np.array(dataset, dtype=dtype)
    except ValueError as e:
        if "Insufficient precision" in str(e):
            # 使用 read_direct 绕过类型转换问题
            buf = np.empty(dataset.shape, dtype=dtype)
            dataset.read_direct(buf)
            return buf
        raise


# ===================== 位姿转换工具 =====================

def normalize(vec, eps=1e-12):
    """归一化向量"""
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    norm = np.maximum(norm, eps)
    return vec / norm


def rpy_to_rotmat(rpy: np.ndarray) -> np.ndarray:
    """RPY (roll, pitch, yaw) 转旋转矩阵，ZYX 约定"""
    roll, pitch, yaw = rpy[..., 0], rpy[..., 1], rpy[..., 2]
    
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    
    # ZYX convention: R = Rz * Ry * Rx
    R = np.zeros(rpy.shape[:-1] + (3, 3), dtype=rpy.dtype)
    R[..., 0, 0] = cy * cp
    R[..., 0, 1] = cy * sp * sr - sy * cr
    R[..., 0, 2] = cy * sp * cr + sy * sr
    R[..., 1, 0] = sy * cp
    R[..., 1, 1] = sy * sp * sr + cy * cr
    R[..., 1, 2] = sy * sp * cr - cy * sr
    R[..., 2, 0] = -sp
    R[..., 2, 1] = cp * sr
    R[..., 2, 2] = cp * cr
    return R


def pose6d_to_mat(pose6: np.ndarray) -> np.ndarray:
    """
    6D pose [x, y, z, roll, pitch, yaw] 转 4x4 齐次变换矩阵
    
    Args:
        pose6: (..., 6)
    Returns:
        mat: (..., 4, 4)
    """
    shape = pose6.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=pose6.dtype)
    mat[..., :3, 3] = pose6[..., :3]  # translation
    mat[..., :3, :3] = rpy_to_rotmat(pose6[..., 3:])  # rotation
    mat[..., 3, 3] = 1.0
    return mat


def rotmat_to_rot6d(rotmat: np.ndarray) -> np.ndarray:
    """
    3x3 旋转矩阵转 6D 旋转表示 (前两列展平)
    
    Args:
        rotmat: (..., 3, 3)
    Returns:
        rot6d: (..., 6)
    """
    col0 = rotmat[..., :, 0]  # (..., 3)
    col1 = rotmat[..., :, 1]  # (..., 3)
    return np.concatenate([col0, col1], axis=-1)  # (..., 6)


def mat_to_pose10d(mat: np.ndarray) -> np.ndarray:
    """
    4x4 齐次变换矩阵转 10D pose [x, y, z, rot6d(6)]
    
    Args:
        mat: (..., 4, 4)
    Returns:
        pose10d: (..., 10)
    """
    pos = mat[..., :3, 3]  # (..., 3)
    rotmat = mat[..., :3, :3]  # (..., 3, 3)
    rot6d = rotmat_to_rot6d(rotmat)  # (..., 6)
    return np.concatenate([pos, rot6d], axis=-1)  # (..., 10)


def compute_relative_action(
    current_pose_mat: np.ndarray,
    future_pose_mats: np.ndarray,
    future_gripper_widths: np.ndarray
) -> np.ndarray:
    """
    计算相对动作：未来位姿相对于当前位姿的变换
    
    Args:
        current_pose_mat: (4, 4) 当前帧的世界位姿
        future_pose_mats: (T, 4, 4) 未来 T 帧的世界位姿
        future_gripper_widths: (T,) 未来 T 帧的夹爪宽度
    Returns:
        action: (T, 10) 相对动作 [relative_pos(3), relative_rot6d(6), gripper(1)]
    """
    T = future_pose_mats.shape[0]
    
    # 计算当前帧的逆
    current_inv = np.linalg.inv(current_pose_mat)
    
    # 计算相对变换: T_rel = T_current^{-1} @ T_future
    relative_mats = np.einsum('ij,tjk->tik', current_inv, future_pose_mats)  # (T, 4, 4)
    
    # 转换为 10D 表示
    relative_pose10d = mat_to_pose10d(relative_mats)  # (T, 9) -> 实际是 (T, 10 - 1) 因为少了 gripper
    
    # 拼接夹爪宽度
    action = np.concatenate([
        relative_pose10d,  # (T, 9)
        future_gripper_widths[:, None]  # (T, 1)
    ], axis=-1)  # (T, 10)
    
    return action


# ===================== 图像处理 =====================

def decode_image(x) -> Optional[np.ndarray]:
    """从 HDF5 数据解码图像（仅处理实际图像数据，不处理路径字符串）"""
    # 如果是 bytes，检查是否是 JPEG/PNG 图像数据（以魔数开头）
    if isinstance(x, (bytes, bytearray, memoryview)):
        data = bytes(x)
        # JPEG 魔数: FF D8 FF
        # PNG 魔数: 89 50 4E 47
        if len(data) > 4 and (data[:3] == b'\xff\xd8\xff' or data[:4] == b'\x89PNG'):
            arr = np.frombuffer(data, dtype=np.uint8)
            return cv2.imdecode(arr, cv2.IMREAD_COLOR)
        # 否则可能是路径字符串，返回 None
        return None
    
    # 如果是 numpy 数组，检查是否是 RGB 图像
    try:
        arr = np.array(x)
        if arr.ndim == 3 and arr.dtype == np.uint8:
            if arr.shape[2] == 3:
                return arr.copy()
            if arr.shape[2] == 4:
                return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)
        # 也可能是 1D JPEG bytes 数组
        if arr.ndim == 1 and arr.dtype == np.uint8 and arr.size > 4:
            if arr[0] == 0xff and arr[1] == 0xd8:  # JPEG
                return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception:
        pass
    return None


def read_image_from_hdf5_or_folder(ds, idx: int, episode_dir: Path) -> Optional[np.ndarray]:
    """从 HDF5 dataset 或文件夹读取图像
    
    Args:
        ds: HDF5 dataset
        idx: 帧索引
        episode_dir: episode 根目录 (data.hdf5 所在目录)
    """
    x = ds[idx]
    
    # 尝试直接解码（如果存的是图像数据）
    img = decode_image(x)
    if img is not None:
        return img
    
    # 解析存储的路径字符串
    if isinstance(x, (bytes, bytearray, memoryview)):
        try:
            rel_path = bytes(x).decode('utf-8', 'ignore').strip().strip('\x00')
        except Exception:
            rel_path = ''
    else:
        rel_path = str(x).strip().strip('\x00')
    
    if not rel_path:
        return None
    
    # 构建完整路径
    full_path = episode_dir / rel_path
    if full_path.is_file():
        img = cv2.imread(str(full_path), cv2.IMREAD_COLOR)
        if img is not None:
            return img
    
    # 尝试只用文件名
    filename = Path(rel_path).name
    for ext in ['', '.jpg', '.jpeg', '.png']:
        candidate = filename + ext if ext and '.' not in filename else filename
        for subdir in ['camera/color/pikaDepthCamera_r', 'camera/color/pikaDepthCamera_l']:
            p = episode_dir / subdir / candidate
            if p.is_file():
                img = cv2.imread(str(p), cv2.IMREAD_COLOR)
                if img is not None:
                    return img
    
    return None


def resize_and_concat_stereo(img_left: np.ndarray, img_right: np.ndarray, target_size: int = 384) -> np.ndarray:
    """
    将左右图像 resize 并水平拼接
    
    RDT2 图像拼接逻辑:
    1. 数据存储约定: camera0_rgb = 右臂, camera1_rgb = 左臂
    2. preprocess_data_from_umi 会反转顺序: [camera0, camera1] -> [camera1, camera0] = [左, 右]
    3. rdt_inferencer 拼接: np.concatenate([左, 右], axis=1)
    
    所以最终输入模型的图像拼接顺序是: [左, 右]
    
    Args:
        img_left: 左臂相机图像 (对应 camera1)
        img_right: 右臂相机图像 (对应 camera0)
        target_size: 目标尺寸
    Returns:
        stereo: (target_size, target_size*2, 3) 拼接图像，顺序为 [左, 右]
    """
    # Resize to square
    left_resized = cv2.resize(img_left, (target_size, target_size), interpolation=cv2.INTER_AREA)
    right_resized = cv2.resize(img_right, (target_size, target_size), interpolation=cv2.INTER_AREA)
    
    # RDT2 最终顺序: 左(camera1) 在前，右(camera0) 在后
    stereo = np.concatenate([left_resized, right_resized], axis=1)
    return stereo


def encode_image_to_jpeg(img: np.ndarray, quality: int = 95) -> bytes:
    """将图像编码为 JPEG 字节"""
    # OpenCV 默认是 BGR，JPEG 编码时会处理
    _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buffer.tobytes()


# ===================== Action Token 生成 =====================

class ActionTokenizer:
    """RVQ Action Tokenizer 封装"""
    
    def __init__(self, device: str = "cuda:0", normalizer_path: Optional[str] = None):
        self.device = device
        self.vae = None
        self.normalizer = None
        self.scale = None
        self.offset = None
        self.normalizer_path = normalizer_path
        self._loaded = False
        
    def load(self):
        """延迟加载模型"""
        if self._loaded:
            return
        
        print("Loading RVQ Action Tokenizer...")
        from vqvae.models.multivqvae import MultiVQVAE
        
        self.vae = MultiVQVAE.from_pretrained("robotics-diffusion-transformer/RVQActionTokenizer")
        self.vae = self.vae.to(device=self.device, dtype=torch.float32)
        self.vae.eval()
        
        if self.normalizer_path and os.path.exists(self.normalizer_path):
            print(f"Loading normalizer from {self.normalizer_path}")
            # 直接用 torch.load 加载 ParameterDict
            normalizer = torch.load(self.normalizer_path, weights_only=False)
            self.scale = normalizer['action']['scale'].to(self.device)
            self.offset = normalizer['action']['offset'].to(self.device)
            print(f"Normalizer loaded: scale shape={self.scale.shape}, offset shape={self.offset.shape}")
        else:
            print("Warning: normalizer not loaded, using identity normalization")
            self.scale = None
            self.offset = None
        
        self.valid_action_id_length = self.vae.pos_id_len + self.vae.rot_id_len + self.vae.grip_id_len
        print(f"RVQ loaded. Token length: {self.valid_action_id_length}")
        self._loaded = True
    
    def encode(self, action: np.ndarray) -> np.ndarray:
        """
        将动作编码为 token
        
        Args:
            action: (24, 20) 动作序列
        Returns:
            tokens: (27,) int16 token 序列
        """
        self.load()
        
        # 归一化
        action_tensor = torch.from_numpy(action).float().unsqueeze(0).to(self.device)  # (1, 24, 20)
        
        if self.scale is not None and self.offset is not None:
            # normalized = x * scale + offset
            action_tensor = action_tensor * self.scale + self.offset
        
        # 编码
        with torch.no_grad():
            tokens = self.vae.encode(action_tensor)  # (1, 27)
        
        return tokens[0].cpu().numpy().astype(np.int16)


# ===================== 数据转换主逻辑 =====================

def is_degree(rpy: np.ndarray) -> bool:
    """判断 RPY 是否为角度制"""
    if rpy.size == 0:
        return False
    return np.mean(np.abs(rpy)) > 6.5  # > ~2*pi


def process_episode(
    h5_path: Path,
    action_horizon: int = 24,
    image_size: int = 384,
    gripper_mapping: Optional[dict] = None,
) -> List[dict]:
    """
    处理单个 episode，生成所有样本
    
    Args:
        h5_path: HDF5 文件路径
        action_horizon: 动作预测长度
        image_size: 图像尺寸
        gripper_mapping: gripper 映射配置，用于将 pika gripper 映射到 [0, 0.088]
    
    Returns:
        samples: 样本列表，每个样本包含 image, action, meta
    """
    samples = []
    
    with h5py.File(h5_path, 'r') as f:
        # 读取数据 - 使用 safe_read_hdf5_dataset 处理非标准浮点类型
        pose_l = safe_read_hdf5_dataset(f['localization/pose/pika_l'])
        pose_r = safe_read_hdf5_dataset(f['localization/pose/pika_r'])
        dist_l = safe_read_hdf5_dataset(f['gripper/encoderDistance/pika_l'])
        dist_r = safe_read_hdf5_dataset(f['gripper/encoderDistance/pika_r'])
        
        # 如果提供了 gripper 映射，将 pika gripper 映射到官方范围 [0, 0.088]
        if gripper_mapping is not None:
            # 右臂 gripper
            pika_min_r = gripper_mapping['pika_min_r']
            pika_max_r = gripper_mapping['pika_max_r']
            dist_r = (dist_r - pika_min_r) / (pika_max_r - pika_min_r) * 0.088
            dist_r = np.clip(dist_r, 0.0, 0.088)
            
            # 左臂 gripper
            pika_min_l = gripper_mapping['pika_min_l']
            pika_max_l = gripper_mapping['pika_max_l']
            dist_l = (dist_l - pika_min_l) / (pika_max_l - pika_min_l) * 0.088
            dist_l = np.clip(dist_l, 0.0, 0.088)
        
        # 图像数据集 - 使用鱼眼相机数据
        # 注意：官方 RDT2 使用的是 wrist camera，对应我们的 FisheyeCamera
        # Pika 鱼眼相机命名：pikaFisheyeCamera_l 就是左手相机, pikaFisheyeCamera_r 就是右手相机
        # （与 DepthCamera 不同，鱼眼相机的命名没有反转）
        ds_left = f['camera/color/pikaFisheyeCamera_l']   # 左手 = camera1
        ds_right = f['camera/color/pikaFisheyeCamera_r']  # 右手 = camera0
        
        n_frames = min(len(pose_l), len(pose_r), len(dist_l), len(dist_r), len(ds_left), len(ds_right))
        
        # 检查并转换角度单位
        if is_degree(pose_l[:, 3:]) or is_degree(pose_r[:, 3:]):
            pose_l[:, 3:] = np.radians(pose_l[:, 3:])
            pose_r[:, 3:] = np.radians(pose_r[:, 3:])
        
        # 预计算所有位姿矩阵
        pose_mats_l = pose6d_to_mat(pose_l)  # (N, 4, 4)
        pose_mats_r = pose6d_to_mat(pose_r)  # (N, 4, 4)
        
        # episode 目录
        episode_dir = h5_path.parent
        
        # 遍历每一帧（需要保留足够的未来帧）
        # 注意：action chunk 是从 i+1 到 i+action_horizon 的未来帧（不含当前帧 i）
        for i in range(n_frames - action_horizon):
            # 读取图像
            img_left = read_image_from_hdf5_or_folder(ds_left, i, episode_dir)
            img_right = read_image_from_hdf5_or_folder(ds_right, i, episode_dir)
            
            # 严格检查图像有效性
            if img_left is None or img_right is None:
                continue
            if not isinstance(img_left, np.ndarray) or not isinstance(img_right, np.ndarray):
                continue
            if img_left.ndim != 3 or img_right.ndim != 3:
                continue
            
            # 拼接双目图像
            stereo_img = resize_and_concat_stereo(img_left, img_right, image_size)
            
            # 计算右臂相对动作 (robot0)
            # 官方约定: action chunk 是从 current_idx+1 开始的未来 24 帧
            future_indices = np.arange(i + 1, i + 1 + action_horizon)
            action_r = compute_relative_action(
                pose_mats_r[i],
                pose_mats_r[future_indices],
                dist_r[future_indices]
            )  # (24, 10)
            
            # 计算左臂相对动作 (robot1)
            action_l = compute_relative_action(
                pose_mats_l[i],
                pose_mats_l[future_indices],
                dist_l[future_indices]
            )  # (24, 10)
            
            # 拼接动作: [右臂(10), 左臂(10)] = 20维
            action = np.concatenate([action_r, action_l], axis=-1).astype(np.float32)  # (24, 20)
            
            samples.append({
                'image': stereo_img,
                'action': action,
                'episode': h5_path.parent.name,
                'frame': i,
            })
    
    return samples


def write_shard(
    samples: List[dict],
    output_path: Path,
    shard_idx: int,
    tokenizer: Optional[ActionTokenizer] = None,
    instruction_key: str = "default_instruction",
) -> int:
    """
    将样本写入 WebDataset shard
    
    Args:
        samples: 样本列表
        output_path: 输出目录
        shard_idx: shard 索引
        tokenizer: Action tokenizer (可选)
        instruction_key: 指令键名
    
    Returns:
        写入的样本数
    """
    shard_name = f"shard-{shard_idx:06d}.tar"
    shard_path = output_path / shard_name
    
    written_count = 0
    with tarfile.open(shard_path, 'w') as tar:
        for sample_idx, sample in tqdm(enumerate(samples), desc=f"Writing {shard_name}", total=len(samples)):
            # 验证图像有效性
            img = sample.get('image')
            if img is None or not isinstance(img, np.ndarray) or img.ndim != 3:
                print(f"Warning: Skipping sample {sample_idx} due to invalid image")
                continue
            
            prefix = str(written_count)
            
            # 1. 写入图像
            img_data = encode_image_to_jpeg(img)
            img_info = tarfile.TarInfo(name=f"{prefix}.image.jpg")
            img_info.size = len(img_data)
            tar.addfile(img_info, io.BytesIO(img_data))
            
            # 2. 写入动作
            action_buffer = io.BytesIO()
            np.save(action_buffer, sample['action'])
            action_data = action_buffer.getvalue()
            action_info = tarfile.TarInfo(name=f"{prefix}.action.npy")
            action_info.size = len(action_data)
            tar.addfile(action_info, io.BytesIO(action_data))
            
            # 3. 写入 action token
            if tokenizer is not None:
                try:
                    tokens = tokenizer.encode(sample['action'])
                except Exception as e:
                    print(f"Warning: failed to encode action token for sample {sample_idx}: {e}")
                    tokens = np.zeros(27, dtype=np.int16)
            else:
                tokens = np.zeros(27, dtype=np.int16)
            
            token_buffer = io.BytesIO()
            np.save(token_buffer, tokens)
            token_data = token_buffer.getvalue()
            token_info = tarfile.TarInfo(name=f"{prefix}.action_token.npy")
            token_info.size = len(token_data)
            tar.addfile(token_info, io.BytesIO(token_data))
            
            # 4. 写入元数据
            meta = {
                "sub_task_instruction_key": instruction_key,
                "episode": sample['episode'],
                "frame": sample['frame'],
            }
            meta_data = json.dumps(meta).encode('utf-8')
            meta_info = tarfile.TarInfo(name=f"{prefix}.meta.json")
            meta_info.size = len(meta_data)
            tar.addfile(meta_info, io.BytesIO(meta_data))
            
            written_count += 1
    
    return written_count


def convert_pika_to_rdt2(
    input_dir: Path,
    output_dir: Path,
    instruction: str,
    samples_per_shard: int = 1000,
    action_horizon: int = 24,
    image_size: int = 384,
    use_tokenizer: bool = True,
    normalizer_path: Optional[str] = None,
    device: str = "cuda:0",
):
    """
    主转换函数
    
    Args:
        input_dir: pika 数据集目录
        output_dir: 输出目录
        instruction: 任务指令
        samples_per_shard: 每个 shard 的样本数
        action_horizon: 动作预测长度
        image_size: 图像尺寸
        use_tokenizer: 是否使用 tokenizer 生成 action_token
        normalizer_path: normalizer 路径
        device: 设备
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载 gripper 映射配置（如果存在）
    gripper_mapping = None
    gripper_config_path = output_dir / "gripper_mapping.json"
    if gripper_config_path.exists():
        with open(gripper_config_path, 'r') as f:
            gripper_mapping = json.load(f)
        print(f"Loaded gripper mapping from {gripper_config_path}")
        print(f"  Right: [{gripper_mapping['pika_min_r']:.4f}, {gripper_mapping['pika_max_r']:.4f}] -> [0, 0.088]")
        print(f"  Left:  [{gripper_mapping['pika_min_l']:.4f}, {gripper_mapping['pika_max_l']:.4f}] -> [0, 0.088]")
    else:
        print("Warning: gripper_mapping.json not found. Gripper values will NOT be remapped.")
        print("         Run --compute-normalizer-only first to generate it.")
    
    # 查找所有 episode
    episode_dirs = sorted([d for d in Path(input_dir).iterdir() if d.is_dir() and d.name.startswith('episode')])
    print(f"Found {len(episode_dirs)} episodes")
    
    # 初始化 tokenizer
    tokenizer = None
    if use_tokenizer:
        try:
            tokenizer = ActionTokenizer(device=device, normalizer_path=normalizer_path)
        except Exception as e:
            print(f"Warning: failed to initialize tokenizer: {e}")
            print("Will generate zero action tokens")
    
    # 收集所有样本
    all_samples = []
    for episode_dir in tqdm(episode_dirs, desc="Processing episodes"):
        h5_path = episode_dir / 'data.hdf5'
        if not h5_path.exists():
            print(f"Warning: {h5_path} not found, skipping")
            continue
        
        try:
            samples = process_episode(h5_path, action_horizon, image_size, gripper_mapping)
            all_samples.extend(samples)
            print(f"  {episode_dir.name}: {len(samples)} samples")
        except Exception as e:
            print(f"Error processing {episode_dir}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\nTotal samples: {len(all_samples)}")
    # from ipdb import set_trace; set_trace()
    # 写入 shards
    instruction_key = "pika_task"
    n_shards = (len(all_samples) + samples_per_shard - 1) // samples_per_shard
    
    total_written = 0
    for shard_idx in tqdm(range(n_shards), desc="Writing shards"):
        start_idx = shard_idx * samples_per_shard
        end_idx = min(start_idx + samples_per_shard, len(all_samples))
        shard_samples = all_samples[start_idx:end_idx]
        
        n_written = write_shard(
            shard_samples,
            output_dir,
            shard_idx,
            tokenizer,
            instruction_key
        )
        total_written += n_written
    
    # 写入 instruction.json
    instructions = {instruction_key: instruction}
    instruction_path = output_dir / "instruction.json"
    with open(instruction_path, 'w') as f:
        json.dump(instructions, f, indent=2)
    
    print(f"\nConversion complete!")
    print(f"  Total samples written: {total_written}")
    print(f"  Total shards: {n_shards}")
    print(f"  Output directory: {output_dir}")
    print(f"  Instruction file: {instruction_path}")
    
    # 生成数据集配置 - 使用项目根目录的官方 normalizer.pt
    project_root = Path(__file__).resolve().parent.parent
    normalizer_abs_path = (project_root / "normalizer.pt").absolute()
    
    config_content = f"""# RDT2 dataset config for pika
# NOTE: Always use the official normalizer.pt in project root!
name: pika/bottle_task
type: single
shards_dir: {output_dir.absolute()}
kwargs:
  instruction_path: {instruction_path.absolute()}
  normalizer_path: {normalizer_abs_path}
"""
    config_path = output_dir / "dataset_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    print(f"  Dataset config: {config_path}")


# ===================== Normalizer 计算 =====================

def compute_action_statistics(input_dir: Path, action_horizon: int = 24) -> dict:
    """
    遍历所有 episode 计算动作统计信息
    
    Returns:
        stats: {'mean': (20,), 'std': (20,), 'min': (20,), 'max': (20,)}
    """
    episode_dirs = sorted([d for d in Path(input_dir).iterdir() 
                          if d.is_dir() and d.name.startswith('episode')])
    
    all_actions = []
    
    for episode_dir in tqdm(episode_dirs, desc="Computing action statistics"):
        h5_path = episode_dir / 'data.hdf5'
        if not h5_path.exists():
            continue
        
        with h5py.File(h5_path, 'r') as f:
            pose_l = safe_read_hdf5_dataset(f['localization/pose/pika_l'])
            pose_r = safe_read_hdf5_dataset(f['localization/pose/pika_r'])
            dist_l = safe_read_hdf5_dataset(f['gripper/encoderDistance/pika_l'])
            dist_r = safe_read_hdf5_dataset(f['gripper/encoderDistance/pika_r'])
            
            # 转弧度
            if is_degree(pose_l[:, 3:]) or is_degree(pose_r[:, 3:]):
                pose_l[:, 3:] = np.radians(pose_l[:, 3:])
                pose_r[:, 3:] = np.radians(pose_r[:, 3:])
            
            pose_mats_l = pose6d_to_mat(pose_l)
            pose_mats_r = pose6d_to_mat(pose_r)
            
            n = min(len(pose_l), len(pose_r), len(dist_l), len(dist_r)) - action_horizon
            
            for i in range(n):
                # action chunk 是从 i+1 开始的未来帧
                future_idx = np.arange(i + 1, i + 1 + action_horizon)
                action_r = compute_relative_action(pose_mats_r[i], pose_mats_r[future_idx], dist_r[future_idx])
                action_l = compute_relative_action(pose_mats_l[i], pose_mats_l[future_idx], dist_l[future_idx])
                action = np.concatenate([action_r, action_l], axis=-1)
                all_actions.append(action.reshape(-1, 20))  # (24, 20)
    
    all_actions = np.concatenate(all_actions, axis=0)  # (N*24, 20)
    
    stats = {
        'mean': all_actions.mean(axis=0).astype(np.float32),
        'std': all_actions.std(axis=0).astype(np.float32),
        'min': all_actions.min(axis=0).astype(np.float32),
        'max': all_actions.max(axis=0).astype(np.float32),
    }
    
    return stats


def create_rdt2_compatible_normalizer(stats: dict, output_path: Path):
    """
    创建与 RDT2 官方格式兼容的 normalizer
    
    官方的归一化公式: normalized = x * scale + offset
    反归一化公式: x = (normalized - offset) / scale
    
    目标：所有维度归一化到 [-1, 1]
    - Position (dim 0-2, 10-12): min-max 归一化
    - Rot6d (dim 3-8, 13-18): 不归一化 (scale=1, offset=0)，因为已经是单位向量
    - Gripper (dim 9, 19): 映射到 [-1, 1]
    
    为了复用官方 RVQ Tokenizer：
    1. 将 Pika gripper 范围映射到官方范围 [0, 0.088]
    2. 使用与官方相同的归一化参数
    """
    action_min = stats['min']
    action_max = stats['max']
    
    # 初始化 offset 和 scale（20维）
    # 公式: y = x * scale + offset
    # 要使 [min, max] -> [-1, 1]
    # scale = 2 / (max - min)
    # offset = -1 - min * scale = -1 - min * 2 / (max - min) = -(min + max) / (max - min)
    offset = np.zeros(20, dtype=np.float32)
    scale = np.ones(20, dtype=np.float32)
    
    # === Position 归一化 (dim 0-2 右臂, dim 10-12 左臂) ===
    for dims in [(0, 3), (10, 13)]:
        start, end = dims
        pos_min = action_min[start:end]
        pos_max = action_max[start:end]
        pos_range = pos_max - pos_min
        pos_range = np.maximum(pos_range, 1e-6)  # 防止除零
        
        scale[start:end] = 2.0 / pos_range
        offset[start:end] = -1.0 - pos_min * scale[start:end]
    
    # === Rot6d 不归一化 (dim 3-8 右臂, dim 13-18 左臂) ===
    # scale = 1, offset = 0 (已经是默认值)
    
    # === Gripper 归一化 (dim 9 右臂, dim 19 左臂) ===
    # 官方使用: [0, 0.088] -> [-1, 1]
    # scale = 2 / 0.088 = 22.7273
    # offset = -1 - 0 * scale = -1
    # 
    # 但 Pika gripper 范围约 [0.028, 0.097]，需要先映射到 [0, 0.088]
    # 在 process_episode 中会处理这个映射
    scale[9] = 22.7273  # 2 / 0.088
    scale[19] = 22.7273
    offset[9] = -1.0
    offset[19] = -1.0
    
    # 验证归一化范围
    print("\n=== 验证归一化范围 ===")
    print("公式: normalized = x * scale + offset")
    
    # Position 验证（使用原始 pika 范围）
    for name, start in [("右臂 Position", 0), ("左臂 Position", 10)]:
        test_min = action_min[start:start+3]
        test_max = action_max[start:start+3]
        norm_min = test_min * scale[start:start+3] + offset[start:start+3]
        norm_max = test_max * scale[start:start+3] + offset[start:start+3]
        print(f"{name}: [{test_min}] -> [{norm_min}]")
        print(f"          [{test_max}] -> [{norm_max}]")
    
    # Gripper 验证（假设已映射到 [0, 0.088]）
    grip_test = np.array([0.0, 0.088])
    grip_norm = grip_test * scale[9] + offset[9]
    print(f"Gripper (映射后): [0.0, 0.088] -> [{grip_norm[0]:.2f}, {grip_norm[1]:.2f}]")
    
    # 创建 ParameterDict 格式
    normalizer_dict = torch.nn.ParameterDict({
        'action': torch.nn.ParameterDict({
            'offset': torch.nn.Parameter(torch.from_numpy(offset).float(), requires_grad=False),
            'scale': torch.nn.Parameter(torch.from_numpy(scale).float(), requires_grad=False),
            'input_stats': torch.nn.ParameterDict({
                'min': torch.nn.Parameter(torch.from_numpy(stats['min']).float(), requires_grad=False),
                'max': torch.nn.Parameter(torch.from_numpy(stats['max']).float(), requires_grad=False),
                'mean': torch.nn.Parameter(torch.from_numpy(stats['mean']).float(), requires_grad=False),
                'std': torch.nn.Parameter(torch.from_numpy(stats['std']).float(), requires_grad=False),
            })
        })
    })
    
    # 保存 gripper 映射参数
    gripper_mapping = {
        'pika_min_r': float(stats['min'][9]),
        'pika_max_r': float(stats['max'][9]),
        'pika_min_l': float(stats['min'][19]),
        'pika_max_l': float(stats['max'][19]),
        'target_min': 0.0,
        'target_max': 0.088,
    }
    
    torch.save(normalizer_dict, output_path)
    
    gripper_config_path = output_path.parent / "gripper_mapping.json"
    with open(gripper_config_path, 'w') as f:
        json.dump(gripper_mapping, f, indent=2)
    
    print(f"\nNormalizer saved to: {output_path}")
    print(f"Gripper mapping saved to: {gripper_config_path}")
    print()
    print("=== Gripper 映射 ===")
    print(f"Pika 右臂: [{gripper_mapping['pika_min_r']:.4f}, {gripper_mapping['pika_max_r']:.4f}] -> [0, 0.088]")
    print(f"Pika 左臂: [{gripper_mapping['pika_min_l']:.4f}, {gripper_mapping['pika_max_l']:.4f}] -> [0, 0.088]")
    print()
    print("⚠️  重要：在处理数据时，会自动将 gripper 值映射到 [0, 0.088] 范围")
    print("⚠️  这样可以复用官方的 RVQ Tokenizer")
    
    return normalizer_dict, gripper_mapping


def main():
    parser = argparse.ArgumentParser(description='Convert Pika dataset to RDT2 WebDataset format')
    parser.add_argument('--input-dir', type=Path, required=True,
                        help='Input directory containing pika episodes (e.g., pika_raw_data)')
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Output directory for WebDataset shards')
    parser.add_argument('--instruction', type=str, 
                        default="Put the bottle into the tape ring, and then take it out with the other hand.",
                        help='Task instruction')
    parser.add_argument('--samples-per-shard', type=int, default=1000,
                        help='Number of samples per shard')
    parser.add_argument('--action-horizon', type=int, default=24,
                        help='Action prediction horizon (frames)')
    parser.add_argument('--image-size', type=int, default=384,
                        help='Target image size')
    parser.add_argument('--no-tokenizer', action='store_true',
                        help='Skip action tokenization (faster but generates zero tokens)')
    parser.add_argument('--normalizer-path', type=str, default='normalizer.pt',
                        help='Path to normalizer checkpoint (MUST use project root normalizer.pt!)')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device for tokenizer')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # 检查 normalizer 路径
    if not os.path.exists(args.normalizer_path):
        print(f"ERROR: Normalizer not found at {args.normalizer_path}")
        print("Please ensure normalizer.pt exists in the project root directory.")
        sys.exit(1)
    
    # 警告用户确保使用官方 normalizer
    if 'pika_normalizer' in args.normalizer_path:
        print("\n" + "!" * 60)
        print("ERROR: Do NOT use pika_normalizer.pt!")
        print("You MUST use the official normalizer.pt from project root!")
        print("!" * 60 + "\n")
        sys.exit(1)
    
    convert_pika_to_rdt2(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        instruction=args.instruction,
        samples_per_shard=args.samples_per_shard,
        action_horizon=args.action_horizon,
        image_size=args.image_size,
        use_tokenizer=not args.no_tokenizer,
        normalizer_path=args.normalizer_path,
        device=args.device,
    )


if __name__ == '__main__':
    main()
