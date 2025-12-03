#!/usr/bin/env python
"""
Pika 夹爪映射工具

数据流：
1. 训练时：pika [0.028, 0.097] → 官方标准 [0, 0.088] → normalize → VAE encode
2. 推理时：VAE decode → unnormalize → 官方标准 [0, 0.088] → 需要映射回 pika [0.028, 0.097]

官方部署代码的映射（针对 UMI 夹爪）：
    action / 0.088 * 0.10  # [0, 0.088] → [0, 0.10]

你的 pika 需要的映射：
    action / 0.088 * (pika_max - pika_min) + pika_min  # [0, 0.088] → [pika_min, pika_max]
"""

import numpy as np
import json
from pathlib import Path


class PikaGripperMapper:
    """Pika 夹爪映射器"""
    
    def __init__(self, gripper_mapping_path: str = None):
        """
        Args:
            gripper_mapping_path: gripper_mapping.json 的路径
        """
        if gripper_mapping_path is not None:
            with open(gripper_mapping_path, 'r') as f:
                mapping = json.load(f)
            self.pika_min_r = mapping['pika_min_r']
            self.pika_max_r = mapping['pika_max_r']
            self.pika_min_l = mapping['pika_min_l']
            self.pika_max_l = mapping['pika_max_l']
        else:
            # 默认值（你的 pika 数据的实际范围）
            self.pika_min_r = 0.028322532773017883
            self.pika_max_r = 0.09670703858137131
            self.pika_min_l = 0.02773468941450119
            self.pika_max_l = 0.09670703858137131
        
        self.official_min = 0.0
        self.official_max = 0.088
    
    def to_official(self, pika_gripper: np.ndarray, arm: str = 'right') -> np.ndarray:
        """
        训练时：pika 夹爪 → 官方标准范围
        
        Args:
            pika_gripper: 原始 pika 夹爪宽度
            arm: 'right' 或 'left'
        Returns:
            official_gripper: 官方标准范围 [0, 0.088]
        """
        if arm == 'right':
            pika_min, pika_max = self.pika_min_r, self.pika_max_r
        else:
            pika_min, pika_max = self.pika_min_l, self.pika_max_l
        
        # [pika_min, pika_max] → [0, 0.088]
        normalized = (pika_gripper - pika_min) / (pika_max - pika_min)
        official = normalized * self.official_max
        return np.clip(official, self.official_min, self.official_max)
    
    def to_pika(self, official_gripper: np.ndarray, arm: str = 'right') -> np.ndarray:
        """
        推理时：官方标准范围 → pika 夹爪
        
        ⚠️ 在真机部署时调用这个函数！
        
        Args:
            official_gripper: 模型输出的官方标准范围 [0, 0.088]
            arm: 'right' 或 'left'
        Returns:
            pika_gripper: pika 夹爪宽度 [pika_min, pika_max]
        """
        if arm == 'right':
            pika_min, pika_max = self.pika_min_r, self.pika_max_r
        else:
            pika_min, pika_max = self.pika_min_l, self.pika_max_l
        
        # [0, 0.088] → [pika_min, pika_max]
        normalized = official_gripper / self.official_max
        pika = normalized * (pika_max - pika_min) + pika_min
        return np.clip(pika, pika_min, pika_max)


def rescale_action_gripper_for_pika(
    action: np.ndarray,
    gripper_mapping_path: str = None,
) -> np.ndarray:
    """
    把模型输出的 action 中的夹爪从官方范围映射到 pika 范围
    
    ⚠️ 在真机部署时，替换官方的 `action / 0.088 * 0.10` 逻辑
    
    Args:
        action: 模型输出的 action，形状 (T, 20) 或 (T, 14) 等
                其中 dim 9 是右臂夹爪，dim 19 是左臂夹爪（如果有）
        gripper_mapping_path: gripper_mapping.json 的路径
    
    Returns:
        action: 夹爪已映射到 pika 范围的 action
    """
    mapper = PikaGripperMapper(gripper_mapping_path)
    action = action.copy()
    
    # 右臂夹爪 (dim 9)
    if action.shape[-1] > 9:
        action[..., 9] = mapper.to_pika(action[..., 9], arm='right')
    
    # 左臂夹爪 (dim 19)
    if action.shape[-1] > 19:
        action[..., 19] = mapper.to_pika(action[..., 19], arm='left')
    
    return action


# 示例用法
if __name__ == '__main__':
    print("=== Pika Gripper 映射测试 ===")
    print()
    
    # 创建映射器
    mapper = PikaGripperMapper()
    
    # 测试：训练时的映射
    print("训练时映射 (pika → official):")
    pika_values = np.array([0.028, 0.05, 0.097])
    for v in pika_values:
        official = mapper.to_official(v, arm='right')
        print(f"  pika {v:.4f} → official {official:.4f}")
    
    print()
    
    # 测试：推理时的映射
    print("推理时映射 (official → pika):")
    official_values = np.array([0.0, 0.044, 0.088])
    for v in official_values:
        pika = mapper.to_pika(v, arm='right')
        print(f"  official {v:.4f} → pika {pika:.4f}")
    
    print()
    
    # 验证往返转换
    print("验证往返转换 (pika → official → pika):")
    original = 0.06
    official = mapper.to_official(original, arm='right')
    recovered = mapper.to_pika(official, arm='right')
    print(f"  原始: {original:.4f}")
    print(f"  官方: {official:.4f}")
    print(f"  恢复: {recovered:.4f}")
    print(f"  误差: {abs(original - recovered):.6f}")
