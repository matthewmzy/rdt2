#!/usr/bin/env python3
"""
RDT2 ROS2 推理节点 (Topic 模式)

该节点订阅左右手相机图像，进行推理并发布动作指令。

订阅话题:
    - /umi/left/image_raw (sensor_msgs/Image): 左手相机图像
    - /umi/right/image_raw (sensor_msgs/Image): 右手相机图像

发布话题:
    - /rdt2/action_full (std_msgs/Float32MultiArray): 完整动作
        - use_rpy=True (默认):  [right(7), left(7)] = 14D
        - use_rpy=False:        [right(10), left(10)] = 20D
    - /rdt2/status (std_msgs/String): 节点状态

参数:
    - fm_checkpoint: Flow Matching checkpoint 路径
    - vlm_model: VLM 模型路径
    - config_path: 配置文件路径
    - normalizer_path: normalizer 路径
    - instruction: 任务指令
    - publish_rate: 发布频率 (Hz)
    - image_size: 图像 resize 尺寸 (默认 384)
    - use_rpy: 输出格式 (默认 True)
        - True: xyz(3) + rpy(3) + gripper(1) = 7D per arm
        - False: xyz(3) + rot6d(6) + gripper(1) = 10D per arm
    - action_horizon: 预测动作帧数 (默认 24)
    - execute_horizon: 每次执行帧数 (默认 12)

用法:
    python pika_test_scripts/ros2_inference_node.py --ros-args \
        -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000

    # 设置更多参数
    python pika_test_scripts/ros2_inference_node.py --ros-args \
        -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
        -p publish_rate:=30.0 \
        -p use_rpy:=true
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import cv2
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from cv_bridge import CvBridge

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from models.rdt_inferencer import RDTInferencer


# ==================== 坐标变换工具函数 ====================

def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """将 6D 旋转表示转换为旋转矩阵 (3, 3)"""
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)  # (3, 3)


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 6D 旋转表示"""
    return np.concatenate([mat[:, 0], mat[:, 1]], axis=-1)


def action10d_to_mat(action_10d: np.ndarray) -> np.ndarray:
    """
    将 10D 动作 [pos(3), rot6d(6), gripper(1)] 转换为 4x4 变换矩阵
    注意：gripper 不参与变换计算
    """
    pos = action_10d[:3]
    rot6d = action_10d[3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    mat = np.eye(4, dtype=action_10d.dtype)
    mat[:3, :3] = rotmat
    mat[:3, 3] = pos
    return mat


def mat_to_action10d(mat: np.ndarray, gripper: float) -> np.ndarray:
    """将 4x4 变换矩阵转换回 10D 动作格式"""
    pos = mat[:3, 3]
    rotmat = mat[:3, :3]
    rot6d = mat_to_rot6d(rotmat)
    return np.concatenate([pos, rot6d, [gripper]], axis=-1)


def convert_actions_to_frame_relative(actions: np.ndarray) -> np.ndarray:
    """
    将相对于当前帧的动作序列转换为相对于上一帧的动作序列
    
    输入定义 (模型输出):
        action[t] = T_current^{-1} @ T_{current+t+1}
        即 action[t] 表示第 (current + t + 1) 帧相对于 current 帧的变换
    
    输出定义 (帧间增量):
        delta[t] = T_{current+t}^{-1} @ T_{current+t+1}
        即 delta[t] 表示第 (current + t + 1) 帧相对于第 (current + t) 帧的变换
    
    转换公式:
        delta[0] = action[0]  (第一帧没有上一帧，直接使用)
        delta[t] = action[t-1]^{-1} @ action[t]  (t > 0)
    
    Args:
        actions: (T, 20) 相对于当前帧的动作序列，格式 [right(10), left(10)]
    
    Returns:
        deltas: (T, 20) 相对于上一帧的动作序列
    """
    T = actions.shape[0]
    deltas = np.zeros_like(actions)
    
    # 第一帧直接复制（相对于当前帧 = 相对于上一帧）
    deltas[0] = actions[0].copy()
    
    # 后续帧需要计算帧间变换
    for t in range(1, T):
        # 右臂
        right_prev = actions[t-1, :10]
        right_curr = actions[t, :10]
        right_prev_mat = action10d_to_mat(right_prev)
        right_curr_mat = action10d_to_mat(right_curr)
        # delta = prev^{-1} @ curr
        right_delta_mat = np.linalg.inv(right_prev_mat) @ right_curr_mat
        right_delta = mat_to_action10d(right_delta_mat, right_curr[9])  # 使用当前帧的 gripper
        
        # 左臂
        left_prev = actions[t-1, 10:20]
        left_curr = actions[t, 10:20]
        left_prev_mat = action10d_to_mat(left_prev)
        left_curr_mat = action10d_to_mat(left_curr)
        left_delta_mat = np.linalg.inv(left_prev_mat) @ left_curr_mat
        left_delta = mat_to_action10d(left_delta_mat, left_curr[9])
        
        deltas[t] = np.concatenate([right_delta, left_delta])
    
    return deltas


class RDT2InferenceNode(Node):
    """RDT2 ROS2 推理节点"""
    
    def __init__(self):
        super().__init__('rdt2_inference_node')
        
        # 声明参数
        self.declare_parameter('fm_checkpoint', 'outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000')
        self.declare_parameter('vlm_model', 'robotics-diffusion-transformer/RDT2-VQ')
        self.declare_parameter('config_path', 'configs/rdt/post_train.yaml')
        self.declare_parameter('normalizer_path', 'normalizer.pt')
        self.declare_parameter('instruction', 'Put the bottle into the tape ring, and then take it out with the other hand.')
        self.declare_parameter('publish_rate', 30.0)  # Hz
        self.declare_parameter('image_size', 384)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('action_horizon', 24)
        self.declare_parameter('execute_horizon', 12)  # 每次执行前多少帧
        self.declare_parameter('use_rpy', True)  # True: xyz+rpy+gripper (7D), False: xyz+rot6d+gripper (10D)
        
        # 获取参数
        self.fm_checkpoint = self.get_parameter('fm_checkpoint').value
        self.vlm_model = self.get_parameter('vlm_model').value
        self.config_path = self.get_parameter('config_path').value
        self.normalizer_path = self.get_parameter('normalizer_path').value
        self.instruction = self.get_parameter('instruction').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.image_size = self.get_parameter('image_size').value
        self.device = self.get_parameter('device').value
        self.action_horizon = self.get_parameter('action_horizon').value
        self.execute_horizon = self.get_parameter('execute_horizon').value
        self.use_rpy = self.get_parameter('use_rpy').value
        
        self.get_logger().info(f"FM Checkpoint: {self.fm_checkpoint}")
        self.get_logger().info(f"VLM Model: {self.vlm_model}")
        self.get_logger().info(f"Instruction: {self.instruction}")
        self.get_logger().info(f"Output format: {'xyz+rpy+gripper (7D)' if self.use_rpy else 'xyz+rot6d+gripper (10D)'}")
        
        # 初始化 CV Bridge
        self.bridge = CvBridge()
        
        # 图像缓存
        self.img_left: Optional[np.ndarray] = None
        self.img_right: Optional[np.ndarray] = None
        self.img_left_timestamp = None
        self.img_right_timestamp = None
        self.img_lock = threading.Lock()
        
        # 动作队列 (用于平滑输出)
        self.action_queue = deque(maxlen=self.action_horizon)
        self.action_idx = 0
        self.action_lock = threading.Lock()
        
        # 推理状态
        self.inferencer: Optional[RDTInferencer] = None
        self.config = None
        self.is_ready = False
        
        # QoS 配置
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        # 创建订阅者
        self.sub_left = self.create_subscription(
            Image,
            '/umi/left/image_raw',
            self.callback_left_image,
            qos_profile
        )
        
        self.sub_right = self.create_subscription(
            Image,
            '/umi/right/image_raw',
            self.callback_right_image,
            qos_profile
        )
        
        # 创建发布者 - 只发布完整动作
        self.pub_full = self.create_publisher(
            Float32MultiArray,
            '/rdt2/action_full',
            10
        )
        
        # 发布当前状态
        self.pub_status = self.create_publisher(
            String,
            '/rdt2/status',
            10
        )
        
        # 创建定时器
        timer_period = 1.0 / self.publish_rate
        self.timer = self.create_timer(timer_period, self.timer_callback)
        
        # 初始化模型 (在后台线程)
        self.init_thread = threading.Thread(target=self.initialize_model)
        self.init_thread.start()
        
        self.get_logger().info("RDT2 Inference Node (Topic mode) started. Initializing model...")
    
    def initialize_model(self):
        """初始化推理模型"""
        try:
            self.get_logger().info(f"Loading config from {self.config_path}...")
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            self.get_logger().info(f"Loading FM model from {self.fm_checkpoint}...")
            self.get_logger().info(f"Loading VLM from {self.vlm_model}...")
            
            self.inferencer = RDTInferencer(
                config=self.config,
                pretrained_path=self.fm_checkpoint,
                normalizer_path=self.normalizer_path,
                pretrained_vision_language_model_name_or_path=self.vlm_model,
                device=self.device,
                dtype=torch.bfloat16,
            )
            
            self.is_ready = True
            self.get_logger().info("Model initialized successfully!")
            
            # 发布状态
            status_msg = String()
            status_msg.data = "ready"
            self.pub_status.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
    
    def rot6d_to_euler(self, rot6d: np.ndarray) -> np.ndarray:
        """
        将 6D 旋转表示转换为欧拉角 (roll, pitch, yaw)
        
        Args:
            rot6d: (6,) 6D rotation representation (两个列向量)
        Returns:
            euler: (3,) roll, pitch, yaw in radians
        """
        # 6D rotation: [col0(3), col1(3)]
        col0 = rot6d[:3]
        col1 = rot6d[3:6]
        
        # Gram-Schmidt 正交化
        col0 = col0 / (np.linalg.norm(col0) + 1e-8)
        col1 = col1 - np.dot(col1, col0) * col0
        col1 = col1 / (np.linalg.norm(col1) + 1e-8)
        col2 = np.cross(col0, col1)
        
        # 构建旋转矩阵
        rotmat = np.stack([col0, col1, col2], axis=1)  # (3, 3)
        
        # 转换为欧拉角 (xyz 顺序，即 roll-pitch-yaw)
        euler = R.from_matrix(rotmat).as_euler('xyz')
        return euler
    
    def convert_action_to_rpy(self, action_10d: np.ndarray) -> np.ndarray:
        """
        将 10D 动作 (xyz + rot6d + gripper) 转换为 7D (xyz + rpy + gripper)
        
        Args:
            action_10d: (10,) = [pos(3), rot6d(6), gripper(1)]
        Returns:
            action_7d: (7,) = [pos(3), rpy(3), gripper(1)]
        """
        pos = action_10d[:3]
        rot6d = action_10d[3:9]
        gripper = action_10d[9:10]
        
        rpy = self.rot6d_to_euler(rot6d)
        
        return np.concatenate([pos, rpy, gripper])
    
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """
        将图像 resize 到目标尺寸
        
        Args:
            img: 输入图像 (H, W, 3)，原始尺寸 640x480
        Returns:
            resized: (image_size, image_size, 3)
        """
        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    
    def callback_left_image(self, msg: Image):
        """左手相机图像回调"""
        try:
            # 将 ROS Image 转换为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # BGR -> RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            with self.img_lock:
                self.img_left = cv_image
                self.img_left_timestamp = msg.header.stamp
                
        except Exception as e:
            self.get_logger().error(f"Error processing left image: {e}")
    
    def callback_right_image(self, msg: Image):
        """右手相机图像回调"""
        try:
            # 将 ROS Image 转换为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # BGR -> RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            with self.img_lock:
                self.img_right = cv_image
                self.img_right_timestamp = msg.header.stamp
                
        except Exception as e:
            self.get_logger().error(f"Error processing right image: {e}")
    
    def run_inference(self) -> bool:
        """执行一次推理"""
        if not self.is_ready or self.inferencer is None:
            return False
        
        with self.img_lock:
            if self.img_left is None or self.img_right is None:
                self.get_logger().warn("Missing images for inference")
                return False
            
            img_left = self.img_left.copy()
            img_right = self.img_right.copy()
        
        try:
            # Resize 图像到 384x384
            img_left_resized = self.resize_image(img_left)
            img_right_resized = self.resize_image(img_right)
            
            # 准备输入
            # RDT2 期望的输入格式: {'left_stereo': (H, W, 3), 'right_stereo': (H, W, 3)}
            # 最终会在模型内部拼接成 [left, right]
            observations = {
                'images': {
                    'left_stereo': img_left_resized,
                    'right_stereo': img_right_resized,
                },
                'state': np.zeros(self.config['common']['state_dim'], dtype=np.float32)
            }
            
            # 推理
            with torch.no_grad():
                action_pred = self.inferencer.step(observations, self.instruction)
                action_pred = action_pred.cpu().numpy()  # (24, 20)
            
            # 将相对于当前帧的动作转换为相对于上一帧的动作
            action_pred = convert_actions_to_frame_relative(action_pred)
            
            # 更新动作队列
            with self.action_lock:
                self.action_queue.clear()
                for i in range(self.action_horizon):
                    self.action_queue.append(action_pred[i])
                self.action_idx = 0
            
            self.get_logger().debug("Inference completed successfully")
            return True
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def timer_callback(self):
        """定时发布动作"""
        if not self.is_ready:
            # 发布状态
            status_msg = String()
            status_msg.data = "initializing"
            self.pub_status.publish(status_msg)
            return
        
        # 检查是否需要进行新的推理
        with self.action_lock:
            need_inference = len(self.action_queue) == 0 or self.action_idx >= self.execute_horizon
        
        if need_inference:
            # 进行推理
            if self.run_inference():
                with self.action_lock:
                    self.action_idx = 0
        
        # 获取当前动作
        with self.action_lock:
            if len(self.action_queue) == 0:
                return
            
            # 获取当前帧动作
            if self.action_idx < len(self.action_queue):
                current_action = self.action_queue[self.action_idx]
                self.action_idx += 1
            else:
                # 使用最后一帧
                current_action = self.action_queue[-1]
        
        # 解析动作
        # 原始格式: [right_arm(10), left_arm(10)] = 20D
        # 每臂原始: [pos(3), rot6d(6), gripper(1)]
        right_arm_action_raw = current_action[:10]   # 右臂 (10D)
        left_arm_action_raw = current_action[10:]    # 左臂 (10D)
        
        # 根据 use_rpy 参数转换格式
        if self.use_rpy:
            # 转换为 xyz + rpy + gripper (7D)
            right_arm_action = self.convert_action_to_rpy(right_arm_action_raw)
            left_arm_action = self.convert_action_to_rpy(left_arm_action_raw)
            arm_dim = 7
        else:
            # 保持原始 xyz + rot6d + gripper (10D)
            right_arm_action = right_arm_action_raw
            left_arm_action = left_arm_action_raw
            arm_dim = 10
        
        # 发布完整动作
        full_action = np.concatenate([right_arm_action, left_arm_action])
        full_msg = Float32MultiArray()
        full_msg.layout.dim = [MultiArrayDimension(label='action', size=arm_dim*2, stride=arm_dim*2)]
        full_msg.data = full_action.astype(np.float32).tolist()
        self.pub_full.publish(full_msg)
        
        # 发布状态
        status_msg = String()
        status_msg.data = f"running:action_idx={self.action_idx-1}"
        self.pub_status.publish(status_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = RDT2InferenceNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
