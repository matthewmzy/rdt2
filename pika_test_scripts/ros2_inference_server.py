#!/usr/bin/env python3
"""
RDT2 ROS2 推理服务节点 (Service 模式)

使用 Service 模式实现同步的请求-响应推理。
客户端发送左右相机图像，服务端返回动作。

服务接口 (RdtUmi.srv):
    Request:
        - sensor_msgs/Image umi_left_image: 左手相机图像
        - sensor_msgs/Image umi_right_image: 右手相机图像
    Response:
        - std_msgs/Float32MultiArray action_full: 完整动作序列 (24, 14) 或 (24, 20) 展平

输出格式:
    - use_rpy=True (默认):  (24, 14) 展平为 336D
        - 每帧: [right(7), left(7)] = 14D
        - 每臂: [pos(3), rpy(3), gripper(1)]
    - use_rpy=False:        (24, 20) 展平为 480D
        - 每帧: [right(10), left(10)] = 20D
        - 每臂: [pos(3), rot6d(6), gripper(1)]

参数:
    - fm_checkpoint: Flow Matching checkpoint 路径
    - vlm_model: VLM 模型路径
    - config_path: 配置文件路径
    - normalizer_path: normalizer 路径
    - instruction: 任务指令
    - image_size: 图像 resize 尺寸 (默认 384)
    - use_rpy: 输出格式 (默认 True)

用法:
    # 1. 首先编译自定义消息 (在你的 ROS2 workspace 中)
    # 将 RdtUmi.srv 放到 srv/ 目录下，然后 colcon build

    # 2. 启动服务节点
    python pika_test_scripts/ros2_inference_server.py --ros-args \
        -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000

    # 3. 客户端调用示例 (Python)
    # from rdt2_interfaces.srv import RdtUmi
    # client = node.create_client(RdtUmi, '/rdt2/inference')
    # request = RdtUmi.Request()
    # request.umi_left_image = left_img_msg
    # request.umi_right_image = right_img_msg
    # response = client.call(request)
    # action = np.array(response.action_full.data).reshape(24, 14)  # use_rpy=True
"""

import os
import sys
import threading
from pathlib import Path
from typing import Optional

import numpy as np
import cv2
import torch
import yaml
from scipy.spatial.transform import Rotation as R

# ROS2 imports
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, MultiArrayDimension, String
from cv_bridge import CvBridge

# 添加项目根目录到路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from models.rdt_inferencer import RDTInferencer

# 导入自定义服务消息
from rdt2_interfaces.srv import RdtUmi


class RDT2InferenceServer(Node):
    """RDT2 ROS2 推理服务节点 (Service 模式)"""
    
    def __init__(self):
        super().__init__('rdt2_inference_server')
        
        # 声明参数
        self.declare_parameter('fm_checkpoint', 'outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000')
        self.declare_parameter('vlm_model', 'robotics-diffusion-transformer/RDT2-VQ')
        self.declare_parameter('config_path', 'configs/rdt/post_train.yaml')
        self.declare_parameter('normalizer_path', 'normalizer.pt')
        self.declare_parameter('instruction', 'Put the bottle into the tape ring, and then take it out with the other hand.')
        self.declare_parameter('image_size', 384)
        self.declare_parameter('device', 'cuda:0')
        self.declare_parameter('use_rpy', True)
        self.declare_parameter('action_horizon', 24)  # 返回的动作帧数
        
        # 获取参数
        self.fm_checkpoint = self.get_parameter('fm_checkpoint').value
        self.vlm_model = self.get_parameter('vlm_model').value
        self.config_path = self.get_parameter('config_path').value
        self.normalizer_path = self.get_parameter('normalizer_path').value
        self.instruction = self.get_parameter('instruction').value
        self.image_size = self.get_parameter('image_size').value
        self.device = self.get_parameter('device').value
        self.use_rpy = self.get_parameter('use_rpy').value
        self.action_horizon = self.get_parameter('action_horizon').value
        
        self.get_logger().info(f"FM Checkpoint: {self.fm_checkpoint}")
        self.get_logger().info(f"VLM Model: {self.vlm_model}")
        self.get_logger().info(f"Instruction: {self.instruction}")
        self.get_logger().info(f"Output format: {'(24,14) xyz+rpy+gripper' if self.use_rpy else '(24,20) xyz+rot6d+gripper'}")
        
        # 初始化 CV Bridge
        self.bridge = CvBridge()
        
        # 推理状态
        self.inferencer: Optional[RDTInferencer] = None
        self.config = None
        self.is_ready = False
        self.inference_lock = threading.Lock()
        
        # 发布状态
        self.pub_status = self.create_publisher(String, '/rdt2/status', 10)
        
        # 创建服务
        self.srv = self.create_service(
            RdtUmi,
            '/rdt2/inference',
            self.callback_inference_service,
            callback_group=ReentrantCallbackGroup()
        )
        self.get_logger().info("Service '/rdt2/inference' (RdtUmi) created")
        
        # 初始化模型
        self.get_logger().info("Initializing model (this may take a while)...")
        self.initialize_model()
    
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
            self.get_logger().info("Model initialized successfully! Ready to serve.")
            
            # 发布状态
            status_msg = String()
            status_msg.data = "ready"
            self.pub_status.publish(status_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to initialize model: {e}")
            import traceback
            traceback.print_exc()
    
    def rot6d_to_euler(self, rot6d: np.ndarray) -> np.ndarray:
        """将 6D 旋转表示转换为欧拉角 (roll, pitch, yaw)"""
        col0 = rot6d[:3]
        col1 = rot6d[3:6]
        
        # Gram-Schmidt 正交化
        col0 = col0 / (np.linalg.norm(col0) + 1e-8)
        col1 = col1 - np.dot(col1, col0) * col0
        col1 = col1 / (np.linalg.norm(col1) + 1e-8)
        col2 = np.cross(col0, col1)
        
        rotmat = np.stack([col0, col1, col2], axis=1)
        euler = R.from_matrix(rotmat).as_euler('xyz')
        return euler
    
    def convert_action_to_rpy(self, action_10d: np.ndarray) -> np.ndarray:
        """将 10D 动作转换为 7D (xyz + rpy + gripper)"""
        pos = action_10d[:3]
        rot6d = action_10d[3:9]
        gripper = action_10d[9:10]
        rpy = self.rot6d_to_euler(rot6d)
        return np.concatenate([pos, rpy, gripper])
    
    def resize_image(self, img: np.ndarray) -> np.ndarray:
        """将图像 resize 到目标尺寸"""
        return cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
    
    def run_inference(self, img_left: np.ndarray, img_right: np.ndarray) -> np.ndarray:
        """
        执行推理
        
        Args:
            img_left: 左相机图像 (H, W, 3) RGB
            img_right: 右相机图像 (H, W, 3) RGB
        
        Returns:
            action: 动作数组，已 flatten
                - use_rpy=True: (336,) = (24, 14) flattened, 每帧 [right(7), left(7)]
                - use_rpy=False: (480,) = (24, 20) flattened, 每帧 [right(10), left(10)]
        """
        with self.inference_lock:
            # Resize 图像
            img_left_resized = self.resize_image(img_left)
            img_right_resized = self.resize_image(img_right)
            
            # 准备输入
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
            
            # 获取 action_horizon 帧的动作
            horizon = min(self.action_horizon, action_pred.shape[0])
            actions_chunk = action_pred[:horizon]  # (horizon, 20)
            
            # 转换格式
            if self.use_rpy:
                # 将每帧的 (20,) 转换为 (14,)
                converted_actions = []
                for t in range(horizon):
                    current_action = actions_chunk[t]  # (20,)
                    right_arm_raw = current_action[:10]
                    left_arm_raw = current_action[10:]
                    right_arm = self.convert_action_to_rpy(right_arm_raw)  # (7,)
                    left_arm = self.convert_action_to_rpy(left_arm_raw)    # (7,)
                    converted_actions.append(np.concatenate([right_arm, left_arm]))  # (14,)
                actions_output = np.stack(converted_actions, axis=0)  # (horizon, 14)
            else:
                actions_output = actions_chunk  # (horizon, 20)
            
            # Flatten 并返回
            return actions_output.flatten().astype(np.float32)
    
    def callback_inference_service(self, request, response):
        """
        服务回调：接收图像，返回动作
        
        Request:
            - umi_left_image: sensor_msgs/Image
            - umi_right_image: sensor_msgs/Image
        
        Response:
            - action_full: std_msgs/Float32MultiArray
                - use_rpy=True: (24, 14) flattened = 336D
                - use_rpy=False: (24, 20) flattened = 480D
        """
        if not self.is_ready:
            self.get_logger().warn("Model not ready, rejecting request")
            response.action_full = Float32MultiArray()
            return response
        
        try:
            # 转换图像
            img_left = self.bridge.imgmsg_to_cv2(request.umi_left_image, desired_encoding='rgb8')
            img_right = self.bridge.imgmsg_to_cv2(request.umi_right_image, desired_encoding='rgb8')
            
            # 推理
            action = self.run_inference(img_left, img_right)
            
            # 计算实际的 shape
            action_dim = 14 if self.use_rpy else 20
            horizon = len(action) // action_dim
            
            # 构建响应 (with 2D layout info)
            response.action_full = Float32MultiArray()
            response.action_full.layout.dim = [
                MultiArrayDimension(label='horizon', size=horizon, stride=horizon * action_dim),
                MultiArrayDimension(label='action', size=action_dim, stride=action_dim)
            ]
            response.action_full.data = action.tolist()
            
            self.get_logger().debug(f"Inference done, action shape=({horizon}, {action_dim})")
            
        except Exception as e:
            self.get_logger().error(f"Inference failed: {e}")
            import traceback
            traceback.print_exc()
            response.action_full = Float32MultiArray()
        
        return response


def main(args=None):
    rclpy.init(args=args)
    
    node = RDT2InferenceServer()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
