# RDT2 ROS2 推理节点

本目录包含两个 ROS2 推理节点，用于将 RDT2 Flow Matching 模型部署为 ROS2 服务。

## 概述

| 脚本 | 模式 | 特点 |
|------|------|------|
| `ros2_inference_node.py` | Topic | 异步通信，持续发布动作 |
| `ros2_inference_server.py` | Service | 同步通信，请求-响应模式 |

## 前置条件

### 1. 编译 rdt2_interfaces 包

```bash
# 创建 ROS2 workspace（如果不存在）
mkdir -p ~/ros2_ws/src
cd ~/ros2_ws/src

# 创建包
ros2 pkg create --build-type ament_cmake rdt2_interfaces \
    --dependencies std_msgs sensor_msgs rosidl_default_generators

# 复制服务定义
mkdir -p rdt2_interfaces/srv
cp /home/ubuntu/mzy/RDT2/RdtUmi.srv rdt2_interfaces/srv/

# 编辑 CMakeLists.txt，添加：
# rosidl_generate_interfaces(${PROJECT_NAME}
#   "srv/RdtUmi.srv"
#   DEPENDENCIES std_msgs sensor_msgs
# )

# 编译
cd ~/ros2_ws
pip install empy==3.3.4 catkin_pkg lark  # 编译依赖
colcon build --packages-select rdt2_interfaces

# Source workspace
source ~/ros2_ws/install/setup.bash  # Bash
source ~/ros2_ws/install/setup.zsh   # Zsh
```

### 2. 服务接口定义 (RdtUmi.srv)

```
# Request
sensor_msgs/Image umi_left_image
sensor_msgs/Image umi_right_image
---
# Response
std_msgs/Float32MultiArray action_full
```

---

## 方式一：Topic 模式 (`ros2_inference_node.py`)

### 特点
- **异步通信**：持续订阅图像，持续发布动作
- **滚动执行**：每次推理预测 24 帧，执行 12 帧后重新推理
- **适用场景**：流式控制、实时跟踪

### 接口

| 类型 | 话题名 | 消息类型 | 说明 |
|------|--------|----------|------|
| 订阅 | `/umi/left/image_raw` | sensor_msgs/Image | 左手相机图像 |
| 订阅 | `/umi/right/image_raw` | sensor_msgs/Image | 右手相机图像 |
| 发布 | `/rdt2/action_full` | std_msgs/Float32MultiArray | 动作输出 |
| 发布 | `/rdt2/status` | std_msgs/String | 节点状态 |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fm_checkpoint` | `outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000` | FM checkpoint 路径 |
| `vlm_model` | `robotics-diffusion-transformer/RDT2-VQ` | VLM 模型 |
| `config_path` | `configs/rdt/post_train.yaml` | 配置文件 |
| `normalizer_path` | `normalizer.pt` | Normalizer 路径 |
| `instruction` | `Put the bottle into the tape ring...` | 任务指令 |
| `publish_rate` | `30.0` | 发布频率 (Hz) |
| `image_size` | `384` | 图像 resize 尺寸 |
| `device` | `cuda:0` | 推理设备 |
| `use_rpy` | `true` | 输出格式 (见下方) |
| `action_horizon` | `24` | 预测帧数 |
| `execute_horizon` | `12` | 执行帧数 |

### 启动

```bash
cd /home/ubuntu/mzy/RDT2

# 基本启动
python pika_test_scripts/ros2_inference_node.py --ros-args \
    -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000

# 自定义参数
python pika_test_scripts/ros2_inference_node.py --ros-args \
    -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000 \
    -p publish_rate:=30.0 \
    -p use_rpy:=true \
    -p instruction:="Pick up the cup."
```

---

## 方式二：Service 模式 (`ros2_inference_server.py`)

### 特点
- **同步通信**：每次请求发送图像，返回对应动作
- **请求-响应**：保证图像和动作一一对应
- **适用场景**：精确控制、步进执行

### 接口

| 类型 | 名称 | 接口类型 | 说明 |
|------|------|----------|------|
| 服务 | `/rdt2/inference` | rdt2_interfaces/srv/RdtUmi | 推理服务 |
| 发布 | `/rdt2/status` | std_msgs/String | 节点状态 |

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fm_checkpoint` | `outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000` | FM checkpoint 路径 |
| `vlm_model` | `robotics-diffusion-transformer/RDT2-VQ` | VLM 模型 |
| `config_path` | `configs/rdt/post_train.yaml` | 配置文件 |
| `normalizer_path` | `normalizer.pt` | Normalizer 路径 |
| `instruction` | `Put the bottle into the tape ring...` | 任务指令 |
| `image_size` | `384` | 图像 resize 尺寸 |
| `device` | `cuda:0` | 推理设备 |
| `use_rpy` | `true` | 输出格式 (见下方) |
| `action_index` | `0` | 返回第几帧动作 |

### 启动

```bash
cd /home/ubuntu/mzy/RDT2

# 先 source workspace
source ~/ros2_ws/install/setup.bash

# 启动服务
python pika_test_scripts/ros2_inference_server.py --ros-args \
    -p fm_checkpoint:=outputs/rdt2-fm-pika-bottle-fm/checkpoint-10000
```

### 客户端示例

```python
import rclpy
from rclpy.node import Node
from rdt2_interfaces.srv import RdtUmi
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class RDT2Client(Node):
    def __init__(self):
        super().__init__('rdt2_client')
        self.client = self.create_client(RdtUmi, '/rdt2/inference')
        self.bridge = CvBridge()
        
        # 等待服务可用
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for service...')
        self.get_logger().info('Service ready!')
    
    def call_inference(self, img_left: np.ndarray, img_right: np.ndarray):
        """
        调用推理服务
        
        Args:
            img_left: 左相机图像 (H, W, 3) RGB
            img_right: 右相机图像 (H, W, 3) RGB
        
        Returns:
            action: list, 动作数据
        """
        request = RdtUmi.Request()
        request.umi_left_image = self.bridge.cv2_to_imgmsg(img_left, encoding='rgb8')
        request.umi_right_image = self.bridge.cv2_to_imgmsg(img_right, encoding='rgb8')
        
        future = self.client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        response = future.result()
        return list(response.action_full.data)

def main():
    rclpy.init()
    client = RDT2Client()
    
    # 示例：创建测试图像
    img_left = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    img_right = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    action = client.call_inference(img_left, img_right)
    print(f"Action: {action}")
    print(f"Action dim: {len(action)}")
    
    client.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
```

---

## 输出格式

### `use_rpy=true` (默认)

每臂 7D：`[x, y, z, roll, pitch, yaw, gripper]`

```
action_full = [
    # Right arm (7D)
    right_x, right_y, right_z,           # 位置 (m)
    right_roll, right_pitch, right_yaw,  # 欧拉角 (rad)
    right_gripper,                        # 夹爪宽度
    # Left arm (7D)
    left_x, left_y, left_z,
    left_roll, left_pitch, left_yaw,
    left_gripper
]
# 总共 14D
```

### `use_rpy=false`

每臂 10D：`[x, y, z, rot6d(6), gripper]`

```
action_full = [
    # Right arm (10D)
    right_x, right_y, right_z,           # 位置 (m)
    right_rot6d[0:6],                     # 6D 旋转表示
    right_gripper,                        # 夹爪宽度
    # Left arm (10D)
    left_x, left_y, left_z,
    left_rot6d[0:6],
    left_gripper
]
# 总共 20D
```

---

## 性能指标

在 RTX 4090 上测试：

| 指标 | 数值 |
|------|------|
| 推理延迟 | ~130 ms |
| 推理频率 | ~7.6 Hz |
| 预测帧数 | 24 帧 |
| 有效控制率 | 182 actions/s (使用全部预测) |
| 显存占用 | ~16.4 GB |

---

## 常见问题

### 1. ImportError: No module named 'rdt2_interfaces'

确保已编译并 source workspace：
```bash
source ~/ros2_ws/install/setup.bash
```

### 2. 模型加载慢

首次加载需要下载 VLM 权重，后续会使用缓存。

### 3. CUDA out of memory

- 减小 `image_size` 参数
- 使用 `device:=cuda:1` 切换 GPU

### 4. 话题没有数据

检查相机话题名称是否匹配：
```bash
ros2 topic list
ros2 topic echo /umi/left/image_raw --no-arr
```

---

## 文件结构

```
pika_test_scripts/
├── README_ROS2.md              # 本文档
├── ros2_inference_node.py      # Topic 模式节点
├── ros2_inference_server.py    # Service 模式节点
├── benchmark_fm_inference.py   # 推理性能测试
├── inference_offline_fm.py     # 离线推理测试
└── inference_offline_fm_full.py # 全 episode 离线推理
```
