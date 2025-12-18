import os
import json
from turtle import left

# INSERT_YOUR_CODE
import numpy as np
import cv2
import math
from scipy.spatial.transform import Rotation as R


def rot6d_to_mat(d6: np.ndarray) -> np.ndarray:
    """将 6D 旋转表示转换为旋转矩阵 (3, 3)"""
    a1, a2 = d6[:3], d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-12)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-12)
    b3 = np.cross(b1, b2)
    out = np.stack([b1, b2, b3], axis=-1)  # (3, 3)
    return out


def mat_to_rot6d(mat: np.ndarray) -> np.ndarray:
    """将旋转矩阵转换为 6D 旋转表示"""
    col0 = mat[:, 0]
    col1 = mat[:, 1]
    return np.concatenate([col0, col1], axis=-1)


def pose10d_to_mat(pose10d: np.ndarray) -> np.ndarray:
    """
    将 10D 绝对位姿 [pos(3), rot6d(6), gripper(1)] 转换为 4x4 变换矩阵
    """
    pos = pose10d[:3]
    rot6d = pose10d[3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    mat = np.eye(4, dtype=pose10d.dtype)
    mat[:3, :3] = rotmat
    mat[:3, 3] = pos
    return mat


def mat_to_pose10d(mat: np.ndarray, gripper: float) -> np.ndarray:
    """
    将 4x4 变换矩阵转换回 10D 位姿格式
    """
    pos = mat[:3, 3]
    rotmat = mat[:3, :3]
    rot6d = mat_to_rot6d(rotmat)
    return np.concatenate([pos, rot6d, [gripper]], axis=-1)


def compute_relative_pose(pose_prev_mat: np.ndarray, pose_curr_mat: np.ndarray) -> np.ndarray:
    """
    计算从 prev 到 curr 的相对变换（在 prev 坐标系下）
    
    T_rel = T_prev^{-1} @ T_curr
    
    这与 convert_pika_to_rdt2.py 中的定义一致
    """
    rel_mat = np.linalg.inv(pose_prev_mat) @ pose_curr_mat
    return rel_mat


def relative_mat_to_rpy(rel_mat: np.ndarray) -> np.ndarray:
    """
    将相对变换矩阵转换为 [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
    
    这是在前一帧 TCP 坐标系下的表示
    """
    delta_pos = rel_mat[:3, 3]
    delta_rotmat = rel_mat[:3, :3]
    delta_rpy = R.from_matrix(delta_rotmat).as_euler('xyz')
    return np.concatenate([delta_pos, delta_rpy])


def get_delta_ee(filepath='delta_poses_new.npy'):
    """
    读取 delta_pose.npy 文件并返回 numpy 数组
    """
    delta_pose = np.load(filepath, allow_pickle=True)
    return delta_pose

def get_joint_pose(filepath='arm_joint.npy'):
    """
    读取 delta_pose.npy 文件并返回 numpy 数组
    """
    joint_pose = np.load(filepath, allow_pickle=True)
    return joint_pose
# INSERT_YOUR_CODE
def rot6d_to_euler(rot6d: np.ndarray) -> np.ndarray:
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

def convert_action_to_rpy(action_10d: np.ndarray) -> np.ndarray:
    """将 10D 动作转换为 7D (xyz + rpy + gripper)"""
    pos = action_10d[:3]
    rot6d = action_10d[3:9]
    gripper = action_10d[9:10]
    rpy = rot6d_to_euler(rot6d)
    return np.concatenate([pos, rpy, gripper])


def degree_to_radian(joints):
    """度转弧度，并保留两位小数。"""
    return [round(math.radians(j), 2) for j in joints]


def get_full_rdt2_data(base_path = '/home/ubuntu/robot/dataset/full_episode_episode0_20251201_161411.npz'):
    """
    读取 full rdt2 数据，返回解压后的内容。
    
    重要：npz 文件中的 gt_trajectory 和 pred_trajectory 是**绝对位姿序列**
    （从单位矩阵原点开始累积得到的），格式为 (N, 20)，其中每行是：
    [right_pos(3), right_rot6d(6), right_gripper(1), left_pos(3), left_rot6d(6), left_gripper(1)]
    
    要得到每一帧相对于上一帧的 delta pose（在上一帧 TCP 坐标系下），需要：
    T_delta = T_{i-1}^{-1} @ T_i
    
    这与 convert_pika_to_rdt2.py 中 compute_relative_action 的定义一致。
    
    Args:
        base_path: str, 文件路径
    Returns:
        dict (key为变量名，value为list)
    """
    if not os.path.exists(base_path):
        print(f"文件不存在: {base_path}")
        return None
    data = np.load(base_path, allow_pickle=True)
    result = dict(data)
    pred_trajectory = result['pred_trajectory']  # (N, 20) 绝对位姿
    gt_trajectory = result['gt_trajectory']       # (N, 20) 绝对位姿
    
    pre_right_action = []
    pre_left_action = []
    gt_right_action = []
    gt_left_action = []
    
    # 第一帧：相对于原点（单位矩阵）的变换
    # 由于轨迹从单位矩阵开始累积，第一帧的绝对位姿就是相对于原点的变换
    pred_right_mat_0 = pose10d_to_mat(pred_trajectory[0, 0:10])
    pred_left_mat_0 = pose10d_to_mat(pred_trajectory[0, 10:20])
    gt_right_mat_0 = pose10d_to_mat(gt_trajectory[0, 0:10])
    gt_left_mat_0 = pose10d_to_mat(gt_trajectory[0, 10:20])
    
    # 第一帧的 delta = 从原点到第一帧的变换
    pre_right_action.append(relative_mat_to_rpy(pred_right_mat_0).tolist())
    pre_left_action.append(relative_mat_to_rpy(pred_left_mat_0).tolist())
    gt_right_action.append(relative_mat_to_rpy(gt_right_mat_0).tolist())
    gt_left_action.append(relative_mat_to_rpy(gt_left_mat_0).tolist())
    
    for i in range(1, len(pred_trajectory)):
        # 计算前一帧和当前帧的 4x4 变换矩阵
        pred_right_mat_prev = pose10d_to_mat(pred_trajectory[i-1, 0:10])
        pred_right_mat_curr = pose10d_to_mat(pred_trajectory[i, 0:10])
        pred_left_mat_prev = pose10d_to_mat(pred_trajectory[i-1, 10:20])
        pred_left_mat_curr = pose10d_to_mat(pred_trajectory[i, 10:20])
        
        gt_right_mat_prev = pose10d_to_mat(gt_trajectory[i-1, 0:10])
        gt_right_mat_curr = pose10d_to_mat(gt_trajectory[i, 0:10])
        gt_left_mat_prev = pose10d_to_mat(gt_trajectory[i-1, 10:20])
        gt_left_mat_curr = pose10d_to_mat(gt_trajectory[i, 10:20])
        
        # 计算相对变换：T_rel = T_prev^{-1} @ T_curr
        # 这是在前一帧 TCP 坐标系下的相对变换
        pred_right_rel = compute_relative_pose(pred_right_mat_prev, pred_right_mat_curr)
        pred_left_rel = compute_relative_pose(pred_left_mat_prev, pred_left_mat_curr)
        gt_right_rel = compute_relative_pose(gt_right_mat_prev, gt_right_mat_curr)
        gt_left_rel = compute_relative_pose(gt_left_mat_prev, gt_left_mat_curr)
        
        # 转换为 [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw]
        pre_right_action.append(relative_mat_to_rpy(pred_right_rel).tolist())
        pre_left_action.append(relative_mat_to_rpy(pred_left_rel).tolist())
        gt_right_action.append(relative_mat_to_rpy(gt_right_rel).tolist())
        gt_left_action.append(relative_mat_to_rpy(gt_left_rel).tolist())

    result = {
        "pred_left_action": pre_left_action,
        "pred_right_action": pre_right_action,
        "gt_left_action": gt_left_action,
        "gt_right_action": gt_right_action,
    }

    return result


def read_gt_data(shard_idx=0, base_path='/home/ubuntu/robot/dataset/umi-shard-000000'):
    """
    按顺序读取路径下编号为shard_idx的数据文件（如 0.npz），返回解压后的内容。
    若不存在则返回None。

    Args:
        shard_idx: int, shard编号，对应 0.npz, 1.npz, ...
        base_path: str, 文件夹路径，默认 umi-shard-000000 目录

    Returns:
        dict 或 None (如果文件不存在)
    """
    action_file = os.path.join(base_path, f"{shard_idx}.action.npy")
    image_file = os.path.join(base_path, f"{shard_idx}.image.jpg")
    if not os.path.exists(action_file) or not os.path.exists(image_file):
        print(f"文件不存在: {action_file} 或 {image_file}")
        return None
    # 读取动作
    image = cv2.imread(image_file)

    _, w, _ = image.shape
    mid = w // 2
    left_image = image[:, :mid, :].copy()
    right_image = image[:, mid:, :].copy()
    
    # bgr转化为rgb
    left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
    right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
    
    action = np.load(action_file, allow_pickle=True)

    right_action = convert_action_to_rpy(action[0, 0:10])
    left_action = convert_action_to_rpy(action[0,10:20])
    # import pdb; pdb.set_trace()
    result = {
        "left_action": left_action,
        "right_action": right_action,
        "left_image": left_image,
        "right_image": right_image,
    }
    return result


if __name__ == '__main__':
    # delta_pose = get_delta_ee()
    # import pdb; pdb.set_trace()
    # left_delta_ee = delta_pose[:, 0, :]
    # right_delta_ee = delta_pose[:, 1, :]
    # import pdb; pdb.set_trace()
    # # INSERT_YOUR_CODE
    # # 找到left_delta_ee最大值及其索引
    # # np.random.shuffle(left_delta_ee)
    # left_max_val = np.max(left_delta_ee)
    # np.rad2deg(left_max_val)
    # left_max_idx = np.unravel_index(np.argmax(left_delta_ee), left_delta_ee.shape)
    # print(f"left_delta_ee max value: {left_max_val}, index: {left_max_idx}")

    # # 找到right_delta_ee最大值及其索引
    # right_max_val = np.max(right_delta_ee)
    # right_max_idx = np.unravel_index(np.argmax(right_delta_ee), right_delta_ee.shape)
    # print(f"right_delta_ee max value: {right_max_val}, index: {right_max_idx}")
    # print(left_delta_ee.shape)
    # print(right_delta_ee.shape)
    # print(delta_pose.shape)
    # joint_pose = get_joint_pose()
    # import pdb; pdb.set_trace()
    # left_arm_joint = joint_pose[0]
    # right_arm_joint = joint_pose[1]
    # import pdb; pdb.set_trace()
    # read_gt_data(shard_idx=0)
    data = get_full_rdt2_data()
    gt_data_10 = read_gt_data(shard_idx = 10)
    gt_data_1 = read_gt_data(shard_idx = 1)
    gt_data_0 = read_gt_data(shard_idx = 0)
    gt_data_23 = read_gt_data(shard_idx = 23)
    gt_data_24 = read_gt_data(shard_idx = 24)
    import pdb; pdb.set_trace()
    print(np.array(data["gt_left_action"][1]) - np.array(data["gt_left_action"][0]))
    print("第一帧：\n")
    print(data["gt_left_action"][0])
    print(data["pred_left_action"][0][0:6])
    print(gt_data_0["left_action"])
    print(gt_data_0["left_action"][0:6] - data["gt_left_action"][0])
    
    print("第二帧：\n")
    print(np.array(data["gt_left_action"][1]) - np.array(data["gt_left_action"][0]))
    print(data["gt_left_action"][1])
    print(data["pred_left_action"][1][0:6])
    print(gt_data_1["left_action"])
    print(gt_data_1["left_action"][0:6] - data["gt_left_action"][1])
    
    print("第十一帧：\n")
    print(np.array(data["gt_left_action"][10]) - np.array(data["gt_left_action"][9]))
    print(data["gt_left_action"][10])
    print(data["pred_left_action"][10][0:6])
    print(gt_data_10["left_action"])
    print(gt_data_10["left_action"][0:6] - data["gt_left_action"][10])
    print("第二十三帧：\n")
    print(np.array(data["gt_left_action"][23]) - np.array(data["gt_left_action"][22]))
    print(data["gt_left_action"][23])
    print(data["pred_left_action"][23][0:6])
    print(gt_data_23["left_action"])
    print(gt_data_23["left_action"][0:6] - data["gt_left_action"][23])
    print("第二十四帧：\n")
    print(np.array(data["gt_left_action"][24]) - np.array(data["gt_left_action"][23]))
    print(data["gt_left_action"][24])
    print(data["pred_left_action"][24][0:6])
    print(gt_data_24["left_action"])
    print(gt_data_24["left_action"][0:6] - data["gt_left_action"][24])