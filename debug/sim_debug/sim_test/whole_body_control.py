from decimal import Clamped
import os
import sys
import numpy as np
import qpsolvers

import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer
import time
import math
try:
    from loop_rate_limiters import RateLimiter
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Examples use loop rate limiters, "
        "try `[conda|pip] install loop-rate-limiters`"
    ) from exc

import pinocchio as pin
import meshcat_shapes  
import pink 

from get_delta_ee import get_delta_ee, read_gt_data, get_full_rdt2_data

def load_r1_pro_urdf(urdf_path: str):
    """加载 R1 Pro 机器人的 URDF 文件
    
    Args:
        urdf_path: URDF 文件的路径
        
    Returns:
        RobotWrapper 对象
    """
    package_dirs = [os.path.dirname(urdf_path)]
    
    robot = pin.RobotWrapper.BuildFromURDF(
        filename=urdf_path,
        package_dirs=package_dirs,
        root_joint=None,  # 固定基座
    )
    
    return robot

def degree_to_radian(joints):
    converted = [round(math.radians(j), 2) for j in joints]
    return converted


def create_initial_configuration(robot, cur_torse_q, cur_left_arm_q, cur_right_arm_q):
    """创建初始关节配置
    
    Args:
        robot: RobotWrapper 对象
        
    Returns:
        初始配置向量
    """
    # 使用 custom_configuration_vector 设置初始关节角度（弧度）
    # 注意：只设置存在的关节，如果关节不存在会跳过
    
    config_dict = {}
    
    # 设置腰部初始位置（可以根据需要调整）

    torso_joints = {
        "torso_joint1": cur_torse_q[0],
        "torso_joint2": cur_torse_q[1],
        "torso_joint3": cur_torse_q[2],
        "torso_joint4": cur_torse_q[3],
    }
    for joint_name, value in torso_joints.items():
        if robot.model.existJointName(joint_name):
            config_dict[joint_name] = value

    # 设置左臂初始位置（弧度）
    # 这里设置一个合理的初始姿态，避免奇异配置
    left_arm_config = {
        "left_arm_joint1": cur_left_arm_q[0],
        "left_arm_joint2": cur_left_arm_q[1],
        "left_arm_joint3": cur_left_arm_q[2],
        "left_arm_joint4": cur_left_arm_q[3],
        "left_arm_joint5": cur_left_arm_q[4],
        "left_arm_joint6": cur_left_arm_q[5],
        "left_arm_joint7": cur_left_arm_q[6],
    }
    for joint_name, value in left_arm_config.items():
        if robot.model.existJointName(joint_name):
            config_dict[joint_name] = value
    
    right_arm_config = {
        "right_arm_joint1": cur_right_arm_q[0],
        "right_arm_joint2": cur_right_arm_q[1],
        "right_arm_joint3": cur_right_arm_q[2],
        "right_arm_joint4": cur_right_arm_q[3],
        "right_arm_joint5": cur_right_arm_q[4],
        "right_arm_joint6": cur_right_arm_q[5],
        "right_arm_joint7": cur_right_arm_q[6],
    }
    

    for joint_name, value in right_arm_config.items():
        if robot.model.existJointName(joint_name):
            config_dict[joint_name] = value
    
    # 使用 custom_configuration_vector 创建配置向量
    # 注意：函数内部使用 robot.model，所以可以传入 RobotWrapper
    try:
        q = custom_configuration_vector(robot, **config_dict)
    except Exception as e:
        # 如果出错，使用中性配置
        print(f"警告：使用 custom_configuration_vector 时出错: {e}")
        print("使用中性配置作为初始配置")
        q = pin.neutral(robot.model)
    
    return q

class WholeBodyControl:
    def __init__(self, cur_torse_q, cur_left_arm_q, cur_right_arm_q):
        urdf_path = '/home/ubuntu/mzy/RDT2/debug/sim_debug/sim_test/r1pro_umi/urdf/r1pro_original_umi.urdf'
        self.robot = load_r1_pro_urdf(urdf_path)
        self.tasks = []
        self.solver = qpsolvers.available_solvers[2]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"
        self.rate = RateLimiter(frequency=30.0, warn=False)
        self.dt = self.rate.period
        self.t = 0.0
        left_ee_frame = "left_tcp_link"
        right_ee_frame = "right_tcp_link"
        arm_cost = 1.0 # 1: 完全使用arm移动
        torsor_cost = 0.1
        lm_damping = 1.0
        self.visulization = True
        self.left_ee_task = FrameTask(
            left_ee_frame,
            position_cost=arm_cost,      # [cost] / [m]
            orientation_cost=arm_cost,    # [cost] / [rad]
            lm_damping=lm_damping         # Levenberg-Marquardt 阻尼
        )
        # 右臂末端执行器任务
        self.right_ee_task = FrameTask(
            right_ee_frame,
            position_cost=arm_cost,       # [cost] / [m]
            orientation_cost=arm_cost,    # [cost] / [rad]
            lm_damping=lm_damping,         # Levenberg-Marquardt 阻尼
        )
        self.torsor_1_task = FrameTask(
            "torso_link1",
            position_cost=torsor_cost,      # [cost] / [m]
            orientation_cost=torsor_cost,    # [cost] / [rad]
            lm_damping=lm_damping,         # Levenberg-Marquardt 阻尼
        )        
        self.torsor_2_task = FrameTask(
            "torso_link2",
            position_cost=torsor_cost,      # [cost] / [m]
            orientation_cost=torsor_cost,    # [cost] / [rad]
            lm_damping=lm_damping,         # Levenberg-Marquardt 阻尼
        )
        self.torsor_3_task = FrameTask(
            "torso_link3",
            position_cost=torsor_cost,      # [cost] / [m]
            orientation_cost=torsor_cost,    # [cost] / [rad]
            lm_damping=lm_damping,         # Levenberg-Marquardt 阻尼
        )
        self.torsor_4_task = FrameTask(
            "torso_link4",
            position_cost=torsor_cost,      # [cost] / [m]
            orientation_cost=torsor_cost,    # [cost] / [rad]
            lm_damping=lm_damping,         # Levenberg-Marquardt 阻尼
        )
        # 姿态任务（用于正则化，避免奇异配置）
        self.posture_task = PostureTask(
            cost=1e-2,  # [cost] / [rad]
        )
        self.tasks = [self.left_ee_task, self.right_ee_task, self.torsor_1_task, self.torsor_2_task, self.torsor_3_task, self.torsor_4_task, self.posture_task]
        self.viz = start_meshcat_visualizer(self.robot)
        self.viewer = self.viz.viewer
        
        q0 = create_initial_configuration(self.robot, cur_torse_q, cur_left_arm_q, cur_right_arm_q)
        self.configuration = pink.Configuration(self.robot.model, self.robot.data, q0)

        # 设置可视化 frame（如果 meshcat_shapes 可用）
        if meshcat_shapes and self.visulization:
            meshcat_shapes.frame(self.viewer["left_ee_target"], opacity=0.5)
            meshcat_shapes.frame(self.viewer["left_ee"], opacity=1.0)
            meshcat_shapes.frame(self.viewer["right_ee_target"], opacity=0.5)
            meshcat_shapes.frame(self.viewer["right_ee"], opacity=1.0)
            self.viz.display(self.configuration.q) 
            

         # 从当前配置设置任务目标
        for task in self.tasks:
            if isinstance(task, FrameTask):
                task.set_target_from_configuration(self.configuration)
            elif isinstance(task, PostureTask):
                task.set_target(self.configuration.q)

                  
    def solve(self, current_torse_q: np.ndarray, current_left_arm_q: np.ndarray, current_right_arm_q: np.ndarray, delta_left_ee, delta_right_ee):
    # INSERT_YOUR_CODE
        """
        求解全身逆运动学，输出新的torse/arms关节角度
        Args:
            current_torse_q: np.ndarray, shape=(4,)
            current_left_arm_q: np.ndarray, shape=(7,)
            current_right_arm_q: np.ndarray, shape=(7,)
            target_left_ee: dict or tuple (xyz, rotation) 或 np.ndarray(len=7)
            target_right_ee: dict or tuple (xyz, rotation) 或 np.ndarray(len=7)
        Returns:
            torse_q: np.ndarray, shape=(4,)
            left_arm_q: np.ndarray, shape=(7,)
            right_arm_q: np.ndarray, shape=(7,)
        """

        # 拼接机器人全身当前配置q
        # 这里假设q顺序为 [torso(4), left_arm(7), right_arm(7)]
        
        # INSERT_YOUR_CODE
        # 将输入的 current_torse_q, current_left_arm_q, current_right_arm_q 设置到当前 configuration.q
        # 假定 configuration.q 顺序: 前8位保留不变, 8~11为torso, 12~18为left_arm, 19~25为right_arm（注意左闭右开）

        # 由于 self.configuration.q 可能是只读数组（见 file_context_0），这里做法是创建新q然后赋值
        # import pdb; pdb.set_trace()
        # self.viz.display(self.configuration.q) 
        q_full = self.configuration.q.copy()
        # import pdb; pdb.set_trace()
        
        q_full[9:13] = current_torse_q
        q_full[13:20] = current_left_arm_q
        q_full[22:29] = current_right_arm_q
        self.configuration.q = q_full

        delta_l_trans = delta_left_ee[0:3]
        delta_r_trans = delta_right_ee[0:3]
        
        left_ee_target = self.left_ee_task.transform_target_to_world
        right_ee_target = self.right_ee_task.transform_target_to_world
        
        left_R = left_ee_target.rotation
        right_R = right_ee_target.rotation
            
        delta_l_rot = delta_left_ee[3:6]
        delta_r_rot = delta_right_ee[3:6]

        # # 变换到world系增量：R_g2w @ delta_g
        world_delta_l = left_R  @ delta_l_trans
        world_delta_r = right_R  @ delta_r_trans
        
        left_ee_target.translation += world_delta_l
        right_ee_target.translation += world_delta_r


        left_rot_update = pin.utils.rpyToMatrix(*(delta_l_rot))
        # print("left_delta_rpy: ", left_delta_rpy)
        left_ee_target.rotation = left_ee_target.rotation  @ left_rot_update

        right_rot_update = pin.utils.rpyToMatrix(*(delta_r_rot))
        right_ee_target.rotation = right_ee_target.rotation  @ right_rot_update

        if meshcat_shapes and self.visulization:
            self.viewer["left_ee_target"].set_transform(left_ee_target.np)
            self.viewer["left_ee"].set_transform(
                self.configuration.get_transform_frame_to_world(
                    self.left_ee_task.frame
                ).np
            )
            self.viewer["right_ee_target"].set_transform(right_ee_target.np)
            self.viewer["right_ee"].set_transform(
                self.configuration.get_transform_frame_to_world(
                    self.right_ee_task.frame
                ).np
            )
        # import pdb; pdb.set_trace()           
        # 调用pink的solve_ik获得关节速度dq
        dq = pink.solve_ik(self.configuration, self.tasks, self.dt, solver=self.solver)

        # 更新关节配置
        self.configuration.integrate_inplace(dq, self.dt)
        self.viz.display(self.configuration.q)
        # 拆分torso/arms
        torse_q = self.configuration.q[9:13].copy()
        left_arm_q = self.configuration.q[13:20].copy()
        right_arm_q = self.configuration.q[22:29].copy()
        return torse_q, left_arm_q, right_arm_q


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Whole Body Control Replay')
    parser.add_argument('--mode', type=str, default='tar', choices=['tar', 'npz_gt', 'npz_pred'],
                        help='Data source mode: tar (from shard tar files), npz_gt (GT from npz), npz_pred (predictions from npz)')
    parser.add_argument('--tar-path', type=str, 
                        default='/home/ubuntu/mzy/RDT2/debug/sim_debug/sim_test/umi-shard-000000',
                        help='Path to extracted tar shard directory')
    parser.add_argument('--npz-path', type=str,
                        default='/home/ubuntu/mzy/RDT2/inference_outputs_fm_full/full_episode_episode0_20251201_161411.npz',
                        help='Path to npz file containing trajectory data')
    parser.add_argument('--episode', type=str, default=None,
                        help='Specific episode to replay (e.g., "episode0"). If not specified, replay all frames.')
    parser.add_argument('--max-frames', type=int, default=447,
                        help='Maximum number of frames to replay')
    parser.add_argument('--output-log', type=str, default=None,
                        help='Output log file path (optional)')
    parser.add_argument('--no-debug', action='store_true',
                        help='Skip the pdb breakpoint')
    args = parser.parse_args()
    
    # 初始化关节配置
    cur_torse_q = np.array([degree_to_radian([30.0]), degree_to_radian([-60.0]), degree_to_radian([-45.0]), degree_to_radian([0.0])])
    cur_left_arm_q = np.array([-0.3, 0.1, 0.0, -1.0, 0.0, 0.0, -0.2])
    cur_right_arm_q = np.array([-0.50, -0.01, 0.0, -0.81, 0.0, -0.1, 0.2])
    wbc = WholeBodyControl(cur_torse_q, cur_left_arm_q, cur_right_arm_q)
    
    if not args.no_debug:
        import pdb; pdb.set_trace()
    
    print(f"Running replay with mode: {args.mode}")
    
    if args.mode == 'tar':
        # ======================== TAR 数据回放 ========================
        # 从 shard tar 文件中读取 delta pose（相对于当前帧的未来第一帧变换）
        print(f"Loading data from tar shard: {args.tar_path}")
        
        # 如果指定了 episode，先扫描找到该 episode 的帧范围
        if args.episode:
            import json
            print(f"Filtering for episode: {args.episode}")
            episode_frames = []
            for i in range(10000):  # 假设最多 10000 帧
                meta_file = os.path.join(args.tar_path, f"{i}.meta.json")
                if not os.path.exists(meta_file):
                    break
                with open(meta_file) as f:
                    meta = json.load(f)
                if meta.get('episode') == args.episode:
                    episode_frames.append(i)
            if not episode_frames:
                print(f"Error: Episode '{args.episode}' not found in {args.tar_path}")
                exit(1)
            print(f"Found {len(episode_frames)} frames for {args.episode}: {min(episode_frames)}-{max(episode_frames)}")
            frame_indices = episode_frames[:args.max_frames]
        else:
            frame_indices = range(args.max_frames)
        
        for idx, i in enumerate(frame_indices):
            gt_data = read_gt_data(shard_idx=i, base_path=args.tar_path)
            if gt_data is None:
                print(f"Warning: No data at index {i}, stopping.")
                break
            delta_left_ee = gt_data["left_action"][0:6]
            delta_right_ee = gt_data["right_action"][0:6]
            
            if args.output_log:
                with open(args.output_log, "a") as f:
                    f.write(f"============================{i}==============================\n")
                    f.write(f"Left EE: {delta_left_ee}\n")
                    f.write(f"Right EE: {delta_right_ee}\n")
            
            cur_torse_q = cur_torse_q.reshape(4,)
            torse_q, left_arm_q, right_arm_q = wbc.solve(cur_torse_q, cur_left_arm_q, cur_right_arm_q, delta_left_ee, delta_right_ee)
            cur_torse_q = torse_q
            cur_left_arm_q = left_arm_q
            cur_right_arm_q = right_arm_q
            time.sleep(0.03)
    
    elif args.mode in ['npz_gt', 'npz_pred']:
        # ======================== NPZ 数据回放 ========================
        # 从 npz 文件中读取累积绝对轨迹，计算帧间 delta pose
        print(f"Loading data from npz: {args.npz_path}")
        full_rdt2_data = get_full_rdt2_data(base_path=args.npz_path)
        if full_rdt2_data is None:
            print("Error: Failed to load npz data")
            exit(1)
        
        # 根据模式选择 GT 或 Pred 数据
        if args.mode == 'npz_gt':
            left_actions = full_rdt2_data["gt_left_action"]
            right_actions = full_rdt2_data["gt_right_action"]
            print("Using GT trajectory from npz")
        else:  # npz_pred
            left_actions = full_rdt2_data["pred_left_action"]
            right_actions = full_rdt2_data["pred_right_action"]
            print("Using predicted trajectory from npz")
        
        max_frames = min(args.max_frames, len(left_actions))
        for i in range(max_frames):
            delta_left_ee = np.array(left_actions[i][0:6])
            delta_right_ee = np.array(right_actions[i][0:6])
            
            if args.output_log:
                with open(args.output_log, "a") as f:
                    f.write(f"============================{i}==============================\n")
                    f.write(f"Left EE: {delta_left_ee}\n")
                    f.write(f"Right EE: {delta_right_ee}\n")
            
            cur_torse_q = cur_torse_q.reshape(4,)
            torse_q, left_arm_q, right_arm_q = wbc.solve(cur_torse_q, cur_left_arm_q, cur_right_arm_q, delta_left_ee, delta_right_ee)
            cur_torse_q = torse_q
            cur_left_arm_q = left_arm_q
            cur_right_arm_q = right_arm_q
            time.sleep(0.03)
    
    print("Replay finished!")
