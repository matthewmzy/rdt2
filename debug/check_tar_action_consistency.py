#!/usr/bin/env python3
"""
Debug 脚本：检查 tar 数据集中 action[1:24] 的正确性

关键问题：
- action[0] 是 frame i+1 相对于 frame i 的变换
- action[t] 应该是 frame i+t+1 相对于 frame i 的变换

验证方法：
- 用 action[t] from frame i 累积
- 与 逐帧 action[0] 累积 比较
- 如果不一致，说明 tar 数据集的 action[1:24] 有问题
"""

import os
import sys
import io
import json
import tarfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def rot6d_to_mat(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = a1 / (np.linalg.norm(a1, axis=-1, keepdims=True) + 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=-1, keepdims=True) + 1e-12)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-1)


def mat_to_rot6d(mat):
    return np.concatenate([mat[..., :, 0], mat[..., :, 1]], axis=-1)


def pose9d_to_mat(pose9d):
    pos = pose9d[..., :3]
    rot6d = pose9d[..., 3:9]
    rotmat = rot6d_to_mat(rot6d)
    
    shape = pos.shape[:-1]
    mat = np.zeros(shape + (4, 4), dtype=np.float64)
    mat[..., :3, :3] = rotmat
    mat[..., :3, 3] = pos
    mat[..., 3, 3] = 1
    return mat


def mat_to_pose9d(mat):
    pos = mat[..., :3, 3]
    rotmat = mat[..., :3, :3]
    rot6d = mat_to_rot6d(rotmat)
    return np.concatenate([pos, rot6d], axis=-1)


def load_tar_actions(shard_dir, episode):
    actions = {}
    for shard_path in sorted(shard_dir.glob('shard-*.tar')):
        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()
            sample_indices = {int(m.name.split('.')[0]) for m in members if m.name.endswith('.meta.json')}
            
            for idx in sample_indices:
                try:
                    meta = json.load(tar.extractfile(f'{idx}.meta.json'))
                    if meta['episode'] != episode:
                        continue
                    action = np.load(io.BytesIO(tar.extractfile(f'{idx}.action.npy').read()))
                    actions[meta['frame']] = action
                except:
                    continue
    return actions


def main():
    print("=" * 80)
    print("检查 tar 数据集中 action[1:24] 的正确性")
    print("=" * 80)
    
    shard_dir = Path(PROJECT_ROOT / 'rdt2_pika_shards')
    actions = load_tar_actions(shard_dir, 'episode0')
    print(f"Loaded {len(actions)} frames")
    
    sorted_frames = sorted(actions.keys())
    
    # ================================================================
    # 检验1: action[t] from frame i 是否等于 逐帧 action[0] 累积
    # ================================================================
    print("\n" + "=" * 60)
    print("检验1: action[t] vs 逐帧 action[0] 累积")
    print("=" * 60)
    
    # 选择几个 frame 来测试
    test_frames = [0, 10, 50, 100]
    
    for base_frame in test_frames:
        if base_frame not in actions:
            continue
        
        base_action = actions[base_frame]  # (24, 20)
        
        print(f"\n从 frame {base_frame} 开始:")
        
        # 方法A: 直接用 action[t]
        # 方法B: 逐帧累积 action[0]
        
        errors_pos_r = []
        errors_pos_l = []
        
        # 初始化为单位矩阵（当前帧位姿）
        cum_mat_r = np.eye(4)
        cum_mat_l = np.eye(4)
        
        for t in range(24):
            target_frame = base_frame + t + 1
            
            if target_frame not in actions and t > 0:
                # 如果没有数据，用上一帧的 action[0] 近似
                continue
            
            # 方法A: 直接用 base_action[t]
            mat_A_r = pose9d_to_mat(base_action[t, :9])
            mat_A_l = pose9d_to_mat(base_action[t, 10:19])
            pos_A_r = mat_A_r[:3, 3]
            pos_A_l = mat_A_l[:3, 3]
            
            # 方法B: 累积 action[0]
            if t == 0:
                # 第一帧：直接用 base_action[0]
                cum_mat_r = pose9d_to_mat(base_action[0, :9])
                cum_mat_l = pose9d_to_mat(base_action[0, 10:19])
            else:
                # 后续帧：用 (base_frame + t) 的 action[0]
                prev_frame = base_frame + t
                if prev_frame in actions:
                    delta_r = pose9d_to_mat(actions[prev_frame][0, :9])
                    delta_l = pose9d_to_mat(actions[prev_frame][0, 10:19])
                    
                    # 这里是关键：怎么累积？
                    # action[0] from prev_frame 是 T_{prev_frame}^{-1} @ T_{target_frame}
                    # 所以 T_{target_frame} = T_{prev_frame} @ action[0]
                    # 但我们需要 T_{target_frame} 相对于 T_{base_frame}
                    
                    # 实际上，我们需要：
                    # T_{target} 相对于 T_{base} = T_{base}^{-1} @ T_{target}
                    # = T_{base}^{-1} @ T_{prev} @ action[0]_{prev}
                    # = (T_{base}^{-1} @ T_{prev}) @ action[0]_{prev}
                    # = cum_mat @ action[0]_{prev}
                    
                    # 不对！cum_mat 是 T_{prev} 相对于 T_{base}
                    # 而 action[0]_{prev} 是 T_{target} 相对于 T_{prev}
                    # 所以正确的累积应该是 cum_mat @ action[0]_{prev}
                    
                    # 但这与 action[t] 应该不同！
                    # 因为 action[t] = T_{base}^{-1} @ T_{target}
                    # 而 cum_mat @ action[0]_{prev} 
                    #   = (T_{base}^{-1} @ T_{prev}) @ (T_{prev}^{-1} @ T_{target})
                    #   = T_{base}^{-1} @ T_{target}
                    # 所以应该相等！
                    
                    cum_mat_r = cum_mat_r @ delta_r
                    cum_mat_l = cum_mat_l @ delta_l
            
            pos_B_r = cum_mat_r[:3, 3]
            pos_B_l = cum_mat_l[:3, 3]
            
            err_r = np.linalg.norm(pos_A_r - pos_B_r)
            err_l = np.linalg.norm(pos_A_l - pos_B_l)
            
            errors_pos_r.append(err_r)
            errors_pos_l.append(err_l)
            
            if t < 5 or err_r > 0.001:
                print(f"  t={t} (frame {target_frame}):")
                print(f"    action[{t}] pos_r: {pos_A_r}")
                print(f"    累积 pos_r:        {pos_B_r}")
                print(f"    误差: {err_r:.6f}")
        
        print(f"\n  位置误差统计:")
        print(f"    右臂: max={max(errors_pos_r):.6f}, mean={np.mean(errors_pos_r):.6f}")
        print(f"    左臂: max={max(errors_pos_l):.6f}, mean={np.mean(errors_pos_l):.6f}")
    
    # ================================================================
    # 检验2: 直接比较 action[t] from frame i 与 action[0] from frame i+t
    # 并分析差异的原因
    # ================================================================
    print("\n" + "=" * 60)
    print("检验2: action[t] from frame i vs action[0] from frame i+t")
    print("理论分析:")
    print("  action[t] from frame i = T_i^{-1} @ T_{i+t+1}")
    print("  action[0] from frame i+t = T_{i+t}^{-1} @ T_{i+t+1}")
    print("  它们描述的是同一个目标帧，但参考帧不同！")
    print("=" * 60)
    
    base_frame = 0
    base_action = actions[base_frame]
    
    print(f"\n从 frame {base_frame}:")
    for t in [0, 1, 5, 11, 23]:
        target_frame = base_frame + t + 1
        src_frame = base_frame + t
        
        if src_frame not in actions:
            continue
        
        # action[t] from base_frame
        action_t = base_action[t]  # T_{base}^{-1} @ T_{target}
        
        # action[0] from src_frame
        action_0 = actions[src_frame][0]  # T_{src}^{-1} @ T_{target}
        
        # 这两个不同是因为参考帧不同
        # 如果我们知道 T_{src} 相对于 T_{base} (即 base_action[t-1] when t>0 or identity when t=0)
        # 那么可以转换
        
        print(f"\n  t={t} (target frame {target_frame}):")
        print(f"    action[{t}] from frame {base_frame}: pos={action_t[:3]}")
        print(f"    action[0] from frame {src_frame}:    pos={action_0[:3]}")
        
        if t > 0:
            # 计算 T_{src} 相对于 T_{base}
            # 这是 base_action[t-1]（如果 t>=1）
            # 但这只给出 frame src-1 相对于 base 的位姿
            # 不对，应该是 sum(action[0]...action[t-1]) from base
            
            # 其实更简单：
            # action[t] = T_{base}^{-1} @ T_{target}
            # 而 T_{target} = T_{src} @ action[0]
            # 所以 action[t] = T_{base}^{-1} @ T_{src} @ action[0]
            # 如果我们定义 T_rel_src = T_{base}^{-1} @ T_{src}
            # 那么 action[t] = T_rel_src @ action[0]
            
            # T_rel_src 是 frame src 相对于 frame base 的位姿
            # 这不就是 base_action[t-1] 吗？
            # 因为 base_action[t-1] = T_{base}^{-1} @ T_{src}
            
            if t >= 1:
                T_rel_src = pose9d_to_mat(base_action[t-1, :9])
                action_0_mat = pose9d_to_mat(action_0[:9])
                
                # 计算 T_rel_src @ action_0 应该等于 action[t]
                expected_mat = T_rel_src @ action_0_mat
                expected_pos = expected_mat[:3, 3]
                
                print(f"    T_rel_src @ action[0]: pos={expected_pos}")
                print(f"    与 action[{t}] 的误差: {np.linalg.norm(expected_pos - action_t[:3]):.10f}")


if __name__ == '__main__':
    main()
