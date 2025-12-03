#!/usr/bin/env python
"""
Export delta 6D poses from episode 0.

This script reads the left and right hand end-effector 6D poses from episode 0
and computes delta poses relative to the first frame (starting from frame 2).

The 6D pose is assumed to be [x, y, z, roll, pitch, yaw] format.
Delta pose is computed as:
- Position: p_delta = p_current - p_first
- Rotation: Using proper SO(3) delta (R_delta = R_current @ R_first^-1)
  then convert back to euler angles.

Output: numpy array file with shape (num_frames-1, 2, 6) where:
- axis 0: frames (starting from frame 2)
- axis 1: 0 = left hand, 1 = right hand
- axis 2: 6D pose [dx, dy, dz, d_roll, d_pitch, d_yaw]
"""

import argparse
import numpy as np
import h5py
from scipy.spatial.transform import Rotation as R


def pose_to_matrix(pose):
    # xyz in meters, rpy in radians
    rot = R.from_euler('xyz', [pose['roll'], pose['pitch'], pose['yaw']])
    mat = np.eye(4)
    mat[:3, :3] = rot.as_matrix()
    mat[:3, 3] = [pose['x'], pose['y'], pose['z']]
    return mat

def matrix_to_pose(mat):
    xyz = mat[:3, 3]
    rot = R.from_matrix(mat[:3, :3])
    roll, pitch, yaw = rot.as_euler('xyz')
    return {
        'x': float(xyz[0]),
        'y': float(xyz[1]),
        'z': float(xyz[2]),
        'roll': float(roll),
        'pitch': float(pitch),
        'yaw': float(yaw),
    }

def compute_delta_pose(pose1, pose0):
    # pose0 as "gripper" frame, compute pose1 w.r.t. pose0
    T0 = pose_to_matrix(pose0)
    T1 = pose_to_matrix(pose1)
    T0_inv = np.linalg.inv(T0)
    delta_T = T0_inv @ T1
    return matrix_to_pose(delta_T)


def compute_delta_poses_batch(poses: dict) -> list:
    """
    Compute delta poses for a sequence of poses relative to last frame.
    
    Args:
        poses: Array of shape (num_frames, 6) with 6D poses
    
    Returns:
        Delta poses of shape (num_frames-1, 6) starting from frame 2
    """
    num_frames = poses.shape[0]
    
    delta_poses = []
    for i in range(1, num_frames):
        delta = compute_delta_pose(poses[i], poses[i - 1])
        delta_poses.append(delta)
    
    return delta_poses


def main():
    parser = argparse.ArgumentParser(
        description='Export delta 6D poses from a pika episode.'
    )
    parser.add_argument(
        '--episode_path',
        type=str,
        default='/home/ubuntu/mzy/RDT2/pika_raw_data/episode0/data.hdf5',
        help='Path to the episode HDF5 file'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/home/ubuntu/mzy/RDT2/pika_raw_data/episode0/delta_poses.npy',
        help='Output path for the numpy array file'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose information'
    )
    args = parser.parse_args()
    
    # Load poses from HDF5
    print(f"Loading poses from: {args.episode_path}")
    with h5py.File(args.episode_path, 'r') as f:
        poses_left = np.array(f['localization/pose/pika_l'])
        poses_right = np.array(f['localization/pose/pika_r'])
    
    print(f"Left hand poses shape: {poses_left.shape}")
    print(f"Right hand poses shape: {poses_right.shape}")
    
    if args.verbose:
        print(f"\nFirst frame left pose: {poses_left[0]}")
        print(f"First frame right pose: {poses_right[0]}")
        print(f"Second frame left pose: {poses_left[1]}")
        print(f"Second frame right pose: {poses_right[1]}")
    
    # Compute delta poses
    print("\nComputing delta poses relative to first frame...")
    delta_poses_left = compute_delta_poses_batch(poses_left)
    delta_poses_right = compute_delta_poses_batch(poses_right)
    
    print(f"Delta poses left shape: {delta_poses_left.shape}")
    print(f"Delta poses right shape: {delta_poses_right.shape}")
    
    # Stack left and right: shape (num_frames-1, 2, 6)
    # axis 1: 0 = left, 1 = right
    delta_poses = np.stack([delta_poses_left, delta_poses_right], axis=1)
    print(f"Combined delta poses shape: {delta_poses.shape}")
    
    if args.verbose:
        print(f"\nFirst delta pose (frame 2 - frame 1):")
        print(f"  Left:  {delta_poses[0, 0]}")
        print(f"  Right: {delta_poses[0, 1]}")
        print(f"\nLast delta pose (frame {delta_poses.shape[0]+1} - frame 1):")
        print(f"  Left:  {delta_poses[-1, 0]}")
        print(f"  Right: {delta_poses[-1, 1]}")
    
    # Save to numpy file
    np.save(args.output_path, delta_poses)
    print(f"\nSaved delta poses to: {args.output_path}")
    print(f"Array shape: {delta_poses.shape}")
    print("  - axis 0: frames (frame 2 to frame N, total {0} frames)".format(delta_poses.shape[0]))
    print("  - axis 1: 0 = left hand, 1 = right hand")
    print("  - axis 2: [dx, dy, dz, d_roll, d_pitch, d_yaw]")


if __name__ == '__main__':
    main()
