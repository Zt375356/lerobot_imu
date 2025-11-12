#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Adapter class to use hardcoded kinematics (lerobot_Kinematics) with the existing processor pipeline.

This adapter wraps the hardcoded kinematics implementation from lerobot_kinematics
to provide a RobotKinematics-like interface for use with existing processor steps.
"""

import logging
from typing import Any

import numpy as np

try:
    from lerobot_kinematics.lerobot.lerobot_Kinematics import (
        get_robot,
        lerobot_FK,
        lerobot_IK,
    )
except ImportError:
    raise ImportError(
        "lerobot_kinematics is required for hardcoded kinematics. "
        "Please install it: pip install -e lerobot-kinematics"
    )

from scipy.spatial.transform import Rotation as R

logger = logging.getLogger(__name__)


class HardcodedKinematicsAdapter:
    """
    Adapter class that provides a RobotKinematics-like interface using hardcoded kinematics.
    
    This class wraps the lerobot_Kinematics functions to work with the existing processor
    pipeline that expects a RobotKinematics object.
    """
    
    def __init__(
        self,
        robot_type: str,
        joint_names: list[str],
    ):
        """
        Initialize the hardcoded kinematics adapter.
        
        Args:
            robot_type: Type of robot ("so100", "so100_plus", "so101")
            joint_names: List of joint names (excluding gripper) in order
        """
        self.robot_type = robot_type
        self.joint_names = [name for name in joint_names if name != "gripper"]
        
        # Map robot type to lerobot_kinematics robot type
        robot_type_map = {
            "so100_follower": "so100",
            "so100_plus_follower": "so100_plus",
            "so101_follower": "so101",
        }
        
        lerobot_robot_type = robot_type_map.get(robot_type, robot_type)
        
        # Get the hardcoded robot model
        self.robot_model = get_robot(lerobot_robot_type)
        if self.robot_model is None:
            raise ValueError(
                f"Unsupported robot type '{robot_type}'. "
                f"Supported types: {list(robot_type_map.keys())}"
            )
        
        # Verify joint count matches
        expected_joints = len(self.robot_model.qlim[0])
        if len(self.joint_names) != expected_joints:
            raise ValueError(
                f"Joint count mismatch: robot model expects {expected_joints} joints, "
                f"but got {len(self.joint_names)} joints: {self.joint_names}. "
                f"For SO100Plus, expected joints are typically: "
                f"shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, shoulder_roll, wrist_roll"
            )
        
        # Map joint names to indices for correct ordering
        # The hardcoded IK model expects joints in a specific order based on the robot model definition
        # From create_so100_plus(): joint 1 (Rz), joint 2 (Ry), joint 3 (Ry), joint 4 (Ry flip), joint 5 (Rz flip), joint 6 (Rx)
        # Robot order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, shoulder_roll
        # Model order based on ET chain: shoulder_pan (Rz), shoulder_lift (Ry), elbow_flex (Ry), 
        #                                 wrist_flex (Ry flip), wrist_roll (Rz flip), shoulder_roll (Rx)
        joint_order_map = {
            "so100_plus": ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "shoulder_roll"],
            "so100": ["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            "so101": ["shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        }
        
        # Get expected order for this robot type
        expected_order = joint_order_map.get(lerobot_robot_type, None)
        if expected_order:
            # Create mapping from joint name to index in expected order
            self.joint_index_map = {}
            for i, joint_name in enumerate(expected_order):
                if joint_name in self.joint_names:
                    self.joint_index_map[joint_name] = i
            # Verify all joints are mapped
            if len(self.joint_index_map) != len(self.joint_names):
                logger.warning(
                    f"Some joints from robot ({self.joint_names}) are not in expected order ({expected_order}). "
                    "Using provided order as-is."
                )
                self.joint_index_map = {name: i for i, name in enumerate(self.joint_names)}
        else:
            # No specific order known, use as-is
            self.joint_index_map = {name: i for i, name in enumerate(self.joint_names)}
        
        logger.info(
            f"Initialized hardcoded kinematics adapter for '{robot_type}' "
            f"with {len(self.joint_names)} joints: {self.joint_names}"
        )
    
    def _reorder_joints_to_model_order(self, joint_pos_deg: np.ndarray, joint_names: list[str]) -> np.ndarray:
        """Reorder joints from robot order to model order."""
        if len(joint_pos_deg) != len(joint_names):
            raise ValueError(
                f"Joint positions length ({len(joint_pos_deg)}) doesn't match joint names length ({len(joint_names)})"
            )
        
        # Create ordered array based on joint_index_map
        ordered_positions = np.zeros(len(self.joint_names), dtype=float)
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_index_map:
                model_idx = self.joint_index_map[joint_name]
                ordered_positions[model_idx] = joint_pos_deg[i]
            else:
                logger.warning(f"Joint {joint_name} not found in joint_index_map, skipping")
        
        return ordered_positions
    
    def _reorder_joints_from_model_order(self, joint_pos_rad: np.ndarray, joint_names: list[str]) -> np.ndarray:
        """
        Reorder joints from model order back to robot order.
        
        作用说明：
        1. 硬编码的 IK 模型（lerobot_IK）返回的关节顺序是按照模型定义的顺序
        2. 但实际机器人可能需要不同的关节顺序
        3. 这个函数将模型顺序转换回机器人期望的顺序
        
        例如：
        - 模型顺序（IK 返回）：[shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, shoulder_roll]
        - 机器人顺序（需要）：   [shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, shoulder_roll]
        - 如果顺序相同，则直接映射；如果不同，则重新排序
        
        Args:
            joint_pos_rad: 关节位置（弧度），按照模型顺序
            joint_names: 机器人期望的关节名称列表（按照机器人顺序）
            
        Returns:
            关节位置（弧度），按照机器人顺序
        """
        if len(joint_pos_rad) != len(self.joint_names):
            raise ValueError(
                f"Joint positions length ({len(joint_pos_rad)}) doesn't match model joints length ({len(self.joint_names)})"
            )
        
        # Create array in robot order
        robot_positions = np.zeros(len(joint_names), dtype=float)
        for i, joint_name in enumerate(joint_names):
            if joint_name in self.joint_index_map:
                # 从 joint_index_map 获取该关节在模型中的索引位置
                model_idx = self.joint_index_map[joint_name]
                # 将模型顺序中的值放到机器人顺序的对应位置
                robot_positions[i] = joint_pos_rad[model_idx]
        
        return robot_positions
    
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """
        Compute forward kinematics for given joint configuration.
        
        Args:
            joint_pos_deg: Joint positions in degrees (numpy array), in robot's joint order
            
        Returns:
            4x4 transformation matrix of the end-effector pose
        """
        # Reorder joints to model order (if needed)
        # joint_pos_deg is expected to be in the order of self.joint_names
        joint_pos_ordered = joint_pos_deg[:len(self.joint_names)]
        
        # Convert degrees to radians
        joint_pos_rad = np.deg2rad(joint_pos_ordered)
        
        # Use lerobot_FK which expects radians and returns [x, y, z, gamma, beta, alpha]
        # where gamma, beta, alpha are Euler angles (roll, pitch, yaw in ZYX order)
        ee_pose = lerobot_FK(joint_pos_rad, robot=self.robot_model)
        
        # Convert to 4x4 transformation matrix
        x, y, z, gamma, beta, alpha = ee_pose
        
        # Convert Euler angles (ZYX) to rotation matrix
        r = R.from_euler('zyx', [gamma, beta, alpha], degrees=False)
        R_mat = r.as_matrix()
        
        # Build 4x4 transformation matrix
        T = np.eye(4, dtype=float)
        T[:3, :3] = R_mat
        T[:3, 3] = [x, y, z]
        
        return T
    
    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """
        Compute inverse kinematics using hardcoded solver.
        
        Args:
            current_joint_pos: Current joint positions in degrees (used as initial guess), in robot's joint order
            desired_ee_pose: Target end-effector pose as a 4x4 transformation matrix
            position_weight: Not used (kept for compatibility)
            orientation_weight: Not used (kept for compatibility)
            
        Returns:
            Joint positions in degrees that achieve the desired end-effector pose, in robot's joint order
        """
        # Get current joint positions (excluding gripper if present)
        current_joint_deg = current_joint_pos[:len(self.joint_names)]
        
        # Convert current joint positions from degrees to radians
        current_joint_rad = np.deg2rad(current_joint_deg)
        
        # Extract position and rotation from 4x4 matrix
        pos = desired_ee_pose[:3, 3]
        R_mat = desired_ee_pose[:3, :3]
        
        # Convert rotation matrix to Euler angles (ZYX order: roll, pitch, yaw)
        r = R.from_matrix(R_mat)
        euler_angles = r.as_euler('zyx', degrees=False)
        gamma, beta, alpha = euler_angles  # roll, pitch, yaw
        
        # Build target pose in format expected by lerobot_IK: [x, y, z, roll, pitch, yaw]
        target_pose = np.array([
            pos[0], pos[1], pos[2],  # x, y, z
            gamma, beta, alpha        # roll, pitch, yaw (ZYX Euler angles)
        ])
        
        # Solve IK (lerobot_IK expects radians and returns radians in model order)
        q_target_rad, success = lerobot_IK(current_joint_rad, target_pose, robot=self.robot_model)
        
        if not success:
            # If IK fails, return current position (or handle error as needed)
            logger.warning("IK solution failed, keeping current joint positions")
            return current_joint_pos.copy()
        
        # Convert result back to degrees (already in model order from lerobot_IK)
        q_target_deg = np.rad2deg(q_target_rad)
        
        # Create result array in robot's joint order
        # Preserve gripper position if present in current_joint_pos
        if len(current_joint_pos) > len(self.joint_names):
            result = np.zeros_like(current_joint_pos)
            result[:len(self.joint_names)] = q_target_deg  # IK solution is already in correct order
            result[len(self.joint_names):] = current_joint_pos[len(self.joint_names):]  # Preserve gripper
            return result
        else:
            return q_target_deg


# Create a wrapper class that matches RobotKinematics interface exactly
class HardcodedRobotKinematics:
    """
    Wrapper class that provides the same interface as RobotKinematics
    but uses hardcoded kinematics internally.
    """
    
    def __init__(
        self,
        robot_type: str,
        target_frame_name: str = "gripper_frame_link",
        joint_names: list[str] | None = None,
    ):
        """
        Initialize hardcoded kinematics wrapper.
        
        Args:
            robot_type: Type of robot (e.g., "so100_follower", "so100_plus_follower")
            target_frame_name: Not used (kept for compatibility)
            joint_names: List of joint names (excluding gripper)
        """
        if joint_names is None:
            raise ValueError("joint_names must be provided for hardcoded kinematics")
        
        self.target_frame_name = target_frame_name
        self.joint_names = [name for name in joint_names if name != "gripper"]
        
        # Create the adapter
        self._adapter = HardcodedKinematicsAdapter(robot_type, joint_names)
    
    def forward_kinematics(self, joint_pos_deg: np.ndarray) -> np.ndarray:
        """Compute forward kinematics."""
        return self._adapter.forward_kinematics(joint_pos_deg)
    
    def inverse_kinematics(
        self,
        current_joint_pos: np.ndarray,
        desired_ee_pose: np.ndarray,
        position_weight: float = 1.0,
        orientation_weight: float = 0.01,
    ) -> np.ndarray:
        """Compute inverse kinematics."""
        return self._adapter.inverse_kinematics(
            current_joint_pos, desired_ee_pose, position_weight, orientation_weight
        )


if __name__ == "__main__":
    """
    Test script for SO100Plus hardcoded kinematics adapter.
    """
    print("=" * 60)
    print("Testing SO100Plus Hardcoded Kinematics Adapter")
    print("=" * 60)
    
    # SO100Plus joint names (excluding gripper)
    so100_plus_joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
        "shoulder_roll",
    ]
    
    try:
        # Initialize the adapter
        print("\n1. Initializing HardcodedRobotKinematics for SO100Plus...")
        kinematics = HardcodedRobotKinematics(
            robot_type="so100_plus_follower",
            target_frame_name="gripper_frame_link",
            joint_names=so100_plus_joint_names,
        )
        print(f"   ✅ Successfully initialized with {len(so100_plus_joint_names)} joints")
        print(f"   Joint names: {so100_plus_joint_names}")
        
        # Test 1: Forward Kinematics
        print("\n2. Testing Forward Kinematics (FK)...")
        # Use a test configuration (all joints at 0 degrees)
        test_joint_pos_deg = np.array([0.0, -45.0, 45.0, 0.0, 0.0, 0.0], dtype=float)
        print(f"   Input joint positions (degrees): {test_joint_pos_deg}")
        
        ee_pose = kinematics.forward_kinematics(test_joint_pos_deg)
        print(f"   ✅ FK computed successfully")
        print(f"   End-effector position (x, y, z): {ee_pose[:3, 3]}")
        print(f"   End-effector rotation matrix:")
        print(f"   {ee_pose[:3, :3]}")
        
        # Test 2: Inverse Kinematics
        print("\n3. Testing Inverse Kinematics (IK)...")
        # Use the result from FK as target
        print(f"   Target end-effector pose:")
        print(f"   Position: {ee_pose[:3, 3]}")
        print(f"   Trying to solve IK back to original joint positions...")
        
        # Use a nearby initial guess (slightly perturbed)
        initial_guess = test_joint_pos_deg + np.array([5.0, -5.0, 5.0, -5.0, 5.0, -5.0])
        print(f"   Initial guess (degrees): {initial_guess}")
        
        solved_joint_pos = kinematics.inverse_kinematics(
            current_joint_pos=initial_guess,
            desired_ee_pose=ee_pose,
        )
        print(f"   ✅ IK computed successfully")
        print(f"   Solved joint positions (degrees): {solved_joint_pos}")
        
        # Verify IK solution by computing FK again
        print("\n4. Verifying IK solution with FK...")
        verify_ee_pose = kinematics.forward_kinematics(solved_joint_pos)
        position_error = np.linalg.norm(ee_pose[:3, 3] - verify_ee_pose[:3, 3])
        print(f"   Position error: {position_error:.6f} meters")
        print(f"   Original position: {ee_pose[:3, 3]}")
        print(f"   Verified position:  {verify_ee_pose[:3, 3]}")
        
        if position_error < 0.01:  # 1cm tolerance
            print(f"   ✅ IK solution verified! (error < 1cm)")
        else:
            print(f"   ⚠️  IK solution has larger error than expected")
        
        # Test 3: Multiple configurations
        print("\n5. Testing multiple joint configurations...")
        test_configs = [
            np.array([0.0, -30.0, 30.0, 0.0, 0.0, 0.0], dtype=float),
            np.array([45.0, -60.0, 60.0, -30.0, 0.0, 0.0], dtype=float),
            np.array([-30.0, -45.0, 90.0, -45.0, 90.0, 0.0], dtype=float),
        ]
        
        for i, config in enumerate(test_configs, 1):
            print(f"\n   Test configuration {i}: {config}")
            try:
                ee = kinematics.forward_kinematics(config)
                print(f"   ✅ FK: Position = {ee[:3, 3]}")
                
                # Try IK
                solved = kinematics.inverse_kinematics(
                    current_joint_pos=config + 10.0,  # Perturbed initial guess
                    desired_ee_pose=ee,
                )
                verify_ee = kinematics.forward_kinematics(solved)
                error = np.linalg.norm(ee[:3, 3] - verify_ee[:3, 3])
                print(f"   ✅ IK: Error = {error:.6f}m")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("   Please make sure lerobot-kinematics is installed:")
        print("   pip install -e lerobot-kinematics")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

