# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
Remote teleoperation of SO101 robot in MuJoCo simulation using ZMQ EEF data.

This script receives end-effector pose data via ZMQ and controls a SO101 robot
in MuJoCo simulation. The EEF pose is converted to joint angles using inverse kinematics.

Example:

```shell
lerobot-teleoperate-remote-sim \
    --zmq_host=localhost \
    --zmq_port=5559 \
    --display_data=true
```

"""
import os
import logging
import pickle
import time
import json
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Optional

import mujoco
import numpy as np
import scipy.spatial.transform as st

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
from lerobot.processor.core import TransitionKey
from lerobot.processor.delta_action_processor import MapDeltaActionToRobotActionStep
from lerobot.processor.pipeline import (
    PipelineFeatureType,
    PolicyFeature,
    RobotActionProcessorStep,
    ProcessorStepRegistry,
)
from lerobot.robots.so100_follower.robot_kinematic_processor import (
    EEBoundsAndSafety,
    EEReferenceAndDelta,
    GripperVelocityToJoint,
    InverseKinematicsEEToJoints,
)
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import init_logging
from lerobot.teleoperators.remote_imu_leader.command_client import ZmqCommandClient
import mujoco.viewer as mjviewer



# Build processor pipeline for EEF pose -> robot joints
@ProcessorStepRegistry.register("ik_solution_validator")
class IKSolutionValidator(RobotActionProcessorStep):
    """
    Validates IK solution by checking:
    1. Joint angles are within reasonable limits
    2. Joint angle changes are not too large (smooth motion)
    
    This ensures safe and smooth robot motion by filtering out invalid or dangerous IK solutions.
    """
    
    def __init__(
        self,
        kinematics: RobotKinematics,
        motor_names: list[str],
        joint_limits: dict[str, tuple[float, float]] | None = None,
        max_joint_step_deg: float = 30.0,
        use_last_solution_on_failure: bool = True,
    ):
        """
        Args:
            kinematics: Robot kinematics solver (for potential future use)
            motor_names: List of motor/joint names
            joint_limits: Dictionary mapping joint names to (min, max) limits in degrees.
                        If None, uses default conservative limits.
            max_joint_step_deg: Maximum allowed change in joint angle per step (degrees)
            use_last_solution_on_failure: If True, use last valid solution when validation fails
        """
        self.kinematics = kinematics
        self.motor_names = motor_names
        self.max_joint_step_deg = max_joint_step_deg
        self.use_last_solution_on_failure = use_last_solution_on_failure
        
        # Default joint limits (conservative, in degrees)
        # These are approximate limits for SO101 robot
        if joint_limits is None:
            self.joint_limits = {
                "shoulder_pan": (-180.0, 180.0),
                "shoulder_lift": (-90.0, 90.0),
                "elbow_flex": (-90.0, 90.0),
                "wrist_flex": (-180.0, 180.0),
                "wrist_roll": (-180.0, 180.0),
            }
        else:
            self.joint_limits = joint_limits
        
        self._last_valid_action: RobotAction | None = None
    
    def action(self, action: RobotAction) -> RobotAction:
        # Extract joint angles
        joint_angles_deg = []
        for name in self.motor_names:
            if name != "gripper":
                key = f"{name}.pos"
                if key not in action:
                    logging.warning(f"Missing joint angle for {name} in IK solution")
                    if self.use_last_solution_on_failure and self._last_valid_action:
                        return self._last_valid_action.copy()
                    return action
                joint_angles_deg.append(float(action[key]))
        
        joint_angles_deg = np.array(joint_angles_deg)
        
        # Check 1: Joint limits
        for i, name in enumerate(self.motor_names):
            if name != "gripper" and name in self.joint_limits:
                min_limit, max_limit = self.joint_limits[name]
                angle = joint_angles_deg[i]
                if angle < min_limit or angle > max_limit:
                    logging.warning(
                        f"IK solution invalid: Joint {name} angle {angle:.2f}° "
                        f"outside limits [{min_limit:.2f}°, {max_limit:.2f}°]"
                    )
                    if self.use_last_solution_on_failure and self._last_valid_action:
                        logging.info(f"Using last valid solution due to joint limit violation")
                        return self._last_valid_action.copy()
                    # Clip to limits as fallback
                    joint_angles_deg[i] = np.clip(angle, min_limit, max_limit)
                    action[f"{name}.pos"] = joint_angles_deg[i]
        
        # Check 2: Joint angle changes (smoothness)
        if self._last_valid_action is not None:
            last_joint_angles = np.array([
                float(self._last_valid_action.get(f"{name}.pos", 0.0))
                for name in self.motor_names if name != "gripper"
            ])
            joint_delta = np.abs(joint_angles_deg - last_joint_angles)
            max_delta = np.max(joint_delta)
            if max_delta > self.max_joint_step_deg:
                logging.warning(
                    f"IK solution has large joint change: max delta {max_delta:.2f}° "
                    f"> {self.max_joint_step_deg}°"
                )
                # Scale down the change
                scale = self.max_joint_step_deg / max_delta
                joint_angles_deg = last_joint_angles + (joint_angles_deg - last_joint_angles) * scale
                for i, name in enumerate(self.motor_names):
                    if name != "gripper":
                        action[f"{name}.pos"] = joint_angles_deg[i]
        
        # Store valid solution
        self._last_valid_action = action.copy()
        return action
    
    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Features remain unchanged."""
        return features
    
    def reset(self):
        """Reset internal state."""
        self._last_valid_action = None


@ProcessorStepRegistry.register("map_absolute_to_delta_eef")
class MapAbsoluteToDeltaEEF(RobotActionProcessorStep):
    """
    Converts absolute EEF position to relative delta commands.
    
    This processor converts absolute position/orientation to relative deltas
    by comparing with the previous target action (not the current robot pose).
    The output format matches what EEReferenceAndDelta expects.
    """

    def __init__(self, kinematics: RobotKinematics, motor_names: list[str], position_scale: float = 1.0):
        """
        Args:
            kinematics: Robot kinematics solver for forward kinematics (used only for first action)
            motor_names: List of motor/joint names
            position_scale: Scale factor for position deltas
        """
        self.kinematics = kinematics
        self.motor_names = motor_names
        self.position_scale = position_scale
        self._last_target_pos: np.ndarray | None = None
        self._last_target_rot: st.Rotation | None = None
        self._last_target_gripper: float | None = None

    def action(self, action: RobotAction) -> RobotAction:
        # Extract absolute target position and orientation
        if all(k in action for k in ["eef_pos_x", "eef_pos_y", "eef_pos_z"]):
            target_pos = np.array([
                action.pop("eef_pos_x"),
                action.pop("eef_pos_y"),
                action.pop("eef_pos_z"),
            ])

            # Extract target orientation
            if all(k in action for k in ["eef_ori_roll", "eef_ori_pitch", "eef_ori_yaw"]):
                roll = action.pop("eef_ori_roll")
                pitch = action.pop("eef_ori_pitch")
                yaw = action.pop("eef_ori_yaw")
                target_rot = st.Rotation.from_euler('xyz', [roll, pitch, yaw])
            else:
                # Default orientation (pointing down)
                logging.warning("Missing orientation data, using default orientation (pointing down)")
                target_rot = st.Rotation.from_euler('xyz', [0, np.pi/2, 0])

            # Get gripper value
            gripper_value = action.pop("gripper", 0.5)  # Default if not found

            # Compute delta relative to previous target action
            if self._last_target_pos is not None and self._last_target_rot is not None:
                # Use previous target as reference
                ref_pos = self._last_target_pos
                ref_rot = self._last_target_rot
                ref_gripper = self._last_target_gripper if self._last_target_gripper is not None else 0.5
            else:
                # 第一个动作：以该动作自身为参考（delta=0），相对运动为0
                ref_pos = target_pos.copy()
                ref_rot = target_rot
                ref_gripper = gripper_value

                logging.debug(
                    f"First action: using the first action itself as reference "
                    f"(pos={ref_pos}, rot={ref_rot.as_euler('xyz', degrees=True)})"
                )

            # Compute position delta: current target - reference (zero for first action)
            delta_pos = target_pos - ref_pos
            # Scale the delta
            delta_pos = delta_pos * self.position_scale

            # Compute rotation delta: relative rotation from reference to current target (zero for first action)
            delta_rot = ref_rot.inv() * target_rot
            delta_rotvec = delta_rot.as_rotvec()

            # Compute gripper delta: current target - reference (zero for first action)
            gripper_delta = gripper_value - ref_gripper
            # Convert to velocity-like command (scaled)
            gripper_vel = gripper_delta * 10.0  # Scale factor (adjust as needed)

            # Determine if enabled (if there's significant movement)
            position_magnitude = np.linalg.norm(delta_pos)
            rotation_magnitude = np.linalg.norm(delta_rotvec)
            enabled = position_magnitude > 1e-3 or rotation_magnitude > 1e-3

            # Output format expected by EEReferenceAndDelta
            action["enabled"] = enabled
            action["target_x"] = float(delta_pos[0])
            action["target_y"] = float(delta_pos[1])
            action["target_z"] = float(delta_pos[2])
            action["target_wx"] = float(delta_rotvec[0])
            action["target_wy"] = float(delta_rotvec[1])
            action["target_wz"] = float(delta_rotvec[2])
            action["gripper_vel"] = gripper_vel

            # Store current target as last target for next iteration
            self._last_target_pos = target_pos
            self._last_target_rot = target_rot
            self._last_target_gripper = gripper_value
        else:
            raise ValueError("Missing position data in action")

        print("--------------------------------")
        print(f"Step1: Delta action: {action}")
        print(f"Step1: Move distance (m): {np.linalg.norm(delta_pos)}")
        print("--------------------------------")
        return action

    def reset(self) -> None:
        """Reset internal state (clear last target pose)."""
        self._last_target_pos = None
        self._last_target_rot = None
        self._last_target_gripper = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transform features - convert to delta format."""
        action_features = features.get(PipelineFeatureType.ACTION, {})

        # Remove old format features
        old_keys = ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_ori_roll", "eef_ori_pitch", "eef_ori_yaw", "gripper"]
        for key in old_keys:
            action_features.pop(key, None)

        # Add delta format features (expected by EEReferenceAndDelta)
        delta_keys = ["enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper_vel"]
        for key in delta_keys:
            action_features[key] = PolicyFeature(
                shape=(1,),
                dtype=np.float32,
                names=(key,)
            )

        features[PipelineFeatureType.ACTION] = action_features
        return features

@dataclass
class RemoteSimConfig:
    # ZMQ server host for receiving EEF pose data
    zmq_host: str = "localhost"
    # ZMQ server port for receiving EEF pose data
    zmq_port: int = 5556
    # Limit the maximum frames per second
    fps: int = 30
    # Timeout for ZMQ receive in milliseconds
    zmq_timeout_ms: int = 1000
    # Use PULL mode (passive receive) or REQ mode (active request)
    use_pull_mode: bool = True
    # Initial EEF position [x, y, z] in meters
    initial_position: list[float] = None
    # Initial EEF orientation [roll, pitch, yaw] in degrees
    initial_orientation: list[float] = None

    # DISPLAY environment variable for X11 forwarding (e.g., "10.42.198.38:0.0")
    # display_env: str = "10.47.91.30:0.0"

    enable_visualization: bool = True

    def __post_init__(self):
        if self.initial_position is None:
            self.initial_position = [0.24, 0.0, 0.15]  # Default initial position
        if self.initial_orientation is None:
            self.initial_orientation = [0.0, 0.0, 0.0]  # Default initial orientation

class ZmqEEFReceiver:
    """
    ZMQ EEF 数据接收器，基于 lerobot 规范的 ZmqCommandClient。
    
    支持 PULL 和 REQ 两种模式，自动处理 pickle 和 JSON 格式的数据。

    Args:
        host: ZMQ 服务器地址
        port: ZMQ 服务器端口
        timeout_ms: 接收超时时间（毫秒）
        use_pull_mode: 如果为 True，使用 PULL socket（被动接收）；如果为 False，使用 REQ socket（主动请求）
        initial_position: 初始位置 [x, y, z]
        initial_orientation: 初始姿态 [roll, pitch, yaw]
    """

    def __init__(
        self, 
        host: str = "localhost", 
        port: int = 5556, 
        timeout_ms: int = 1000,
        use_pull_mode: bool = True,
        initial_position: list[float] = None,
        initial_orientation: list[float] = None
    ):

        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.use_pull_mode = use_pull_mode
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation

        # 使用 lerobot 规范的 ZmqCommandClient
        address = f"tcp://{host}:{port}"
        self._zmq_client: Optional[ZmqCommandClient] = None
        self._address = address
        self._last_eef_data: Optional[dict[str, Any]] = None
        self._frame_count = 0
        self._start_time = time.time()

    def start(self) -> None:
        """Start ZMQ client connection."""
        if self._zmq_client is not None:
            logging.warning("ZMQ EEF receiver is already started")
            return

        try:
            self._zmq_client = ZmqCommandClient(address=self._address, timeout_ms=self.timeout_ms, use_pull_mode=self.use_pull_mode)
            logging.info(f"ZMQ EEF receiver started: {self._address} (mode: {'PULL' if self.use_pull_mode else 'REQ'})")
        except Exception as e:
            logging.error(f"Failed to start ZMQ EEF receiver: {e}")
            raise

    def get_latest_eef_data(self) -> Optional[dict[str, Any]]:
        """
        Get the latest EEF data.

        Returns:
            EEF data dictionary, containing position and orientation, or None if no data is available
        """
        if self._zmq_client is None:
            logging.warning("ZMQ client not started. Call start() first.")
            return None

        try:
            # 从 ZMQ 客户端获取最新命令
            command = self._zmq_client.get_latest_command()
            print("--------------------------------")
            print(f"Raw IMU command: {command}")
            print("--------------------------------")
            if command is None:
                # 如果没有收到新数据，返回上次的数据
                return self._last_eef_data
            
            # 解析接收到的数据
            eef_data = self._parse_command(command)
            
            if eef_data is not None:
                self._last_eef_data = eef_data
                self._frame_count += 1
                
                # 每30帧打印一次统计信息
                if self._frame_count % 30 == 0:
                    elapsed = time.time() - self._start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    logging.debug(
                        f"[ZMQ EEF] Received {self._frame_count} frames, "
                        f"FPS: {fps:.1f}"
                    )
                    self._start_time = time.time()
            
            return eef_data
            
        except Exception as e:
            logging.warning(f"Failed to get EEF data: {e}")
            return self._last_eef_data

    def _parse_command(self, command: dict[str, Any]) -> Optional[dict[str, Any]]:
        if "position" in command and "orientation" in command and "gripper" in command:
            return {
                "position": command["position"],
                "orientation": command["orientation"],
                "gripper": command["gripper"]
            }
        else:
            print(f"Invalid command: {command}")
        return None

    
    def stop(self) -> None:
        """Stop ZMQ client and clean up resources."""
        if self._zmq_client is not None:
            try:
                self._zmq_client.close()
                logging.info("ZMQ EEF receiver stopped")
            except Exception as e:
                logging.warning(f"Error closing ZMQ client: {e}")
            finally:
                self._zmq_client = None
                self._last_eef_data = None

class MuJoCoSO101Robot:
    """MuJoCo simulation wrapper for SO101 robot that mimics Robot interface."""

    def __init__(
        self,
        xml_path: Path,
        urdf_path: Path,
        initial_position: Optional[list[float]] = None,
        initial_orientation: Optional[list[float]] = None,
    ):
        """Initialize MuJoCo robot simulation.

        Args:
            xml_path: Path to MuJoCo XML scene file
            urdf_path: Path to URDF file (for kinematics)
            initial_position: Initial EEF position [x, y, z] in meters. If None, uses default joint angles.
            initial_orientation: Initial EEF orientation [roll, pitch, yaw] in radians. If None, uses default joint angles.
        """
        self.xml_path = xml_path
        self.urdf_path = urdf_path
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation

        # MuJoCo needs to load XML from the directory containing the XML file
        # so that relative mesh paths can be resolved correctly
        xml_dir = xml_path.parent
        original_cwd = str(Path.cwd())
        try:
            os.chdir(str(xml_dir))
            self.mjmodel = mujoco.MjModel.from_xml_path(str(xml_path.name))
        finally:
            os.chdir(original_cwd)

        self.mjdata = mujoco.MjData(self.mjmodel)

        # Get joint indices in MuJoCo
        self.joint_indices = []
        self.joint_names = []
        SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
        for joint_name in SO101_JOINT_NAMES:
            try:
                joint_id = self.mjmodel.joint(joint_name).id
                qpos_addr = self.mjmodel.jnt_qposadr[joint_id]
                self.joint_indices.append(qpos_addr)
                self.joint_names.append(joint_name)
            except Exception as e:
                print(f"Warning: Joint '{joint_name}' not found in MuJoCo model: {e}")

        # Initialize joint positions
        if initial_position is not None and initial_orientation is not None:
            # Use IK to compute joint angles from EEF pose
            self._set_initial_pose_from_eef(initial_position, initial_orientation)
        else:
            raise ValueError("Initial position and orientation are required")

        # Create a mock bus object with motors dict
        class MockMotor:
            def __init__(self, name: str):
                self.name = name

        class MockBus:
            def __init__(self, motor_names: list[str]):
                self.motors = {name: MockMotor(name) for name in motor_names}
                self.is_connected = True

        self.bus = MockBus(SO101_JOINT_NAMES)
        self.name = "so101_follower"
        self._is_connected = False

    def _set_initial_pose_from_eef(self, position: list[float], orientation: list[float]) -> None:
        """Set initial joint angles using IK from EEF position and orientation.

        This method computes the inverse kinematics to find the joint angles that
        place the end-effector at the specified position and orientation.

        Args:
            position: EEF position [x, y, z] in meters
            orientation: EEF orientation [roll, pitch, yaw] in radians (Euler angles, xyz order)
        """
        try:
            # Create kinematics solver for IK
            ik_joint_names = [name for name in self.joint_names if name != "gripper"]
            kinematics_solver = RobotKinematics(
                urdf_path=str(self.urdf_path.absolute()),
                target_frame_name="gripper_frame_link",
                joint_names=ik_joint_names,
            )

            # Build target transform matrix from position and orientation
            # Convert Euler angles (roll, pitch, yaw in xyz order) to rotation matrix
            rot_matrix = st.Rotation.from_euler('xyz', orientation, degrees=False).as_matrix()
            t_target = np.eye(4, dtype=float)
            t_target[:3, :3] = rot_matrix
            t_target[:3, 3] = position

            # Use default joint angles as initial guess for IK solver
            q_init_rad = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, -np.pi / 2], dtype=float)
            q_init_deg = np.rad2deg(q_init_rad)

            # Compute inverse kinematics to get joint angles
            q_target_deg = kinematics_solver.inverse_kinematics(q_init_deg, t_target)
            print(f"initial_pose_from_eef_q_target (degrees): {q_target_deg}")

            # For FK testing: verify IK solution
            t_ee = kinematics_solver.forward_kinematics(q_target_deg)
            print(f"initial_pose_from_eef_t_ee (FK result): {t_ee[:3,3]}")
            print(f"initial_pose_from_eef_t_target (desired): {t_target[:3,3]}")
            pos_error = np.linalg.norm(t_ee[:3,3] - t_target[:3,3])
            print(f"initial_pose_from_eef_position_error (m): {pos_error:.6f}")

            # Convert joint angles from degrees to radians for MuJoCo
            q_target_rad = np.deg2rad(q_target_deg)

            # Set joint positions in MuJoCo (excluding gripper)
            # Note: MuJoCo uses radians for joint positions
            for i, joint_name in enumerate(ik_joint_names):
                if joint_name in self.joint_names:
                    idx = self.joint_names.index(joint_name)
                    if idx < len(self.joint_indices):
                        self.mjdata.qpos[self.joint_indices[idx]] = q_target_rad[i]

            # Gripper remains at 0 (initial position)
            if "gripper" in self.joint_names:
                gripper_idx = self.joint_names.index("gripper")
                if gripper_idx < len(self.joint_indices):
                    self.mjdata.qpos[self.joint_indices[gripper_idx]] = 0.0

            logging.info(
                f"Initialized robot pose from EEF position {position} m "
                f"and orientation (roll={np.rad2deg(orientation[0]):.1f}°, "
                f"pitch={np.rad2deg(orientation[1]):.1f}°, "
                f"yaw={np.rad2deg(orientation[2]):.1f}°)"
            )
            logging.info(f"Computed joint angles (degrees): {q_target_deg}")
            logging.info(f"Position error: {pos_error:.6f} m")
        except Exception as e:
            logging.error(f"Failed to set initial pose from EEF using IK: {e}")

    def connect(self, calibrate: bool = True) -> None:
        """Connect to MuJoCo simulation (no-op for simulation)."""
        self._is_connected = True
        mujoco.mj_forward(self.mjmodel, self.mjdata)

    def disconnect(self) -> None:
        """Disconnect from simulation."""
        self._is_connected = False

    @property
    def is_connected(self) -> bool:
        """Check if connected."""
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        """Simulation is always 'calibrated'."""
        return True

    def configure(self) -> None:
        """Configure robot (no-op for simulation)."""
        pass

    def calibrate(self) -> None:
        """Calibrate robot (no-op for simulation)."""
        pass

    def get_observation(self) -> dict[str, Any]:
        """Get current robot observation (joint positions in degrees)."""
        obs = {}
        for i, joint_name in enumerate(self.joint_names):
            if i < len(self.joint_indices):
                pos_deg = self.mjdata.qpos[self.joint_indices[i]]
                obs[f"{joint_name}.pos"] = np.rad2deg(pos_deg)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot (set joint positions in degrees)."""
        for key, value in action.items():
            # Handle both "motor_name.pos" and "motor_name" formats
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
            else:
                motor_name = key

            if motor_name in self.joint_names:
                idx = self.joint_names.index(motor_name)
                if idx < len(self.joint_indices):
                    # mujoco uses radians for joint positions
                    self.mjdata.qpos[self.joint_indices[idx]] = float(np.deg2rad(value))

        # Step simulation
        mujoco.mj_step(self.mjmodel, self.mjdata)
        return action

    @property
    def observation_features(self) -> dict[str, type | tuple]:
        """Observation features."""
        return {f"{name}.pos": float for name in self.joint_names}

    @property
    def action_features(self) -> dict[str, type]:
        """Action features."""
        return {f"{name}.pos": float for name in self.joint_names}


def teleop_remote_sim_loop(
    robot: MuJoCoSO101Robot,
    zmq_eef_receiver: ZmqEEFReceiver,
    kinematics_solver: RobotKinematics,
    ik_joint_names: list[str],
    ee_to_joints_pipeline: RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction],
    fps: int,
    enable_visualization: bool = False,
    viewer: Any | None = None,
):
    """
    Remote teleoperation simulation loop that receives EEF data via ZMQ and controls MuJoCo robot.

    Args:
        robot: MuJoCo SO101 robot instance
        zmq_eef_receiver: ZMQ EEF data receiver
        kinematics_solver: Robot kinematics solver
        ik_joint_names: Joint names for inverse kinematics
        ee_to_joints_pipeline: Pipeline to convert EEF pose to joint angles
        fps: Target frames per second
    """

    frame_count = 0
    last_eef_data = None

    # Position output frequency (every N frames)
    position_output_interval = max(1, int(fps / 2))  # Output every 0.5 seconds

    while True:
        loop_start = time.perf_counter()

        # Get latest EEF data from ZMQ
        eef_data = zmq_eef_receiver.get_latest_eef_data()
        if eef_data is not None:
            last_eef_data = eef_data

        # If no EEF data received yet, skip this iteration
        if last_eef_data is None:
            logging.debug("No EEF data received yet, waiting...")
            dt_s = time.perf_counter() - loop_start
            busy_wait(max(1 / fps - dt_s, 0.0))
            continue

        # Get robot observation (for pipeline processing)
        robot_obs = robot.get_observation()
        print(f"Now robot obs (get_observation): {robot_obs}")
        # Convert EEF pose to robot action format
        # EEF data should contain position (x,y,z) and orientation (roll, pitch, yaw in radians)
        position = last_eef_data["position"]  # [x, y, z] in meters
        orientation = last_eef_data["orientation"]  # [roll, pitch, yaw] in radians
        gripper = last_eef_data["gripper"]  # Gripper value

        # Validate orientation format
        if len(orientation) != 3:
            logging.warning(
                f"Invalid orientation format: expected 3 elements (roll, pitch, yaw), "
                f"got {len(orientation)}. Using default orientation."
            )
            orientation = [0.0, 0.0, 0.0]  # Default to zero orientation

        # Create action dict in the format expected by the pipeline
        ee_action = {
            "eef_pos_x": position["x"],
            "eef_pos_y": position["y"],
            "eef_pos_z": position["z"],
            "eef_ori_roll": orientation["roll"],   # Roll in radians
            "eef_ori_pitch": orientation["pitch"],  # Pitch in radians
            "eef_ori_yaw": orientation["yaw"],    # Yaw in radians
            "gripper": gripper,
        }

        # Process through IK pipeline: EEF pose -> robot joints
        robot_action = ee_to_joints_pipeline((ee_action, robot_obs))
        print(f"Final: Action: {robot_action}")

        # Send action to MuJoCo robot
        robot.send_action(robot_action)

        # Render with MuJoCo viewer (if enabled)
        if viewer is not None:
            try:
                viewer.sync()
            except Exception as e:
                logging.debug(f"Viewer sync error: {e}")

        frame_count += 1

        # Calculate and output current position
        if frame_count % position_output_interval == 0:
            # Get current joint positions (in degrees)
            joint_positions_deg = np.array([
                robot_obs.get(f"{name}.pos", 0.0) for name in ik_joint_names
            ])

            # Calculate end-effector position using forward kinematics
            try:
                ee_transform = kinematics_solver.forward_kinematics(joint_positions_deg)
                ee_position = ee_transform[:3, 3]  # Extract translation
                ee_rotation = ee_transform[:3, :3]  # Extract rotation matrix

                # Convert rotation matrix to Euler angles (xyz order)
                euler_angles = st.Rotation.from_matrix(ee_rotation).as_euler('xyz', degrees=True)

            except Exception as e:
                print(f"\r[EEF Control] Error calculating EE position: {e}", end='', flush=True)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)


def _can_use_glfw() -> bool:
    """Return True if a glfw context can be initialized (basic headless safety check)."""
    try:
        import glfw  # type: ignore
        if not glfw.init():
            return False
        glfw.terminate()
        return True
    except Exception:
        return False

@parser.wrap()
def teleoperate_remote_sim(cfg: RemoteSimConfig):
    init_logging()
    logging.info(pformat(vars(cfg)))

    # Configure display for visualization
    #if cfg.display_env:
    #    os.environ["DISPLAY"] = cfg.display_env
    #    logging.info(f"Set DISPLAY={cfg.display_env}")

    # Robust GL backend selection
    if cfg.enable_visualization:
        if _can_use_glfw():
            os.environ["MUJOCO_GL"] = "glfw"
            logging.info("Using MUJOCO_GL=glfw (interactive viewer)")
        else:
            logging.warning("glfw backend unavailable (headless or missing display). Falling back to MUJOCO_GL=egl.")
            os.environ["MUJOCO_GL"] = "egl"
            cfg.enable_visualization = False
    else:
        if os.environ.get("MUJOCO_GL") != "egl":
            os.environ["MUJOCO_GL"] = "egl"
        logging.info("Visualization disabled: MUJOCO_GL=egl (offscreen)")

    # Configuration
    FPS = cfg.fps
    # Get paths relative to lerobot directory
    LEROBOT_DIR = Path(__file__).parent.parent
    SO101_DIR = LEROBOT_DIR / "SO101"
    URDF_PATH = SO101_DIR / "so101_new_calib.urdf"
    MUJOCO_XML_PATH = SO101_DIR / "scene_so101.xml"

    # Motor names for LeRobot (matching SO101 follower)
    SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

    # Check if files exist
    if not MUJOCO_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo XML file not found: {MUJOCO_XML_PATH}")
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")

    # Create MuJoCo robot with initial position from config
    # Convert initial_orientation from degrees to radians if provided
    initial_orientation_rad = None
    if cfg.initial_orientation is not None:
        initial_orientation_rad = [np.deg2rad(angle) for angle in cfg.initial_orientation]
    
    robot = MuJoCoSO101Robot(
        MUJOCO_XML_PATH,
        URDF_PATH,
        initial_position=cfg.initial_position,
        initial_orientation=initial_orientation_rad,
    )

    # Create kinematics solver
    print(f"Loading kinematics from URDF: {URDF_PATH}")
    ik_joint_names = [name for name in SO101_JOINT_NAMES if name != "gripper"]

    # Use absolute path for kinematics solver
    kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH.absolute()),  # Use absolute path
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Pipeline流程: 绝对位置 → 相对增量 → 绝对EEF位姿 → 安全检查 → IK求解 → 关节角度
    ee_to_joints_pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            # Step 1: 将绝对位置转换为相对增量(以IMU初始化位置为基准，后续动作都以该位置为基准)
            # 输入: {"eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_ori_roll", "eef_ori_pitch", "eef_ori_yaw", "gripper"}
            # 输出: {"enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper_vel"}
            MapAbsoluteToDeltaEEF(
                kinematics=kinematics_solver,
                motor_names=ik_joint_names,
                position_scale=1.0,  # 位置增量缩放因子
            ),
            # Step 2: 将相对增量转换为绝对EEF位姿（带参考）
            # 输入: {"enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper_vel"}
            # 输出: {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_vel"}
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 1.0, "y": 1.0, "z": 1.0},  # 步长缩放（1.0 = 不缩放，因为已经在MapAbsoluteToDeltaEEF中处理）
                motor_names=ik_joint_names,
                use_latched_reference=True,  # 使用当前位姿作为参考（每步更新）
            ),
            # Step 3: 安全检查（工作空间边界和步长限制）
            # 输入: {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_vel"}
            # 输出: {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_vel"} (裁剪后)
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-1.0, -1.0, -1.0], "max": [1.0, 1.0, 1.0]},
                max_ee_step_m=0.10,  # 最大步长 (10cm)
            ),
            # Step 4: 将Gripper速度转换为位置
            # 输入: {"ee.gripper_vel", ...}
            # 输出: {"ee.gripper_pos", ...}
            GripperVelocityToJoint(
                speed_factor=20.0,  # Gripper速度缩放因子
            ),
            # Step 4: 逆运动学求解（EEF位姿 → 关节角度）
            # 输入: {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"}
            # 输出: {"shoulder_pan.pos", "shoulder_lift.pos", ..., "gripper.pos"}
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=SO101_JOINT_NAMES,
                initial_guess_current_joints=True,  # 使用当前关节角度作为IK初始猜测
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Initialize ZMQ EEF receiver
    zmq_eef_receiver = ZmqEEFReceiver(
        host=cfg.zmq_host,
        port=cfg.zmq_port,
        timeout_ms=cfg.zmq_timeout_ms,
        use_pull_mode=cfg.use_pull_mode
    )
    zmq_eef_receiver.start()

    # Connect robot
    robot.connect()

    # Launch MuJoCo viewer if enabled
    viewer = None
    if cfg.enable_visualization:
        try:
            viewer = mjviewer.launch_passive(robot.mjmodel, robot.mjdata)
            logging.info("Launched MuJoCo viewer (glfw)")
        except Exception as e:
            logging.error(f"Failed to launch MuJoCo viewer: {e}")
            logging.warning("Auto-switching to headless mode (egl). Disable --enable_visualization to suppress.")
            viewer = None

    print("\n" + "=" * 60)
    print("Remote SO101 Simulation Teleoperation")
    print("=" * 60)
    print(f"Waiting for EEF data from ZMQ: tcp://{cfg.zmq_host}:{cfg.zmq_port}")
    if cfg.enable_visualization and viewer is not None:
        print("Visualization: MuJoCo viewer (GLFW window)")
    else:
        print("Visualization disabled")
    print("Press Ctrl+C to exit\n")
    
    try:
        teleop_remote_sim_loop(
            robot=robot,
            zmq_eef_receiver=zmq_eef_receiver,
            kinematics_solver=kinematics_solver,
            ik_joint_names=ik_joint_names,
            ee_to_joints_pipeline=ee_to_joints_pipeline,
            fps=FPS,
            enable_visualization=cfg.enable_visualization,
            viewer=viewer,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        zmq_eef_receiver.stop()
        robot.disconnect()
        print("Cleanup completed.")

def main():
    teleoperate_remote_sim()


if __name__ == "__main__":
    main()
