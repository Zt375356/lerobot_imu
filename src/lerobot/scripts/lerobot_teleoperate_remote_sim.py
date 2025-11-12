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
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from pprint import pformat
from typing import Any

import mujoco
import numpy as np
import scipy.spatial.transform as st
import zmq
import rerun as rr

from lerobot.configs import parser
from lerobot.model.kinematics import RobotKinematics
from lerobot.processor import RobotAction, RobotObservation, RobotProcessorPipeline
from lerobot.processor.converters import (
    robot_action_observation_to_transition,
    transition_to_robot_action,
)
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
from lerobot.utils.visualization_utils import init_rerun, log_rerun_data

# Build processor pipeline for EEF pose -> robot joints
# Convert EEF pose to the format expected by InverseKinematicsEEToJoints
@ProcessorStepRegistry.register("eef_pose_to_ik_format")
class EEFToIKFormatProcessor(RobotActionProcessorStep):
    """Converts EEF pose format to the format expected by InverseKinematicsEEToJoints."""

    def action(self, action: RobotAction) -> RobotAction:
        # Convert from our format to IK processor format
        if all(k in action for k in ["eef_pos_x", "eef_pos_y", "eef_pos_z"]):
            # Position
            action["ee.x"] = action.pop("eef_pos_x")
            action["ee.y"] = action.pop("eef_pos_y")
            action["ee.z"] = action.pop("eef_pos_z")

            # Orientation - assume euler angles and convert to axis-angle format
            if all(k in action for k in ["eef_ori_roll", "eef_ori_pitch", "eef_ori_yaw"]):
                roll = np.deg2rad(action.pop("eef_ori_roll"))
                pitch = np.deg2rad(action.pop("eef_ori_pitch"))
                yaw = np.deg2rad(action.pop("eef_ori_yaw"))

                # Convert Euler angles to rotation vector (axis-angle)
                rotation = st.Rotation.from_euler('xyz', [roll, pitch, yaw])
                rotvec = rotation.as_rotvec()  # Returns [wx, wy, wz]

                action["ee.wx"] = rotvec[0]
                action["ee.wy"] = rotvec[1]
                action["ee.wz"] = rotvec[2]
            else:
                # Default orientation (pointing down) - 90 degrees around Y axis
                action["ee.wx"] = 0.0
                action["ee.wy"] = np.pi / 2
                action["ee.wz"] = 0.0

            # Gripper
            gripper_value = action.get("gripper", 0.5)  # Default half-open
            action["ee.gripper_pos"] = gripper_value

        return action

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        """Transform features - convert EEF pose features to IK format."""
        action_features = features.get(PipelineFeatureType.ACTION, {})

        # Remove old format features
        old_keys = ["eef_pos_x", "eef_pos_y", "eef_pos_z", "eef_ori_roll", "eef_ori_pitch", "eef_ori_yaw"]
        for key in old_keys:
            action_features.pop(key, None)

        # Add IK format features
        ik_keys = ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "ee.gripper_pos"]
        for key in ik_keys:
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
    # Timeout for ZMQ receive in seconds
    zmq_timeout_s: float = 1.0

    display_env: str = "10.42.198.38:0.0"

    enable_visualization: bool = True
    # Rerun web viewer port (for remote viewing when DISPLAY is not available)
    rerun_web_port: int = 9876


class ZmqEEFReceiver:
    """异步接收EEF位姿数据的ZMQ接收器"""

    def __init__(self, host: str = "localhost", port: int = 5559, timeout_s: float = 1.0):
        """
        初始化 ZMQ EEF 数据接收器。

        Args:
            host: ZMQ 服务器地址
            port: ZMQ 服务器端口
            timeout_s: 接收超时时间（秒）
        """
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.queue = queue.Queue(maxsize=1)  # 只保留最新数据
        self.running = False
        self.thread: threading.Thread | None = None
        self.context: zmq.Context | None = None
        self.socket: zmq.Socket | None = None
        self.last_eef_data = None

    def start(self) -> None:
        """启动接收线程"""
        if self.running:
            logging.warning("ZMQ EEF receiver is already running")
            return

        self.running = True
        self.thread = threading.Thread(target=self._receive_loop, daemon=True)
        self.thread.start()
        logging.info(f"ZMQ EEF receiver started: tcp://{self.host}:{self.port}")

    def get_latest_eef_data(self) -> dict[str, Any] | None:
        """
        获取最新的EEF数据（非阻塞）。

        Returns:
            EEF位姿数据字典，包含position和orientation，如果没有新数据则返回None
        """
        try:
            return self.queue.get_nowait()
        except queue.Empty:
            return None

    def _receive_loop(self) -> None:
        """后台线程：从ZMQ接收EEF数据"""
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.setsockopt(zmq.RCVHWM, 1)  # 接收缓冲区只保留1帧
        self.socket.setsockopt(zmq.LINGER, 0)  # 关闭时不等待
        self.socket.connect(f"tcp://{self.host}:{self.port}")

        # 设置接收超时
        self.socket.setsockopt(zmq.RCVTIMEO, int(self.timeout_s * 1000))

        logging.info(f"ZMQ EEF receiver connected to tcp://{self.host}:{self.port}")

        frame_count = 0
        start_time = time.time()

        while self.running:
            try:
                # 接收数据
                serialized_data = self.socket.recv()
                data = pickle.loads(serialized_data)

                # 提取EEF数据
                if "eef_pose" in data:
                    eef_data = data["eef_pose"]
                elif "position" in data and "orientation" in data:
                    # 如果数据直接包含position和orientation
                    eef_data = {
                        "position": data["position"],
                        "orientation": data["orientation"]
                    }
                else:
                    logging.warning(f"Received data does not contain EEF pose: {data.keys()}")
                    continue

                # 放入队列（只保留最新数据）
                try:
                    self.queue.put_nowait(eef_data)
                    self.last_eef_data = eef_data
                except queue.Full:
                    # 队列满时，移除旧数据并放入新数据
                    try:
                        self.queue.get_nowait()
                        self.queue.put_nowait(eef_data)
                        self.last_eef_data = eef_data
                    except queue.Empty:
                        pass

                frame_count += 1

                # 每30帧打印一次统计信息
                if frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    logging.debug(
                        f"[ZMQ EEF] Received {frame_count} frames, "
                        f"FPS: {fps:.1f}, data size: {len(serialized_data)} bytes"
                    )
                    start_time = time.time()

            except zmq.Again:
                # 超时，继续等待
                continue
            except Exception as e:
                logging.warning(f"Failed to receive EEF data via ZMQ: {e}")
                time.sleep(0.1)  # 短暂等待后重试

        # 清理资源
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        logging.info("ZMQ EEF receiver thread stopped")

    def stop(self) -> None:
        """停止接收线程并清理资源"""
        if not self.running:
            return

        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logging.info("ZMQ EEF receiver stopped")


class MuJoCoSO101Robot:
    """MuJoCo simulation wrapper for SO101 robot that mimics Robot interface."""

    def __init__(self, xml_path: Path, urdf_path: Path):
        """Initialize MuJoCo robot simulation.

        Args:
            xml_path: Path to MuJoCo XML scene file
            urdf_path: Path to URDF file (for kinematics)
        """
        self.xml_path = xml_path
        self.urdf_path = urdf_path

        # MuJoCo needs to load XML from the directory containing the XML file
        # so that relative mesh paths can be resolved correctly
        xml_dir = xml_path.parent
        original_cwd = str(Path.cwd())
        try:
            import os
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

        # Initialize joint positions (in radians for MuJoCo)
        # Set initial pose: shoulder_pan=0, shoulder_lift=-90deg, elbow_flex=90deg, wrist_flex=0, wrist_roll=-90deg
        init_positions_rad = np.array([0.0, -np.pi / 2, np.pi / 2, 0.0, -np.pi / 2, 0.0])
        for i, idx in enumerate(self.joint_indices):
            if i < len(init_positions_rad):
                self.mjdata.qpos[idx] = init_positions_rad[i]

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
                # Convert from radians to degrees
                pos_rad = self.mjdata.qpos[self.joint_indices[i]]
                obs[f"{joint_name}.pos"] = np.rad2deg(pos_rad)
        return obs

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send action to robot (set joint positions in degrees, convert to radians for MuJoCo)."""
        for key, value in action.items():
            # Handle both "motor_name.pos" and "motor_name" formats
            if key.endswith(".pos"):
                motor_name = key.removesuffix(".pos")
            else:
                motor_name = key

            if motor_name in self.joint_names:
                idx = self.joint_names.index(motor_name)
                if idx < len(self.joint_indices):
                    # Convert from degrees to radians and set position
                    self.mjdata.qpos[self.joint_indices[idx]] = np.deg2rad(float(value))

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

    display_len = max(len(key) for key in robot.action_features)
    start = time.perf_counter()
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

        # Convert EEF pose to robot action format
        # EEF data should contain position (x,y,z) and orientation (quaternion or euler)
        position = last_eef_data["position"]  # [x, y, z]
        orientation = last_eef_data["orientation"]  # quaternion [w,x,y,z] or euler [roll,pitch,yaw]

        # Create action dict in the format expected by the pipeline
        # This simulates what a teleoperator would provide
        ee_action = {
            "delta_x": 0.0,  # We'll use absolute positioning, not deltas
            "delta_y": 0.0,
            "delta_z": 0.0,
            "eef_pos_x": position[0],
            "eef_pos_y": position[1],
            "eef_pos_z": position[2],
            "eef_ori_roll": orientation[0] if len(orientation) == 3 else 0.0,  # Assume euler for now
            "eef_ori_pitch": orientation[1] if len(orientation) == 3 else 0.0,
            "eef_ori_yaw": orientation[2] if len(orientation) == 3 else 0.0,
            "gripper": 1.0,  # Default gripper state
        }

        # Process through IK pipeline: EEF pose -> robot joints
        robot_action = ee_to_joints_pipeline((ee_action, robot_obs))

        # Send action to MuJoCo robot
        robot.send_action(robot_action)

        # Log data to Rerun for visualization
        if enable_visualization:
            # Prepare observation data (joint positions)
            observation_data = {}
            for name in ik_joint_names:
                observation_data[f"{name}.pos"] = robot_obs.get(f"{name}.pos", 0.0)
            
            # Prepare action data (joint targets)
            action_data = {}
            for name in ik_joint_names:
                action_data[f"{name}.pos"] = robot_action.get(f"{name}.pos", 0.0)
            
            # Add EEF target and current position
            try:
                joint_positions_deg = np.array([
                    robot_obs.get(f"{name}.pos", 0.0) for name in ik_joint_names
                ])
                ee_transform = kinematics_solver.forward_kinematics(joint_positions_deg)
                ee_position = ee_transform[:3, 3]
                
                # Log EEF positions
                observation_data["eef_pos_x"] = ee_position[0]
                observation_data["eef_pos_y"] = ee_position[1]
                observation_data["eef_pos_z"] = ee_position[2]
                
                action_data["eef_target_x"] = position[0]
                action_data["eef_target_y"] = position[1]
                action_data["eef_target_z"] = position[2]
            except Exception as e:
                logging.debug(f"Error calculating EE position for visualization: {e}")
            
            # Log to Rerun
            log_rerun_data(observation=observation_data, action=action_data)

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

                # Print position info
                print(f"\r[EEF Control] Target: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} | "
                      f"Current: x={ee_position[0]:.3f}, y={ee_position[1]:.3f}, z={ee_position[2]:.3f} | "
                      f"Error: {np.linalg.norm(position - ee_position):.3f}m | "
                      f"Joints: {', '.join([f'{name}={robot_obs.get(f"{name}.pos", 0.0):.1f}°' for name in ik_joint_names])}",
                      end='', flush=True)
            except Exception as e:
                print(f"\r[EEF Control] Error calculating EE position: {e}", end='', flush=True)

        dt_s = time.perf_counter() - loop_start
        busy_wait(1 / fps - dt_s)


@parser.wrap()
def teleoperate_remote_sim(cfg: RemoteSimConfig):
    init_logging()
    logging.info(pformat(vars(cfg)))

    # Set up MuJoCo render backend (EGL for headless rendering)
    # This is needed even without visualization, as MuJoCo still needs to render internally
    if os.environ.get("MUJOCO_GL") != "egl":
        os.environ["MUJOCO_GL"] = "egl"
        logging.info("Set MUJOCO_GL=egl for offscreen rendering")
    
    # Initialize Rerun visualization if enabled
    if cfg.enable_visualization:
        logging.info("Initializing Rerun visualization...")
        try:
            # Check if DISPLAY is set (X11 forwarding available)
            has_display = os.getenv("DISPLAY") is not None
            
            if has_display:
                # Use local viewer with X11 forwarding
                init_rerun(session_name="remote_sim_teleoperation")
                logging.info("Rerun initialized with local viewer (X11 forwarding)")
            else:
                # Use web viewer for remote access
                batch_size = os.getenv("RERUN_FLUSH_NUM_BYTES", "8000")
                os.environ["RERUN_FLUSH_NUM_BYTES"] = batch_size
                rr.init("remote_sim_teleoperation", spawn=False)
                memory_limit = os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "10%")
                rr.serve_web_viewer(open_browser=False, web_port=cfg.rerun_web_port)
                logging.info(f"Rerun web viewer started on port {cfg.rerun_web_port}")
                logging.info(f"Connect from local machine: rerun ws://localhost:{cfg.rerun_web_port}")
                logging.info(f"Or forward port and access: http://localhost:{cfg.rerun_web_port}")
        except Exception as e:
            logging.error(f"Failed to initialize Rerun: {e}")
            logging.warning("Continuing without visualization. Set --enable_visualization=false to suppress this warning.")
            cfg.enable_visualization = False  
    # Configuration
    FPS = cfg.fps
    # Get paths relative to lerobot directory
    LEROBOT_DIR = Path(__file__).parent.parent.parent.parent
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

    # Create MuJoCo robot
    robot = MuJoCoSO101Robot(MUJOCO_XML_PATH, URDF_PATH)

    # Create kinematics solver
    print(f"Loading kinematics from URDF: {URDF_PATH}")
    ik_joint_names = [name for name in SO101_JOINT_NAMES if name != "gripper"]

    # Use absolute path for kinematics solver
    kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH.absolute()),  # Use absolute path
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Build IK processor pipeline
    ee_to_joints_pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            EEFToIKFormatProcessor(),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=SO101_JOINT_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Initialize ZMQ EEF receiver
    zmq_eef_receiver = ZmqEEFReceiver(
        host=cfg.zmq_host,
        port=cfg.zmq_port,
        timeout_s=cfg.zmq_timeout_s
    )
    zmq_eef_receiver.start()

    # Connect robot
    robot.connect()

    print("\n" + "=" * 60)
    print("Remote SO101 Simulation Teleoperation")
    print("=" * 60)
    print(f"Waiting for EEF data from ZMQ: tcp://{cfg.zmq_host}:{cfg.zmq_port}")
    if cfg.enable_visualization:
        has_display = os.getenv("DISPLAY") is not None
        if has_display:
            print("Visualization: Rerun viewer (X11 forwarding)")
        else:
            print(f"Visualization: Rerun web viewer on port {cfg.rerun_web_port}")
            print(f"  Connect with: rerun ws://localhost:{cfg.rerun_web_port}")
            print(f"  Or forward port and access: http://localhost:{cfg.rerun_web_port}")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            loop_start = time.perf_counter()
            
            # 执行仿真逻辑
            teleop_remote_sim_loop(
                robot=robot,
                zmq_eef_receiver=zmq_eef_receiver,
                kinematics_solver=kinematics_solver,
                ik_joint_names=ik_joint_names,
                ee_to_joints_pipeline=ee_to_joints_pipeline,
                fps=FPS,
                enable_visualization=cfg.enable_visualization,
            )
            
            # 控制循环帧率
            dt = time.perf_counter() - loop_start
            if dt < 1.0 / FPS:
                time.sleep(1.0 / FPS - dt)
                
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    finally:
        if cfg.enable_visualization:
            try:
                rr.rerun_shutdown()
            except:
                pass
        zmq_eef_receiver.stop()
        robot.disconnect()
        print("Cleanup completed.")

def main():
    teleoperate_remote_sim()


if __name__ == "__main__":
    main()
