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
Keyboard teleoperation of SO101 robot in MuJoCo simulation with end-effector control.

This script allows you to control a SO101 robot in MuJoCo simulation using keyboard
commands. The control is done at the end-effector level, with inverse kinematics
automatically converting end-effector commands to joint angles.

Usage:
    python teleoperate.py

Keyboard controls:
    - w/s: Move end-effector forward/backward (X axis)
    - a/d: Move end-effector left/right (Y axis)
    - r/f: Move end-effector up/down (Z axis)
    - q/e: Rotate end-effector roll +/-
    - g/t: Rotate end-effector pitch +/-
    - z/c: Open/close gripper
    - 0: Reset to initial position
"""

import os
import sys
import time
import termios
import tty
import select
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import scipy.spatial.transform as st

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
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardEndEffectorTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardEndEffectorTeleop
from lerobot.utils.robot_utils import busy_wait

# Set up MuJoCo render backend
# Only use EGL for headless mode, otherwise use default (glfw) which works better with X11
if "MUJOCO_GL" not in os.environ:
    # Check if we're in headless mode
    has_display = os.getenv("DISPLAY") is not None or os.getenv("WAYLAND_DISPLAY") is not None
    if not has_display:
        # Headless mode: use EGL for offscreen rendering
        os.environ["MUJOCO_GL"] = "egl"
    # If display is available, don't set MUJOCO_GL - let MuJoCo use default (glfw)
    # which handles X11 forwarding better

# Configuration
FPS = 30
# Get paths relative to lerobot examples directory (2 levels up from this file)
LEROBOT_EXAMPLES = Path(__file__).parent.parent.parent
SO101_DIR = LEROBOT_EXAMPLES / "examples" / "SO101"
URDF_PATH = SO101_DIR / "so101_new_calib.urdf"
MUJOCO_XML_PATH = SO101_DIR / "scene_so101.xml"

# Motor names for LeRobot (matching SO101 follower)
SO101_JOINT_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


class HeadlessKeyboardInput:
    """Simple keyboard input handler for headless mode using terminal input."""
    
    def __init__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        self.pressed_keys = {}
        self.setup_terminal()
    
    def setup_terminal(self):
        """Set terminal to raw mode for non-blocking input."""
        tty.setraw(self.fd)
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    
    def restore_terminal(self):
        """Restore terminal to normal mode."""
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self):
        """Get a single key press (non-blocking)."""
        if select.select([sys.stdin], [], [], 0)[0]:
            key = sys.stdin.read(1)
            return key
        return None
    
    def get_action(self):
        """Get keyboard action in the same format as KeyboardEndEffectorTeleop."""
        # Read available keys
        while True:
            key = self.get_key()
            if key is None:
                break
            
            # Handle special keys
            if key == '\x03':  # Ctrl+C
                raise KeyboardInterrupt()
            elif key == '\x1b':  # ESC sequence start
                # Try to read more for arrow keys
                if select.select([sys.stdin], [], [], 0.01)[0]:
                    seq = sys.stdin.read(2)
                    if seq == '[A':  # Up arrow -> w
                        self.pressed_keys['w'] = True
                    elif seq == '[B':  # Down arrow -> s
                        self.pressed_keys['s'] = True
                    elif seq == '[C':  # Right arrow -> d
                        self.pressed_keys['d'] = True
                    elif seq == '[D':  # Left arrow -> a
                        self.pressed_keys['a'] = True
                else:
                    # Just ESC
                    pass
            else:
                # Regular character keys
                key_lower = key.lower()
                if key_lower in ['w', 's', 'a', 'd', 'r', 'f', 'q', 'e', 'g', 't', 'z', 'c', '0']:
                    self.pressed_keys[key_lower] = True
        
        # Generate action based on pressed keys
        # Match KeyboardEndEffectorTeleop format: delta_x, delta_y, delta_z, gripper
        delta_x = 0.0
        delta_y = 0.0
        delta_z = 0.0
        gripper_action = 1.0  # Default: stay
        
        # Map keys to actions (matching KeyboardEndEffectorTeleop behavior)
        # Note: KeyboardEndEffectorTeleop uses arrow keys, but we use WASD for compatibility
        if self.pressed_keys.get('w', False):
            delta_y = -1.0  # Forward (matches up arrow)
        if self.pressed_keys.get('s', False):
            delta_y = 1.0  # Backward (matches down arrow)
        if self.pressed_keys.get('a', False):
            delta_x = 1.0  # Left (matches left arrow)
        if self.pressed_keys.get('d', False):
            delta_x = -1.0  # Right (matches right arrow)
        if self.pressed_keys.get('r', False):
            delta_z = 1.0  # Up (matches shift_r)
        if self.pressed_keys.get('f', False):
            delta_z = -1.0  # Down (matches shift)
        if self.pressed_keys.get('z', False):
            gripper_action = 0.0  # Close
        if self.pressed_keys.get('c', False):
            gripper_action = 2.0  # Open
        
        # Clear pressed keys (they are momentary, not held)
        self.pressed_keys.clear()
        
        return {
            "delta_x": delta_x,
            "delta_y": delta_y,
            "delta_z": delta_z,
            "gripper": gripper_action,
        }
    
    def disconnect(self):
        """Clean up terminal settings."""
        self.restore_terminal()


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
        original_cwd = os.getcwd()
        try:
            os.chdir(str(xml_dir))
            self.mjmodel = mujoco.MjModel.from_xml_path(str(xml_path.name))
        finally:
            os.chdir(original_cwd)
        
        self.mjdata = mujoco.MjData(self.mjmodel)

        # Get joint indices in MuJoCo
        self.joint_indices = []
        self.joint_names = []
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


def main():
    """Main teleoperation loop."""
    print("Initializing MuJoCo SO101 simulation...")
    
    # Initialize headless keyboard (will be set if needed)
    headless_keyboard = None

    # Check if files exist
    if not MUJOCO_XML_PATH.exists():
        raise FileNotFoundError(f"MuJoCo XML file not found: {MUJOCO_XML_PATH}")
    if not URDF_PATH.exists():
        raise FileNotFoundError(f"URDF file not found: {URDF_PATH}")

    # Create MuJoCo robot
    robot = MuJoCoSO101Robot(MUJOCO_XML_PATH, URDF_PATH)

    # Create keyboard teleoperator
    teleop_config = KeyboardEndEffectorTeleopConfig(id="keyboard_control")
    teleop = KeyboardEndEffectorTeleop(teleop_config)

    # Create kinematics solver
    print(f"Loading kinematics from URDF: {URDF_PATH}")
    ik_joint_names = [name for name in SO101_JOINT_NAMES if name != "gripper"]
    
    # placo resolves relative mesh paths in URDF relative to the URDF file's directory
    # So we use absolute path, and placo will automatically use URDF's directory as base
    kinematics_solver = RobotKinematics(
        urdf_path=str(URDF_PATH.absolute()),  # Use absolute path - placo resolves relative paths from URDF's directory
        target_frame_name="gripper_frame_link",
        joint_names=ik_joint_names,
    )

    # Build processor pipeline for keyboard EE -> robot joints
    # Step 1: Map keyboard_ee gripper format
    @ProcessorStepRegistry.register("map_keyboard_gripper_to_discrete")
    class MapKeyboardGripperToDiscrete(RobotActionProcessorStep):
        """Maps keyboard_ee gripper values to discrete gripper format."""

        def action(self, action: RobotAction) -> RobotAction:
            if "gripper" in action:
                gripper_val = action["gripper"]
                # Map: 0(close)->1, 1(stay)->2, 2(open)->0
                if gripper_val == 0:  # close -> 1
                    action["gripper"] = 1
                elif gripper_val == 1:  # stay -> 2
                    action["gripper"] = 2
                elif gripper_val == 2:  # open -> 0
                    action["gripper"] = 0
            return action
        
        def transform_features(
            self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
        ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
            """Transform features - gripper feature remains the same."""
            # This step doesn't change the feature structure, just the values
            return features

    # Build IK processor pipeline
    ee_to_joints_pipeline = RobotProcessorPipeline[tuple[RobotAction, RobotObservation], RobotAction](
        steps=[
            MapKeyboardGripperToDiscrete(),
            MapDeltaActionToRobotActionStep(position_scale=1.0, noise_threshold=1e-3),
            EEReferenceAndDelta(
                kinematics=kinematics_solver,
                end_effector_step_sizes={"x": 0.01, "y": 0.01, "z": 0.01},
                motor_names=ik_joint_names,
                use_latched_reference=True,
            ),
            EEBoundsAndSafety(
                end_effector_bounds={"min": [-0.5, -0.5, 0.0], "max": [0.5, 0.5, 0.5]},
                max_ee_step_m=0.05,
            ),
            GripperVelocityToJoint(speed_factor=20.0, discrete_gripper=True),
            InverseKinematicsEEToJoints(
                kinematics=kinematics_solver,
                motor_names=SO101_JOINT_NAMES,
                initial_guess_current_joints=True,
            ),
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    # Connect
    robot.connect()
    
    # Check if keyboard teleop can connect (requires display/pynput)
    # Note: connect() may succeed but is_connected may still be False if pynput is not available
    try:
        teleop.connect()
        # Check if actually connected (pynput may not be available even if connect() succeeds)
        keyboard_available = teleop.is_connected
        if not keyboard_available:
            print("\n" + "="*60)
            print("Warning: Keyboard input is not available!")
            print("This may be because:")
            print("  1. DISPLAY environment variable is not set")
            print("  2. pynput library cannot access the display")
            print("\nTo fix:")
            print("  - Set DISPLAY: export DISPLAY=:0")
            print("  - Or use X11 forwarding: ssh -X user@host")
            print("  - Or use VSCode's built-in terminal (may have display access)")
            print("="*60 + "\n")
            print("Simulation will run but robot will remain stationary without keyboard input.")
    except Exception as e:
        print(f"\nWarning: Keyboard teleop could not connect: {e}")
        print("Keyboard input may not work.")
        keyboard_available = False

    # MuJoCo visualization will be handled by the viewer

    print("\n" + "=" * 60)
    print("Keyboard Controls:")
    print("  w/s: Move end-effector forward/backward (X axis)")
    print("  a/d: Move end-effector left/right (Y axis)")
    print("  r/f: Move end-effector up/down (Z axis)")
    print("  q/e: Rotate end-effector roll +/-")
    print("  g/t: Rotate end-effector pitch +/-")
    print("  z/c: Open/close gripper")
    print("  0: Reset to initial position")
    print("=" * 60)
    print("\nStarting teleoperation loop...")
    print("Press Ctrl+C to exit\n")

    # Check if display is available
    has_display = os.getenv("DISPLAY") is not None or os.getenv("WAYLAND_DISPLAY") is not None
    
    if not has_display:
        print("\n" + "="*60)
        print("Warning: No display detected!")
        print("MuJoCo viewer requires a display to visualize the simulation.")
        print("\nTo enable visualization:")
        print("  1. Use X11 forwarding: ssh -X user@host (or ssh -Y for trusted X11)")
        print("  2. Or set DISPLAY environment variable: export DISPLAY=:0")
        print("  3. Or use VNC/Xvfb for remote display")
        print("  4. Or use virtual display: Xvfb :99 -screen 0 1024x768x24 & export DISPLAY=:99")
        print("="*60 + "\n")
        print("Simulation will run but you won't see the visualization without a display.")
        print("Press Ctrl+C to exit or set up display forwarding.\n")
    
    # Run simulation loop
    # Position output frequency (every N frames)
    position_output_interval = max(1, int(FPS / 2))  # Output every 0.5 seconds
    frame_count = 0
    
    try:
        if has_display:
            # Launch MuJoCo viewer if display is available
            with mujoco.viewer.launch_passive(robot.mjmodel, robot.mjdata) as viewer:
                start_time = time.perf_counter()

                while viewer.is_running():
                    loop_start = time.perf_counter()
                    frame_count += 1

                    # Get robot observation
                    robot_obs = robot.get_observation()

                    # Get keyboard action
                    if keyboard_available:
                        if headless_keyboard is not None:
                            raw_action = headless_keyboard.get_action()
                        else:
                            raw_action = teleop.get_action()
                    else:
                        # Fallback if keyboard not available
                        raw_action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

                    # Process through IK pipeline: keyboard EE -> robot joints
                    robot_action = ee_to_joints_pipeline((raw_action, robot_obs))

                    # Send action to MuJoCo
                    robot.send_action(robot_action)

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
                            print(f"\r[Position] EE: x={ee_position[0]:.3f}, y={ee_position[1]:.3f}, z={ee_position[2]:.3f} | "
                                  f"Roll={euler_angles[0]:.1f}°, Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}° | "
                                  f"Joints: {', '.join([f'{name}={robot_obs.get(f"{name}.pos", 0.0):.1f}°' for name in ik_joint_names])}",
                                  end='', flush=True)
                        except Exception as e:
                            print(f"\r[Position] Error calculating EE position: {e}", end='', flush=True)

                    # Sync viewer
                    with viewer.lock():
                        viewer.sync()

                    # Maintain FPS
                    dt = time.perf_counter() - loop_start
                    busy_wait(max(1.0 / FPS - dt, 0.0))
        else:
            # Headless mode - run simulation without viewer
            print("Running simulation loop (headless mode - no visualization)...")
            print("Note: To see the visualization, set up display forwarding (see instructions above).")
            start_time = time.perf_counter()
            
            while True:
                loop_start = time.perf_counter()
                frame_count += 1

                # Get robot observation
                robot_obs = robot.get_observation()

                # Get keyboard action
                if keyboard_available:
                    if headless_keyboard is not None:
                        raw_action = headless_keyboard.get_action()
                    else:
                        raw_action = teleop.get_action()
                else:
                    # In headless mode without keyboard, use zero action
                    raw_action = {"delta_x": 0.0, "delta_y": 0.0, "delta_z": 0.0, "gripper": 1.0}

                # Process through IK pipeline: keyboard EE -> robot joints
                robot_action = ee_to_joints_pipeline((raw_action, robot_obs))

                # Send action to MuJoCo
                robot.send_action(robot_action)

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
                        
                        # Convert rotation matrix to Euler angles (ZYX order)
                        import scipy.spatial.transform as st
                        euler_angles = st.Rotation.from_matrix(ee_rotation).as_euler('xyz', degrees=True)
                        
                        # Print position info
                        print(f"\r[Position] EE: x={ee_position[0]:.3f}, y={ee_position[1]:.3f}, z={ee_position[2]:.3f} | "
                              f"Roll={euler_angles[0]:.1f}°, Pitch={euler_angles[1]:.1f}°, Yaw={euler_angles[2]:.1f}° | "
                              f"Joints: {', '.join([f'{name}={robot_obs.get(f"{name}.pos", 0.0):.1f}°' for name in ik_joint_names])}",
                              end='', flush=True)
                    except Exception as e:
                        print(f"\r[Position] Error calculating EE position: {e}", end='', flush=True)

                # Maintain FPS
                dt = time.perf_counter() - loop_start
                busy_wait(max(1.0 / FPS - dt, 0.0))

    except KeyboardInterrupt:
        print("\n\nTeleoperation interrupted by user.")
    finally:
        # Print newline to clear the position output line
        print()
        if keyboard_available:
            if headless_keyboard is not None:
                headless_keyboard.disconnect()
            else:
                try:
                    teleop.disconnect()
                except Exception:
                    pass  # Ignore disconnect errors if not connected
        robot.disconnect()
        print("Disconnected from simulation.")


if __name__ == "__main__":
    main()

