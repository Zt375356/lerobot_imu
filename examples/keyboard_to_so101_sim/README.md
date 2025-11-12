# Keyboard Teleoperation of SO101 in MuJoCo Simulation

This example demonstrates how to control a SO101 robot arm in MuJoCo simulation using keyboard commands with end-effector (EE) control. The script uses inverse kinematics to automatically convert end-effector position commands into joint angles.

## Requirements

- `mujoco` - MuJoCo physics simulator
- `lerobot` - LeRobot framework
- `placo` - For inverse kinematics (URDF-based)

## Files

- `teleoperate.py` - Main teleoperation script

## Usage

Run the teleoperation script:

```bash
cd lerobot/examples/keyboard_to_so101_sim
python teleoperate.py
```

## Remote Visualization

If you're running on a remote server without a display, you can visualize the simulation in several ways:

### Option 1: VSCode Remote (Easiest)

If you're using VSCode with Remote SSH extension:

1. **Run the script** - it will start Rerun on port 9876
2. **VSCode will automatically detect** the port and show a notification, OR:
3. **Manually forward the port**:
   - Open the **Ports** tab at the bottom of VSCode
   - Click **"Forward a Port"** button (or press `Ctrl+Shift+P` and search "Ports: Focus on Ports View")
   - Enter `9876` and press Enter
4. **Open in browser**:
   - Right-click port 9876 → **"Open in Browser"**
   - Or click the browser icon next to the port
   - The Rerun web interface will open automatically

**Quick Steps:**
- Look for the **Ports** tab at the bottom of VSCode
- Find port **9876** (it may auto-appear)
- Right-click → **"Open in Browser"**

### Option 2: Rerun Command Line

1. **On the remote server**, run the script - it will print connection info
2. **On your local machine**, install Rerun viewer:
   ```bash
   pip install rerun-sdk
   ```
3. **Set up SSH port forwarding**:
   ```bash
   ssh -L 9876:localhost:9876 user@hostname
   ```
4. **Connect to the remote server**:
   ```bash
   rerun --connect rerun+http://127.0.0.1:9876/proxy
   ```

### Option 3: X11 Forwarding (for MuJoCo Viewer)

To see the MuJoCo 3D viewer window:

1. **SSH with X11 forwarding**:
   ```bash
   ssh -X user@hostname
   # Or for trusted X11:
   ssh -Y user@hostname
   ```

2. **Set DISPLAY** (if needed):
   ```bash
   export DISPLAY=:0
   ```

3. Run the script - the MuJoCo viewer window will appear on your local machine

**Note**: X11 forwarding may not work well with VSCode Remote. Use Rerun web interface instead.

## Keyboard Controls

- **w/s**: Move end-effector forward/backward (X axis)
- **a/d**: Move end-effector left/right (Y axis)
- **r/f**: Move end-effector up/down (Z axis)
- **q/e**: Rotate end-effector roll +/-
- **g/t**: Rotate end-effector pitch +/-
- **z/c**: Open/close gripper
- **0**: Reset to initial position
- **Ctrl+C**: Exit

## How It Works

1. **MuJoCo Simulation**: The script creates a MuJoCo simulation environment using the SO101 model from `scene_so101.xml`.

2. **Keyboard Teleoperator**: Uses LeRobot's `KeyboardEndEffectorTeleop` to capture keyboard inputs and convert them to end-effector delta commands.

3. **Inverse Kinematics Pipeline**: The processor pipeline includes:
   - `MapKeyboardGripperToDiscrete`: Converts keyboard gripper format
   - `MapDeltaActionToRobotActionStep`: Converts delta actions to robot format
   - `EEReferenceAndDelta`: Computes target end-effector pose from delta and current pose
   - `EEBoundsAndSafety`: Applies safety bounds and checks for large jumps
   - `GripperVelocityToJoint`: Converts gripper velocity to joint position
   - `InverseKinematicsEEToJoints`: Solves IK to get joint positions

4. **Visualization**: 
   - MuJoCo viewer shows the robot simulation in real-time
   - Rerun viewer (optional) logs the teleoperation data

## Configuration

The script uses the following default paths (relative to the lerobot root):
- URDF: `lerobot-kinematics/examples/so101_new_calib.urdf`
- MuJoCo XML: `lerobot-kinematics/examples/scene_so101.xml`

You can modify these paths in the script if needed.

## Notes

- The simulation runs at 30 FPS by default
- End-effector step sizes are set to 0.01m (1cm) per key press
- Safety bounds limit the workspace to [-0.5, 0.5]m in X/Y and [0.0, 0.5]m in Z
- Maximum end-effector step size is limited to 0.05m for safety

