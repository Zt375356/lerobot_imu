## Reinforcenment Learning (RL)

You can try [lerobot-sim2real (by Stone Tao)](https://github.com/StoneT2000/lerobot-sim2real) with Maniskill, or [huggingface official tutorial on HIL-SERL](https://huggingface.co/docs/lerobot/hilserl) on single SO101 arm first. The offcial code for complete XLeRobot RL is coming soon. The demo below shows the implementation of [lerobot-sim2real](https://github.com/StoneT2000/lerobot-sim2real), with minor changes to the camera direction and sim-object distribution. 


<video width="100%" controls>
  <source src="https://vector-wangel.github.io/XLeRobot-assets/videos/Real_demos/sim2real_2.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

## VLA

You can follow [huggingface official VLA tutorial](https://huggingface.co/docs/lerobot/smolvla) on single SO101 arm first.  

# Vision-Language-Action (VLA) Training for XLeRobot

This tutorial will guide you through the complete process of training a Vision-Language-Action (VLA) model to control your XLeRobot autonomously using imitation learning. This isn't the official tutorial for the XLeRobot VLA part. If you have any questions, please feel free to open an issue.

## What You'll Learn

1. How to teleoperate and record demonstration datasets for XLeRobot
2. How to train and evaluate your policy on real-world tasks
3. How to make the policy work effectively

By following these steps, you'll be able to train your XLeRobot to autonomously perform various household tasks using LeRobot policies (such as ACT), including picking up objects, wiping tables, or organizing items.

---

## Table of Contents

1. [Hardware Setup and Check](#1-hardware-setup-and-check)
2. [Record Dataset for XLeRobot with VR](#2-record-dataset-for-xlerobot-with-vr)
3. [Tips for Better Performance](#3-tips-for-better-performance)
4. [Train and Deploy the Model](#4-train-and-deploy-the-model)

---

## 1. Hardware Setup and Check

### 1.1 Verify Motor and Controller Connections

Ensure that all motors and controllers are properly connected by running the test script:

```bash
# Test script from XLeRobot repository
python examples/4_xlerobot_teleop_keyboard.py
```

### 1.2 Check Camera Status

Use the LeRobot [cameras](https://huggingface.co/docs/lerobot/cameras) tutorial to verify your camera setup:

```bash
lerobot-find-cameras opencv  # or use 'realsense' for Intel RealSense cameras
```

You should see output similar to this:

```
--- Detected Cameras ---
Camera #0:
  Name: OpenCV Camera @ 0
  Type: OpenCV
  Id: 0
  Backend api: AVFOUNDATION
  Default stream profile:
    Format: 16.0
    Width: 1920
    Height: 1080
    Fps: 15.0
--------------------
(more cameras ...)
```

XLeRobot has three cameras: two wrist cameras and one head camera. Make sure all three are detected.

### 1.3 Verify VR Device Network Connection

Ensure your VR device is connected to the same WLAN as your host computer.

---

## 2. Record Dataset for XLeRobot with VR

Before the official merge into LeRobot, copy the required code from XLeRobot to LeRobot:
```bash
```bash
cp your_dir/XLeRobot/software/src/record.py your_dir/lerobot/src/lerobot/record.py
cp your_dir/XLeRobot/software/src/teleporators/xlerobot_vr your_dir/lerobot/src/lerobot/teleporators/xlerobot_vr -r
```

### Recording Script Example

Run the following script to start recording:

```bash
python /your_dir/lerobot/src/lerobot/record.py \
  --robot.type=xlerobot \
  --robot.cameras="{ head: {type: intelrealsense, serial_number_or_name: 935422072196, width: 640, height: 480, fps: 30, use_depth: True}, right: {type: opencv, index_or_path: '/dev/video6', width: 640, height: 480, fps: 20}, left: {type: opencv, index_or_path: '/dev/video8', width: 640, height: 480, fps: 20} }" \
  --dataset.repo_id=your_huggingface_id/clear_table \
  --dataset.single_task="Clear the table." \
  --dataset.root=your_dir/clear_table \
  --display_data=true \
  --teleop.type=xlerobot_vr
```

### Important Notes

1. **Camera Configuration**: The `robot.cameras` parameter should match the output from section 1.2. If you encounter camera timeout errors, reduce the FPS (e.g., from 30 to 20).

2. **VR Connection**: At the beginning of the script, it will wait for VR connection. Visit the URL displayed in the terminal output using your VR device. Data collection will start automatically once the connection is established.

3. **VR Controls**: The left-hand controller has four functions (practice several times to become familiar):
   - **Reset Position**: Returns the robot arms to their zero position
   - **Early Exit**: Ends the current episode collection (use when you've completed the task)
   - **Delete Episode**: Deletes the current episode (use if the task failed)
   - **Stop Recording**: Stops the dataset recording session

<p align="center">
  <img src="https://github.com/user-attachments/assets/4b9004d7-6d4c-47c6-9d87-043e2a120bad" width="45%">
  <img src="https://github.com/user-attachments/assets/b2bddd83-1a95-4aee-bbb5-4f13e927f7c7" width="45%">
</p>

---

## 3. Tips for Better Performance

1. **Check for Dropped Frames**: Review [these examples](https://gold-column-7d2.notion.site/Some-examples-for-VLA-dataset-2a2e20e657ad8037aa09d1228a2bf4bf?pvs=73) to understand what dropped frames look like. Monitor your bandwidth and CPU usage during recording. If issues occur, optimize your system accordingly.

2. **Avoid Redundant Frames**: Use the early exit function when the task is completed, rather than letting the script continue logging static robot data.

3. **Maintain Object Visibility**: Ensure objects stay within the camera's field of view. Randomize object positions and collect more than 50 episodes for robust training.

4. **Maintain Scene Consistency**: Avoid having additional moving objects or people in the camera view during recording.
---

## 4. Train and Deploy the Model

After collecting your datasets, refer to the LeRobot [training tutorial](https://huggingface.co/docs/lerobot/il_robots) to select and train a policy.

### Deployment Script Example

Use the following script to deploy your trained model:
```
The script is like:
```python
python /your_dir/lerobot/src/lerobot/record.py \
  --robot.type=xlerobot \
  --robot.cameras="{ head: {type: intelrealsense, serial_number_or_name: 935422072196, width: 640, height: 480, fps: 30, use_depth: True}, right: {type: opencv, index_or_path: '/dev/video6', width: 640, height: 480, fps: 20}, left: {type: opencv, index_or_path: '/dev/video8', width: 640, height: 480, fps: 20} }" \
  --dataset.repo_id=your_huggingface_id/clear_table \
  --dataset.single_task="Clear the table." \
  --dataset.root=your_dir/clear_table \
  --display_data=true \
  --teleop.type=xlerobot_vr
```