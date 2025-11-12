# Robot Kinematic Processor 数据流分析

## 完整 Pipeline 数据流

```
ZMQ输入 (绝对位置)
  ↓
MapAbsoluteToDeltaEEF (绝对 → 相对增量)
  ↓
EEReferenceAndDelta (相对增量 → 绝对EEF位姿)
  ↓
EEBoundsAndSafety (安全检查)
  ↓
GripperVelocityToJoint (Gripper速度 → 位置)
  ↓
InverseKinematicsEEToJoints (EEF位姿 → 关节角度)
  ↓
输出 (关节角度)
```

---

## 1. MapAbsoluteToDeltaEEF

### 输入 (RobotAction):
```python
{
    "eef_pos_x": float,        # 绝对位置 X (米)
    "eef_pos_y": float,        # 绝对位置 Y (米)
    "eef_pos_z": float,        # 绝对位置 Z (米)
    "eef_ori_roll": float,     # 绝对旋转 Roll (弧度, Euler角)
    "eef_ori_pitch": float,    # 绝对旋转 Pitch (弧度, Euler角)
    "eef_ori_yaw": float,      # 绝对旋转 Yaw (弧度, Euler角)
    "gripper": float,          # Gripper绝对位置
}
```

### 处理过程:
1. 从 observation 获取当前关节角度
2. 使用正向运动学 (FK) 计算当前 EEF 位姿: `t_curr = FK(q_current)`
3. 计算位置增量: `delta_pos = target_pos - curr_pos`
4. 计算旋转增量: `delta_rot = curr_rot.inv() * target_rot` (转换为 rotvec)
5. 计算 gripper 增量并转换为速度: `gripper_vel = (target_gripper - curr_gripper) / speed_factor`

### 输出 (RobotAction):
```python
{
    "enabled": bool,           # 是否有显著运动
    "target_x": float,         # 相对位置增量 X (米)
    "target_y": float,         # 相对位置增量 Y (米)
    "target_z": float,         # 相对位置增量 Z (米)
    "target_wx": float,        # 相对旋转增量 X (弧度, rotvec)
    "target_wy": float,        # 相对旋转增量 Y (弧度, rotvec)
    "target_wz": float,        # 相对旋转增量 Z (弧度, rotvec)
    "gripper_vel": float,      # Gripper速度命令
}
```

### 关键逻辑:
- **位置增量**: `delta = target - current` (直接差值)
- **旋转增量**: `delta_rot = current_rot.inv() * target_rot` (相对旋转)
- **启用判断**: 如果位置或旋转变化 > 1e-3，则 `enabled=True`

---

## 2. EEReferenceAndDelta

### 输入 (RobotAction):
```python
{
    "enabled": bool,
    "target_x": float,         # 相对增量
    "target_y": float,
    "target_z": float,
    "target_wx": float,        # 相对旋转增量 (rotvec)
    "target_wy": float,
    "target_wz": float,
    "gripper_vel": float,
}
```

### 处理过程:
1. 从 observation 获取当前关节角度
2. 使用 FK 计算当前 EEF 位姿: `t_curr = FK(q_current)`
3. 根据 `use_latched_reference` 决定参考位姿:
   - **True**: 锁定参考（启用时锁定，后续相对此参考）
   - **False**: 每步使用当前位姿作为参考
4. 计算目标位姿:
   - 位置: `target_pos = ref_pos + delta_pos * step_size`
   - 旋转: `target_rot = ref_rot @ delta_rot_matrix`

### 输出 (RobotAction):
```python
{
    "ee.x": float,             # 绝对位置 X (米)
    "ee.y": float,             # 绝对位置 Y (米)
    "ee.z": float,             # 绝对位置 Z (米)
    "ee.wx": float,            # 绝对旋转 X (弧度, rotvec)
    "ee.wy": float,            # 绝对旋转 Y (弧度, rotvec)
    "ee.wz": float,            # 绝对旋转 Z (弧度, rotvec)
    "ee.gripper_vel": float,   # Gripper速度（保持不变）
}
```

### 关键参数:
- `end_effector_step_sizes`: 增量缩放因子
  - `{"x": 1.0, "y": 1.0, "z": 1.0}` = 不缩放
  - `{"x": 0.5, "y": 0.5, "z": 0.5}` = 缩放50%
- `use_latched_reference`:
  - **True**: 锁定参考（适合连续控制）
  - **False**: 动态参考（每步更新，更接近绝对控制）

### 数学公式:
```python
# 位置
delta_p = [tx * step_x, ty * step_y, tz * step_z]
target_pos = ref_pos + delta_p

# 旋转
delta_rot_matrix = Rotation.from_rotvec([wx, wy, wz]).as_matrix()
target_rot_matrix = ref_rot_matrix @ delta_rot_matrix
target_rotvec = Rotation.from_matrix(target_rot_matrix).as_rotvec()
```

---

## 3. EEBoundsAndSafety

### 输入 (RobotAction):
```python
{
    "ee.x": float,
    "ee.y": float,
    "ee.z": float,
    "ee.wx": float,
    "ee.wy": float,
    "ee.wz": float,
    "ee.gripper_vel": float,
}
```

### 处理过程:
1. **位置裁剪**: 限制在工作空间内
   ```python
   pos = np.clip(pos, bounds["min"], bounds["max"])
   ```

2. **步长限制**: 检查位置变化是否过大
   ```python
   if |pos - last_pos| > max_ee_step_m:
       # 限制步长
       pos = last_pos + (pos - last_pos) * (max_ee_step_m / |pos - last_pos|)
       raise ValueError("EE jump too large")
   ```

### 输出 (RobotAction):
```python
{
    "ee.x": float,      # 裁剪后的位置
    "ee.y": float,
    "ee.z": float,
    "ee.wx": float,     # 旋转（不变）
    "ee.wy": float,
    "ee.wz": float,
    "ee.gripper_vel": float,  # Gripper速度（不变）
}
```

### 关键参数:
- `end_effector_bounds`: 工作空间边界 `{"min": [x,y,z], "max": [x,y,z]}`
- `max_ee_step_m`: 最大允许步长（默认 0.05m = 5cm）

---

## 4. GripperVelocityToJoint

### 输入 (RobotAction):
```python
{
    "ee.gripper_vel": float,   # Gripper速度命令
    # ... 其他EEF字段
}
```

### 处理过程:
1. 从 observation 获取当前 gripper 位置
2. 计算新位置: `gripper_pos = current_pos + gripper_vel * speed_factor`
3. 裁剪到限制范围: `gripper_pos = clip(gripper_pos, clip_min, clip_max)`

### 输出 (RobotAction):
```python
{
    "ee.x": float,
    "ee.y": float,
    "ee.z": float,
    "ee.wx": float,
    "ee.wy": float,
    "ee.wz": float,
    "ee.gripper_pos": float,   # Gripper位置（从速度积分得到）
}
```

### 关键参数:
- `speed_factor`: 速度缩放因子（默认 20.0）
- `clip_min`, `clip_max`: Gripper位置限制

### 数学公式:
```python
delta = gripper_vel * speed_factor
gripper_pos = clip(current_pos + delta, clip_min, clip_max)
```

---

## 5. InverseKinematicsEEToJoints

### 输入 (RobotAction):
```python
{
    "ee.x": float,             # 绝对位置 X
    "ee.y": float,             # 绝对位置 Y
    "ee.z": float,             # 绝对位置 Z
    "ee.wx": float,            # 绝对旋转 X (rotvec)
    "ee.wy": float,            # 绝对旋转 Y (rotvec)
    "ee.wz": float,            # 绝对旋转 Z (rotvec)
    "ee.gripper_pos": float,   # Gripper位置
}
```

### 处理过程:
1. 从 observation 获取当前关节角度（作为IK初始猜测）
2. 构建目标变换矩阵:
   ```python
   t_des = [
       [R11, R12, R13, x],
       [R21, R22, R23, y],
       [R31, R32, R33, z],
       [0,   0,   0,   1 ]
   ]
   ```
   其中 `R = Rotation.from_rotvec([wx, wy, wz]).as_matrix()`
3. 调用 IK 求解器: `q_target = IK(q_curr, t_des)`
4. 将关节角度写入 action

### 输出 (RobotAction):
```python
{
    "shoulder_pan.pos": float,      # 关节角度 (度)
    "shoulder_lift.pos": float,
    "elbow_flex.pos": float,
    "wrist_flex.pos": float,
    "wrist_roll.pos": float,
    "gripper.pos": float,           # Gripper位置（直接传递）
}
```

### 关键参数:
- `initial_guess_current_joints`:
  - **True**: 使用当前关节角度（推荐，闭环控制）
  - **False**: 使用上次IK解（开环，适合回放）

---

## 完整数据流示例

### 输入 (ZMQ接收):
```python
{
    "eef_pos_x": 0.25,
    "eef_pos_y": 0.14,
    "eef_pos_z": 0.20,
    "eef_ori_roll": 0.1,
    "eef_ori_pitch": 1.57,
    "eef_ori_yaw": 0.0,
    "gripper": 0.8,
}
```

### Step 1: MapAbsoluteToDeltaEEF
假设当前位姿: `curr_pos = [0.20, 0.14, 0.18]`, `curr_rot = [0.0, 1.57, 0.0]`

```python
delta_pos = [0.25 - 0.20, 0.14 - 0.14, 0.20 - 0.18] = [0.05, 0.0, 0.02]
delta_rot = curr_rot.inv() * target_rot = [0.1, 0.0, 0.0] (approx)
gripper_delta = 0.8 - 0.5 = 0.3
gripper_vel = 0.3 / 20.0 = 0.015

输出: {
    "enabled": True,
    "target_x": 0.05,
    "target_y": 0.0,
    "target_z": 0.02,
    "target_wx": 0.1,
    "target_wy": 0.0,
    "target_wz": 0.0,
    "gripper_vel": 0.015,
}
```

### Step 2: EEReferenceAndDelta
假设 `use_latched_reference=False`, `step_sizes = {"x": 1.0, "y": 1.0, "z": 1.0}`

```python
ref_pos = curr_pos = [0.20, 0.14, 0.18]
ref_rot = curr_rot

delta_p = [0.05 * 1.0, 0.0 * 1.0, 0.02 * 1.0] = [0.05, 0.0, 0.02]
target_pos = ref_pos + delta_p = [0.25, 0.14, 0.20]

delta_rot_matrix = Rotation.from_rotvec([0.1, 0.0, 0.0]).as_matrix()
target_rot_matrix = ref_rot_matrix @ delta_rot_matrix
target_rotvec = Rotation.from_matrix(target_rot_matrix).as_rotvec()

输出: {
    "ee.x": 0.25,
    "ee.y": 0.14,
    "ee.z": 0.20,
    "ee.wx": ~0.1,
    "ee.wy": ~1.57,
    "ee.wz": ~0.0,
    "ee.gripper_vel": 0.015,
}
```

### Step 3: EEBoundsAndSafety
```python
# 裁剪位置到边界内
pos = clip([0.25, 0.14, 0.20], [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]) = [0.25, 0.14, 0.20]

# 检查步长（假设 last_pos = [0.20, 0.14, 0.18]）
step = |[0.25, 0.14, 0.20] - [0.20, 0.14, 0.18]| = |[0.05, 0.0, 0.02]| = 0.054m
# 0.054m < 0.10m (max_ee_step_m), 通过

输出: {
    "ee.x": 0.25,
    "ee.y": 0.14,
    "ee.z": 0.20,
    "ee.wx": ~0.1,
    "ee.wy": ~1.57,
    "ee.wz": ~0.0,
    "ee.gripper_vel": 0.015,
}
```

### Step 4: GripperVelocityToJoint
```python
# 假设当前 gripper_pos = 0.5
delta = 0.015 * 20.0 = 0.3
gripper_pos = clip(0.5 + 0.3, 0.0, 100.0) = 0.8

输出: {
    "ee.x": 0.25,
    "ee.y": 0.14,
    "ee.z": 0.20,
    "ee.wx": ~0.1,
    "ee.wy": ~1.57,
    "ee.wz": ~0.0,
    "ee.gripper_pos": 0.8,
}
```

### Step 5: InverseKinematicsEEToJoints
```python
# 构建目标变换矩阵
t_des = build_transform_matrix([0.25, 0.14, 0.20], [0.1, 1.57, 0.0])

# 调用IK求解器
q_target = IK(q_current, t_des)

输出: {
    "shoulder_pan.pos": 10.5,
    "shoulder_lift.pos": 45.2,
    "elbow_flex.pos": -30.1,
    "wrist_flex.pos": 15.8,
    "wrist_roll.pos": -90.0,
    "gripper.pos": 0.8,
}
```

---

## 关键设计决策

### 1. use_latched_reference 的选择

**True (锁定参考)**:
- 启用时锁定参考位姿
- 后续所有命令相对于此固定参考
- 适合: 连续控制，保持相对位置关系

**False (动态参考)**:
- 每步使用当前位姿作为参考
- 更接近绝对位置控制
- 适合: 从绝对位置转换的场景

### 2. end_effector_step_sizes 的作用

- 控制相对增量的缩放
- `1.0` = 不缩放，直接使用增量
- `< 1.0` = 缩小增量，使运动更平滑
- `> 1.0` = 放大增量，使运动更快

### 3. Gripper 处理

- 输入: 绝对位置
- MapAbsoluteToDeltaEEF: 计算 delta，转换为速度
- GripperVelocityToJoint: 积分速度得到位置
- 最终: 输出绝对位置

---

## 与 Phone/Keyboard 实现的对比

| 特性 | Phone/Keyboard | 当前实现 (ZMQ) |
|------|---------------|----------------|
| 输入格式 | 相对增量 | 绝对位置 → 相对增量 |
| MapAbsoluteToDeltaEEF | ❌ 不需要 | ✅ 需要（转换绝对→相对） |
| EEReferenceAndDelta | ✅ | ✅ |
| EEBoundsAndSafety | ✅ | ✅ |
| GripperVelocityToJoint | ✅ | ✅ |
| InverseKinematicsEEToJoints | ✅ | ✅ |

---

## 注意事项

1. **Gripper速度计算**: 需要与 `GripperVelocityToJoint` 的 `speed_factor` 匹配
2. **步长限制**: `EEBoundsAndSafety` 会限制位置变化，但不会限制旋转
3. **参考位姿**: `use_latched_reference=False` 时，每步都更新参考，行为更接近绝对控制
4. **IK初始猜测**: `initial_guess_current_joints=True` 使用当前关节角度，更适合闭环控制

