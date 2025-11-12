from dataclasses import dataclass, field
from typing import List

from ..config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("imu")
@dataclass
class IMUTeleopConfig(TeleoperatorConfig):
    """
    IMU Teleoperator 配置类。
    配置 ZMQ 服务器连接参数，用于从远程服务器获取 IMU 数据。
    """
    # ZMQ 服务器配置
    server_b_host: str = "localhost"  # 服务器地址（可通过 SSH 隧道连接）
    server_b_port_command: int = 5556  # 命令端口，用于接收控制命令
    use_pull_mode: bool = True  # 如果为 True，使用 PULL socket（被动接收）；如果为 False，使用 REQ socket（主动请求）
    
    # 初始状态配置
    initial_position: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 初始位置 [x, y, z]
    initial_rotation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # 初始旋转 [roll, pitch, yaw]（度）
    initial_gripper: float = 0.0  # 初始 gripper 值


@TeleoperatorConfig.register_subclass("imu_ee")
@dataclass
class IMUEndEffectorTeleopConfig(IMUTeleopConfig):
    """
    支持末端执行器的 IMU Teleoperator 配置类。
    """
    use_gripper: bool = True  # 是否使用 gripper 控制
