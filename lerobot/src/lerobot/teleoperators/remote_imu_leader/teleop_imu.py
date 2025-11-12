import logging
import time
import numpy as np
import math
from typing import Any

from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

from ..teleoperator import Teleoperator
from .configuration_imu import IMUEndEffectorTeleopConfig, IMUTeleopConfig
from .command_client import ZmqCommandClient

logger = logging.getLogger(__name__)


# imu_data_structure = {
#     "position": np.ndarray,
#     "rotation": np.ndarray, #[roll, pitch, yaw]
#     "gripper": np.float64,
#     "timestamp": np.float64,
# }



class IMUTeleop(Teleoperator):
    """
    通过 ZMQ 从远程服务器获取 IMU 数据的 Teleoperator。
    从服务器接收position, rotation, gripper数据。
    """
    config_class = IMUTeleopConfig
    name = "imu"

    def __init__(self, config: IMUTeleopConfig):
        super().__init__(config)
        self.config = config
        self.robot_type = config.type
        self.logs = {}
        self._zmq_client: ZmqCommandClient | None = None
        self._latest_command = None
        
        # 初始化默认状态（从配置中获取）
        self._initial_position = np.array(self.config.initial_position, dtype=np.float32)
        self._initial_rotation = np.array(self.config.initial_rotation, dtype=np.float32)
        self._initial_gripper = float(self.config.initial_gripper)

    @property
    def action_features(self) -> dict:
        return {
            "imu.pos": np.ndarray,
            "imu.rot": np.ndarray, #[roll, pitch, yaw]
            "imu.gripper": np.float64,
            "timestamp": np.float64,
        }

    @property
    def feedback_features(self) -> dict:
        # TODO 在这一部分加入video feedback 
        return {}

    @property
    def is_connected(self) -> bool:
        return self._zmq_client is not None

    @property
    def is_calibrated(self) -> bool:
        # IMU 不需要校准，始终返回 True
        return True

    def connect(self, calibrate: bool = True) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self} already connected")
        
        # 构建 ZMQ 服务器地址
        address = f"tcp://{self.config.server_b_host}:{self.config.server_b_port_command}"
        logger.info(f"连接到 IMU 服务器: {address}")
        
        try:
            # 默认使用 PULL 模式（与 C_real_video_reverse.py 兼容）
            use_pull_mode = getattr(self.config, 'use_pull_mode', True)
            self._zmq_client = ZmqCommandClient(address, use_pull_mode=use_pull_mode)
            logger.info(f"IMU 连接成功（模式: {'PULL' if use_pull_mode else 'REQ'}）")
        except Exception as e:
            logger.error(f"连接 IMU 服务器失败: {e}")
            raise

    def calibrate(self) -> None:
        # IMU 不需要校准
        pass

    def configure(self) -> None:
        pass

    def get_action(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "IMUTeleop is not connected. You need to run `connect()` before `get_action()`."
            )

        before_read_t = time.perf_counter()

        # 从 ZMQ 服务器获取最新的命令
        command = self._zmq_client.get_latest_command()
        
        if command is None:
            # 如果没有收到新数据，使用上次的命令
            if self._latest_command is None:
                # 如果从未收到过数据，返回初始状态
                position = self._initial_position.copy()
                rotation = self._initial_rotation.copy()
                gripper = self._initial_gripper
                timestamp = time.time()
            else:
                position = self._latest_command.get("position")
                rotation = self._latest_command.get("rotation")
                gripper = self._latest_command.get("gripper", self._initial_gripper)
                timestamp = self._latest_command.get("timestamp", time.time())
        else:
            # 更新最新命令
            self._latest_command = command
            position = command.get("position")
            rotation = command.get("rotation")
            gripper = command.get("gripper", self._initial_gripper)
            timestamp = command.get("timestamp", time.time())

        # 确保 position 是 numpy 数组，如果没有则使用初始值
        if position is None:
            position = self._initial_position.copy()
        elif not isinstance(position, np.ndarray):
            position = np.array(position, dtype=np.float32)
        else:
            position = position.astype(np.float32)

        # 确保 rotation 是 numpy 数组，如果没有则使用初始值
        if rotation is None:
            rotation = self._initial_rotation.copy()
        elif not isinstance(rotation, np.ndarray):
            rotation = np.array(rotation, dtype=np.float32)
        else:
            rotation = rotation.astype(np.float32)

        # 构建动作字典
        action = {
            "imu.pos": position,
            "imu.rot": rotation,
            "imu.gripper": np.float64(gripper),
            "timestamp": np.float64(timestamp),
        }

        self.logs["read_pos_dt_s"] = time.perf_counter() - before_read_t

        return action

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        # IMU teleoperator 不支持反馈
        pass

    def disconnect(self) -> None:
        if not self.is_connected:
            raise DeviceNotConnectedError(
                "IMUTeleop is not connected. You need to run `connect()` before `disconnect()`."
            )
        
        if self._zmq_client is not None:
            self._zmq_client.close()
            self._zmq_client = None
            logger.info("IMU 已断开连接")

