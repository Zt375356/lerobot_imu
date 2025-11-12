import json
import pickle
import traceback
import zmq
from typing import Any, Dict, Optional

from .config import DEFAULT_CMD_ADDR, REQUEST_PAYLOAD, REQUEST_TIMEOUT_MS

class TorchSerializer:
    @staticmethod
    def to_bytes(obj) -> bytes:
        # 将 Python 对象序列化为字节
        return pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def from_bytes(data: bytes):
        """
        将字节反序列化为 Python 对象
        支持 pickle 和 JSON 两种格式
        """
        # 首先尝试 pickle 反序列化
        try:
            return pickle.loads(data)
        except (pickle.UnpicklingError, ValueError, TypeError):
            # 如果 pickle 失败，尝试 JSON 反序列化
            try:
                # 尝试解码为字符串
                if isinstance(data, bytes):
                    data_str = data.decode('utf-8')
                else:
                    data_str = data
                return json.loads(data_str)
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                # 如果都失败，打印原始数据的前100个字符以便调试
                data_preview = str(data)[:100] if len(str(data)) > 100 else str(data)
                raise ValueError(
                    f"无法反序列化数据。既不是有效的 pickle 格式，也不是有效的 JSON 格式。\n"
                    f"数据预览: {data_preview}\n"
                    f"原始错误: {e}"
                )



class ZmqCommandClient:
    """
    ZMQ 命令客户端，支持两种模式：
    1. PULL 模式：被动接收服务器推送的命令（与 C_real_video_reverse.py 兼容）
    2. REQ 模式：主动请求服务器返回命令
    """
    def __init__(
        self, 
        address: str = DEFAULT_CMD_ADDR, 
        timeout_ms: int = REQUEST_TIMEOUT_MS,
        use_pull_mode: bool = True
    ) -> None:
        """
        初始化 ZMQ 客户端。
        
        Args:
            address: ZMQ 服务器地址
            timeout_ms: 超时时间（毫秒）
            use_pull_mode: 如果为 True，使用 PULL socket（被动接收）；如果为 False，使用 REQ socket（主动请求）
        """
        self._ctx = zmq.Context()
        self._timeout_ms = timeout_ms
        self._address = address
        self._use_pull_mode = use_pull_mode
        
        if use_pull_mode:
            # PULL 模式：被动接收服务器推送的命令
            self._sock = self._ctx.socket(zmq.PULL)
            # 设置 socket 的关闭行为为立即关闭，不等待未发送的消息（防止阻塞）
            self._sock.setsockopt(zmq.LINGER, 0)
            # 开启 CONFLATE，只保留最新的一条消息，避免消息堆积（适合仅关心最新状态的命令流）
            self._sock.setsockopt(zmq.CONFLATE, 1)
            self._sock.connect(address)
            print(f"[命令线程] 已连接到 B: {address}")
        else:
            # REQ 模式：主动请求服务器返回命令
            self._sock = self._ctx.socket(zmq.REQ)
            self._sock.setsockopt(zmq.LINGER, 0)
            self._sock.connect(address)
            self._poller = zmq.Poller()
            self._poller.register(self._sock, zmq.POLLIN)

    def get_latest_command(self, payload: bytes = REQUEST_PAYLOAD) -> Optional[Dict[str, Any]]:
        """
        获取最新的命令。
        
        Args:
            payload: 请求负载（仅在 REQ 模式下使用）
            
        Returns:
            命令字典，如果超时或出错则返回 None
        """
        try:
            # REQ 模式：先发送请求
            if not self._use_pull_mode:
                self._sock.send(payload)
            
            # 接收数据（PULL 和 REQ 模式都需要）
            data = self._sock.recv()
            # print(f"[命令线程] 接收到原始消息: {data}")
            
            # 反序列化命令
            command = TorchSerializer.from_bytes(data)
            # print(f"[命令线程] 接收到命令: {command}")
            return command
            
        except zmq.Again:
            # 非阻塞模式下没有数据可读
            print(f"[命令线程] 警告: 接收超时或没有数据可读")
            return None
        except Exception as e:
            # 其他错误，打印详细错误信息以便调试
            print(f"[命令线程] 错误: 获取命令时发生异常: {type(e).__name__}: {e}")
            traceback.print_exc()
            return None

    def close(self) -> None:
        """关闭连接并清理资源"""
        try:
            if self._sock is not None:
                self._sock.close()
        finally:
            if self._ctx is not None:
                self._ctx.term()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
