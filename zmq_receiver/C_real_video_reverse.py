# C_real_video_reverse.py - 反向连接版本（C 主动连接 B）
import time
import threading
import logging
from datetime import datetime
import pickle
import json
import zmq
import cv2
import numpy as np

# --- 配置 ---
# B 服务器的地址（C 需要主动连接）
# 使用 SSH 隧道连接跳板机 Docker 容器
SERVER_B_HOST = "localhost"  # 通过 SSH 隧道连接
SERVER_B_PORT_COMMAND = 5556  # 连接到 B 接收控制命令
SERVER_B_PORT_VIDEO = 5558    # 连接到 B 推送视频流

# 摄像头配置
CAMERA_ID = 0  # 默认摄像头，可以改为视频文件路径
VIDEO_FPS = 30  # 目标帧率
VIDEO_WIDTH = 640  # 视频宽度
VIDEO_HEIGHT = 480  # 视频高度
JPEG_QUALITY = 80  # JPEG 压缩质量 (1-100)
# ------------

# 全局状态：存储最新的控制命令
latest_command = {
    "euler_angles": {"roll": 0, "pitch": 0, "yaw": 0},
    "throttle": 0,
    "timestamp": time.time()
}
command_lock = threading.Lock()

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


def thread_receive_commands():
    """
    线程1：接收来自 B 的控制命令
    反向模式：C 主动连接到 B
    """
    context = zmq.Context()
    
    # 主动连接到 B（PULL socket - 拉取命令）
    socket = context.socket(zmq.PULL)
    socket.connect(f"tcp://{SERVER_B_HOST}:{SERVER_B_PORT_COMMAND}")
    
    print(f"[命令线程] 已连接到 B: {SERVER_B_HOST}:{SERVER_B_PORT_COMMAND}")
    print(f"[命令线程] 等待接收控制命令...")
    
    try:
        while True:
            # 接收命令
            message = socket.recv()
            
            # 打印原始消息的前100个字符以便调试
            try:
                message_preview = message[:100] if len(message) > 100 else message
                if isinstance(message_preview, bytes):
                    message_preview = message_preview.decode('utf-8', errors='replace')
                logging.debug(f"[命令线程] 接收到原始消息 (前100字符): {message_preview}")
            except:
                pass
                
            try:
                command = TorchSerializer.from_bytes(message)
                with command_lock:
                    latest_command.update(command)
                    latest_command["timestamp"] = time.time()
                print(f"\n[命令] 从 B 接收命令：{command}")
            except:
                pass
    except Exception as e:
        print(f"[命令线程] 错误: {e}")
        import traceback
        traceback.print_exc()
    finally:
        socket.close()
        context.term()


def thread_send_video():
    """
    线程2：采集并发送视频流给 B
    反向模式：C 主动连接到 B 并推送视频
    """
    context = zmq.Context()
    
    # 主动连接到 B（PUSH socket - 推送视频）
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.SNDHWM, 1)  # 发送缓冲区只保留1帧，减少延迟
    socket.setsockopt(zmq.LINGER, 0)   # 关闭时不等待
    socket.connect(f"tcp://{SERVER_B_HOST}:{SERVER_B_PORT_VIDEO}")
    
    print(f"[视频线程] 已连接到 B: {SERVER_B_HOST}:{SERVER_B_PORT_VIDEO}")
    
    # 短暂等待确保连接建立（解决 PUB-SUB slow-joiner 问题）
    time.sleep(0.5)
    
    # 打开摄像头
    cap = cv2.VideoCapture(CAMERA_ID)
    
    if not cap.isOpened():
        print(f"❌ 无法打开摄像头 {CAMERA_ID}")
        return
    
    # 设置摄像头参数
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
    
    print(f"[视频线程] 摄像头已打开，分辨率: {VIDEO_WIDTH}x{VIDEO_HEIGHT}, FPS: {VIDEO_FPS}")
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[视频] 无法读取帧，尝试重新打开摄像头...")
                cap.release()
                time.sleep(1)
                cap = cv2.VideoCapture(CAMERA_ID)
                continue
            
            # 获取当前控制命令
            with command_lock:
                current_command = latest_command.copy()
            
            # 在帧上绘制 OSD（欧拉角等信息）
            euler = current_command.get("euler_angles", {})
            throttle = current_command.get("throttle", 0)
            
            # 绘制半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (5, 5), (350, 85), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
            
            # 绘制文字
            cv2.putText(frame, f"Roll:  {euler.get('roll', 0):6.2f} deg", 
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Pitch: {euler.get('pitch', 0):6.2f} deg", 
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Yaw:   {euler.get('yaw', 0):6.2f} deg", 
                       (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"Throttle: {throttle:5.2f}", 
                       (200, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # JPEG 编码（压缩）
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
            result, encoded_frame = cv2.imencode('.jpg', frame, encode_param)
            
            if not result:
                print("[视频] 编码失败")
                continue
            
            # 准备数据包（格式要匹配 A 的解码逻辑）
            capture_time = time.time()  # 记录采集时间用于延迟测量
            frame_data = {
                "image": encoded_frame.tobytes(),  # A 期待 "image" 字段
                "encoding": "jpeg",  # A 期待 "encoding" 字段
                "resolution": f"{VIDEO_WIDTH}x{VIDEO_HEIGHT}",
                "timestamp": capture_time,  # 帧采集时间戳
                "frame_count": frame_count,
                "width": VIDEO_WIDTH,
                "height": VIDEO_HEIGHT
            }
            
            # 发送视频帧
            send_start = time.time()
            socket.send(TorchSerializer.to_bytes(frame_data))
            send_time = (time.time() - send_start) * 1000  # 发送耗时(ms)
            
            frame_count += 1
            
            # 计算并显示 FPS（每 30 帧）
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed
                print(f"[视频] 发送帧 {frame_count}, FPS: {fps:.1f}, "
                      f"帧大小: {len(encoded_frame)} bytes, 发送耗时: {send_time:.2f}ms")
                start_time = time.time()
            
            # 控制帧率
            time.sleep(1.0 / VIDEO_FPS)
            
    except KeyboardInterrupt:
        print("\n[视频线程] 停止中...")
    except Exception as e:
        print(f"[视频线程] 错误: {e}")
    finally:
        cap.release()
        socket.close()
        context.term()


def main():
    """
    主函数：启动两个线程
    - 线程1：接收来自 B 的控制命令
    - 线程2：采集并发送视频流给 B
    反向连接模式：适用于 C 在 NAT 后面
    """
    print("=" * 60)
    print("服务器 C 启动 - 反向连接模式（真实视频采集）")
    print("C 主动连接到 B，适用于 NAT 环境")
    print("=" * 60)
    print(f"\n连接到服务器 B: {SERVER_B_HOST}")
    print(f"  - 命令端口: {SERVER_B_PORT_COMMAND}")
    print(f"  - 视频端口: {SERVER_B_PORT_VIDEO}")
    print()
    
    # 启动线程1：接收命令
    command_thread = threading.Thread(target=thread_receive_commands, daemon=True)
    command_thread.start()
    
    # 启动线程2：发送视频
    # video_thread = threading.Thread(target=thread_send_video, daemon=True)
    # video_thread.start()
    
    print("两个线程已启动，按 Ctrl+C 停止...\n")
    
    try:
        # 保持主线程运行
        command_thread.join()
        # video_thread.join()
    except KeyboardInterrupt:
        print("\n\n服务器 C 正在关闭...")
        print("服务器 C 已关闭。")


if __name__ == "__main__":
    main()
