import glfw
import time

# 初始化GLFW
if not glfw.init():
    raise Exception("❌ GLFW初始化失败！")

# 创建窗口（800x600分辨率，标题"GLFW测试窗口"）
window = glfw.create_window(800, 600, "GLFW Window Test (树莓派)", None, None)
if not window:
    glfw.terminate()
    raise Exception("❌ GLFW创建窗口失败！")

# 设置窗口为当前渲染上下文
glfw.make_context_current(window)

# 窗口循环（保持窗口显示3秒）
start_time = time.time()
while time.time() - start_time < 3:
    # 处理窗口事件（如关闭）
    glfw.poll_events()
    # 交换缓冲区（更新窗口显示）
    glfw.swap_buffers(window)

# 清理资源
glfw.destroy_window(window)
glfw.terminate()
print("✅ GLFW窗口创建成功！有头模式基础依赖已打通～")