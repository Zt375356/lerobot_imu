import glfw
import time
import os

# 再次确认环境变量（脚本内双重保险）
os.environ['EGL_PLATFORM'] = 'wayland'
os.environ['GLFW_CONTEXT_CREATION_API'] = 'egl'
os.environ['XDG_RUNTIME_DIR'] = f"/run/user/{os.getuid()}"
if 'DISPLAY' in os.environ:
    del os.environ['DISPLAY']  # 彻底删除DISPLAY，避免干扰

print("🔍 手动强制配置：Wayland+EGL（禁用GLX）")
print(f"EGL_PLATFORM: {os.environ['EGL_PLATFORM']}")
print(f"GLFW_CONTEXT_CREATION_API: {os.environ['GLFW_CONTEXT_CREATION_API']}")

# -------------------------- 核心：GLFW窗口提示（硬限制EGL） --------------------------
# 初始化GLFW（必须在设置window_hint之前）
if not glfw.init():
    raise Exception("❌ GLFW初始化失败（环境变量已正确，可能是GLFW版本过旧）")
print("✅ GLFW初始化成功")

# 1. 强制GLFW使用EGL创建上下文（禁用GLX，优先级最高）
glfw.window_hint(glfw.CONTEXT_CREATION_API, glfw.EGL_CONTEXT_API)

# 2. 可选：指定EGL客户端API（OpenGL ES，Wayland更兼容）
glfw.window_hint(glfw.CLIENT_API, glfw.OPENGL_ES_API)
glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

# 3. 禁用不必要的功能（减少干扰）
glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
glfw.window_hint(glfw.DECORATED, glfw.TRUE)  # 显示窗口边框

# -------------------------- 创建窗口（此时绝对不会用GLX） --------------------------
window = glfw.create_window(640, 480, "Wayland+EGL", None, None)
#if not window:
#    glfw.terminate()
#    # 最后尝试：打印GLFW支持的上下文API，确认是否有EGL
#    supported_apis = glfw.get_supported_context_creation_apis()
#    print(f"❌ 窗口创建失败！GLFW支持的上下文API：{supported_apis}")
#    print("   （正常应包含 EGL_CONTEXT_API=0，若没有则GLFW版本过旧）")
#    raise Exception("❌ 窗口创建失败（GLFW已强制EGL，可能是版本不支持）")

# -------------------------- 运行测试 --------------------------
glfw.make_context_current(window)
print("🎉 窗口创建成功！3秒后关闭...")

start_time = time.time()
while time.time() - start_time < 3 and not glfw.window_should_close(window):
    glfw.swap_buffers(window)
    glfw.poll_events()

# 清理资源
glfw.destroy_window(window)
glfw.terminate()
print("✅ 测试完成！手动配置Wayland+EGL成功～")

