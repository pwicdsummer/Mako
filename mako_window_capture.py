"""
mako_window_capture.py
======================
【v4.0 · 重构版】绝对前台跟随 + mss BBox 物理裁切捕获模块

核心架构变更（v3.x → v4.0）：
  废除生产力窗口标题白名单 + PrintWindow 句柄导出的复合寻址策略，
  改用纯粹的「前台物理跟随」机制：
  
  1. 废除标题过滤 —— 不再检查窗口标题是否匹配 VS Code / Chrome 等关键字
  2. 绝对前台跟随 —— win32gui.GetForegroundWindow() 获取用户当前操作的
     最前台窗口（游戏 / 播放器 / 编辑器 一视同仁）
  3. 安全 BBox 裁切 —— 用 GetWindowRect() 获取前台窗口的物理屏幕坐标，
     直接调用 mss.sct.grab(bbox) 从屏幕缓冲区抠出该矩形容器
  4. 桌面回退 —— 如果焦点在桌面（Progman/WorkerW/hwnd==0），回退全屏 mss

修复目标：
  当用户将《千恋＊万花》放在前台时，发给 Qwen 的图必须是完美裁剪、
  没有任何桌面背景的高清游戏窗口截图。不再出现因白名单跳过导致的 2.5K
  全屏严重压缩和代码幻觉问题。

降级链：
  mss BBox 前台窗口截取 → mss 全屏截图（桌面 / 异常降级）

依赖：
  pywin32, mss, Pillow
"""

from __future__ import annotations

from PIL import Image

# ---- pywin32 ----
import win32gui


# ============================================================
# 桌面窗口类名（当用户焦点在桌面时，需要绕过）
# ============================================================
DESKTOP_CLASSNAMES = frozenset({
    "Progman",       # 经典桌面
    "WorkerW",       # 桌面图标层（Win10+）
    "Shell_TrayWnd", # 任务栏（防止误抓）
})


# ============================================================
# 核心：WindowCapture — 绝对前台跟随 + mss BBox 物理裁切
# ============================================================

class WindowCapture:
    """
    纯前台物理跟随捕获引擎。

    核心策略：
      1. win32gui.GetForegroundWindow() 获取最前台窗口句柄
      2. win32gui.GetWindowRect(hwnd) 获取窗口在物理屏幕上的坐标
      3. mss.sct.grab(bbox) 从屏幕缓冲区中抠出精确矩形
      4. 桌面回退：焦点在桌面时 → 全屏 mss 截图

    对外接口：
      - capture_best_window() -> (Image | None, str)    —— 主入口
      - capture_foreground_window() -> (Image | None, str) —— 前台 BBox 截取
      - capture_fullscreen() -> Image | None              —— 全屏保底
    """

    def __init__(self):
        self._last_title: str = ""   # 上次成功捕获的窗口标题（调试用）
        self._last_hwnd: int = 0     # 上次成功捕获的窗口句柄（供 BBox 降级使用）

    # ---------------------------------------------------------------
    # 前台窗口信息获取
    # ---------------------------------------------------------------

    @staticmethod
    def _get_foreground_window_rect() -> tuple[int, dict | None]:
        """
        获取当前前台窗口的句柄和物理屏幕坐标矩形。

        Returns
        -------
        tuple[int, dict | None]
            (hwnd, bbox)
            hwnd : 前台窗口句柄，0 表示无有效窗口
            bbox : dict 格式 {'left': x, 'top': y, 'width': w, 'height': h}
                   如果窗口无效 / 无法获取 / 是桌面 / 尺寸为 0 则返回 None
        """
        try:
            hwnd = win32gui.GetForegroundWindow()
            if hwnd == 0:
                return 0, None

            # 跳过桌面窗口（焦点在桌面时直接回退全屏）
            class_name = win32gui.GetClassName(hwnd)
            if class_name in DESKTOP_CLASSNAMES:
                return hwnd, None

            # 获取窗口在物理屏幕上的矩形坐标
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top

            # 排除最小化或尺寸无效的窗口
            if width <= 0 or height <= 0:
                return hwnd, None

            bbox = {
                "left": left,
                "top": top,
                "width": width,
                "height": height,
            }
            return hwnd, bbox

        except Exception as e:
            print(f"[WindowCapture] ⚠ _get_foreground_window_rect 异常: {e}")
            return 0, None

    # ---------------------------------------------------------------
    # 核心：mss BBox 前台窗口截取（防黑屏终极方案）
    # ---------------------------------------------------------------

    def capture_foreground_window(self) -> tuple[Image.Image | None, str]:
        """
        【核心捕获方法】安全 BBox 物理裁切提取。

        流程：
          1. GetForegroundWindow() → 获取最前台窗口句柄
          2. GetWindowRect(hwnd) → 获取该窗口在物理屏幕上的精确坐标
          3. mss.sct.grab(bbox) → 从物理屏幕缓冲区中抠出该矩形图像
          4. 桌面回退：如果焦点在桌面，返回 None 让调用方降级全屏

        为什么用 mss 而不是 PrintWindow？
          - DirectX/OpenGL 游戏和渲染器通过 PrintWindow 常返回黑屏
          - mss 直接从帧缓冲区读取像素，100% 兼容所有窗口类型
          - mss BBox grab 只读取指定矩形区域，不产生全屏缩放

        Returns
        -------
        tuple[Image.Image | None, str]
            (pil_image, source_desc)
            pil_image   : PIL RGB 格式的窗口截图；失败返回 None
            source_desc : 描述字符串
                "foreground:窗口标题" — mss BBox 成功
                "desktop"             — 焦点在桌面，需降级
                "bbox_empty"          — 窗口矩形无效
                "bbox_failed"         — mss 捕获异常
        """
        hwnd, bbox = self._get_foreground_window_rect()
        self._last_hwnd = hwnd  # 保存句柄，供 BBox 降级使用

        if bbox is None:
            if hwnd != 0:
                # 是桌面或无效窗口
                return None, "desktop"
            return None, "bbox_empty"

        try:
            import mss
            with mss.MSS() as sct:
                sct_img = sct.grab(bbox)
                img = Image.frombytes("RGB", sct_img.size, sct_img.rgb)

                # 记录标题供调试
                try:
                    title = win32gui.GetWindowText(hwnd)
                    self._last_title = title
                except Exception:
                    title = "unknown"

                return img, f"foreground:{title}"

        except Exception as e:
            print(f"[WindowCapture] ❌ mss BBox grab 异常: {e}")
            return None, "bbox_failed"

    # ---------------------------------------------------------------
    # 保底降级：mss 全屏截图
    # ---------------------------------------------------------------

    @staticmethod
    def capture_fullscreen() -> Image.Image | None:
        """
        保底降级方案：使用 mss 截取全屏全景。

        当：
          - 用户焦点在桌面（Progman / WorkerW）
          - mss BBox 捕获异常
        时调用此方法。

        Returns
        -------
        Image.Image | None
            PIL RGB 格式的全屏截图；失败返回 None。
        """
        try:
            import mss
            with mss.MSS() as sct:
                monitor = sct.monitors[1]  # 主显示器
                sct_img = sct.grab(monitor)
                return Image.frombytes("RGB", sct_img.size, sct_img.rgb)

        except Exception as e:
            print(f"[WindowCapture] ⚠ 全屏降级截图失败: {e}")
            return None

    # ---------------------------------------------------------------
    # 组合方法：前台 BBox → 降级全屏
    # ---------------------------------------------------------------

    def capture_best_window(self) -> tuple[Image.Image | None, str]:
        """
        一键组合：前台 BBox 截取 → 桌面 / 失败时降级全屏。

        这是外部模块调用的主入口。
        v4.0 行为变更：
          - 不再依赖生产力窗口白名单
          - 永远跟随当前最前台窗口（用户正在操作的那个）
          - 从屏幕缓冲区用 mss BBox 物理裁切，精确抠出窗口区域

        Returns
        -------
        tuple[Image.Image | None, str]
            (pil_image, source_desc)
            pil_image   : PIL RGB 图像或 None
            source_desc : 描述字符串
                "foreground:窗口标题" — 前台窗口 mss BBox 成功
                "fullscreen"         — 降级为 mss 全屏
                "failed"             — 彻底失败
        """
        # ---- 1. 前台 BBox 截取 ----
        img, desc = self.capture_foreground_window()
        if img is not None:
            return img, desc

        # ---- 2. 降级：全屏保底 ----
        # 如果是因为焦点在桌面而回退，打印提示
        if desc == "desktop":
            print("[WindowCapture] ℹ 焦点在桌面，回退全屏截图")
        else:
            print(f"[WindowCapture] ⚠ 前台 BBox 捕获失败 ({desc}), 降级到全屏")

        fallback_img = self.capture_fullscreen()
        if fallback_img is not None:
            return fallback_img, "fullscreen"

        return None, "failed"

    # ---------------------------------------------------------------
    # 属性访问
    # ---------------------------------------------------------------

    @property
    def last_window_title(self) -> str:
        """上次成功捕获的窗口标题。"""
        return self._last_title


# ============================================================
# 单例快捷函数（与旧版接口兼容）
# ============================================================

_capture_instance: WindowCapture | None = None


def get_window_capture() -> WindowCapture:
    """获取全局 WindowCapture 单例。"""
    global _capture_instance
    if _capture_instance is None:
        _capture_instance = WindowCapture()
    return _capture_instance


def capture_best_window() -> tuple[Image.Image | None, str]:
    """
    全局快捷函数：取最佳窗口截图。
    返回 (pil_image, source_desc)。
    """
    return get_window_capture().capture_best_window()


# ============================================================
# 自测试入口
# ============================================================
if __name__ == "__main__":
    import time

    print("=" * 70)
    print("  🧪 WindowCapture v4.0 自测试 — 绝对前台跟随 + mss BBox 物理裁切")
    print("=" * 70)
    print("  请确保前台运行着一个窗口（任意程序），测试将：")
    print("    · 获取当前前台窗口的句柄和坐标")
    print("    · 使用 mss BBox 从屏幕缓冲区抠出窗口图像")
    print("    · 如果焦点在桌面则降级全屏")
    print("=" * 70)

    wc = WindowCapture()

    # 测试 1：前台窗口信息
    print("\n[测试 1] 前台窗口信息获取...")
    hwnd, bbox = wc._get_foreground_window_rect()
    if bbox is not None:
        try:
            title = win32gui.GetWindowText(hwnd)
        except Exception:
            title = "???"
        print(f"  ✅ 前台窗口: HWND={hwnd}, 标题='{title}'")
        print(f"     BBox: left={bbox['left']}, top={bbox['top']}, "
              f"width={bbox['width']}, height={bbox['height']}")
    else:
        print(f"  ⚠ 前台窗口无效或为桌面 (hwnd={hwnd})")

    # 测试 2：mss BBox 捕获
    print("\n[测试 2] mss BBox 前台窗口截取...")
    img, desc = wc.capture_foreground_window()
    if img is not None:
        print(f"  ✅ 捕获成功: 来源={desc}, 尺寸={img.size[0]}x{img.size[1]}")
        save_path = "test_window_capture.png"
        img.save(save_path)
        print(f"  📁 已保存到 {save_path}")
    else:
        print(f"  ⚠ {desc}，将测试全屏降级")

        # 测试 3：全屏降级
        print("\n[测试 3] 保底降级：全屏截图...")
        full_img = wc.capture_fullscreen()
        if full_img is not None:
            print(f"  ✅ 全屏捕获成功: {full_img.size[0]}x{full_img.size[1]}")
            save_path = "test_fullscreen.png"
            full_img.save(save_path)
            print(f"  📁 已保存到 {save_path}")
        else:
            print("  ❌ 全屏捕获失败")

    # 测试 4：组合方法
    print("\n[测试 4] capture_best_window 一键组合...")
    img, desc = wc.capture_best_window()
    if img is not None:
        print(f"  ✅ 捕获成功: 来源={desc}, 尺寸={img.size[0]}x{img.size[1]}")
    else:
        print(f"  ❌ 彻底失败: {desc}")

    # 测试 5：切换到桌面再测（仅提示）
    print("\n[测试 5] 提示：点击桌面后重新运行可测试桌面降级场景")

    print("\n" + "=" * 70)
    print("  WindowCapture v4.0 自测试完成")
    print("=" * 70)
