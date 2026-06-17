"""
deep_thinker.py
===============
认知沙盒：DeepThinker — 10 回合 VQA 套娃循环（双图实时流版）。
负责驱动【里人格 DeepSeek A】与【Qwen2-VL 眼睛】之间的多轮视觉思维链。

v3.0 核心升级：光学变焦路由（动态高清 ROI 裁切替换）
  - 检测 <mako_look> 问题中的关键字，自动切换图 2 为对应部位的高清微距特写
  - 手相关 → get_hands_roi_image() / 眼相关 → get_eye_roi_image() / 脸相关 → get_face_roi_image()
  - 安全降级：若裁切失败，静默回退全景摄像头帧

v2.0 双图升级：
  - 每轮追问时重新捕获【最新桌面截图 + 最新摄像头帧】，形成动态时间流感知
  - 里人格可以通过图1（桌面）和图2（摄像头）分别观察数字世界与物理世界

线程安全设计：
- 所有涉及 GPU 推理（Qwen2-VL）和云端 API（DeepSeek）的操作
  都封装为同步函数，由外部调用者通过 run_in_executor 安全调度。
- DeepThinker 本身不创建线程，只提供纯同步的 think_about_scene() 方法。
"""
import re
import cv2
import numpy as np
from PIL import Image
from typing import Callable

# ---- Fix 2: HWND 黑屏检测 & BBox mss 降级用 ----
import win32gui
import mss


# ============================================================
# 常量
# ============================================================
MAX_TURNS = 10

# 初探 prompt：宽泛扫描（temperature=0.0 的 Qwen2-VL 用）
INITIAL_SCOUT_PROMPT = (
    "你是一个极其精准、没有任何感情色彩的计算机视觉雷达模块。\n"
    "你现在拥有两张相互隔离的实时视图：\n"
    "- 图 1 (Image 1) 是真寻当前最关注的【核心应用窗口】——\n"
    "  这是通过系统底层直接抓取到的 100% 原始渲染像素，没有全屏壁纸或其他无关元素的稀释。\n"
    "- 图 2 (Image 2) 是真寻的【摄像头画面】——这是他的真实人脸和物理环境。\n\n"
    "请对两张图像进行地毯式的全景扫描，尽可能详尽、丰富地列出你看到的所有客观物理事实。\n\n"
    "【扫描维度】\n"
    "1. 图1（核心视窗）：客观描述窗口内部显示的大致内容。例如：如果当前是代码编辑器，描述大致的代码结构或错误信息（红色报错）；如果是浏览器，描述网页内容；如果是其他软件，如实汇报即可。注意这是最核心的视窗，不含其他桌面元素。\n"
    "2. 图2（摄像头）：人物状态——他的肢体姿态、双手的精确位置、面部神态以及视线方向、周围环境。\n"
    "3. 综合：他正在做的事（如盯着控制台敲代码、阅读文档、浏览网页等）。\n\n"
    "【防幻觉与防死循环铁律 - 必须遵守】\n"
    "1. 绝对禁止顺着常规逻辑进行脑补。\n"
    "2. 只能描述画面像素中【切实可见】的东西。如果图1中没有出现红色报错，就不要说有错误；是什么软件就描述什么软件。\n"
    "3. 绝对禁止逐字抄写图1中的大段文字或密集菜单！请以客观描述为主，切勿进行主观心理推测。\n"
    "4. 请将回答内容按照图1（核心视窗）和图2（摄像头）两个维度分别描述，并将总字数稳定控制在【300 到 500 字】之间，以确保每次报告的信息粒度一致且丰富。"
)

# ---- 追问 prompt 模板基座 + 默认图2描述 ----
_FOLLOWUP_PROMPT_PREFIX = (
    "你是一个计算机视觉传感器。\n"
    "你现在拥有两张相互隔离的实时视图：\n"
    "- 图 1 (Image 1) 是真寻当前最关注的【主力生产力软件窗口】（代码编辑器或浏览器）——\n"
    "  这是通过系统底层直接抓取到的 100% 原始像素渲染，不含全屏壁纸稀释。\n"
)
_DEFAULT_CAMERA_PROMPT = (
    "- 图 2 (Image 2) 是真寻的【全景摄像头画面】。\n\n"
)

# ---- 光学变焦关键词映射 ----
_ZOOM_KEYWORDS_HANDS = ["手", "指", "敲"]
_ZOOM_KEYWORDS_EYES  = ["眼", "目", "盯", "看"]
_ZOOM_KEYWORDS_FACE  = ["脸", "表情", "嘴", "笑"]

_ZOOM_CAMERA_PROMPTS = {
    "hands": (
        "- 图 2 (Image 2) 已经切换为【右手/双手的高清微距特写】。"
        "请只针对该局部画面回答问题，不要寻找人脸。\n\n"
    ),
    "eyes": (
        "- 图 2 (Image 2) 已经切换为【眼部高清微距特写】。"
        "请仔细观察眼皮开合与视线方向。\n\n"
    ),
    "face": (
        "- 图 2 (Image 2) 已经切换为【面部高清微距特写】。"
        "请仔细观察面部微表情。\n\n"
    ),
}

_ZOOM_GETTERS = {
    "hands": "get_hands_roi_image",
    "eyes":  "get_eye_roi_image",
    "face":  "get_face_roi_image",
}


# ============================================================
# ★ Fix 2 辅助函数：HWND 黑屏检测 & BBox 精准物理裁切
# ============================================================

def _is_black_screen(pil_img: Image.Image,
                     brightness_threshold: int = 5,
                     black_pixel_ratio: float = 0.95) -> bool:
    """
    判定 PIL 图像是否为纯黑屏（或接近纯黑）。

    策略：转灰度后统计亮度 < brightness_threshold 的像素占比，
    若超过 black_pixel_ratio 则认为是黑屏。

    Parameters
    ----------
    pil_img : Image.Image
        待检测的 PIL RGB 图像。
    brightness_threshold : int
        亮度阈值（0-255），低于此值视为"纯黑"像素。
    black_pixel_ratio : float
        纯黑像素占比阈值。

    Returns
    -------
    bool
        True 表示检测到黑屏。
    """
    if pil_img is None:
        return False
    # 转灰度
    gray = pil_img.convert("L")
    pixels = np.array(gray, dtype=np.uint8)
    if pixels.size == 0:
        return False
    dark_count = int(np.sum(pixels < brightness_threshold))
    ratio = dark_count / pixels.size
    return ratio >= black_pixel_ratio


def _capture_window_bbox(hwnd: int) -> Image.Image | None:
    """
    使用 mss 从物理屏幕精准截取 HWND 窗口区域（BBox 模式）。

    绝不截全屏 — 只取目标窗口的绝对屏幕矩形坐标，
    以保证 1:1 原始分辨率无损捕获。

    Parameters
    ----------
    hwnd : int
        目标窗口句柄。

    Returns
    -------
    Image.Image | None
        PIL RGB 窗口区域截图；失败返回 None。
    """
    if hwnd == 0:
        return None
    try:
        # 获取窗口在屏幕上的绝对矩形坐标（包括窗口边框）
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            print(f"[BBox] ⚠ 窗口尺寸无效 ({width}x{height})")
            return None

        # 使用 mss 精准截取该窗口区域（绝不扩散到全屏）
        bbox = {"left": left, "top": top, "width": width, "height": height}
        with mss.mss() as sct:
            sct_img = sct.grab(bbox)
            return Image.frombytes("RGB", sct_img.size, sct_img.rgb)
    except Exception as e:
        print(f"[BBox] ❌ mss 窗口区域截图异常: {e}")
        return None


class DeepThinker:
    """
    认知沙盒：里人格思考循环（双图实时流版）。

    负责：
    - 从 MakoVisionDeep 捕获最新【桌面截图 + 摄像头帧】
    - 通过 Qwen2-VL 进行初探
    - 驱动【思考版 DeepSeek A】与【Qwen2-VL 眼睛】间的多轮追问
      （**每一轮都重新捕获最新画面，形成动态时间流**）
    - ★ v3.0 光学变焦：根据 <mako_look> 关键字自动切换高清微距裁切
    - 收敛并输出 visual_thought_digest（视觉思维纪要）

    用法（推荐在后台线程中执行）：
        digest = thinker.think_about_scene()
    """

    def __init__(
        self,
        vision,
        thinking_deepseek_fn: Callable,
    ):
        """
        Parameters
        ----------
        vision : MakoVisionDeep
            已初始化的 MakoVisionDeep 实例（包含摄像头 + Qwen2-VL + 截屏能力）。
        thinking_deepseek_fn : Callable
            思考版 Letta 接口函数，签名：
                fn(visual_report: str, chat_history: list[dict] | None) -> str
            注意：记忆已由 Letta mako_think_agent 内部管理，不再需要外部 memory_fn。
        """
        self.vision = vision
        self.think = thinking_deepseek_fn


    # ---------------------------------------------------------------
    # 捕获双图（HWND 最优窗口内存直出 + 摄像头帧）
    # ---------------------------------------------------------------
    def _capture_frames(self) -> list[Image.Image] | None:
        """
        捕获最新【HWND 最优窗口直出 + 摄像头帧】，返回 list[Image.Image]。

        v3.0 核心变更：
          - Image 1 从 mss 全屏截图 → 升级为 WindowCapture.PW_RENDERFULLCONTENT 句柄内存直出
          - 零全屏壁纸稀释，100% 原始渲染像素

        Returns
        -------
        list[Image.Image] | None
            [hwnd_pil, camera_pil] 格式的列表，两者都成功才返回；否则返回 None。
        """
        # 1. HWND 最优窗口内存直出（Z-Order 寻址 + PrintWindow）
        hwnd_pil, source_desc = self.vision._capture_active_window()
        if hwnd_pil is not None:
            # ---- ★ Fix 2: HWND 黑屏检测 — 纯黑方块判定 ----
            if _is_black_screen(hwnd_pil):
                print(f"[DeepThinker] ⚠ HWND 捕获结果为纯黑 ({source_desc})，"
                      "尝试 BBox mss 精准窗口降级…")
                bbox_hwnd = self.vision._window_capture._last_hwnd
                bbox_pil = _capture_window_bbox(bbox_hwnd)
                if bbox_pil is not None:
                    hwnd_pil = bbox_pil
                    source_desc = "mss_bbox"
                    print(f"[DeepThinker] ✅ BBox 降级成功: 尺寸 {hwnd_pil.size}")
                else:
                    print("[DeepThinker] ⚠ BBox 降级也失败，保留原始 black frame")

        if hwnd_pil is None:
            print(f"[DeepThinker] ⚠ HWND 窗口捕获失败 ({source_desc})，尝试降级全屏…")
            hwnd_pil = self.vision._capture_screenshot()
            if hwnd_pil is None:
                print("[DeepThinker] ⚠ 降级全屏截图也失败")
                return None

        # 2. 摄像头帧
        data, frame = self.vision.check_presence()
        if frame is None:
            print("[DeepThinker] ⚠ 摄像头画面未就绪")
            return None

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera = Image.fromarray(rgb)

        return [hwnd_pil, camera]  # 图1=HWND直出，图2=摄像头

    # ---------------------------------------------------------------
    # ★ v3.0 光学变焦路由：根据问题关键字动态替换图2
    # ---------------------------------------------------------------
    def _try_optical_zoom(self, question: str,
                          default_pil: Image.Image) -> tuple[Image.Image, str]:
        """
        根据 <mako_look> 问题中的关键字，尝试将图 2 切换为高清微距特写。

        匹配策略（优先级：手 > 眼 > 脸）：
          - 手相关: ["手", "指", "敲"] → get_hands_roi_image()
          - 眼相关: ["眼", "目", "盯", "看"] → get_eye_roi_image()
          - 脸相关: ["脸", "表情", "嘴", "笑"] → get_face_roi_image()

        安全降级：若裁切返回 None，静默回退到 default_pil 与默认描述。

        Parameters
        ----------
        question : str
            从 <mako_look> 提取的纯中文提问文本。
        default_pil : Image.Image
            默认的全景摄像头帧 PIL 图像（回退基底）。

        Returns
        -------
        tuple[Image.Image, str]
            (camera_pil, camera_prompt_desc)
            camera_pil         : 用于投喂 Qwen 的图 2 图像（高清微距或全景回退）
            camera_prompt_desc : 对应的图 2 描述文本（追加在 _FOLLOWUP_PROMPT_PREFIX 后）
        """
        # ---- ★ Fix 1: 前置严格拦截 — 只有明确提及 "图2" 或 "摄像头" 才允许变焦 ----
        # 如果提问只针对图1或根本未提及图2，绝对禁止触发光学变焦，直接返回全景
        if not re.search(r'图\s*2|摄像头', question):
            return default_pil, _DEFAULT_CAMERA_PROMPT

        # ---- 检测触发关键字 ----
        zoom_type = None
        for kw in _ZOOM_KEYWORDS_HANDS:
            if kw in question:
                zoom_type = "hands"
                break
        if zoom_type is None:
            for kw in _ZOOM_KEYWORDS_EYES:
                if kw in question:
                    zoom_type = "eyes"
                    break
        if zoom_type is None:
            for kw in _ZOOM_KEYWORDS_FACE:
                if kw in question:
                    zoom_type = "face"
                    break

        if zoom_type is None:
            # 无匹配 → 使用默认全景
            return default_pil, _DEFAULT_CAMERA_PROMPT

        # ---- 尝试获取高清微距裁切 ----
        getter_name = _ZOOM_GETTERS[zoom_type]
        getter = getattr(self.vision, getter_name, None)
        if getter is None:
            return default_pil, _DEFAULT_CAMERA_PROMPT

        try:
            roi_pil = getter()
        except Exception as e:
            print(f"[DeepThinker] ⚠ 光学变焦 {zoom_type} 裁切异常: {e}，回退全景")
            return default_pil, _DEFAULT_CAMERA_PROMPT

        if roi_pil is None:
            # 裁切失败（如人脸/手未检测到），静默回退
            print(f"[DeepThinker] ⚠ 光学变焦 {zoom_type} 裁切无结果（目标未检测到），回退全景")
            return default_pil, _DEFAULT_CAMERA_PROMPT

        # ---- 成功获取高清微距 ----
        zoom_desc = _ZOOM_CAMERA_PROMPTS[zoom_type]
        print(f"[DeepThinker] 🔬 光学变焦激活: {zoom_type} (裁切尺寸: {roi_pil.size[0]}x{roi_pil.size[1]})")
        return roi_pil, zoom_desc

    # ---------------------------------------------------------------
    # 调用 Qwen2-VL（同步，GPU 推理）
    # ---------------------------------------------------------------
    def _ask_qwen(self, pil_images: list[Image.Image], question: str) -> str:
        """
        向 Qwen2-VL 提问（支持双图），获取客观回答。

        Parameters
        ----------
        pil_images : list[Image.Image]
            当前帧图像列表，如 [desktop_pil, camera_pil]。
        question : str
            问题文本。

        Returns
        -------
        str
            Qwen2-VL 的回答；失败返回空字符串。
        """
        if not pil_images:
            return ""
        result = self.vision.analyze_scene(pil_images, question)
        return result if result else ""

    # ---------------------------------------------------------------
    # 提取 <mako_look> 标签
    # ---------------------------------------------------------------
    @staticmethod
    def _extract_look_tag(text: str) -> str | None:
        """
        从 DeepSeek A 回复中提取第一个 <mako_look>...</mako_look> 标签内容。

        Parameters
        ----------
        text : str
            DeepSeek A 的原始回复。

        Returns
        -------
        str | None
            标签内的文本，未找到返回 None。
        """
        m = re.search(r'<mako_look>(.*?)</mako_look>', text, re.DOTALL)
        return m.group(1).strip() if m else None

    # ---------------------------------------------------------------
    # 主入口：认知沙盒循环（双图实时流 + 光学变焦）
    # ---------------------------------------------------------------
    def think_about_scene(self) -> dict:
        """
        执行完整认知沙盒循环。

        v3.0 核心变更：
          - 每轮追问携带光学变焦路由（根据关键字切换图2为高清微距）
          - 安全降级机制：裁切失败自动回退

        返回结构（纯数据，不含任何 GUI 或异步对象）：
        {
            "digest": str,          # visual_thought_digest（最终思维纪要）
            "full_log": [           # 完整的追问日志（用于调试）
                {"role": "初探", "content": "..."},
                {"role": "追问1", "question": "...", "answer": "..."},
                ...
            ],
            "rounds_used": int,     # 实际使用的回合数（1~MAX_TURNS）
            "has_frame": bool,      # 是否成功捕获到画面
        }
        """
        # ---- 1. 捕获初探双图 ----
        init_frames = self._capture_frames()
        if init_frames is None:
            return {
                "digest": "【カメラ未起動】カメラがまだ準備できていないみたい。",
                "full_log": [],
                "rounds_used": 0,
                "has_frame": False,
            }

        # ---- 2. 初探：Qwen2-VL 宽泛扫描 ----
        initial_report = self._ask_qwen(init_frames, INITIAL_SCOUT_PROMPT)
        if not initial_report:
            return {
                "digest": "【視覚異常】映像が取得できたけど、何も認識できなかった…",
                "full_log": [{"role": "初探", "content": "(空返回)"}],
                "rounds_used": 1,
                "has_frame": True,
            }

        full_log = [{"role": "初探", "content": initial_report}]
        print(f"\n[DeepThinker] 🔍 初探報告: {initial_report}")

        # ---- 3. CoT messages：维护完整的思考链 ----
        # 不再注入外部对话历史。Letta mako_think_agent 内部管理。
        cot_messages = [
            {
                "role": "user",
                "content": f"这是视觉传感器对真寻当前状态的描述：{initial_report}"
            }
        ]


        # ---- 4. 多轮追问循环（每轮重新捕获最新画面 + 光学变焦） ----
        for turn in range(MAX_TURNS):
            # 调用思考版 DeepSeek A（传入累积的 CoT 历史）
            thought = self.think(
                visual_report="",
                chat_history=cot_messages,
            )

            # 提取 <mako_look> 标签
            next_question = self._extract_look_tag(thought)

            if next_question is None:
                # --- 强制轮次拦截（前 8 轮不允许收敛） ---
                if turn < 8:  # 0-indexed，对应前 8 个回合
                    print(f"[DeepThinker] ⚠️ 警告: 模型试图在第 {turn+1} 轮提前收敛，已拦截！")
                    warning_msg = (
                        "【System 拦截】未检测到 <mako_look> 标签！\n"
                        f"当前仅完成第 {turn+1} 轮追问，尚未达到 3+5 的最低思考轮次。\n"
                        "初探报告提供的信息不计入分数。请立刻反省，重新输出进度看板，并提出带有 <mako_look>…</mako_look> 标签的具体物理提问！"
                    )
                    # 使用 "user" 角色以确保 API 兼容性
                    cot_messages.append({"role": "user", "content": warning_msg})
                    continue  # 强迫模型重新生成

                # 不再需要追问 → 用 thought 作为最终思维纪要
                print(f"[DeepThinker] ✅ {turn+1} 回合后收敛")
                full_log.append({
                    "role": "思考",
                    "turn": turn + 1,
                    "content": thought,
                    "has_look_tag": False,
                })
                return {
                    "digest": thought,
                    "full_log": full_log,
                    "rounds_used": turn + 1,
                    "has_frame": True,
                }

            print(f"[DeepThinker] 🔄 第{turn+1}回追问: {next_question}")

            # ---- 【关键】每轮追问前重新捕获最新双图，形成动态时间流 ----
            current_frames = self._capture_frames()
            if current_frames is None:
                # 捕获失败，用上一轮的 frames（降级方案）
                current_frames = init_frames
                print("[DeepThinker] ⚠ 本轮捕获失败，使用上轮画面降级")

            # ---- ★ v3.0 光学变焦路由：检测关键字并替换图2 ----
            # 取出图1（桌面截图，不变）
            desktop_pil = current_frames[0]
            # 图2（全景摄像头帧，作为回退基底）
            default_camera_pil = current_frames[1]

            # 尝试光学变焦（若匹配 + 裁切成功，返回高清微距；否则回退全景）
            zoomed_camera_pil, zoom_camera_desc = self._try_optical_zoom(
                next_question, default_camera_pil
            )

            # 组装动态 prompt：前缀 + 图2描述（由 zoom 决定） + 问题后缀
            followup_prompt = (
                _FOLLOWUP_PROMPT_PREFIX
                + zoom_camera_desc
                + "用户提出了以下关于图像的特定问题，请仅根据图像像素回答，\n"
                  "不要添加任何多余信息。\n\n"
                + f"问题：{next_question}"
            )

            # 替换图2（可能为全景或高清微距）
            zoomed_frames = [desktop_pil, zoomed_camera_pil]
            answer = self._ask_qwen(zoomed_frames, followup_prompt)
            print(f"[DeepThinker] 📸 Qwen回答: {answer}")

            # 记录日志（包含光学变焦信息）
            log_entry = {
                "role": "追问",
                "turn": turn + 1,
                "question": next_question,
                "answer": answer,
                "deepseek_thought": thought,
            }
            # 如果 zoom_camera_desc != _DEFAULT_CAMERA_PROMPT，说明启用了变焦
            if zoom_camera_desc != _DEFAULT_CAMERA_PROMPT:
                log_entry["optical_zoom"] = True
                log_entry["crop_size"] = zoomed_camera_pil.size
            full_log.append(log_entry)

            # 更新 CoT messages：
            # - DeepSeek A 这轮的思考（含 <mako_look>）作为 assistant
            # - Qwen2-VL 的回答作为 user
            cot_messages.append({"role": "assistant", "content": thought})
            cot_messages.append({"role": "user", "content": f"补充观察：{answer}"})

            # 更新 init_frames 引用，为下一轮降级做准备
            init_frames = current_frames

        # ---- 5. 达到 MAX_TURNS 仍未收敛 → 强制输出 ----
        print(f"[DeepThinker] ⚠ 达到最大{MAX_TURNS}回合，强制收敛")
        force_final_prompt = (
            "你已经达到最大思考回合数。请直接输出综合视觉思维纪要，"
            "不要使用 <mako_look> 标签。"
        )
        final_thought = self.think(
            visual_report=force_final_prompt,
            chat_history=cot_messages,
        )
        full_log.append({
            "role": "强制收敛",
            "content": final_thought,
        })

        return {
            "digest": final_thought,
            "full_log": full_log,
            "rounds_used": MAX_TURNS,
            "has_frame": True,
        }
