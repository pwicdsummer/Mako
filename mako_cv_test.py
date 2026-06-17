"""
mako_cv_test.py
===============
MakoVision 深度场景分析模块 (Qwen2-VL) — 双图原生输入版

本模块从 mako_cv.py 中剥离，包含 Qwen2-VL-2B-Instruct 深度语义理解层。
v3.0 核心升级：废弃 mss 全屏截图 → 改用【动态最优单窗策略 + OBS级 DWM 内存导出】
  - Image 1：动态寻址出的【最活跃的主力生产力软件窗口】（100% 原始渲染像素，无全屏壁纸稀释）
  - Image 2：物理摄像头最新帧

核心类:
    MakoVisionDeep — 继承 mako_cv.MakoVision，添加 Qwen2-VL 场景分析能力

依赖:
    torch, transformers, qwen_vl_utils, pillow, bitsandbytes, pywin32, mss(降级保底)
"""

import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
from qwen_vl_utils import process_vision_info

# 从轻量层导入 MakoVision 基类
from mako_cv import MakoVision

# 导入单窗感知管线核心
from mako_window_capture import WindowCapture


# ============================================================
# 路径常量 — Qwen2-VL 模型路径
# ============================================================
QWEN2_VL_PATH = "./models/qwen/Qwen2-VL-2B-Instruct"


def _safe_empty_cache():
    """
    防御性 GPU 显存清理。

    在 Python 进程退出或析构时，全局变量可能已被回收，
    必须通过 sys.modules 安全获取 torch 模块。
    """
    import sys
    torch_mod = sys.modules.get('torch', None)
    if torch_mod is not None and hasattr(torch_mod, 'cuda') and torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


class MakoVisionDeep(MakoVision):
    """
    Mako 深度视觉皮层 — 在轻量层基础上增加 Qwen2-VL 场景分析能力。

    v3.0 核心变更：
      - Image 1 来源从 `mss` 全屏截图 → 升级为 `WindowCapture` 单窗 HWND 内存直出
      - 窗口寻址算法：Z-Order 前台优先 + 后台降级 + 窗口锁定
      - 安全降级：PrintWindow 失败 → BitBlt → mss 全屏保底

    用法:
        vision = MakoVisionDeep(camera_id=0)
        data, frame = vision.check_presence()                          # 轻量层
        result = vision.analyze_scene([hwnd_pil, cam_pil], prompt)     # 深度层（双图）
        result = vision.analyze_scene(single_pil, prompt)              # 深度层（单图，向后兼容）
    """

    def __init__(self, camera_id: int = 0):
        """
        初始化深度视觉皮层。

        加载 MakoVision 轻量层 + Qwen2-VL-2B-Instruct 深度模型 + WindowCapture 单窗捕获。
        """
        # 初始化基类（摄像头 + MediaPipe + 3D 追踪）
        super().__init__(camera_id=camera_id)

        # 初始化 WindowCapture 单窗感知引擎
        self._window_capture = WindowCapture()

        # 加载 Qwen2-VL 深度模型
        self._load_qwen2vl()

        print("[MakoVisionDeep] ✅ 深度场景分析层初始化完成（单窗 HWND 直出 + 双图就绪）")

    # ---------------------------------------------------------------
    # 加载 Qwen2-VL
    # ---------------------------------------------------------------
    def _load_qwen2vl(self):
        """
        加载 Qwen2-VL-2B-Instruct 模型与处理器。

        显存优化策略 (RTX 5060 8GB, 实际可用 ~4.5G):
          - BitsAndBytesConfig 4-bit 量化 (nf4 + 双量化)
          - torch_dtype=torch.float16
          - max_pixels=1024*28*28 = 802,816（单图上限 ~1024 tokens）
          - 双图总视觉 Token ~2048，推理后立即 _safe_empty_cache()
        """
        print("[MakoVisionDeep] ⏳ 正在加载 Qwen2-VL-2B-Instruct (4-bit, GPU)…")

        self.processor = AutoProcessor.from_pretrained(
            QWEN2_VL_PATH,
            min_pixels=256 * 28 * 28,
            max_pixels=1024 * 28 * 28,
        )

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            QWEN2_VL_PATH,
            torch_dtype=torch.float16,
            device_map="auto",
            quantization_config=quantization_config,
        )

        print("[MakoVisionDeep] ✅ Qwen2-VL 加载完成")

    # ---------------------------------------------------------------
    # v3.0 核心：HWND 窗口捕获（取代旧版 mss 全屏截图）
    # ---------------------------------------------------------------
    def _capture_active_window(self) -> tuple[Image.Image | None, str]:
        """
        v3.0 核心视觉增强：OBS级句柄无损捕获。

        取代旧版 _capture_screenshot（mss 全屏截图），改用纯粹的 DWM 内存图层导出：
          1. Z-Order 寻址 → 找到最活跃的生产力软件窗口句柄
          2. PrintWindow(hwnd, hdc, 3) → 从内存直接导出 HWND 渲染为 PIL Image
          3. 安全降级：失败回退到 mss 全屏全景

        Returns
        -------
        tuple[Image.Image | None, str]
            (pil_image, source_desc)
            pil_image  : PIL RGB 图像或 None
            source_desc: 描述字符串
                "hwnd:窗口标题" — HWND 内存直出
                "fullscreen"   — 降级为 mss 全屏
                "failed"       — 彻底失败
        """
        return self._window_capture.capture_best_window()

    # ---------------------------------------------------------------
    # 保留旧版本 _capture_screenshot 作为降级备用的公开接口别名
    # （注：内部不再使用，仅保留给外部引用兼容）
    # ---------------------------------------------------------------
    def _capture_screenshot(self) -> Image.Image | None:
        """
        旧版 mss 全屏截图（保留兼容）。
        内部 v3.0 管线已改用 _capture_active_window，此方法仅作兼容。
        """
        return self._window_capture.capture_fullscreen()

    # ---------------------------------------------------------------
    # 深度场景分析（双图原生输入）
    # ---------------------------------------------------------------
    @torch.no_grad()
    def analyze_scene(
        self,
        pil_images: Image.Image | list[Image.Image],
        prompt: str,
        temperature: float = 0.0,
        do_sample: bool = False,
        max_new_tokens: int = 1000,
    ) -> str | None:
        """
        使用 Qwen2-VL 对场景图像进行深度语义理解。
        支持单图或双图输入。

        v2.0 变化：
          - 参数名从 pil_image → pil_images
          - 接受 Image.Image 或 list[Image.Image]
          - 多张图按顺序组装进 messages content 列表

        参数:
            pil_images: PIL Image 对象（单图）或 PIL Image 列表（多图，如 [桌面, 摄像头]）
            prompt:    文字提示词
            temperature: 生成温度（默认 0.0，视觉工具人模式，绝对客观）
                         设为 > 0.0 且 do_sample=True 可开启创造性发散。
            do_sample:   是否进行采样（默认 False，贪心解码）
                         视觉工具人场景保持 False，角色扮演场景可设为 True。
            max_new_tokens: 最大生成 token 数

        返回:
            str | None: 模型生成的文本描述；失败返回 None
        """
        if self.model is None:
            print("[MakoVisionDeep] ⚠ 深度模型未加载，跳过场景分析")
            return None

        output_text = None

        try:
            # ---- 统一为 list[Image.Image] ----
            if isinstance(pil_images, Image.Image):
                image_list = [pil_images]
            else:
                image_list = pil_images

            # ---- 组装多图消息体 ----
            content = []
            for img in image_list:
                content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": prompt})

            messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]

            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            # 构建生成参数（根据 temperature/do_sample 自动配置）
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": 1.15,
            }

            if do_sample:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temperature
                gen_kwargs["top_p"] = 0.9
            else:
                # 贪心解码：视觉工具人模式，确定性输出
                gen_kwargs["do_sample"] = False
                gen_kwargs["temperature"] = 0.0  # 确保零随机性

            generated_ids = self.model.generate(**inputs, **gen_kwargs)

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

        except Exception as e:
            print(f"[MakoVisionDeep] ❌ analyze_scene 出错: {e}")
            import traceback
            traceback.print_exc()

        finally:
            _safe_empty_cache()

        return output_text

    # ---------------------------------------------------------------
    # 资源释放（扩展基类方法）
    # ---------------------------------------------------------------
    def release(self):
        """释放所有资源，包括 GPU 显存。"""
        # 释放基类资源（摄像头 + MediaPipe）
        super().release()

        # 清理 GPU 缓存
        _safe_empty_cache()

        print("[MakoVisionDeep] ✅ 深度层资源已释放")


# ============================================================
# 独立测试入口（v3.0：使用 HWND 窗口捕获）
# ============================================================
if __name__ == "__main__":
    import cv2
    import time
    print("=" * 70)
    print("  🧪 MakoVisionDeep v3.0 独立测试 — 单窗HWND直出 + 摄像头")
    print("=" * 70)

    print("\n[1/3] 正在初始化 MakoVisionDeep…")
    vision = MakoVisionDeep(camera_id=0)
    print("[2/3] ✅ 初始化完成\n")

    print("📸 每 20 秒自动捕捉【最优生产力窗口 HWND 直出 + 摄像头帧】\n")

    DUAL_PROMPT = """你是一个高精度的计算机视觉环境感知模块。
你现在拥有两张相互隔离的实时视图：
- 图 1 (Image 1) 是真寻当前最关注的【主力生产力软件窗口】（代码编辑器或浏览器）
- 图 2 (Image 2) 是真寻的【摄像头人脸画面】

请仔细观察两张图像，用一句话客观、精准地综合描述画面中男性的【动作】、【面部表情】、
【他面对的屏幕内容（从图1中看）】以及【他眼前的环境/物品（从图2中看）】。

【严格限制】
1. 绝对禁止任何形式的文学创作、心理推测、角色扮演或情感润色。
2. 只陈述你从画面中看到的物理事实。
3. 字数控制在 80 字以内，直接输出描述，不要说任何客套话。"""

    try:
        round_num = 0
        while True:
            round_num += 1

            # 短暂等待摄像头就绪 + 同时让出 CPU
            time.sleep(0.3)

            # ---- v3.0：捕获双图：HWND 最优窗口直出 + 摄像头帧 ----
            hwnd_pil, source_desc = vision._capture_active_window()
            if hwnd_pil is None:
                print("⏳ 窗口捕获失败，尝试旧版全屏截图降级…")
                hwnd_pil = vision._capture_screenshot()
                source_desc = "fullscreen(fallback)"
            if hwnd_pil is None:
                print("⏳ 截图完全失败，跳过本轮…")
                time.sleep(20)
                continue

            data, frame = vision.check_presence()
            if frame is None:
                print("⏳ 等待摄像头画面…跳过本轮")
                time.sleep(20)
                continue

            # BGR → RGB → PIL
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_pil = Image.fromarray(rgb_frame)

            print(f"\n[{round_num}] 双图分析 (窗口[{source_desc}]: {hwnd_pil.size}, 摄像头: {camera_pil.size})…")
            result = vision.analyze_scene(
                [hwnd_pil, camera_pil],  # 顺序：图1=HWND直出，图2=摄像头
                DUAL_PROMPT
            )

            print("=" * 70)
            if result:
                print(f"  🎯 第 {round_num} 轮双图分析结果:")
                print(f"  {result}")
            else:
                print("  ⚠ 分析失败（请检查模型/显存状态）")
            print("=" * 70)
            print(f"⏳ 等待 20 秒后进入下一轮…")

            time.sleep(20)

    except KeyboardInterrupt:
        print("\n\n🛑 用户中断循环")

    vision.release()
    print("\n MakoVisionDeep v3.0 测试结束")
    print("=" * 70)
