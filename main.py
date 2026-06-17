"""
main.py
=======
最终总装：茉子桌宠（MakoFinalPet）
- 纯文本 UI（继承 PetWidget）
- DeepSeek AI 对话（DS_test.py）
- GPT-SoVITS 语音合成（内联 FD_TTS.py 的 speak 函数）
- ★ v2.1 多单元流式播放队列（段落拆分 + 第 x/y 段日志）

v2.1 核心升级：
  - DS_test.py / DS_proactive.py 返回 Unit 列表（多段落块级解析 + 段落拆分）
  - main.py 引入异步播放队列，按顺序逐单元播放
  - 每单元：更新气泡 → TTS 下载 → 语音播放 → 立绘切换 → 自动递进下一单元
  - 日志清晰显示 "第 1/3 段"、"第 2/3 段"

运行要求：
  qasync 已安装（venv 中已装）
  GPT-SoVITS 服务运行在 http://127.0.0.1:9880
  DeepSeek API Key 已配置（DS_test.py 中）

启动：
  python main.py
"""

# ---- 加载 .env 环境变量（必须在所有其他导入之前） ----
from dotenv import load_dotenv
load_dotenv()

# ============================================================
# 强制环境修复（Monkey Patch）— 兼容新版 torchaudio
# 必须在任何其他导入之前执行，防止 wespeaker/torchaudio 报错
# ============================================================
import torchaudio
import types
import sys

if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

# wespeaker / torchaudio-sox 兼容补丁
if 'torchaudio.sox_effects' not in sys.modules:
    sox_mock = types.ModuleType('torchaudio.sox_effects')
    sox_mock.SoxEffectsChain = type(
        'SoxEffectsChain',
        (),
        {
            '__init__': lambda self: None,
            '__enter__': lambda self: self,
            '__exit__': lambda self, *a: None,
            'append_effect_to_chain': lambda self, *a, **kw: None,
        },
    )
    sox_mock.apply_effects_tensor = lambda *a, **kw: (None, None)
    sox_mock.apply_effects_file = lambda *a, **kw: (None, None)
    sox_mock.make_output_file = lambda *a, **kw: None
    sys.modules['torchaudio.sox_effects'] = sox_mock
    torchaudio.sox_effects = sox_mock

# ============================================================

import asyncio
import time
import os
from datetime import datetime
import concurrent.futures
import requests
import pygame
import re

# ---- PyQt5 + qasync ----
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QTimer, Qt
from qasync import QEventLoop

# ---- 自有模块 ----
from simple_chat_ui import PetWidget
from DS_test import get_mako_reply
from DS_thinking import get_thinking_reply
from DS_proactive import get_proactive_reply
from chat_logger import save_chat_to_json
from voice_listener import VoiceListener, clean_funasr_output
import mako_image_loader
import mako_timeline

# ---- 闹钟模块（v2.3 静默监听） ----
from alarm_parser import parse_alarm_text
from alarm_scheduler import AlarmScheduler

#----小工具函数----
def text_without_emotion_labels(text):
    clean_text = re.sub("\[.*?\]","",text)
    clean_text = re.sub(" ","",clean_text)
    return clean_text


# ============================================================
# 全局情绪变量 — 随播放进度同步更新
# ============================================================
_current_emotion = "normal"


# ============================================================
# ★ v3.0 打断触发接口（物理打断与记忆回流机制）
# ============================================================
INTERRUPT_EVENT = asyncio.Event()  # 打断信号（asyncio 安全）

def trigger_mako_interruption():
    """
    全局打断函数：物理掐断声音 + 强制唤醒播放线程 + 激活打断信号。
    可从任意线程安全调用（pygame 接口线程安全，asyncio.Event 在主线程设置）。
    """
    print("\n[打断] 🔴 触发 Mako 语音打断！")
    # 1. 物理掐断声音
    try:
        pygame.mixer.music.stop()
    except Exception as e:
        print(f"[打断] ⚠ music.stop() 异常: {e}")
    # 2. 发送事件强制唤醒因 pygame.event.wait() 阻塞的播放线程
    try:
        pygame.event.post(pygame.event.Event(pygame.USEREVENT))
    except Exception as e:
        print(f"[打断] ⚠ event.post() 异常: {e}")
    # 3. 激活打断信号（异步消费者检测后执行切片回流）
    INTERRUPT_EVENT.set()


# ============================================================
# 内联 speak（从 FD_TTS.py 提取，零改动逻辑）
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VOICE_DIR = os.path.join(PROJECT_ROOT, "voice_data")
_VOICE_REF_DIR = os.path.join(PROJECT_ROOT, "gptsovit_voice_needed")
TTS_URL = "http://127.0.0.1:9880/tts"

def _download_tts(text: str, emotion):
    """调用 GPT-SoVITS TTS 服务，下载并保存 WAV 文件。返回文件路径。"""
    g_s_api_json = {}
    if emotion == "normal":
        _normal_dir = os.path.join(_VOICE_REF_DIR, "normal")
        g_s_api_json = {
            "ref_audio_path": os.path.join(_normal_dir, "mak208_032.wav"),
            "prompt_text": "とりあえず、お弁当にしましょう。このままでは食べる時間も無くなってしまいますから",
            "prompt_lang":  "ja",
            "text": text,
            "text_lang": "ja",
            "aux_ref_audio_paths":  [
                os.path.join(_normal_dir, "mak201_095.wav"),
                os.path.join(_normal_dir, "mak201_099.wav"),
                os.path.join(_normal_dir, "mak201_114.wav"),
                os.path.join(_normal_dir, "mak202_002.wav"),
                os.path.join(_normal_dir, "mak206_184.wav"),
                os.path.join(_normal_dir, "mak208_027.wav"),
                os.path.join(_normal_dir, "mak208_031.wav"),
                os.path.join(_normal_dir, "mak208_032.wav"),
                os.path.join(_normal_dir, "mak208_151.wav"),
                os.path.join(_normal_dir, "mak209_030.wav")
            ],
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "speed_factor": 1.0,
        }

    if emotion == "shy":
        _shy_dir = os.path.join(_VOICE_REF_DIR, "shy")
        g_s_api_json = {
            "ref_audio_path": os.path.join(_shy_dir, "mak207_029.wav"),
            "prompt_text": "ドキドキするんですが、力強いから安心也できて、殿方的手だな……って",
            "prompt_lang":  "ja",
            "text": text,
            "text_lang": "ja",
            "aux_ref_audio_paths":  [
                os.path.join(_shy_dir, "mak207_008.wav"),
                os.path.join(_shy_dir, "mak207_009.wav"),
                os.path.join(_shy_dir, "mak207_015.wav"),
                os.path.join(_shy_dir, "mak207_024.wav"),
                os.path.join(_shy_dir, "mak207_025.wav"),
                os.path.join(_shy_dir, "mak207_026.wav"),
                os.path.join(_shy_dir, "mak207_028.wav"),
                os.path.join(_shy_dir, "mak207_029.wav"),
                os.path.join(_shy_dir, "mak207_030.wav"),
                os.path.join(_shy_dir, "mak207_032.wav")
            ],
            "top_k": 15,
            "top_p": 1,
            "temperature": 1,
            "speed_factor": 1.0,
        }



    response = requests.post(TTS_URL, json=g_s_api_json, timeout=30)

    if response.status_code == 200:
        os.makedirs(VOICE_DIR, exist_ok=True)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        file_path = f"{VOICE_DIR}/Mako{current_time}.wav"
        with open(file_path, "wb") as f:
            f.write(response.content)
        return file_path
    else:
        print(f"TTS 请求失败 ({response.status_code})，检查 GPT-SoVITS 是否已启动？")
        return None


def _play_wav(wav_path: str, max_retries: int = 2):
    """
    播放 WAV 音频（纯事件驱动，零循环零 sleep，必须跑在后台线程）。

    实现原理：
      - pygame.mixer.music.load(wav_path)  加载 WAV
      - set_endevent(pygame.USEREVENT)     注册"播放完毕"事件
      - music.play()                       开始播放（异步）
      - pygame.event.wait()                原生阻塞线程，直到 USEREVENT 触发后自动唤醒

    内置重试机制：若播放因设备变更中断，自动重试最多 max_retries 次。
    重试时的等待使用 pygame.time.wait()，零 time.sleep。
    """
    _AUDIO_DONE = pygame.USEREVENT + 1  # 自定义音频结束事件类型

    for attempt in range(max_retries + 1):
        try:
            pygame.mixer.music.load(wav_path)
            pygame.mixer.music.set_endevent(_AUDIO_DONE)
            pygame.mixer.music.play()
            # 原生阻塞线程，直到音频播放完毕触发 _AUDIO_DONE 后自动唤醒
            pygame.event.wait()
            return  # 播放成功
        except pygame.error as e:
            if attempt < max_retries:
                print(f"[TTS] ⚠ 播放失败（第{attempt+1}次），重试中: {e}")
                pygame.time.wait(500)  # 零 time.sleep
            else:
                print(f"[TTS] ❌ 播放失败（已达最大重试次数）: {e}")
        except Exception as e:
            if attempt < max_retries:
                print(f"[TTS] ⚠ 播放失败（第{attempt+1}次），重试中: {e}")
                pygame.time.wait(500)
            else:
                print(f"[TTS] ❌ 播放失败（已达最大重试次数）: {e}")


# ============================================================
# 全局视觉模块引用（惰性初始化）
# ============================================================
_deep_vision = None
_deep_thinker = None


# ============================================================
# MakoFinalPet — 继承 PetWidget，注入 AI + TTS + 主动视觉 + 播放队列
# ============================================================
class MakoFinalPet(PetWidget):
    """
    最终桌宠：右键聊天 → AI 回复 → 单元队列 → 逐段气泡显示 + 语音播放 + 立绘切换。

    ★ v2.1 多单元流式播放队列（段落拆分版）：
      - get_mako_reply() / get_proactive_reply() 返回 Unit 列表（段落级别）
      - 所有 Unit 入队后，由 _process_unit_queue() 逐一取出播放
      - 上一段播放完毕（winsound 同步阻塞返回）自动触发下一段
      - 日志清晰显示 "第 1/3 段"、"第 2/3 段"、"第 3/3 段"
      - 气泡内容依次更新为各单元 CN 文本
    """

    AUTO_DISMISS_MS = 90000  # 显示 90 秒足够阅读

    def __init__(self):
        super().__init__()
        self.current_path = mako_image_loader.get_mako_image_path()

        # ---- 主动搭话状态 ----
        self._is_speaking = False        # 播放锁：防止重叠
        self._last_digest = None         # 上次视觉纪要（变化检测）
        self._unchanged_count = 0        # 连续无变化计数器
        self._vision_ready = False       # 视觉模块就绪标志
        self._proactive_enabled = True   # 主动搭话总开关

        # ---- ★ v2.1 播放队列系统 ----
        self._unit_queue: asyncio.Queue = asyncio.Queue()   # 单元队列
        self._queue_processing = False                      # 队列处理锁
        self._queue_total = 0                               # 当前批次总单元数
        self._queue_processed = 0                           # 当前批次已处理数

        # ---- ★ v3.0 打断与记忆回流状态 ----
        self._current_playing_idx = -1                      # 当前正在播放的单元索引（用于打断切片）
        self._interrupt_spoken_text: str = ""               # 打断时已说的中文文本（缓存供回流）
        self._interrupt_unspoken_text: str = ""             # 打断时未说的中文文本（缓存供回流）

        # ---- ★ v2.3 闹钟静默监听 ----
        self.alarm_scheduler = AlarmScheduler()

        # 哨兵轮询：50ms 检查一次图片路径是否有变
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(self._poll_image_path)
        self._poll_timer.start(50)

        # ---- 主动搭话定时器（300 秒周期） ----
        self._proactive_timer = QTimer(self)
        self._proactive_timer.timeout.connect(self._on_proactive_tick)
        self._proactive_timer.setInterval(90_000)  # 90 秒（修正：之前误写为 480_000）
        # 注意：在视觉模块就绪后才 start()

        # ============================================================
        # ★ 启动动画兼容：临时降级窗口层级
        #    若 launch_mv.mp4 正在播放，PetWidget 的 SubWindow 级别会
        #    盖住视频。此处先降级为普通无边框窗口，22 秒后再恢复置顶。
        # ============================================================
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.show()
        QTimer.singleShot(25000, self._restore_topmost)

    def _restore_topmost(self):
        """25 秒后恢复 PetWidget 的 SubWindow + 置顶标志"""
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.SubWindow
        )
        self.show()
        print("[MakoFinalPet] ✅ 已恢复 SubWindow + 置顶层级")

    def set_vision_ready(self, vision, thinker):
        """由 _post_init 在后台线程初始化完成后调用，激活视觉模块。"""
        global _deep_vision, _deep_thinker
        _deep_vision = vision
        _deep_thinker = thinker
        self._vision_ready = True
        if self._proactive_enabled:
            self._proactive_timer.start()
            print("[MakoFinalPet] ✅ 主动视觉搭话定时器已启动（周期 90s）")

    # ---------------------------------------------------------------
    # ★ 运行中视觉模块开关（右键菜单触发）
    # ---------------------------------------------------------------
    def _toggle_vision_module(self, enabled: bool):
        """
        运行中切换视觉认知模块的启用/禁用。
        由右键菜单 vision_toggled 信号触发，通过 QTimer 安全转移到 UI 线程。

        禁用流程：
          1. 停止主动搭话定时器
          2. 释放 GPU 显存 + 摄像头 + MediaPipe 资源
          3. 将模块引用置空
          4. 标记 _vision_ready = False

        启用流程：
          1. 标记 _vision_ready = False（加载中）
          2. 创建后台任务重新初始化视觉模块
          3. 加载完成后自动重启定时器
        """
        global _deep_vision, _deep_thinker

        if not enabled:
            # ---- 禁用视觉模块 ----
            print("[MakoFinalPet] 👁️ 正在禁用视觉认知模块…")
            # 停止主动搭话定时器
            if self._proactive_timer.isActive():
                self._proactive_timer.stop()
            self._proactive_enabled = False
            self._vision_ready = False

            # 释放视觉资源（GPU 显存 + 摄像头）
            if _deep_vision is not None:
                try:
                    _deep_vision.release()
                except Exception as e:
                    print(f"[MakoFinalPet] ⚠ 释放视觉资源异常: {e}")
                _deep_vision = None
            if _deep_thinker is not None:
                _deep_thinker = None

            # 清理缓存引用
            self._last_digest = None
            self._unchanged_count = 0

            print("[MakoFinalPet] ✅ 视觉认知模块已禁用（GPU 显存 + 摄像头已释放）")
        else:
            # ---- 启用视觉模块 ----
            if self._vision_ready:
                print("[MakoFinalPet] 👁️ 视觉模块已在运行中，跳过重复初始化")
                return

            print("[MakoFinalPet] 👁️ 正在启用视觉认知模块（后台加载）…")

            # 创建异步任务在后台重新加载
            asyncio.create_task(self._async_reload_vision())

    async def _async_reload_vision(self):
        """
        后台重新加载视觉模块。
        在后台线程中执行 Qwen2-VL 模型加载（不阻塞 UI），
        加载完成后自动激活主动搭话。
        """
        from deep_thinker import DeepThinker
        from mako_cv_test import MakoVisionDeep

        loop = asyncio.get_running_loop()
        try:
            def _load_vision():
                vision = MakoVisionDeep(camera_id=0)
                thinker = DeepThinker(
                    vision=vision,
                    thinking_deepseek_fn=get_thinking_reply,
                )
                return vision, thinker

            _vision, _thinker = await loop.run_in_executor(None, _load_vision)
            self.set_vision_ready(_vision, _thinker)
            self._proactive_enabled = True
            print("[MakoFinalPet] ✅ 视觉认知模块重新加载完成")
            # 气泡提示用户加载成功
            self._show_reply("👁️ 视觉认知模块已开启")
        except Exception as e:
            print(f"[MakoFinalPet] ⚠ 视觉模块重新加载失败: {e}")
            import traceback
            traceback.print_exc()
            self._vision_ready = False
            # 修复：加载失败时 _proactive_enabled 应设为 False，
            # 避免 _vision_ready=False 与 _proactive_enabled=True 的矛盾状态
            self._proactive_enabled = False
            # 气泡提示用户加载失败
            self._show_reply("⚠️ 视觉模块加载失败，请检查控制台日志")

    # ---------------------------------------------------------------
    # ★ 开机主动重逢搭讪（首次启动时静默发送系统事件到 Letta）
    # ---------------------------------------------------------------
    async def _send_startup_greeting(self):
        """
        启动完成后静默向 Letta 发送系统事件，触发茉子的主动重逢搭讪。
        该事件文本不渲染到前端气泡中——只是作为触发上下文送入 Letta。
        让 Letta 后端自行根据当前时间段（morning/afternoon/evening/night）
        生成合适的主动搭讪台词，然后通过播放队列正常播放。
        """
        loop = asyncio.get_running_loop()
        try:
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            system_event = (
                f"[当前系统时间: {now_str}] "
                "【系统事件】用户重新打开了客户端。"
                "风早真寻（Mahiro）已经回到了你身边。"
                "请你（常陆茉子）感知到这次久别重逢，"
                "并根据当前时间段向他主动发起搭讪问候。"
            )
            print(f"\n{'='*20}\n[启动搭讪] 📤 向 Letta 发送开机系统事件（不显示气泡）\n{'='*20}")
            units = await loop.run_in_executor(
                None, get_mako_reply, system_event
            )
            if not units:
                print("[启动搭讪] ⚠ Letta 返回空回复，跳过主动搭讪")
                return

            # 全部 Unit 入队 → 启动队列播放
            total = len(units)
            self._queue_total = total
            for idx, unit in enumerate(units):
                await self._unit_queue.put(unit)
                print(f"[Queue] 📥 [启动搭讪] 第 {idx+1}/{total} 段已入队")

            if not self._queue_processing:
                asyncio.create_task(self._process_unit_queue())
            print("[启动搭讪] ✅ 开机系统事件已发送，茉子即将主动搭讪")
        except Exception as e:
            print(f"[启动搭讪] ❌ 异常: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------
    # 图片轮询
    # ---------------------------------------------------------------
    def _poll_image_path(self):
        """定时轮询：如果 mako_image_loader 返回了新路径，自动触发平滑切换。"""
        new_path = mako_image_loader.get_mako_image_path()
        if new_path != self.current_path:
            self.current_path = new_path
            self.fade_switch_image(new_path)

    # ---------------------------------------------------------------
    # 用户输入路径
    # ---------------------------------------------------------------
    def _on_user_send(self, text: str):
        """
        用户发送：显示气泡 + 物理打断进行中的播放 + 启动异步 AI+TTS 任务。

        ★ v3.0 打断机制：
          - 若当前有播放队列正在处理（_queue_processing 或 _is_speaking），
            立刻调用 trigger_mako_interruption()：物理掐断声音 + 唤醒播放线程 + 设打断信号。
          - 消费者协程检测到打断信号后，执行切片 + 清空队列。
          - 随后 _handle_mako_logic 会读取打断缓存，构造回流上下文。
        """
        self.chat_bubbles.add_message(text, msg_type="sent")
        QTimer.singleShot(self.AUTO_DISMISS_MS, self.chat_bubbles.del_first_msg)

        # ---- ★ v3.0 打断正在进行的播放（如果存在） ----
        if self._queue_processing or self._is_speaking:
            trigger_mako_interruption()
            print("[打断] 📢 用户新输入 → 打断正在播放的语音")

        # ---- ★ v2.3 闹钟静默监听（旁路，不阻塞不修改聊天流） ----
        self._silent_alarm_check(text)

        asyncio.create_task(self._handle_mako_logic(text))

    # ---------------------------------------------------------------
    # ★ v2.3 闹钟静默监听旁路
    # ---------------------------------------------------------------
    def _silent_alarm_check(self, text: str):
        """
        静默旁路：解析用户文本中的闹钟意图，启动后台倒计时。
        无论解析成功与否，都**不**阻塞、不修改、不影响当前聊天流。
        """
        result = parse_alarm_text(text)
        if result is None:
            return  # 没有闹钟意图，直接放行

        delay_seconds, event_text = result
        if delay_seconds <= 0:
            return  # 过去的时间，不设置闹钟

        # 启动后台倒计时（asyncio.sleep 非阻塞）
        # 注意：_on_alarm_triggered 是 async def，必须通过 lambda 包成 asyncio.create_task
        self.alarm_scheduler.add_alarm(
            delay_seconds,
            event_text,
            lambda text: asyncio.create_task(self._on_alarm_triggered(text)),
        )
        print(f"[闹钟] ⏰ 已设置闹钟: {event_text}（{delay_seconds:.0f} 秒后触发）")

    # ---------------------------------------------------------------
    # ★ v2.3 闹钟到点触发回调
    # ---------------------------------------------------------------
    async def _on_alarm_triggered(self, event_text: str):
        """
        闹钟到点异步回调——由 alarm_scheduler 的后台协程在 QEventLoop 上下文中调用。

        逻辑：
          1. 防冲突守卫：若 _is_speaking 为 True（正在对话/播放），跳过本次提醒
          2. 构造系统提示文本，调用 get_mako_reply() 获取 Unit 列表
          3. 复用播放队列：Unit 入队 → 启动 _process_unit_queue()
        """
        if self._is_speaking:
            print(f"[闹钟] ⏰ 提醒「{event_text}」被跳过（正在说话中）")
            return

        loop = asyncio.get_running_loop()
        self._is_speaking = True
        try:
            # ---- 1. 构造主动搭话文本（寄生时间戳） ----
            now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            prompt = (
                f"[当前系统时间: {now_str}] "
                f"【系统提醒：时间到了！Edward 之前让你提醒他：{event_text}。"
                f"请立刻用你充满活力的语气主动向他搭话提醒！】"
            )
            print(f"\n[闹钟] 🔔 触发提醒: {event_text}")

            # ---- 2. 调用 Letta Chat Agent 获取 Unit 列表 ----
            units = await loop.run_in_executor(
                None, get_mako_reply, prompt
            )

            if not units:
                print("[闹钟] ⚠ Letta 返回空回复，跳过提醒播报")
                return

            # ---- 3. 全部 Unit 入队 ----
            total = len(units)
            self._queue_total = total
            for idx, unit in enumerate(units):
                await self._unit_queue.put(unit)
                print(f"[Queue] 📥 [闹钟] 第 {idx+1}/{total} 段已入队")

            # ---- 4. 启动队列处理 ----
            if not self._queue_processing:
                asyncio.create_task(self._process_unit_queue())

        except Exception as e:
            print(f"[闹钟] ❌ 闹钟回调异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_speaking = False

    async def _handle_mako_logic(self, text: str):
        """
        ★ v3.0 MVP 硬路由 — 异步处理用户输入（打断回流传入 + Letta 透传 + 多单元队列）。

        流程：
          0. 打断记忆回流：若 INTERRUPT_EVENT 已激活，将【已说/未说文本】包装为系统事件
             强制拼接在用户输入最前面，随后重置打断信号
          1. 调用 get_mako_reply() 获取 Unit 列表
          2. 更新队列总计数并全部入队
          3. 启动队列处理（如尚未启动）
          4. 持久化聊天日志（使用第一段 cn 作为摘要）
        """
        loop = asyncio.get_running_loop()

        self._is_speaking = True
        try:
            # ---- 0. ★ v3.0 神经上下文回流 ----
            if INTERRUPT_EVENT.is_set():
                spoken = self._interrupt_spoken_text
                unspoken = self._interrupt_unspoken_text
                print(f"[回流] 📤 检测到打断记忆缓存，准备注入上下文")
                print(f"[回流]   已说文本: {spoken[:100] if spoken else '（空）'}...")
                print(f"[回流]   未说文本: {unspoken[:100] if unspoken else '（空）'}...")
                # 构造打断系统事件，强制拼接在用户输入最前面
                interrupt_context = (
                    f"【系统事件】真寻在你说话时打断了你。"
                    f"你当时已经说出口的话是：'{spoken}'；"
                    f"你原本打算说但没能说出口的话是：'{unspoken}'。"
                    f"请结合真寻新输入的话，敏锐推断他打断你的原因，给予符合人设的动态回复。"
                )
                text_with_stamp = (
                    f"[当前系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"{interrupt_context} 真寻的新输入：{text}"
                )
                # 使用完回流缓存后清空
                self._interrupt_spoken_text = ""
                self._interrupt_unspoken_text = ""
                print(f"[回流] ✅ 打断上下文已拼接，准备发送至 Letta（打断信号保留，待新单元入队后清除）")
            else:
                # ---- 1. 普通寄生时间戳 ----
                text_with_stamp = f"[当前系统时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {text}"

            units = await loop.run_in_executor(
                None, get_mako_reply, text_with_stamp
            )

            if not units:
                print("[Queue] ⚠ 没有有效回复单元，跳过")
                return

            # ---- 2. 持久化聊天日志（使用第一段 cn 作为摘要） ----
            summary_cn = units[0].get("cn", "")
            await asyncio.to_thread(save_chat_to_json, text, summary_cn)

            # ---- 3. 更新总计数并全部入队 ----
            total = len(units)
            self._queue_total = total
            for idx, unit in enumerate(units):
                await self._unit_queue.put(unit)
                print(f"[Queue] 📥 第 {idx+1}/{total} 段已入队")

            # ★ 注意：不在此处 clear() 打断信号！
            # 旧 consumer/producer 可能还在播放/克隆中，需要它们自己检测到信号后退出。
            # 清除事件的责任交给新 _process_unit_queue 在启动时处理。

            # ---- 4. 启动队列处理（若尚未启动） ----
            if not self._queue_processing:
                asyncio.create_task(self._process_unit_queue())

        finally:
            self._is_speaking = False

    # ---------------------------------------------------------------
    # ★ v2.2 核心：TTS 异步预克隆双缓冲流水线
    # ---------------------------------------------------------------
    async def _prefetch_audio(self, unit: dict) -> str | None:
        """
        阶段 A：纯后台异步预克隆，不涉及任何 UI 操作。

        参数
        ----------
        unit : dict
            包含 "jp"（日语文本）和 "emotion" 字段的单元字典。

        返回
        -------
        str | None
            下载成功的音频文件路径（GS→WAV, FA→MP3），失败返回 None。
        """
        jp_text = unit.get("jp", "")
        emotion = unit.get("emotion", "normal")

        if not jp_text:
            return None  # 没有日语文本，不需要 TTS

        try:
            if self._tts_backend == "FA":
                # Fish Audio 云端：纯文本直调，无视情绪参数
                from fish_tts_plugin import fish_download_tts
                return await fish_download_tts(jp_text)
            else:
                # GPT-SoVITS 本地：带情绪参考音频
                return await asyncio.to_thread(_download_tts, text_without_emotion_labels(jp_text), emotion)
        except Exception as e:
            print(f"[预克隆] ❌ 下载异常: {e}")
            return None

    async def _play_audio_blocking(
        self,
        wav_path: str | None,
        cn_text: str,
        emotion: str = "normal",
        raw_tag: str = "",
        skip_ui: bool = False
    ) -> None:
        """
        阶段 B：前台播放动作——更新气泡 + 切换立绘 + 播放语音 + 时间轴。

        参数
        ----------
        wav_path : str | None
            预克隆好的 WAV 路径。若为 None（无日语或下载失败），仅显示气泡 + 立绘。
        cn_text : str
            中文文本，在气泡中显示。为空则跳过气泡。
        emotion : str
            情绪状态，影响参考音频和图片选择。
        skip_ui : bool
            是否跳过气泡和立绘刷新（用于第一段——文字已提前渲染的场景）。
        """
        # ---- 0. 更新全局情绪变量 ----
        global _current_emotion
        _current_emotion = emotion

        # ---- 情绪动画：用原始标签判断 ----
        if raw_tag == "happy" or raw_tag == "laughing":
            self.jump()
        elif raw_tag == "shy":
            self.shy_movement()

        # ---- 1. 气泡显示中文（skip_ui=True 时跳过，因文字已提前渲染） ----
        if not skip_ui and cn_text:
            self._show_reply(text_without_emotion_labels(cn_text))

        # ---- 2. 表情切换（skip_ui=True 时跳过，因立绘已提前切换） ----
        if not skip_ui:
            img_path = mako_image_loader.provide_default_path(emotion)
            self.current_path = img_path
            self.fade_switch_image(img_path)

        # ---- 3. 如果没有 WAV 路径，就到此为止 ----
        if not wav_path:
            return

        # ---- 4. 时间轴 + 并行播放 ----
        timeline_task = asyncio.create_task(
            self._run_expression_timeline(wav_path, emotion)
        )
        await asyncio.to_thread(_play_wav, wav_path)
        await timeline_task

    # ---------------------------------------------------------------
    # ★ v2.2 双指针流水线调度器
    # ---------------------------------------------------------------
    async def _process_unit_queue(self):
        """
        ★ v2.3 生产者-消费者流水线调度器（修复双缓冲只能预取下一段的缺陷）。

        核心架构：
          - 生产者：从 all_units 逐段取出文本 → 顺序发送 TTS → 下载 WAV → 放入 wav_queue
                    不等播放！克隆完一段立刻开始克隆下一段
          - 消费者：从 wav_queue 取出 WAV → 更新气泡 + 立绘 → 播放
                    播完立刻取下一个 WAV，无需等待克隆
          - 两者通过 asyncio.Queue 解耦，独立运行

        时间线示意（以3段为例）：
          生产者： [克隆0]──[克隆1]──[克隆2]   ← 全部在播放0期间完成
                     ↓ put   ↓ put    ↓ put
          wav_queue: [wav0]  [wav0,wav1]  [wav0,wav1,wav2]
                     ↓ get
          消费者： [播放0 ██████████████] → [播放1 ██████] → [播放2 ████]
        """
        if self._queue_processing:
            print("[Queue] ⚠ 队列已在处理中，跳过")
            return

        # ★ 新队列启动时，清除任何残留的打断信号
        # 旧的打断事件已被旧 consumer/producer 处理完毕，新队列需要干净状态
        INTERRUPT_EVENT.clear()

        self._queue_processing = True
        self._queue_processed = 0

        try:
            # ---- 从队列中预取所有单元到本地列表（确保队列可被提前消费） ----
            all_units = []
            while not self._unit_queue.empty():
                all_units.append(await self._unit_queue.get())

            total = len(all_units)
            self._queue_total = total

            if total == 0:
                return

            # ---- ★ 第一段特殊处理：不等 TTS，立即显示文字 + 立绘 ----
            first_unit = all_units[0]
            cn_text = first_unit.get("cn", "")
            if cn_text:
                self._show_reply(text_without_emotion_labels(cn_text))
                print(f"[Queue] 🖥️ 第 1 段文字已提前渲染（不等 TTS）")
            img_path = mako_image_loader.provide_default_path(
                first_unit.get("emotion", "normal")
            )
            self.current_path = img_path
            self.fade_switch_image(img_path)

            # ---- 创建 WAV 队列（解耦生产者与消费者） ----
            wav_queue: asyncio.Queue = asyncio.Queue()

            # ============================================================
            # 生产者协程：顺序克隆所有段，克隆完一段立刻开始下一段
            # ============================================================
            async def producer():
                for idx, unit in enumerate(all_units):
                    # ★ 打断检测：如果被打断，跳过剩余所有段的克隆
                    if INTERRUPT_EVENT.is_set():
                        remaining = total - idx
                        print(f"[打断] ⏭️ 打断信号已激活，跳过剩余 {remaining} 段克隆")
                        break

                    current = idx + 1
                    jp_text = unit.get("jp", "")
                    cn_text = unit.get("cn", "")
                    print(f"[克隆] 🔄 第 {current}/{total} 段克隆中..."
                          f"  JP: {jp_text[:40] if jp_text else '（无日语）'}")
                    wav_path = await self._prefetch_audio(unit)
                    print(f"[克隆] ✅ 第 {current}/{total} 段克隆完成")
                    await wav_queue.put({
                        "wav_path": wav_path,
                        "cn_text": cn_text,
                        "emotion": unit.get("emotion", "normal"),
                        "raw_tag": unit.get("raw_tag", ""),
                        "index": idx,
                        "total": total,
                        "wait_seconds": unit.get("wait_seconds", 0),
                    })
                # 哨兵：通知消费者无更多数据
                await wav_queue.put(None)

            # ============================================================
            # ★ v3.0 消费者协程：从 wav_queue 取 WAV → 更新 UI → 播放
            # 集成打断检测 + 指针追踪 + 切片回流 + 队列清空
            # ============================================================
            async def consumer():
                while True:
                    # ---- ★ v3.0 打断检测：每段播放前检查 INTERRUPT_EVENT ----
                    if INTERRUPT_EVENT.is_set():
                        print(f"\n[打断] ✂️ 打断信号已激活，立即停止播放队列")
                        # 记录当前播放索引（如果尚未播放任何段则取 -1）
                        current_idx = self._current_playing_idx

                        # ---- 切片：已说文本 vs 未说文本 ----
                        if current_idx >= 0 and current_idx < total:
                            # 已说文本：all_units[0 : current_idx + 1] 的中文
                            spoken_parts = [
                                u.get("cn", "") for u in all_units[0 : current_idx + 1]
                                if u.get("cn", "")
                            ]
                            # 未说文本：all_units[current_idx + 1 : ] 的中文
                            unspoken_parts = [
                                u.get("cn", "") for u in all_units[current_idx + 1 : ]
                                if u.get("cn", "")
                            ]
                        else:
                            # 还没来得及播放任何段，或索引异常
                            spoken_parts = []
                            unspoken_parts = [
                                u.get("cn", "") for u in all_units if u.get("cn", "")
                            ]

                        self._interrupt_spoken_text = "，".join(spoken_parts)
                        self._interrupt_unspoken_text = "，".join(unspoken_parts)
                        print(f"[打断] 📋 已说文本: {self._interrupt_spoken_text[:120]}...")
                        print(f"[打断] 📋 未说文本: {self._interrupt_unspoken_text[:120]}...")

                        # ---- 一键清空 wav_queue（消费掉所有剩余项 + 哨兵） ----
                        # 清空 wav_queue 中所有待消费的 WAV 项，直到遇到 None 哨兵
                        while True:
                            try:
                                remaining = wav_queue.get_nowait()
                                if remaining is None:
                                    break
                            except asyncio.QueueEmpty:
                                break
                        print(f"[打断] ✅ 队列已清空（wav_queue），打断记忆已缓存（注意：不触碰 _unit_queue，其中的新单元由 _handle_mako_logic 放入）")
                        return  # 退出消费者

                    # ---- 正常消费流程 ----
                    item = await wav_queue.get()
                    if item is None:  # 哨兵，生产者已结束
                        break

                    idx = item["index"]
                    current = idx + 1
                    wav_path = item["wav_path"]
                    cn_text = item["cn_text"]
                    emotion = item["emotion"]
                    raw_tag = item.get("raw_tag", "")

                    # ---- ★ v3.0 在每段播放前更新 _current_playing_idx ----
                    self._current_playing_idx = idx

                    print(f"\n[Queue] 🎬 正在播放第 {current}/{total} 段...")
                    jp_text = all_units[idx].get("jp", "")
                    print(f"[Queue]    JP: {jp_text[:60] if jp_text else '（无日语）'}")
                    print(f"[Queue]    CN: {cn_text[:60] if cn_text else '（无中文）'}")

                    # 播放（第一段 skip_ui=True，因为文字+立绘已提前渲染）
                    await self._play_audio_blocking(
                        wav_path=wav_path,
                        cn_text=cn_text,
                        emotion=emotion,
                        raw_tag=raw_tag,
                        skip_ui=(idx == 0),
                    )

                    # 段间间隔 0.25 秒，给用户留出理解时间
                    await asyncio.sleep(0.25)

                    # ---- [Wait: Xs] 等待标签：该段播放完毕后额外等待 X 秒 ----
                    # 将长 sleep 拆分为短轮询，每 0.25s 检查打断信号，确保打断及时响应
                    wait_seconds = item.get("wait_seconds", 0)
                    if wait_seconds > 0:
                        print(f"[Queue] ⏳ 等待 {wait_seconds} 秒（[Wait: {wait_seconds}s] 标签），每 0.25s 检测打断信号")
                        poll_interval = 0.25
                        poll_count = int(wait_seconds / poll_interval)
                        for _ in range(poll_count):
                            if INTERRUPT_EVENT.is_set():
                                print(f"[Queue] ⏭️ 等待被打断信号中断")
                                break
                            await asyncio.sleep(poll_interval)
                        # 处理余数部分（当 wait_seconds 不是 0.25 的整数倍时）
                        remaining = wait_seconds % poll_interval
                        if remaining > 0 and not INTERRUPT_EVENT.is_set():
                            await asyncio.sleep(remaining)

                    self._queue_processed = current
                    print(f"[Queue] ✅ 第 {current}/{total} 段播放完毕")

            # ---- 同时启动生产者 + 消费者 ----
            await asyncio.gather(producer(), consumer())

        except Exception as e:
            print(f"[Queue] ❌ 队列处理异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._queue_processing = False
            remaining = self._unit_queue.qsize()
            if remaining > 0:
                print(f"[Queue] 🔄 队列尚有 {remaining} 个单元未处理，自动重启队列")
                asyncio.create_task(self._process_unit_queue())
            else:
                print(f"[Queue] ✅ 全部 {self._queue_processed}/{self._queue_total} 段播放完成")

    # ---------------------------------------------------------------
    # 表情时间轴
    # ---------------------------------------------------------------
    async def _run_expression_timeline(self, wav_path: str, emotion: str = "normal"):
        """基于音频静音时间轴，定时切换茉子表情图片。"""
        timestamps = await asyncio.to_thread(
            mako_timeline.get_switch_timestamps, wav_path
        )
        start_time = time.time()
        for ts in timestamps:
            wait_time = ts - (time.time() - start_time)
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            new_path = mako_image_loader.provide_default_path(emotion)
            self.current_path = new_path
            self.fade_switch_image(new_path)

    # ---------------------------------------------------------------
    # 【主动搭话决策器】定时器回调
    # ---------------------------------------------------------------
    def _on_proactive_tick(self):
        """QTimer 回调：在事件循环中安全地创建异步任务。"""
        if not self._vision_ready or not self._proactive_enabled:
            return
        if self._is_speaking:
            # 正在说话，不打断当前发言
            return
        asyncio.create_task(self._proactive_think_loop())

    async def _proactive_think_loop(self):
        """
        后台主动思考循环——由 QTimer 周期触发。
        完整流程：
          1. 防冲突守卫
          2. 快速用户在场检测（MediaPipe）
          3. 认知沙盒 DeepThinker（5回合 VQA）
          4. 场景变化检测
          5. 对话版 DeepSeek B 生成台词（Unit 列表）
          6. 全部 Unit 入队 → 启动队列播放
        """
        global _deep_vision, _deep_thinker
        if _deep_vision is None or _deep_thinker is None:
            return

        loop = asyncio.get_running_loop()
        self._is_speaking = True
        try:
            # ---- 1. 快速检查用户是否在场 ----
            data, _ = _deep_vision.check_presence()
            if not data["is_present"]:
                # 用户不在摄像头前，不浪费算力
                return

            # ---- 2. 认知沙盒（后台线程，不阻塞 UI） ----
            think_result = await loop.run_in_executor(
                None, _deep_thinker.think_about_scene
            )
            digest = think_result.get("digest", "")

            # 暂用日志
            print(f"[暂用]qwen2+里人格分析的物理信息{digest}")

            if not digest or not think_result.get("has_frame", False):
                # 没有有效画面摘要
                return

            # ---- 3. 变化检测 ----
            if self._last_digest is not None and digest == self._last_digest:
                self._unchanged_count += 1
                if self._unchanged_count < 3:
                    # 少于 3 次相同 → 跳过，避免过于频繁
                    return
            else:
                self._unchanged_count = 0
                self._last_digest = digest

            # ---- 4. 场景有变化 → Letta Proactive Agent 生成 Unit 列表 ----
            print(f"\n[主动搭话] 💭 场景变化检测到，生成台词...")
            units = await loop.run_in_executor(
                None, get_proactive_reply, digest
            )

            if not units:
                print("[主动搭话] ⚠ 没有有效台词单元，跳过")
                return

            # ---- 5. 更新总计数并全部入队 ----
            total = len(units)
            self._queue_total = total
            for idx, unit in enumerate(units):
                await self._unit_queue.put(unit)
                print(f"[Queue] 📥 [主动搭话] 第 {idx+1}/{total} 段已入队")

            # ---- 6. 启动队列处理 ----
            if not self._queue_processing:
                asyncio.create_task(self._process_unit_queue())

        except Exception as e:
            print(f"[主动搭话] ❌ 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._is_speaking = False


# ============================================================
# 全局资源：FunASR 专用线程池 + 模型实例 + 监听器（惰性初始化）
# ============================================================
_asr_pool: concurrent.futures.ThreadPoolExecutor | None = None
_funasr_model = None
_listener: VoiceListener | None = None


async def _init_funasr_resources():
    """
    在后台线程中异步预加载 FunASR SenseVoice-Small 模型。
    使用 GPU (CUDA) 推理，显著降低显存占用（~200MB vs Whisper ~1.5GB）。
    """
    global _asr_pool, _funasr_model
    from funasr import AutoModel

    _asr_pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="funasr",
    )

    def _load_model():
        return AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model=None,         # 继续使用 Silero-VAD，不启用 FunASR 内置 VAD
            punc_model=None,        # 不启用标点恢复（保持与当前行为一致）
            device="cuda:0",
            disable_update=True,
        )

    loop = asyncio.get_running_loop()
    print("[main] ⏳ 正在预加载 FunASR 模型（iic/SenseVoiceSmall, CUDA）…")
    _funasr_model = await loop.run_in_executor(_asr_pool, _load_model)
    print("[main] ✅ FunASR 模型加载完成（单例，仅启动时加载一次）")


async def _init_voice_listener(widget: MakoFinalPet):
    """初始化 VoiceListener 并连接到 UI。"""
    global _asr_pool, _funasr_model, _listener
    if _funasr_model is None:
        print("[main] ⚠ FunASR 模型未加载，跳过语音监听")
        return

    listener = VoiceListener(
        asr_pool=_asr_pool,
        funasr_model=_funasr_model,
    )
    _listener = listener  # 保存全局引用，供关闭钩子使用

    # 将语音识别文本直接连接至 UI 用户输入入口（零改动 BubbleInput）
    listener.text_recognized.connect(widget._on_user_send)
    # 注意：不调用 listener.start() — 初始状态与 UI 的 _voice_enabled=False 保持一致，
    # 用户需通过右键菜单"语音输入：开"手动开启监听。
    print("[main] ✅ VoiceListener 已创建，初始为停止状态（等待用户右键开启）…")

    # 右键菜单"语音输入"切换连接：
    # 开 → listener.start() 恢复 VAD 检测；关 → listener.stop() 逻辑门控（不关物理流，保护蓝牙连接）
    def _on_voice_toggled(enabled: bool):
        if enabled:
            listener.start()
        else:
            listener.stop()

    widget.voice_toggled.connect(_on_voice_toggled)
    print("[main] ✅ 右键菜单语音输入切换已连接到 VoiceListener")


def _shutdown():
    """
    优雅关闭钩子（连接至 app.aboutToQuit）。
    关闭顺序：逻辑门控停 VAD → 等待 ASR 任务完成 → 物理关闭流 → 关线程池 → 停 pygame。
    杜绝 RuntimeError: cannot schedule new futures after shutdown。
    """
    global _listener, _asr_pool
    print("[main] ⏳ 正在优雅关闭语音模块…")
    # 第一步：逻辑门控，阻止 VAD 继续产生新任务
    if _listener is not None:
        _listener.stop()
    # 第二步：等待线程池中已提交的转写任务全部完成
    if _asr_pool is not None:
        _asr_pool.shutdown(wait=True)
        _asr_pool = None
    # 第三步：此时再无新任务，可以安全地物理关闭音频流
    if _listener is not None:
        _listener.close()
    # 第四步：关闭 pygame 混音器
    try:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
        print("[main] ✅ pygame.mixer 音频引擎已关闭")
    except Exception as e:
        print(f"[main] ⚠ pygame 关闭异常: {e}")
    print("[main] ✅ 语音模块已完全关闭")


# ============================================================
# 单次模型权重对齐 — GPT-SoVITS 服务初始化
# ============================================================
def _align_gptsovit_weights():
    """
    启动时一次性推送 GPT-SoVITS 模型权重路径，确保服务对齐。
    仅在程序启动时执行一次，不影响后续 TTS 调用性能。
    """
    _weights_dir = os.path.join(PROJECT_ROOT, "gptsovit_weights")
    GPT_WEIGHT_PATH = os.path.join(_weights_dir, "Mako_v2pp_v1-e15.ckpt")
    SOVITS_WEIGHT_PATH = os.path.join(_weights_dir, "Mako_v2pp_v1_e8_s208.pth")
    base_url = TTS_URL.replace("/tts", "").rstrip("/")
    # 正确的切换权重写法
    gpt_resp = requests.get(f"{base_url}/set_gpt_weights", params={"weights_path": GPT_WEIGHT_PATH})
    sovits_resp = requests.get(f"{base_url}/set_sovits_weights", params={"weights_path": SOVITS_WEIGHT_PATH})

    if gpt_resp.status_code == 200 and sovits_resp.status_code == 200:
        print("Mako 权重对齐成功！")
    else:
        print(f"对齐失败: GPT({gpt_resp.status_code}), SoVITS({sovits_resp.status_code})")


# ============================================================
# 启动前检查 — me_embedding.npy 是否存在
# ============================================================
def _check_tse_prerequisites():
    """检查 TSE 声纹锁的前置条件。"""
    embedding_path = "my_voice/me_embedding.npy"
    if not os.path.exists(embedding_path):
        print("=" * 60)
        print("⚠️  未检测到声纹文件 me_embedding.npy")
        print("    TSE 目标说话人提取功能将不可用。")
        print()
        print("📝 请先运行以下命令注册您的声纹：")
        print("    python register_me.py")
        print()
        print("   1. 将您的录音（WAV格式，3~30秒）放入 my_voice/ 目录")
        print("   2. 命名为 my_voice.wav")
        print("   3. 运行 python register_me.py")
        print("=" * 60)
    else:
        print(f"✅ 已检测到声纹文件 {embedding_path}")
        print("   TSE 目标说话人提取功能将在语音开启时自动激活。")


# ============================================================
# 异步入口
# ============================================================
if __name__ == "__main__":
    # 启动前检查

    _check_tse_prerequisites()

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    widget = MakoFinalPet()
    widget.show()

    # 音频流 + 线程池优雅关闭（停止 → 等待 → 释放）
    app.aboutToQuit.connect(_shutdown)

    # ============================================================
    # 异步后初始化：模型加载 + 语音监听 + 视觉模块 + 主动搭话
    # ============================================================
    async def _post_init():
        # ---- 第〇步：初始化 pygame 混音器（事件驱动播放引擎） ----
        pygame.mixer.init()
        pygame.init()
        print("[main] ✅ pygame.mixer 音频引擎已初始化（事件驱动模式）")

        from deep_thinker import DeepThinker
        from mako_cv_test import MakoVisionDeep

        # ---- 第一步：单次对齐 GPT-SoVITS 模型权重（仅 GS 模式需要） ----
        if widget._tts_backend == "GS":
            await loop.run_in_executor(None, _align_gptsovit_weights)
        else:
            print("[main] ⏭️ FA 模式，跳过 GPT-SoVITS 权重对齐")

        # ---- 第二步：预加载 FunASR 语音识别模型 ----
        await _init_funasr_resources()

        # ---- 第三步：启动语音监听 ----
        await _init_voice_listener(widget)

        # ---- 第四步：[可选] Qwen2-VL 视觉模块 — 默认关闭，右键菜单手动开启 ----
        # 视觉模型默认不加载，以节省 GPU 显存。
        # 如需使用，请在桌宠上右键 → 开启「视觉认知」。
        # 取消下方注释可在启动时自动加载。
        # try:
        #     def _load_vision():
        #         vision = MakoVisionDeep(camera_id=0)
        #         thinker = DeepThinker(
        #             vision=vision,
        #             thinking_deepseek_fn=get_thinking_reply,
        #         )
        #         return vision, thinker
        # 
        #     _vision, _thinker = await loop.run_in_executor(None, _load_vision)
        #     widget.set_vision_ready(_vision, _thinker)
        #     print("[main] ✅ Qwen2-VL + DeepThinker 视觉沙盒初始化完成")
        # except Exception as e:
        #     print(f"[main] ⚠️  视觉模块初始化失败（跳过主动视觉搭话）: {e}")
        print("[main] 👁️ 视觉模块默认关闭（右键菜单可手动开启）")

        # ---- 第五步：连接右键菜单视觉开关信号 ----
        widget.vision_toggled.connect(widget._toggle_vision_module)
        print("[main] ✅ 右键菜单视觉认知切换已连接")

        # ---- 连接右键菜单 TTS 音源切换信号 ----
        def _on_tts_toggled(backend: str):
            label = "GPT-SoVITS（本地）" if backend == "GS" else "Fish Audio（云端）"
            print(f"[main] 🎙 TTS 音源已切换至：{label}")

        widget.tts_toggled.connect(_on_tts_toggled)
        print("[main] ✅ 右键菜单 TTS 音源切换已连接")

        # ---- ★ 第六步：发送开机系统事件，触发茉子主动重逢搭讪 ----
        await widget._send_startup_greeting()

    loop.create_task(_post_init())


    with loop:
        loop.run_forever()
