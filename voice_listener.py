"""
voice_listener.py
=================
实时语音监听模块 — 基于 silero-vad + FunASR (SenseVoice-Small) + CAM++ 声纹锁。

架构（数据流）：
  麦克风音频（32ms/帧）
    │
    ├─ [VAD 检测] silero-vad VADIterator
    │   实时判定语音活动端点
    │
    ├─ [声纹锁] CAM++ Speaker Verification（cosine similarity）
    │   对比说话人声纹是否与注册的 me_embedding.npy 匹配
    │
    └─ [ASR 转写] FunASR SenseVoice-Small (CUDA)
        输出文本 → 情感标签清洗 → pyqtSignal(str) → UI 主线程

声纹锁特性：
  - 启动时自动加载 me_embedding.npy（由 register_me.py 生成）
  - 若 me_embedding.npy 不存在，程序正常降级运行（弹窗提示，不阻塞）
  - 声纹验证 CAM++ 运行在 CUDA 上（RTX 5060），模型仅在启动时加载一次
  - 声纹匹配使用 cosine similarity，阈值 0.5（可配置）

情感标签：
  - FunASR 输出可能包含 <|HAPPY|><|zh|><|NEUTRAL|> 等标签
  - 清洗函数 clean_funasr_output() 提取标签并返回纯净文本
  - 提取到的情感标签存储于 self.current_emotion，供后续表情切换使用

对接点：
  VoiceListener.text_recognized.connect(pet_widget._on_user_send)
"""

# ============================================================
# 强制环境修复（Monkey Patch）— 兼容新版 torchaudio
# 必须在任何其他导入之前执行
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
import os
import re
import numpy as np
import torch
import concurrent.futures
from PyQt5.QtCore import QObject, pyqtSignal
import sounddevice as sd


# ============================================================
# 常量
# ============================================================
SAMPLE_RATE = 16000          # silero-vad / FunASR / CAM++ 统一采样率
BLOCK_SIZE = 512             # silero-vad 推荐的音频块大小（约 32ms）
VAD_THRESHOLD = 0.5          # VAD 判定为语音的概率阈值
MIN_SILENCE_MS = 300         # 判定说话结束的最小静音时长（毫秒）
EMBEDDING_PATH = "my_voice/me_embedding.npy"  # 声纹指纹路径（项目根目录）

# 声纹锁阈值（cosine similarity，范围 -1 ~ 1）
# 推荐 0.5~0.7：过低易误放，过高易拒真
VOICE_MATCH_THRESHOLD = 0.4

# 降级开关：设为 True 可在 TSE/声纹锁失效时静默透传（默认 False：明确报错）
SILENT_DEGRADE = False


# ============================================================
# 情感标签清洗函数（模块级，可被 main.py 导入）
# ============================================================

# FunASR 情感/语言标签正则：匹配 <|HAPPY|>、<|zh|>、<|NEUTRAL|> 等
_FUNASR_TAG_RE = re.compile(r'<\|[A-Z_]+\|>')


def clean_funasr_output(raw_text: str) -> tuple[str, list[str]]:
    """
    清洗 FunASR 输出，分离情感标签和纯净文本。

    Parameters
    ----------
    raw_text : str
        FunASR 原始输出，例如 "<|HAPPY|><|zh|><|NEUTRAL|>今天天气真好啊"

    Returns
    -------
    tuple[str, list[str]]
        (纯净文本, 提取到的标签列表)
        例如 ("今天天气真好啊", ["HAPPY", "zh", "NEUTRAL"])
    """
    tags = re.findall(_FUNASR_TAG_RE, raw_text)
    pattern = r"<.*?>"
    clean = re.sub(pattern, '', raw_text).strip()

    # 提取标签内容（去掉 <||> 包裹符号）
    tag_values = [tag.strip('<|>') for tag in tags]

    return clean, tag_values


# ============================================================
# VoiceListener 主类
# ============================================================

class VoiceListener(QObject):
    """
    实时语音监听器：VAD 检测 → CAM++ 声纹锁 → FunASR 转写 → pyqtSignal(str) 输出。

    Parameters
    ----------
    asr_pool : concurrent.futures.ThreadPoolExecutor
        专用于 ASR 推理的独立线程池（max_workers=1），与默认线程池隔离。
    funasr_model : funasr.AutoModel
        已预加载的 FunASR SenseVoice-Small 模型实例。
    sample_rate : int
        音频采样率（默认 16000）。
    threshold : float
        VAD 语音概率阈值（默认 0.5）。
    min_silence_duration_ms : int
        判定说话结束的最小静音时长（默认 500ms）。
    """

    # ---- 线程安全信号：ASR 后台线程 → UI 主线程 ----
    text_recognized = pyqtSignal(str)

    def __init__(
        self,
        asr_pool: concurrent.futures.ThreadPoolExecutor,
        funasr_model,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = VAD_THRESHOLD,
        min_silence_duration_ms: int = MIN_SILENCE_MS,
    ):
        super().__init__()
        self._asr_pool = asr_pool
        self._funasr_model = funasr_model
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_duration_ms = min_silence_duration_ms

        # 运行时状态
        self._is_running = False
        self.enabled = False
        self._audio_buffer: list[np.ndarray] = []
        self._vad_model = None
        self._vad_iterator = None
        self._audio_stream: sd.InputStream | None = None

        # ---- CAM++ 声纹锁状态 ----
        self._sv_pipeline = None       # CAM++ SpeakerVerificationPipeline（惰性初始化）
        self._sv_device = None         # CUDA 设备
        self._target_embedding = None  # me_embedding.npy → numpy array
        self._voicelock_available = False  # 声纹锁是否就绪（声纹+模型加载成功）

        # ---- 情感标签状态 ----
        self.current_emotion: list[str] = []  # 最近一次识别的情感标签

        # 在构造函数中预加载声纹 Embedding（同步，不阻塞）
        self._load_embedding()

    # ----------------------------------------------------------
    # 声纹 Embedding 加载
    # ----------------------------------------------------------
    def _load_embedding(self):
        """从 me_embedding.npy 加载目标说话人声纹特征。"""
        if not os.path.exists(EMBEDDING_PATH):
            print(f"[VoiceListener] ⚠️  未找到声纹文件 {EMBEDDING_PATH}")
            print("   语音识别将降级运行（无声纹锁）。")
            print("   请先运行 register_me.py 注册目标说话人声纹。")
            self._target_embedding = None
            return

        try:
            emb = np.load(EMBEDDING_PATH)
            # 确保是 1D 向量
            if emb.ndim == 2:
                emb = emb.flatten()
            self._target_embedding = emb.astype(np.float32)
            print(f"[VoiceListener] ✅ 目标声纹已加载（{EMBEDDING_PATH}，维度: {emb.shape}）")
        except Exception as e:
            print(f"[VoiceListener] ❌ 声纹文件加载失败: {e}")
            self._target_embedding = None

    def _ensure_voicelock(self):
        """
        惰性初始化 CAM++ SpeakerVerificationPipeline。
        仅在首次声纹验证时加载，避免启动阻塞。
        全量计算运行在 CUDA 上（RTX 5060）。

        Returns
        -------
        bool
            True 表示声纹锁就绪，False 表示不可用。
        """
        if self._sv_pipeline is not None:
            return True
        if self._target_embedding is None:
            return False  # 无声纹，声纹锁不可用

        try:
            from modelscope.pipelines import pipeline
            from modelscope.utils.constant import Tasks

            print("[VoiceListener] ⏳ 正在预加载 CAM++ 声纹模型…")
            # 创建 SpeakerVerificationPipeline（首次会联网下载，后续使用缓存）
            self._sv_pipeline = pipeline(
                Tasks.speaker_verification,
                model='damo/speech_campplus_sv_zh-cn_16k-common',
                model_revision='v1.0.0',  # 固定版本，避免缓存失效触发重复下载
            )

            # 强制迁移到 CUDA（RTX 5060）
            if torch.cuda.is_available():
                self._sv_device = torch.device("cuda")
                # 将 pipeline 内部模型迁移到 CUDA
                if hasattr(self._sv_pipeline, 'model'):
                    self._sv_pipeline.model.to(self._sv_device)
                    self._sv_pipeline.model.eval()
                    torch.cuda.empty_cache()
                elif hasattr(self._sv_pipeline, 'pipeline'):
                    for attr_name in dir(self._sv_pipeline.pipeline):
                        attr = getattr(self._sv_pipeline.pipeline, attr_name)
                        if isinstance(attr, torch.nn.Module):
                            attr.to(self._sv_device)
                            attr.eval()
                else:
                    # 兜底：递归遍历 pipeline 中的模块
                    for attr_name in dir(self._sv_pipeline):
                        try:
                            attr = getattr(self._sv_pipeline, attr_name)
                            if isinstance(attr, torch.nn.Module):
                                attr.to(self._sv_device)
                                attr.eval()
                        except Exception:
                            pass
                print(f"[VoiceListener] ✅ CAM++ 声纹模型已迁移至 {self._sv_device} (RTX 5060)")
            else:
                self._sv_device = torch.device("cpu")
                print("[VoiceListener] ⚠️  CUDA 不可用，声纹模型运行在 CPU（推理可能会很慢）")

            self._voicelock_available = True
            print("[VoiceListener] ✅ CAM++ 声纹锁加载完成（单例，仅启动时加载一次）")
            return True

        except Exception as e:
            print(f"[VoiceListener] ❌ 声纹模型加载失败: {e}")
            print("   语音识别将降级运行（无声纹锁功能）。")
            print("   提示：如果网络卡顿，请尝试设置环境变量：")
            print("          set MODELSCOPE_MIRROR=https://modelscope.aliyun.com")
            self._voicelock_available = False
            return False

    # ----------------------------------------------------------
    # 声纹锁推理（后台线程中执行）
    # ----------------------------------------------------------
    def _voicelock_filter(self, audio: np.ndarray) -> np.ndarray | None:
        """
        在后台线程中对语音段执行 CAM++ 声纹验证。
        对比说话人是否与注册的 me_embedding.npy 匹配。

        Parameters
        ----------
        audio : np.ndarray
            VAD 截断后的语音段（float32, 16kHz）。

        Returns
        -------
        np.ndarray | None
            匹配 → 返回原始音频（放行），交由 ASR 转写。
            不匹配 → 返回 None（丢弃）。
        """
        if not self._voicelock_available or self._target_embedding is None:
            return audio  # 降级：无声纹锁，直接透传

        try:
            # 确保输入音频足够长（CAM++ 至少需要 ~1 秒音频）
            if len(audio) < self._sample_rate * 0.5:
                return None  # 太短，无法有效验证，丢弃

            print("[TSE] 正在对比声纹向量...")

            # SpeakerVerificationPipeline 输入格式：
            # __call__(in_audios, output_emb=True)
            #  in_audios: list of (audio_path str or np.ndarray)
            #  output_emb=True: 返回 {'embs': np.ndarray}
            result = self._sv_pipeline([audio], output_emb=True)

            if not isinstance(result, dict) or 'embs' not in result:
                print(f"[TSE] ⚠️ 声纹提取异常（返回值格式不符），降级放行")
                return audio

            # 提取当前说话人 embedding
            current_emb = np.array(result['embs'], dtype=np.float32)
            if current_emb.ndim > 1:
                current_emb = current_emb.flatten()

            # cosine similarity
            ref_norm = self._target_embedding / (np.linalg.norm(self._target_embedding) + 1e-8)
            cur_norm = current_emb / (np.linalg.norm(current_emb) + 1e-8)
            similarity = float(np.dot(ref_norm, cur_norm))

            if similarity >= VOICE_MATCH_THRESHOLD:
                print(f"[TSE] ✅ 声纹匹配（cosine相似度: {similarity:.3f}），放行")
                tse_duration = len(audio) / self._sample_rate
                print(f"[TSE] 处理完成，输出音频长度: {tse_duration:.1f}s")
                return audio
            else:
                print(f"[TSE] ❌ 声纹不匹配（cosine相似度: {similarity:.3f} < {VOICE_MATCH_THRESHOLD}），丢弃")
                return None

        except Exception as e:
            error_msg = f"[VoiceListener] ⚠️ 声纹验证异常: {e}"
            if SILENT_DEGRADE:
                print(f"{error_msg}（降级模式：放行）")
                return audio
            else:
                print(f"{error_msg}（丢弃此段音频）")
                return None

    # ----------------------------------------------------------
    # 公开接口
    # ----------------------------------------------------------
    def start(self):
        """启动麦克风监听（静默运行，后台常驻）。"""
        if self._is_running:
            return
        self._is_running = True

        # ---- 延迟导入 + 初始化 VAD —— 无论流是否已存在，VAD 都必须重新创建 ----
        from silero_vad import load_silero_vad
        from silero_vad.utils_vad import VADIterator

        self._vad_model = load_silero_vad()
        # 保持 VAD 在 CPU 上运行：silero-vad TorchScript 内部 STFT buffer 无法随 .to("cuda") 迁移
        # 且 VAD 推理极轻量（<1ms/帧），CPU 运行完全不影响实时性能
        self._vad_iterator = VADIterator(
            model=self._vad_model,
            threshold=self._threshold,
            sampling_rate=self._sample_rate,
            min_silence_duration_ms=self._min_silence_duration_ms,
            speech_pad_ms=30,
        )

        # ---- 启动/重启音频流（如果之前被 close 彻底关闭了才重新创建） ----
        if self._audio_stream is None:
            self._audio_buffer = []
            self._audio_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=BLOCK_SIZE,
                callback=self._audio_callback,
            )
            self._audio_stream.start()
            print("[VoiceListener] ✅ 麦克风监听已启动（静默后台运行）")
        else:
            # 流还在（蓝牙兼容模式），只清空缓冲区
            self._audio_buffer = []
            print("[VoiceListener] ✅ 麦克风监听已恢复（音频流保持常驻）")

    def stop(self):
        """
        停止麦克风监听 — 逻辑门控模式。
        不物理关闭音频流（不调用 stream.stop/close），
        仅通过 _is_running = False 让 _audio_callback 丢弃所有音频帧。
        这样可以防止蓝牙耳机因协议切换（免提↔立体声）导致 TTS 播放中断。

        线程安全：先设 _is_running = False 让回调立即退出，短暂阻塞确保
        正在执行的回调完成，再清理 _vad_iterator，杜绝竞态条件。
        """
        self._is_running = False
        self._audio_buffer = []
        # 等待声卡回调线程中的残留帧排空（约 32ms 一个周期，等 50ms 足够）
        import time
        time.sleep(0.05)
        self._vad_iterator = None
        print("[VoiceListener] ⏹ 麦克风监听已停止（音频流保持打开，仅逻辑门控）")

    def close(self):
        """
        彻底释放音频流资源（仅程序退出时由 _shutdown 调用）。
        与 stop() 不同，此方法会物理关闭 sounddevice 流。
        """
        self._is_running = False
        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception:
                pass
            self._audio_stream = None
        self._audio_buffer = []
        self._vad_iterator = None
        print("[VoiceListener] 🔇 音频流已彻底关闭")

    # ----------------------------------------------------------
    # 音频回调（sounddevice 音频线程 → 极轻量，不可阻塞）
    # ----------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status):
        """
        sounddevice 实时音频回调。
        每次传入 BLOCK_SIZE=512 个 float32 样本（约 32ms 音频）。
        此函数必须在 32ms 内返回，仅做缓冲区累积。
        声纹锁和 ASR 的推理在后台线程中处理，不阻塞音频流。
        """
        # ---- 严格守卫：逻辑门控 + VAD 已初始化，缺一不可 ----
        if not self._is_running or self._vad_iterator is None:
            return
        if status:
            if status.input_overflow:
                print("[VoiceListener] ⚠ 音频输入溢出")
            return

        # float32 → torch tensor（VAD 输入）
        audio_chunk: np.ndarray = indata.flatten()  # shape: (512,)
        audio_tensor = torch.from_numpy(audio_chunk.copy())

        # ---- VAD 实时检测（保持不变） ----
        speech_dict = self._vad_iterator(audio_tensor, return_seconds=False)

        if speech_dict is not None:
            if "start" in speech_dict:
                # 语音开始：清空旧缓冲区，开始累积
                self._audio_buffer = [audio_chunk]

            elif "end" in speech_dict:
                # 语音结束：追加最后一块 → 提交声纹锁 + FunASR 转写（后台线程）
                self._audio_buffer.append(audio_chunk)
                full_audio = np.concatenate(self._audio_buffer)
                self._audio_buffer = []

                # 提交到独立 ASR 线程池（非阻塞）
                # 捕获 RuntimeError：解释器退出时线程池可能已关闭
                try:
                    self._asr_pool.submit(
                        self._transcribe_with_voicelock, full_audio
                    )
                except RuntimeError:
                    pass

        else:
            # 无 VAD 事件触发：检查是否仍处于说话状态
            if self._vad_iterator.triggered:
                self._audio_buffer.append(audio_chunk)

    # ----------------------------------------------------------
    # CAM++ 声纹锁 → FunASR 流水线（运行在独立后台线程中）
    # ----------------------------------------------------------
    def _transcribe_with_voicelock(self, audio: np.ndarray):
        """
        完整推理流水线（在 ASR 线程池中执行）：
          1. 首次调用时惰性初始化 CAM++ 声纹模型
          2. 声纹锁验证：对比 embedding cosine similarity
          3. FunASR SenseVoice-Small 转写（含情感标签输出）
          4. 清洗情感标签，提取纯净文本
          5. 通过 pyqtSignal 将文本发射回 UI 主线程
        """
        try:
            # ---- 第一步：确保声纹模型已加载 ----
            if self._target_embedding is not None and not self._voicelock_available:
                self._ensure_voicelock()

            # ---- 第二步：CAM++ 声纹锁验证（保持不变） ----
            if self._voicelock_available and self._target_embedding is not None:
                audio = self._voicelock_filter(audio)
                # 声纹不匹配时返回 None
                if audio is None:
                    return

            # ---- 第三步：FunASR 转写（替换 faster-whisper） ----
            audio = audio.astype(np.float32)

            print("[FunASR] 接收到声纹锁过滤后的音频，开始转写...")
            result = self._funasr_model.generate(
                input=audio,
                cache={},
                language="auto",      # 自动检测语言（中/英/日）
                use_itn=True,         # 启用逆文本正则化（数字、标点等）
                ban_emo_unk=False, 
                hotword = ["茉子","makosama","解包","gemini"],      # 允许情感标签输出
            )

            # FunASR 返回格式：[{"text": "...", "key": "...", ...}]
            if isinstance(result, list) and len(result) > 0:
                raw_text = result[0].get("text", "")
            elif isinstance(result, dict):
                raw_text = result.get("text", "")
            else:
                raw_text = str(result) if result else ""
            
            print("【暂用】源文本："+raw_text)
            # ---- 第四步：情感标签清洗 ----
            clean_text, emotion_tags = clean_funasr_output(raw_text)
            print("【暂用】清理后的文本："+clean_text)
            # 保存情感标签供外部读取
            self.current_emotion = emotion_tags

            if clean_text:
                lock_tag = " [VOICELOCK]" if self._voicelock_available else ""
                emotion_tag = ""
                if emotion_tags:
                    emotion_tag = f" [情感:{'|'.join(emotion_tags)}]"
                print(f"[VoiceListener]{lock_tag}{emotion_tag} 🎤 识别: {clean_text}")

                # 发射纯净文本（不含 <|...|> 标签）
                self.text_recognized.emit(clean_text)
            else:
                print("[VoiceListener] ⚪ 识别结果为空")

            # 释放 FunASR 推理占用的 GPU 显存，为 LLM 预留空间
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[VoiceListener] ❌ 转写失败: {e}")
            import traceback
            traceback.print_exc()

    @property
    def is_voicelock_active(self) -> bool:
        """声纹锁是否处于激活状态。"""
        return self._voicelock_available
