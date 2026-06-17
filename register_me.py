"""
register_me.py
==============
声纹注册脚本 — 使用 ModelScope 的 CAM++ 声纹模型从 my_voice.wav 提取目标说话人声纹特征。

用法：
  1. 将目标说话人的录音文件（WAV 格式，约 3–30 秒）放置到 my_voice/ 目录，命名为 my_voice.wav
  2. 运行：python register_me.py
  3. 生成的 me_embedding.npy 将自动被 voice_listener.py 加载，用于 TSE 目标说话人提取。

依赖：
  pip install modelscope soundfile numpy torch
"""

import os
import sys
import numpy as np
import soundfile as sf


# ============================================================
# 强制环境修复（Monkey Patch）
# 兼容新版 torchaudio（移除了 set_audio_backend）
# ============================================================
import torchaudio
import types

if not hasattr(torchaudio, 'set_audio_backend'):
    torchaudio.set_audio_backend = lambda x: None

# wespeaker 兼容补丁（某些模型内部仍可能引用）
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


# ---- 设置 ModelScope 镜像源（可选，提升稳定性） ----
os.environ.setdefault('MODELSCOPE_CACHE', os.path.join(os.path.dirname(__file__), '.modelscope_cache'))
# 如果需要切换镜像源，取消下面一行的注释：
# os.environ['MODELSCOPE_CACHE'] = os.path.join(os.path.dirname(__file__), '.modelscope_cache')


def register(target_wav: str = "my_voice/my_voice.wav", output_npy: str = "me_embedding.npy") -> np.ndarray:
    """
    从指定 WAV 文件中提取声纹 Embedding，保存为 .npy 文件。

    Parameters
    ----------
    target_wav : str
        目标说话人录音路径（默认 my_voice/my_voice.wav）。
    output_npy : str
        输出 .npy 文件路径（默认 me_embedding.npy，项目根目录）。

    Returns
    -------
    np.ndarray
        提取的声纹特征向量。
    """
    # ---- 检查输入文件 ----
    if not os.path.exists(target_wav):
        print(f"❌ 未找到目标录音文件: {target_wav}")
        print("   请将目标说话人的录音（WAV 格式，建议 3~30 秒）放置到 my_voice/ 目录后重试。")
        sys.exit(1)

    # ---- 验证音频格式 ----
    try:
        data, sr = sf.read(target_wav)
        # 如果是立体声（多通道），取左声道
        if data.ndim > 1:
            print(f"📢 检测到多声道音频（{data.shape[1]}声道），将取左声道。")
            data = data[:, 0]
        duration = len(data) / sr
        print(f"📂 读取音频: {target_wav} | 采样率: {sr}Hz | 时长: {duration:.1f}秒")
        if duration < 1.0:
            print("⚠️  录音过短（<1秒），建议使用 3~30 秒的清晰录音以获得最佳声纹效果。")
        # 如果需要重采样到 16kHz
        if sr != 16000:
            print(f"⏳ 正在将音频从 {sr}Hz 重采样至 16000Hz…")
            import torch
            import torchaudio.transforms as T
            audio_tensor = torch.from_numpy(data).float()
            if audio_tensor.ndim == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # (1, T)
            resampler = T.Resample(sr, 16000)
            audio_tensor = resampler(audio_tensor)
            data = audio_tensor.squeeze(0).numpy()
            sr = 16000
            print(f"✅ 重采样完成（16000Hz）")

    except Exception as e:
        print(f"❌ 无法读取音频文件: {e}")
        sys.exit(1)

    # ---- 使用 ModelScope CAM++ 模型提取声纹 ----
    print("⏳ 正在下载模型，请稍候…")
    print("   模型: damo/speech_campplus_sv_zh-cn_16k-common（高精度声纹识别）")
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks

        # 使用 ModelScope pipeline 加载 CAM++ 声纹模型
        # model_revision: 固定版本号，防止缓存失效触发重复下载
        sv_pipeline = pipeline(
            Tasks.speaker_verification,
            model='damo/speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0',
        )
        print("✅ 模型加载完成")

    except Exception as e:
        print(f"❌ ModelScope 模型加载失败: {e}")
        print("   提示1: 确保已安装 modelscope: pip install modelscope")
        print("   提示2: 如果网络卡顿，尝试设置镜像源:")
        print("          pip install modelscope -i https://mirrors.aliyun.com/pypi/simple/")
        print("   提示3: 或设置环境变量: set MODELSCOPE_MIRROR=https://modelscope.aliyun.com")
        sys.exit(1)

    # ---- 提取 Embedding ----
    print("⏳ 正在提取声纹特征…")
    try:
        # SpeakerVerificationPipeline 的 __call__ 签名：
        #   __call__(self, in_audios: Union[np.ndarray, list], output_emb: bool = False)
        # - in_audios: 列表，每个元素可以是音频路径(str)或 numpy 数组(1D)
        # - output_emb=True 时返回 {'outputs': ..., 'embs': np.ndarray}
        # 注意：该 pipeline 不支持 dict 格式输入！
        result = sv_pipeline([data], output_emb=True)

        # 提取 embedding（output_emb=True 时 result['embs'] 形状为 [1, emb_dim]）
        if isinstance(result, dict) and 'embs' in result:
            embedding = np.array(result['embs'], dtype=np.float32)
            if embedding.ndim > 1:
                embedding = embedding.flatten()
        else:
            # 意外格式，尝试兜底
            print(f"⚠️ 返回值结构异常: {type(result)}")
            if isinstance(result, dict):
                print(f"   键: {list(result.keys())}")
                # 尝试取第一个值
                if result:
                    embedding = np.array(list(result.values())[0], dtype=np.float32)
                else:
                    raise RuntimeError("pipeline 返回空 dict")
            else:
                embedding = np.array(result, dtype=np.float32)
            if embedding.ndim > 1:
                embedding = embedding.flatten()

        print(f"✅ 声纹特征提取完成（维度: {embedding.shape}）")


    except Exception as e:
        print(f"❌ 声纹提取失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ---- 保存 ----
    np.save(output_npy, embedding)
    print(f"✅ 声纹指纹已生成 → {output_npy}")
    print(f"   路径: {os.path.abspath(output_npy)}")
    print(f"   维度: {embedding.shape}")
    print(f"   类型: {embedding.dtype}")
    print("\n💡 现在可以启动 main.py，VoiceListener 将自动加载此声纹进行目标说话人提取。")

    return embedding


if __name__ == "__main__":
    register("my_voice/my_voice.wav", "my_voice/me_embedding.npy")
