"""
record_speaker_ref.py
======================
声纹注册录制脚本 — 录制目标说话人的 35 秒参考音频，
用于后续 register_me.py 提取声纹 embedding。

工作流程：
  1. 提示用户切换到免提模式
  2. 按回车开始 35 秒倒计时录音（sounddevice, 16kHz mono Int16）
  3. 保存原始录音 → speaker_ref_raw.wav
  4. 静音裁剪（剔除首尾绝对静音段）→ speaker_ref_final.wav
  5. 验证并打印录制文件信息
"""

import os
import time
import sys
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as write_wav

# ============================================================
# 常量
# ============================================================
SAMPLE_RATE = 16000          # 采样率 16kHz
CHANNELS = 1                  # 单声道
DTYPE = "int16"               # Int16 格式
RECORD_SECONDS = 35           # 录制时长（秒）
OUTPUT_DIR = "my_voice"     # 输出目录

# 静音裁剪阈值（对于 Int16，最大值为 32767，这里取 200 作为静音判定）
SILENCE_THRESHOLD = 200
# 裁剪时至少保留的静音前后 padding（秒）
PADDING_SECONDS = 0.2


def print_countdown_bar(current_sec: int, total_sec: int, bar_length: int = 40):
    """
    打印倒计时进度条。

    Parameters
    ----------
    current_sec : int
        当前已过去的秒数。
    total_sec : int
        总秒数。
    bar_length : int
        进度条字符长度。
    """
    progress = current_sec / total_sec
    filled = int(bar_length * progress)
    bar = "█" * filled + "░" * (bar_length - filled)
    remaining = total_sec - current_sec
    sys.stdout.write(f"\r⏳ 录音中 [{bar}] {current_sec:2d}/{total_sec}s  (剩余 {remaining:2d}s)")
    sys.stdout.flush()


def trim_silence_edges(audio: np.ndarray, threshold: int = SILENCE_THRESHOLD,
                       padding_samples: int = None) -> np.ndarray:
    """
    基于能量检测的首尾静音裁剪。

    从首尾分别向内搜索，找到第一个超过 threshold 的采样点，
    加上 padding_samples 的缓冲后裁剪。

    Parameters
    ----------
    audio : np.ndarray
        输入音频（1D, Int16）。
    threshold : int
        静音判定阈值（绝对值）。
    padding_samples : int, optional
        裁剪后保留的边缘缓冲样本数。默认取 0.2 秒。

    Returns
    -------
    np.ndarray
        裁剪后的音频。
    """
    if padding_samples is None:
        padding_samples = int(PADDING_SECONDS * SAMPLE_RATE)

    # 取绝对值
    audio_abs = np.abs(audio)

    # 找到第一个超过阈值的索引（从头开始）
    start_idx = 0
    for i in range(len(audio_abs)):
        if audio_abs[i] > threshold:
            start_idx = max(0, i - padding_samples)
            break
    else:
        # 整段音频都是静音 → 返回原音频
        print("⚠️  警告：整段音频均为静音，未做裁剪")
        return audio

    # 找到最后一个超过阈值的索引（从尾开始）
    end_idx = len(audio)
    for i in range(len(audio_abs) - 1, -1, -1):
        if audio_abs[i] > threshold:
            end_idx = min(len(audio), i + padding_samples)
            break

    trimmed = audio[start_idx:end_idx]

    trimmed_sec = len(trimmed) / SAMPLE_RATE
    original_sec = len(audio) / SAMPLE_RATE
    print(f"\n🔇 静音裁剪: {original_sec:.2f}s → {trimmed_sec:.2f}s "
          f"(裁剪掉了 {(original_sec - trimmed_sec):.2f}s 首尾静音)")

    return trimmed


def main():
    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("🎙️  声纹注册 — 参考音频录制")
    print("=" * 60)
    print()
    print("📌 请确保蓝牙耳机已切换至通话模式（Hands-free），并保持环境安静。")
    print()
    print(f"⏱️  录制时长: {RECORD_SECONDS} 秒")
    print(f"🎚️  采样率: {SAMPLE_RATE} Hz | 声道: 单声道 (Mono) | 格式: Int16")
    print()

    # 等待用户按回车开始
    input("🟢 准备就绪后按 Enter 键开始录音...")
    print()

    # ============================================================
    # 开始录音
    # ============================================================
    print(f"\n🔴 开始录音（{RECORD_SECONDS}秒）...\n")

    # 用于存储录音数据的列表
    audio_chunks = []

    def callback(indata, frames, time_info, status):
        """sounddevice 流回调：收集音频块。"""
        if status:
            print(f"⚠️  音频状态: {status}")
        audio_chunks.append(indata.copy())

    # 创建音频输入流
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=DTYPE,
        callback=callback,
    )

    stream.start()

    # 倒计时进度条
    try:
        for sec in range(RECORD_SECONDS):
            time.sleep(1)
            print_countdown_bar(sec + 1, RECORD_SECONDS)
    except KeyboardInterrupt:
        print("\n\n⚠️  录音被用户中断")

    # 停止流
    stream.stop()
    stream.close()

    print("\n\n✅ 录音完成！正在处理音频...\n")

    # ============================================================
    # 拼接音频数据并保存原始文件
    # ============================================================
    raw_audio = np.concatenate(audio_chunks)  # shape: (samples, 1)
    raw_audio = raw_audio.flatten()           # shape: (samples,)

    raw_path = os.path.join(OUTPUT_DIR, "speaker_ref_raw.wav")
    write_wav(raw_path, SAMPLE_RATE, raw_audio.astype(np.int16))
    print(f"💾 原始录音已保存: {raw_path}")

    # ============================================================
    # 静音裁剪
    # ============================================================
    try:
        trimmed_audio = trim_silence_edges(raw_audio)
    except Exception as e:
        print(f"⚠️  静音裁剪过程出现异常: {e}")
        print("   将直接使用原始录音作为最终文件。")
        trimmed_audio = raw_audio

    final_path = os.path.join(OUTPUT_DIR, "speaker_ref_final.wav")
    write_wav(final_path, SAMPLE_RATE, trimmed_audio.astype(np.int16))
    print(f"💾 裁剪后音频已保存: {final_path}")

    # ============================================================
    # 验证检查
    # ============================================================
    print()
    print("=" * 60)
    print("📊 录制结果验证")
    print("=" * 60)

    # 验证最终文件
    from scipy.io.wavfile import read as read_wav
    fs, data = read_wav(final_path)
    duration = len(data) / fs

    print(f"  文件路径: {final_path}")
    print(f"  采样率:   {fs} Hz  {'✅ 符合 16000Hz 要求' if fs == 16000 else '❌ 不符合要求!'}")
    print(f"  声道数:   {data.ndim} {'(单声道)' if data.ndim == 1 else '(多声道)'}")
    print(f"  时长:     {duration:.2f} 秒 ({len(data)} 样本)")
    print(f"  数据格式: {data.dtype}")
    print()

    if fs == 16000 and data.ndim == 1:
        print("🎉 所有检查通过！声纹参考录制成功。")
        print()
        print("👉 下一步运行 register_me.py 来提取声纹 embedding：")
        print("   python register_me.py")
    else:
        print("⚠️  检查未通过，请重试录制。")

    # 同时也验证原始文件
    fs_raw, data_raw = read_wav(raw_path)
    duration_raw = len(data_raw) / fs_raw
    print()
    print(f"📎 原始录音文件: {raw_path}")
    print(f"   采样率: {fs_raw} Hz | 时长: {duration_raw:.2f}s | 格式: {data_raw.dtype}")


if __name__ == "__main__":
    main()
