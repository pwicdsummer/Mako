"""
mako_timeline.py
================
音频静音检测 → 表情切换时间轴。
纯算力模块，不包含任何异步 / GUI 代码。
"""

from pydub import AudioSegment
from pydub.silence import detect_silence


def get_switch_timestamps(wav_path: str) -> list[float]:
    """
    分析 WAV 音频的静音区间，返回适合切换表情的时间戳（单位：秒）。

    将每个静音区间的中值点作为切换时间点。

    Parameters
    ----------
    wav_path : str
        WAV 音频文件的绝对路径。

    Returns
    -------
    list[float]
        以秒为单位的时间戳列表，例如 [1.2, 3.8, 5.5]。
    """
    audio = AudioSegment.from_file(wav_path)
    silence_intervals = detect_silence(
        audio,
        min_silence_len=200,
        silence_thresh=-40,
    )
    # 每个静音区间取中值（毫秒），再转换为秒
    timestamps = [(start + end) / 2 / 1000 for start, end in silence_intervals]
    return timestamps
