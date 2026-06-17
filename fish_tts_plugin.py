"""
fish_tts_plugin.py
==================
Fish Audio 云端 TTS 插件。
从 fish_audio_cloud_api_tts_test.py 移植，包装为统一接口供 main.py 调用。

接口：
    async def fish_download_tts(text: str) -> str | None

使用方法：
    from fish_tts_plugin import fish_download_tts
    mp3_path = await fish_download_tts("こんにちは")
"""

import httpx
import asyncio
import os
import time

# ---- 配置 ----
API_KEY = os.getenv("FISH_AUDIO_API_KEY", "")
VOICE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_data")
TTS_URL = "https://api.fish.audio/v1/tts"
REFERENCE_ID = os.getenv("FISH_AUDIO_REFERENCE_ID", "")

# ---- 参考音频映射表（可选扩） ----
# Fish Audio 云端本身不区分情绪，但可以注册多个 reference_id
# 此处预留映射表，方便以后切换不同声线模板
REFERENCE_ID_MAP = {
    "default": REFERENCE_ID,
    # "shy": "另一个reference_id",
    # "happy": "又一个reference_id",
}


async def fish_download_tts(text: str, reference_id: str | None = None) -> str | None:
    """
    调用 Fish Audio 云端 TTS API，异步下载 MP3 音频。

    参数
    ----------
    text : str
        要合成的文本（日语或中文均可，API 自动检测语言）。
    reference_id : str | None
        可选的参考音频 ID。为 None 时使用默认 ID。

    返回
    -------
    str | None
        下载成功的 MP3 文件路径，失败返回 None。
    """
    ref_id = reference_id or REFERENCE_ID_MAP["default"]

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "text": text,
        "reference_id": ref_id,
        "format": "mp3",
        "latency": "normal",
    }

    try:
        async with httpx.AsyncClient(timeout=60.0, trust_env=True) as client:
            async with client.stream(
                "POST", TTS_URL, headers=headers, json=payload
            ) as response:
                if response.status_code == 200:
                    os.makedirs(VOICE_DIR, exist_ok=True)
                    fname = f"MakoFish_{time.strftime('%Y%m%d_%H%M%S')}.mp3"
                    fpath = os.path.join(VOICE_DIR, fname)
                    with open(fpath, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    print(f"[Fish TTS] ✅ 云端合成成功: {fpath}")
                    return fpath
                else:
                    error_msg = await response.aread()
                    print(
                        f"[Fish TTS] ❌ API 错误 ({response.status_code}): "
                        f"{error_msg.decode()}"
                    )
                    return None

    except httpx.ReadTimeout:
        print("[Fish TTS] 🚨 云端 60s 超时，请检查网络或服务器状态")
        return None
    except Exception as e:
        print(f"[Fish TTS] 🚨 异常: {e}")
        return None


# ---- 独立测试入口 ----
if __name__ == "__main__":
    test_text = (
        "[embarrassed] 環境構築、やっと終わりましたね！"
        "[tsundere] 別にあんたのために心配したわけじゃないんだからね！"
    )

    async def test():
        result = await fish_download_tts(test_text)
        if result:
            print(f"🎉 测试成功，文件: {result}")
        else:
            print("❌ 测试失败")

    asyncio.run(test())
