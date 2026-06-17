import httpx
import asyncio
import os

async def fetch_mako_voice(target_text: str, output_file: str):
    # 从环境变量读取 API Key（不要再硬编码）
    API_KEY = os.getenv("FISH_AUDIO_API_KEY", "")
    url = "https://api.fish.audio/v1/tts"

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "text": target_text,
        "reference_id": os.getenv("FISH_AUDIO_REFERENCE_ID", ""),
        "format": "mp3",
        "latency": "normal"
    }
    
    print("🚀 正在向鱼语云端集群发送高情绪自回归请求（已挂载 60s 耐心等待阈值）...")
    
    # 核心重构点：
    # 1. timeout=60.0 彻底击碎 5 秒超时魔咒，给云端前向传播留足时间
    # 2. trust_env=False（可选）：如果你发现充钱后依然报代理错，将其设为 False 可以强制不走本地代理，直连云端
    async with httpx.AsyncClient(timeout=60.0, trust_env=True) as client:
        try:
            async with client.stream("POST", url, headers=headers, json=payload) as response:
                if response.status_code == 200:
                    print("⚡ 云端已成功咬合算力！正在流式下发声学 Token 矩阵...")
                    with open(output_file, "wb") as f:
                        async for chunk in response.aiter_bytes():
                            f.write(chunk)
                    print(f"✨ 满血云端克隆成功！音频已无损降落至: {output_file}")
                else:
                    error_msg = await response.aread()
                    print(f"❌ 触发云端业务拦截: 代码 {response.status_code}, 详情: {error_msg.decode()}")
        except httpx.ReadTimeout:
            print("🚨 极度罕见：云端集群算力在 60 秒内仍未吐字，请检查官方服务器状态或尝试将 trust_env 改为 False 直连。")
        except Exception as e:
            print(f"🚨 捕获到未知的网络拓扑异常: {str(e)}")

if __name__ == "__main__":
    test_text = "[embarrassed] 环境構築、やっと終わりましたね！[tsundere] 別にあんたのために心配したわけじゃないんだからね！"
    asyncio.run(fetch_mako_voice(test_text, "mako_cloud_success.mp3"))