"""
chat_logger.py
==============
聊天记录持久化存储模块。
以 JSON 格式将对话记录保存到 chat_data 目录，与 UI / LLM 逻辑完全解耦。
"""

import json
import os
import time

CHAT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chat_data")


def save_chat_to_json(user_text: str, mako_text: str) -> str | None:
    """
    将一条对话记录保存为 JSON 文件。

    Parameters
    ----------
    user_text : str
        用户的输入文本。
    mako_text : str
        茉子的中文回复文本。

    Returns
    -------
    str | None
        保存成功返回文件绝对路径，失败返回 None。
    """
    try:
        os.makedirs(CHAT_DATA_DIR, exist_ok=True)

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        filename = time.strftime("%Y%m%d_%H%M%S") + ".json"
        filepath = os.path.join(CHAT_DATA_DIR, filename)

        record = {
            "timestamp": timestamp,
            "user_input": user_text,
            "mako_response": mako_text,
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        print(f"[chat_logger] 对话记录已保存 → {filepath}")
        return filepath

    except Exception as e:
        print(f"[chat_logger] 保存失败: {e}")
        return None
