"""
mako_memory.py
==============
纯内存短时记忆模块，基于 collections.deque 实现。
存储最近 20 轮对话，辅助 DeepSeek 对话上下文。
完全与持久化存储（chat_logger.py）解耦。
"""

from collections import deque


class ShortTermMemory:
    """存储最近 N 轮对话的短时记忆队列。"""

    def __init__(self, maxlen: int = 20):
        self.history: deque = deque(maxlen=maxlen)

    def add_turn(self, user_input: str, assistant_reply: str):
        """将一轮对话加入记忆队列。"""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": assistant_reply})

    def get_memory_msgs(self) -> list:
        """返回当前记忆队列的列表副本。"""
        return list(self.history)

    def clear(self):
        """清空记忆（可选）。"""
        self.history.clear()
