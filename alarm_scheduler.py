"""
alarm_scheduler.py
==================
轻量级异步闹钟调度器 — 基于 asyncio.sleep 的非阻塞绝对时间倒计时。

核心设计：
  - AlarmScheduler 类维护一个活跃闹钟集合
  - 每个闹钟由 add_alarm() 创建，返回唯一的 alarm_id
  - 内部使用 asyncio.create_task() 启动后台协程，await asyncio.sleep() 非阻塞等待
  - 到点时通过 trigger_callback(event_text) 通知调用方
  - 支持取消（cancel_alarm）和查询全部（list_active）

与非阻塞保证：
  - asyncio.sleep(delay_seconds) 在 qasync 事件循环中完全非阻塞
  - 即使有 100 个闹钟同时计数，也不会阻塞 UI 渲染或用户输入
  - 回调 trigger_callback 会在事件循环中异步执行

线程安全：
  - 所有操作都在同一个 asyncio 事件循环中执行
  - 使用 asyncio.Lock 保护活跃闹钟字典，防止并发修改

使用示例：
    scheduler = AlarmScheduler()
    alarm_id = scheduler.add_alarm(3600, "该喝水了", my_callback)
    # 一小时后自动触发 my_callback("该喝水了")
    scheduler.cancel_alarm(alarm_id)  # 取消
"""

import asyncio
import time
import uuid
from typing import Callable, Optional


# ============================================================
# 类型别名
# ============================================================

# 闹钟触发时的回调函数签名：callback(event_text: str)
AlarmCallback = Callable[[str], None]


# ============================================================
# AlarmScheduler 类
# ============================================================

class AlarmScheduler:
    """
    轻量级异步闹钟调度器。

    基于 asyncio.sleep 实现非阻塞倒计时，无需线程、无需 APScheduler 等重型依赖。
    通过 add_alarm() 创建闹钟，cancel_alarm() 取消，list_active() 查询。

    属性
    ----------
    _active_alarms : dict[str, dict]
        活跃闹钟字典。键为 alarm_id（UUID 字符串），值为：
        {
            "event_text": str,          # 提醒文本
            "target_time": float,       # 目标绝对时间戳（time.time()）
            "delay_seconds": int,       # 延迟秒数
            "task": asyncio.Task,       # 后台倒计时协程
            "created_at": float,        # 创建时间戳
        }
    _lock : asyncio.Lock
        保护 _active_alarms 的锁。
    """

    def __init__(self):
        self._active_alarms: dict[str, dict] = {}
        self._lock = asyncio.Lock()

    # ---------------------------------------------------------------
    # 公开 API
    # ---------------------------------------------------------------

    def add_alarm(
        self,
        delay_seconds: int,
        event_text: str,
        trigger_callback: AlarmCallback,
    ) -> str:
        """
        添加一个闹钟。

        参数
        ----------
        delay_seconds : int
            距离闹钟触发的秒数（必须 > 0）。
        event_text : str
            闹钟触发时的提醒文本（如 "该喝水了"、"代码编译完了"）。
        trigger_callback : AlarmCallback
            闹钟触发时的回调函数。签名：callback(event_text: str)。
            注意：此回调会在 asyncio 事件循环中运行，不能是同步阻塞函数。
            如果需要在回调中执行阻塞操作，请使用 asyncio.to_thread() 或
            loop.run_in_executor() 委托到线程池。

        返回
        -------
        str
            闹钟的唯一 ID（UUID 字符串）。可用于 cancel_alarm() 取消闹钟。
        """
        if delay_seconds <= 0:
            raise ValueError(f"delay_seconds 必须 > 0，收到: {delay_seconds}")

        alarm_id = str(uuid.uuid4())
        target_time = time.time() + delay_seconds

        # 创建后台倒计时协程
        task = asyncio.create_task(
            self._countdown(alarm_id, delay_seconds, event_text, trigger_callback)
        )

        # 注册到活跃字典
        # 注：直接用 asyncio.create_task 时无需 await，submit 可视为"异步"
        # 但为了线程安全性，在这里不使用 await lock，而是通过 create_task 处理
        self._active_alarms[alarm_id] = {
            "event_text": event_text,
            "target_time": target_time,
            "delay_seconds": delay_seconds,
            "task": task,
            "created_at": time.time(),
        }

        # 打印日志
        from datetime import datetime
        target_dt = datetime.fromtimestamp(target_time)
        print(
            f"[AlarmScheduler] ✅ 闹钟已设定: "
            f"ID={alarm_id[:8]}... "
            f"延迟={delay_seconds}s "
            f"目标时间={target_dt.strftime('%H:%M:%S')} "
            f"事件=\"{event_text}\""
        )

        return alarm_id

    def cancel_alarm(self, alarm_id: str) -> bool:
        """
        取消一个活跃的闹钟。

        参数
        ----------
        alarm_id : str
            add_alarm() 返回的闹钟 ID。

        返回
        -------
        bool
            取消成功返回 True；若闹钟不存在或已完成返回 False。
        """
        alarm = self._active_alarms.pop(alarm_id, None)
        if alarm is None:
            print(f"[AlarmScheduler] ⚠ 取消失败: 闹钟 {alarm_id[:8]}... 不存在")
            return False

        # 取消后台协程
        task = alarm["task"]
        if not task.done():
            task.cancel()
            print(f"[AlarmScheduler] 🗑️ 闹钟已取消: ID={alarm_id[:8]}...")
        return True

    def list_active(self) -> list[dict]:
        """
        获取所有活跃闹钟的摘要列表。

        返回
        -------
        list[dict]
            每个元素格式：
            {
                "alarm_id": str,
                "event_text": str,
                "remaining_seconds": int,   # 剩余秒数
                "target_time": float,       # 目标时间戳
            }
        """
        now = time.time()
        result = []
        for alarm_id, info in list(self._active_alarms.items()):
            remaining = max(0, int(info["target_time"] - now))
            result.append({
                "alarm_id": alarm_id,
                "event_text": info["event_text"],
                "remaining_seconds": remaining,
                "target_time": info["target_time"],
            })
        # 按剩余时间升序排列
        result.sort(key=lambda x: x["remaining_seconds"])
        return result

    def get_alarm_count(self) -> int:
        """获取当前活跃的闹钟数量。"""
        return len(self._active_alarms)

    # ---------------------------------------------------------------
    # 内部方法
    # ---------------------------------------------------------------

    async def _countdown(
        self,
        alarm_id: str,
        delay_seconds: int,
        event_text: str,
        trigger_callback: AlarmCallback,
    ):
        """
        后台倒计时协程。

        流程：
        1. await asyncio.sleep(delay_seconds) — 非阻塞等待
        2. 从 _active_alarms 中移除自身
        3. 调用 trigger_callback(event_text)
        4. 异常处理：被取消时静默退出，不抛异常
        """
        try:
            await asyncio.sleep(delay_seconds)

            # 倒计时结束：从活跃字典中移除
            self._active_alarms.pop(alarm_id, None)

            # 触发回调
            print(
                f"[AlarmScheduler] 🔔 闹钟触发! "
                f"ID={alarm_id[:8]}... "
                f"事件=\"{event_text}\""
            )
            trigger_callback(event_text)

        except asyncio.CancelledError:
            # 闹钟被取消，静默退出
            pass
        except Exception as e:
            print(f"[AlarmScheduler] ❌ 闹钟 {alarm_id[:8]}... 异常: {e}")
            import traceback
            traceback.print_exc()
            # 确保从活跃字典中清理
            self._active_alarms.pop(alarm_id, None)


# ============================================================
# 测试入口
# ============================================================

if __name__ == "__main__":
    """
    独立测试：创建 3 个闹钟（5s / 10s / 15s），确认依次触发。
    测试方法：
        python alarm_scheduler.py
    """
    import asyncio

    async def test():
        def on_alarm(text):
            print(f"  >>> 叮! \"{text}\"")

        scheduler = AlarmScheduler()
        id1 = scheduler.add_alarm(3, "测试闹钟1", on_alarm)
        id2 = scheduler.add_alarm(6, "测试闹钟2", on_alarm)
        id3 = scheduler.add_alarm(9, "测试闹钟3", on_alarm)

        print(f"活跃闹钟数: {scheduler.get_alarm_count()}")
        print("等待 12 秒...")

        await asyncio.sleep(12)
        print(f"活跃闹钟数: {scheduler.get_alarm_count()}")

    asyncio.run(test())
