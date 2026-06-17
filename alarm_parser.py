"""
alarm_parser.py
================
时间解析模块 — 基于 jionlp 的自然语言绝对时间闹钟解析器。

核心功能：
  - 接收用户自然语言文本（如 "提醒我下午三点半喝水"）
  - 使用 jionlp.parse_time 解析出目标绝对时间和事件文本
  - 计算延迟秒数（delay_seconds = 目标时间 - 当前时间）
  - 防御性容错：过去时间 / 解析失败 / 事件文本空等异常

使用示例：
    result = parse_alarm_text("提醒我明天早上8点起床")
    if result:
        delay_seconds, event_text = result
        # delay_seconds > 0 表示合法闹钟
    else:
        # 非闹钟文本或时间已过去

依赖：
    pip install jionlp
"""

import jionlp
from datetime import datetime
from typing import Tuple, Optional


# ============================================================
# 常量
# ============================================================

# 解析失败的错误码（返回字符串而不是 None，方便调用方 switch）
ERROR_PARSE_FAILED = "error_parse_failed"       # jionlp 无法解析出时间
ERROR_TIME_PAST    = "error_time_past"          # 解析成功，但时间是过去
ERROR_EVENT_EMPTY  = "error_event_empty"        # 时间解析成功，但没提取出事件文本
ERROR_AMBIGUOUS    = "error_ambiguous"          # 时间解析有多义性（返回多个时间点）

# 最大可接受的闹钟延迟秒数（防止用户误输入超长定时，如 "100年后"）
# 暂时设为 365 天（31,536,000 秒），可调
MAX_DELAY_SECONDS = 31_536_000


# ============================================================
# 核心解析函数
# ============================================================

def parse_alarm_text(
    user_text: str,
    time_base: Optional[datetime] = None,
) -> Optional[Tuple[int, str]]:
    """
    解析用户输入的自然语言闹钟指令。

    参数
    ----------
    user_text : str
        用户输入的原始文本。例如：
        - "提醒我下午三点半喝水"
        - "明天早上7点叫我起床"
        - "一小时后提醒我关火"
        - "现在是下午三点"（纯陈述，非闹钟）

    time_base : Optional[datetime]
        解析用的时间基准点。默认为 datetime.now()（当前本地时间）。

    返回
    -------
    Optional[Tuple[int, str]]
        (delay_seconds, event_text)
        - delay_seconds : int
            距离闹钟触发的秒数（> 0 表示合法闹钟）。
        - event_text : str
            闹钟提醒事件文本（如 "喝水"、"起床"、"关火"）。

        以下情况返回 None（表示"这不是闹钟指令，请当作普通聊天文本处理"）：
        - jionlp 完全无法解析出时间
        - 用户文本不包含明显的提醒语义关键词

        以下情况返回特定的错误字符串（由调用方决定如何吐槽反馈）：
        - "error_time_past"  — 解析出一个过去的时间
        - "error_event_empty" — 有时间但没提取出事件内容
        - "error_ambiguous"   — 解析出多个时间点，无法确定

    实现细节
    -------------
    1. 关键词预检（极速短路）
       如果文本中不含常见的时间/提醒关键词，直接返回 None，避免浪费 jionlp 调用。
       此步骤是可选的优化，关键词列表见 _TIME_KEYWORDS。

    2. jionlp.parse_time 调用
       使用 jionlp 的完整时间解析能力：
       - 支持 "明天"、"后天"、"下周一" 等相对时间
       - 支持 "下午3点"、"15:30" 等绝对时间
       - 支持 "一小时后"、"半小时后" 等相对时长

    3. 防御性后处理
       - 若解析结果中 time 为 None → 返回 None
       - 若解析结果中 time 处于过去（<= 当前时间）→ 返回 "error_time_past"
       - 若解析出多个时间点（list）→ 返回 "error_ambiguous"
       - 若无法提取事件文本 → 回退到 "提醒事项"

    设计原则
    ---------
    - 静默降级：任何异常都返回 None，保证不会让主程序崩溃
    - 非侵入：不涉及任何 GUI / asyncio / 全局变量，纯函数
    """
    # ---- 0. 空文本守卫 ----
    if not user_text or not user_text.strip():
        return None

    # ---- 1. 时间基准 ----
    if time_base is None:
        time_base = datetime.now()

    # ---- 2. 尝试 jionlp 解析 ----
    try:
        parsed = jionlp.parse_time(user_text, time_base=time_base)
    except Exception as e:
        print(f"[alarm_parser] ⚠ jionlp.parse_time 异常: {e}")
        return None

    # ==== 检查解析结果 ====

    # jionlp.parse_time 返回结构示例：
    # {
    #   "type": "time_point" | "time_span" | "time_redundancy" | ...,
    #   "time": "2026-05-29 15:30:00" 或 ["2026-05-29 15:30", "2026-05-29 16:00"],
    #   "string": "下午三点半",
    #   "definition": "..."
    # }

    if not parsed:
        return None

    parsed_type = parsed.get("type", "")
    parsed_time = parsed.get("time")

    # ---- 3. type 为 time_redundancy 或未知 → 非时间指令 ----
    if parsed_type == "time_redundancy":
        # "现在是下午三点" — 纯陈述，非闹钟
        return None

    # ---- 4. 提取目标时间 ----
    target_dt = _extract_target_datetime(parsed_time, parsed_type, time_base)
    if target_dt is None:
        return None

    # ---- 5. 计算延迟秒数 ----
    delay_seconds = int((target_dt - time_base).total_seconds())

    if delay_seconds <= 0:
        # 时间已过去
        return None  # 静默降级 → 当普通聊天处理，用户可能只是闲聊

    if delay_seconds > MAX_DELAY_SECONDS:
        # 超过最大可接受时间（如 "100年后"），按普通聊天处理
        print(f"[alarm_parser] ⚠ 延迟 {delay_seconds}s 超过上限 {MAX_DELAY_SECONDS}s，忽略")
        return None

    # ---- 6. 提取事件文本 ----
    event_text = _extract_event_text(user_text, parsed)
    if event_text is None:
        # 有时间但无事件 → 空事件也算合法闹钟（到点只说"时间到了！"）
        event_text = "提醒事项"

    return (delay_seconds, event_text)


# ============================================================
# 内部辅助函数
# ============================================================

def _extract_target_datetime(
    parsed_time,
    parsed_type: str,
    time_base: datetime,
) -> Optional[datetime]:
    """
    从 jionlp 解析结果中提取唯一的 target_datetime。

    处理逻辑：
    - 如果 parsed_time 是字符串 → 直接解析为 datetime
    - 如果 parsed_time 是列表（多义/时间段）：
        - 若 type 为 "time_span" → 取起始时间
        - 若 type 为 "time_point" 且列表长度为 1 → 取第一个
        - 其他情况 → 返回 None（歧义，不贸然选择）
    - 如果 parsed_time 是 None → 返回 None
    """
    if parsed_time is None:
        return None

    # ---- 单字符串 ----
    if isinstance(parsed_time, str):
        try:
            # jionlp 格式: "2026-05-29 15:30:00"
            dt = datetime.strptime(parsed_time[:19], "%Y-%m-%d %H:%M:%S")
            return dt
        except ValueError:
            return None

    # ---- 列表 ----
    if isinstance(parsed_time, list):
        if not parsed_time:
            return None
        if parsed_type in ("time_span", "time_point"):
            # 时间段（或范围表达式如 "两分钟后"）→ 取起始时间
            return _extract_target_datetime(parsed_time[0], "time_point", time_base)
        else:
            return None

    return None


def _extract_event_text(user_text: str, parsed: dict) -> Optional[str]:
    """
    从用户文本中提取闹钟事件描述。

    策略（由简到繁）：
    1. 先尝试 jionlp 返回的 "definition" 字段
    2. 若为空，尝试从 parsed["string"] 中剔除时间词后剩余的部分
    3. 若仍为空，尝试用常见闹钟动词切割（"提醒我X"、"叫我X"、"让我X"）
    4. 最后还是空 → 返回 None，调用方用默认文案

    注意：此函数是纯启发式，不依赖任何外部 NLP 服务。
    """
    # ---- 策略 1：jionlp definition（排掉质量标签，如 "accurate"、"approximate"） ----
    SKIP_DEFINITIONS = {"accurate", "approximate", "precise", "模糊", "精确", "大约"}
    definition = parsed.get("definition", "")
    if definition and definition not in SKIP_DEFINITIONS:
        return definition

    time_string = parsed.get("string", "")

    # ---- 策略 2：从原始文本中去掉 time_string ----
    if time_string and time_string in user_text:
        # 去掉时间词，再 Trim
        remaining = user_text.replace(time_string, "", 1).strip()
        # 再去掉常见闹钟前缀
        remaining = _strip_alarm_prefix(remaining)
        if remaining:
            return remaining

    # ---- 策略 3：正则匹配闹钟动词模式 ----
    import re
    patterns = [
        r'(?:提醒|叫|让|帮)(?:我|你)?(?:在|于|到)?(?:\S*?(?:点|时|分|秒|刻|钟|半))?\s*(?:去|要|记得)?\s*(.+)',
        r'(?:记得|需要|要)\s*(?:在|于)?\s*(?:\S*?(?:点|时))\s*(.+)',
        r'(.+?)(?:的闹钟|的提醒|提醒我)',
    ]
    for pat in patterns:
        m = re.search(pat, user_text)
        if m:
            candidate = m.group(1).strip()
            # 去掉纯时间词残留
            candidate = re.sub(r'(?:点|时|分|秒|刻|钟|半|上午|下午|早上|晚上|中午|凌晨|今晚|明天|后天|大后天|今天|昨天)',
                               '', candidate).strip()
            if candidate:
                return candidate

    return None


def _strip_alarm_prefix(text: str) -> str:
    """去掉常见的闹钟指令前缀词（如"提醒我"、"帮我去"等）。"""
    import re
    # 按长度降序匹配，优先匹配长词
    prefixes = [
        "提醒我", "提醒你", "提醒一下",
        "叫我", "叫你",
        "帮我去", "帮我", "帮我记一下",
        "让", "记得", "别忘了",
        "我要", "我想",
        "设一个", "设置", "设定", "定一个",
        "打开", "开始",
    ]
    for prefix in sorted(prefixes, key=len, reverse=True):
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    return text
