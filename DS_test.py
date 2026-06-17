"""
DS_test.py
==========
Letta Chat Agent 对话接口（替代原 DeepSeek SDK 直连）。
v2.1 升级：多单元块级解析 + 段落拆分（JP/CN 块内的多段落拆成独立单元）。

接口变更：
  get_mako_reply() 不再返回单对 {"jp": str, "cn": str}，
  而是返回 List[Dict[str, str]]，例如：
  [
    {"jp": "第一段日文", "cn": "第一段中文"},
    {"jp": "第二段日文", "cn": "第二段中文"},
  ]
  前端的播放队列可据此逐段顺序消费。
"""

import json
import os
import re
import requests

# ---- 读取 Agent ID 映射 ----
_AGENTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letta_agents.json")
_EXAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "letta_agents.json.example")

if os.path.exists(_AGENTS_PATH):
    with open(_AGENTS_PATH, "r", encoding="utf-8") as f:
        _agent_map = json.load(f)
elif os.path.exists(_EXAMPLE_PATH):
    print("⚠️  letta_agents.json 未找到，使用 letta_agents.json.example 模板。")
    print("   请复制 letta_agents.json.example → letta_agents.json 并填入真实 UUID 后重新启动。")
    with open(_EXAMPLE_PATH, "r", encoding="utf-8") as f:
        _agent_map = json.load(f)
else:
    raise FileNotFoundError(
        "letta_agents.json 和 letta_agents.json.example 均未找到。"
        "请创建 letta_agents.json 文件并填入 Letta Agent UUID。"
    )

_CHAT_AGENT_ID = _agent_map.get("chat_agent_id", "")
_LETTA_URL = os.getenv("LETTA_BASE_URL", "http://127.0.0.1:8283")


# ============================================================
# 块级解析器 v2.1：JP/CN 多段落拆分（核心修复）
# ============================================================

def parse_units(content: str) -> list[dict]:
    """
    从 Letta 返回文本中解析（JP, CN）单元列表。

    ★ v2.1 升级：JP/CN 块内的多行文本按 \n 拆分为独立单元。

    输入示例：
        JP: 第一段日文。
        第二段日文。
        CN: 第一段中文。
        第二段中文。

    旧行为（v2.0）：返回 1 个单元 {jp: "第一段日文。\n第二段日文。", cn: "..."}
    新行为（v2.1）：返回 2 个单元
        [
          {"jp": "第一段日文。", "cn": "第一段中文。"},
          {"jp": "第二段日文。", "cn": "第二段中文。"},
        ]

    安全对齐（双重对齐）：
      1. 块级对齐：若 JP 块数 ≠ CN 块数，多余块合并到最后一个块
      2. 段落级对齐：在每个块对内，若段落数不匹配，多余段落合并到最后一段
    """
    if not content or not content.strip():
        return []

    # ---- 第一步：贪婪捕获所有 JP 块和 CN 块 ----
    jp_blocks = re.findall(
        r'JP:\s*(.*?)(?=\nCN:|\nJP:|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    jp_blocks = [b.strip() for b in jp_blocks if b.strip()]

    cn_blocks = re.findall(
        r'CN:\s*(.*?)(?=\nJP:|\Z)',
        content,
        re.DOTALL | re.IGNORECASE
    )
    cn_blocks = [b.strip() for b in cn_blocks if b.strip()]

    # 必须要有 JP 和 CN 才能构成单元
    if not jp_blocks or not cn_blocks:
        # 兜底：尝试将原始内容作为单段文本处理
        cleaned = content.strip()
        if cleaned:
            return [{"jp": cleaned, "cn": cleaned}]
        return []

    # ---- 第二步：块级对齐 ----
    if len(jp_blocks) > len(cn_blocks):
        surplus = jp_blocks[len(cn_blocks):]
        cn_blocks[-1] = cn_blocks[-1] + "\n" + "\n".join(surplus)
        jp_blocks = jp_blocks[:len(cn_blocks)]
    elif len(cn_blocks) > len(jp_blocks):
        surplus = cn_blocks[len(jp_blocks):]
        jp_blocks[-1] = jp_blocks[-1] + "\n" + "\n".join(surplus)
        cn_blocks = cn_blocks[:len(jp_blocks)]

    # ---- 第三步：段落级拆分 + 对齐 ----
    units = []
    for jp_block, cn_block in zip(jp_blocks, cn_blocks):
        # 按 \n 拆分段落，过滤空白行
        jp_paras = [p.strip() for p in jp_block.split('\n') if p.strip()]
        cn_paras = [p.strip() for p in cn_block.split('\n') if p.strip()]

        if not jp_paras or not cn_paras:
            continue

        # 段落级对齐
        if len(jp_paras) > len(cn_paras):
            surplus = jp_paras[len(cn_paras):]
            jp_paras = jp_paras[:len(cn_paras)]
            cn_paras[-1] = cn_paras[-1] + " " + " ".join(surplus)
        elif len(cn_paras) > len(jp_paras):
            surplus = cn_paras[len(jp_paras):]
            cn_paras = cn_paras[:len(jp_paras)]
            jp_paras[-1] = jp_paras[-1] + " " + " ".join(surplus)

        for jp, cn in zip(jp_paras, cn_paras):
            # ---- 情绪标签提取：只读不写，不动 JP/CN 原文 ----
            emotion = "normal"
            raw_tag = ""
            match = re.match(r'\[(.*?)\]\s*', jp)
            if match:
                tag = match.group(1).lower()
                raw_tag = tag
                if tag == "shy" or tag == "clear throat":
                    emotion = "shy"
                # if tag == "happy" or tag == "laughing":
                #     emotion = "happy"
                # 未来可在此扩展更多标签映射

            # ---- 等待时间标签提取（[Wait: Xs] 在 JP 文本末尾，仅提取数字，不删除标签） ----
            wait_seconds = 0
            wait_match = re.search(r'\[Wait:\s*(\d+)\s*s?\]', jp)
            if wait_match:
                wait_seconds = int(wait_match.group(1))

            units.append({"jp": jp, "cn": cn, "emotion": emotion, "raw_tag": raw_tag, "wait_seconds": wait_seconds})

    return units


# ============================================================
# Letta Chat Agent 接口
# ============================================================
def get_mako_reply(user_input: str, memory_msgs: list | None = None) -> list[dict]:
    """
    获取茉子的 AI 回复（通过 Letta mako_chat_agent）。

    Parameters
    ----------
    user_input : str
        用户的最新一句话。
    memory_msgs : list | None
        保留参数（兼容旧调用方），实际已不再使用。
        Letta 内部自动管理对话历史。

    Returns
    -------
    list[dict]
        单元列表，每个单元包含 {"jp": str, "cn": str}。
        按顺序从前到后排列。
        如果请求失败或无有效内容，返回 []。
    """
    # ---- 只发送最新的一句话，不再打包历史 ----
    payload = {
        "messages": [
            {"role": "user", "content": user_input}
        ]
    }

    print(f"\n{'='*20}\n【Letta Chat】发送给 mako_chat_agent: {user_input}\n{'='*20}")

    resp = requests.post(
        f"{_LETTA_URL}/v1/agents/{_CHAT_AGENT_ID}/messages",
        json=payload,
        timeout=90
    )

    if resp.status_code != 200:
        print(f"❌ Letta Chat Agent 请求失败 (HTTP {resp.status_code}): {resp.text[:300]}")
        return []

    result = resp.json()

    # ---- 提取最后一条 assistant_message 的 content ----
    messages = result.get("messages", [])
    content = ""
    for msg in reversed(messages):
        if msg.get("message_type") == "assistant_message":
            content = msg.get("content", "")
            break

    if not content:
        print("⚠ Letta Chat Agent 返回空内容")
        return []

    print(f"\n【Letta Chat】原始回复:\n{content}\n{'='*20}")

    # ---- 块级解析 ----
    units = parse_units(content)

    print(f"\n【Letta Chat】解析为 {len(units)} 个单元:")
    for i, u in enumerate(units):
        print(f"  单元 {i+1}/{len(units)}: JP={u['jp']}  CN={u['cn']}")
    print('='*20)

    return units
