"""
DS_proactive.py
===============
主动搭话 Letta 接口 — 表人格·对话中枢。
收到视觉思维纪要后，通过 mako_proactive_agent 生成茉子真正说出口的台词。
v2.1 升级：使用与 DS_test.py 一致的多段落拆分解析器，返回逐段 Unit 列表。
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

_PROACTIVE_AGENT_ID = _agent_map.get("proactive_agent_id", "")
_LETTA_URL = os.getenv("LETTA_BASE_URL", "http://127.0.0.1:8283")


# ============================================================
# 块级解析器 v2.1（与 DS_test.py 完全一致，多段落拆分修复）
# ============================================================
def parse_units(content: str) -> list[dict]:
    """
    从 Letta 返回文本中解析（JP, CN）单元列表。
    与 DS_test.py 中的 parse_units 逻辑完全一致（v2.1 多段落拆分）。

    ★ v2.1 升级：JP/CN 块内的多行文本按 \n 拆分为独立单元。

    输入示例：
        JP: 第一段日文。
        第二段日文。
        CN: 第一段中文。
        第二段中文。

    返回 2 个单元：
        [
          {"jp": "第一段日文。", "cn": "第一段中文。"},
          {"jp": "第二段日文。", "cn": "第二段中文。"},
        ]
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

    if not jp_blocks or not cn_blocks:
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
        jp_paras = [p.strip() for p in jp_block.split('\n') if p.strip()]
        cn_paras = [p.strip() for p in cn_block.split('\n') if p.strip()]

        if not jp_paras or not cn_paras:
            continue

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
                if tag == "shy":
                    emotion = "shy"
                # 未来可在此扩展更多标签映射

            # ---- 等待时间标签提取（[Wait: Xs] 在 JP 文本末尾，仅提取数字，不删除标签） ----
            wait_seconds = 0
            wait_match = re.search(r'\[Wait:\s*(\d+)\s*s?\]', jp)
            if wait_match:
                wait_seconds = int(wait_match.group(1))

            units.append({"jp": jp, "cn": cn, "emotion": emotion, "raw_tag": raw_tag, "wait_seconds": wait_seconds})

    return units


# ============================================================
# 主动搭话 Letta 接口
# ============================================================
def get_proactive_reply(
    visual_digest: str,
    memory_msgs: list | None = None
) -> list[dict]:
    """
    对话版 Letta 接口：根据视觉思维纪要生成主动搭话台词。

    Parameters
    ----------
    visual_digest : str
        DeepThinker 输出的视觉思维纪要。
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
    import time
    current_time = time.ctime()

    user_content = (
        f"当前时间：{current_time}\n\n"
        f"【视觉思维纪要】\n{visual_digest}\n\n"
        f"请根据以上观察，自然地开启话题。"
    )

    payload = {
        "messages": [
            {"role": "user", "content": user_content}
        ]
    }

    print(f"\n{'='*20}\n【Letta Proactive】发送给 mako_proactive_agent\n{'='*20}")

    resp = requests.post(
        f"{_LETTA_URL}/v1/agents/{_PROACTIVE_AGENT_ID}/messages",
        json=payload,
        timeout=90
    )

    if resp.status_code != 200:
        print(f"❌ Letta Proactive Agent 请求失败 (HTTP {resp.status_code}): {resp.text[:300]}")
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
        print("⚠ Letta Proactive Agent 返回空内容")
        return []

    print(f"\n【Letta Proactive】原始回复:\n{content}\n{'='*20}")

    # ---- 块级解析 ----
    units = parse_units(content)

    print(f"\n【Letta Proactive】解析为 {len(units)} 个单元:")
    for i, u in enumerate(units):
        print(f"  单元 {i+1}/{len(units)}: JP={u['jp']}  CN={u['cn']}")
    print('='*20)

    return units
