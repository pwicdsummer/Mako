"""
DS_thinking.py
==============
思考版 Letta 接口 — 里人格·思考中枢。
不负责与用户对话，只负责分析视觉信息并输出思考过程。
通过本地 Letta REST API 与 mako_think_agent 交互。
"""

import json
import os
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

_THINK_AGENT_ID = _agent_map.get("think_agent_id", "")
_LETTA_URL = os.getenv("LETTA_BASE_URL", "http://127.0.0.1:8283")


def get_thinking_reply(
    visual_report: str,
    chat_history: list[dict] | None = None
) -> str:
    """
    思考版 Letta 接口：分析视觉信息，输出内心想法或追问标签。

    Parameters
    ----------
    visual_report : str
        视觉传感器的客观描述（Qwen2-VL 的输出），
        或深思考沙盒中的强制收敛提示。
    chat_history : list[dict] | None
        前面轮次积累的思考历史（CoT 追问循环），格式为
        [{"role": "assistant", "content": "..."}, {"role": "user", "content": "..."}, ...]
        第一轮可以为空或 None。
        注意：所有消息均发送给 Letta，让 Letta 内部管理上下文的延续。

    Returns
    -------
    str
        Letta think_agent 的原始回复。
        可能包含 <mako_look>...</mako_look> 标签（需要继续追问），
        也可能直接是一段视觉思维纪要（信息已足够）。
    """
    # ---- 组装 messages（不需 system prompt，Letta 内部已管理） ----
    messages = []
    if chat_history:
        messages.extend(chat_history)

    if visual_report:
        messages.append({"role": "user", "content": visual_report})

    # ---- 发送给 Letta mako_think_agent ----
    payload = {"messages": messages}

    print(f"\n{'='*20}\n【Letta Think】发送给 mako_think_agent\n{'='*20}")

    resp = requests.post(
        f"{_LETTA_URL}/v1/agents/{_THINK_AGENT_ID}/messages",
        json=payload,
        timeout=90
    )

    if resp.status_code != 200:
        print(f"❌ Letta Think Agent 请求失败 (HTTP {resp.status_code}): {resp.text[:300]}")
        return ""

    result = resp.json()

    # ---- 提取最后一条 assistant_message 的 content ----
    messages = result.get("messages", [])
    content = ""
    for msg in reversed(messages):
        if msg.get("message_type") == "assistant_message":
            content = msg.get("content", "")
            break

    print(f"\n【Letta Think】回复:\n{content}\n{'='*20}")

    return content
