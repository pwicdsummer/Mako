#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 MakoTalker 三位一体工具包挂载脚本
=============================================================================
 功能：
   为 MakoTalker 的三个 Agent（日常对话 / 视觉搭话·表人格 / 深度思考·里人格）
   补齐系统内建的 Archival Memory 检索工具，释放共 523 条历史羁绊记忆的
   长期归档检索能力。

 发现的问题：
   新初始化的三个 Agent 的 tools 列表中只包含 `conversation_search`，
   缺失了检索共享 Archive "mako_soul_base" 所需的 `archival_memory_search` 工具。
   通过本脚本将核心检索工具强行「派发」给每个分身。

 架构：
   基于 Letta v0.16.8 的 REST API，纯 `requests` 实现，
   与 `letta_one_click_init.py`、`import_mako_shared_source.py` 保持统一通信风格。

 使用方法：
   1. 确认 Letta 服务已启动（http://127.0.0.1:8283）
   2. 在终端执行: python patch_mako_tools.py

 注意：
   - 零外部依赖（仅需 requests）
   - 脚本会自动遍历系统工具列表，找到 `archival_memory_search` 或 `search_passages`
     等代表长期记忆检索的工具（最终工具名以服务器返回为准）
   - 工具已存在时会自动跳过，不会重复挂载

 Author   : MakoTalker Team
 Version  : 1.0.0 — REST API Tool Attach
=============================================================================
"""

import sys
import time

import requests

# ============================================================
# 【配置项】请根据实际情况修改以下全局变量
# ============================================================

# Letta REST API 基地址（与项目中其他脚本一致）
BASE_URL = "http://127.0.0.1:8283"

# 目标 Agent UUID 列表（依次为：日常对话 / 视觉搭话·表人格 / 深度思考·里人格）
TARGET_AGENT_IDS = [
    "agent-6181ee0f-0837-4c5e-9137-b77e71534258",  # mako_chat_agent（日常对话）
    "agent-806c0fd4-ec29-4a6b-a37d-db51f0db1465",  # mako_proactive_agent（视觉搭话·表人格）
    "agent-8d3fe1cb-d04d-4231-8aef-c4b2b13bfc6c",  # mako_think_agent（深度思考·里人格）
]

# 要检索并挂载的系统工具名称（候选列表）
# 如果 `archival_memory_search` 不存在，脚本会自动尝试列表中其他名称
TOOL_CANDIDATE_NAMES = [
    "archival_memory_search",
    "search_passages",
    "archival_search",
]

# 每个 Agent 对应的可读标签（用于回显）
AGENT_LABELS = {
    TARGET_AGENT_IDS[0]: "日常对话 (mako_chat_agent)",
    TARGET_AGENT_IDS[1]: "视觉搭话·表人格 (mako_proactive_agent)",
    TARGET_AGENT_IDS[2]: "深度思考·里人格 (mako_think_agent)",
}


# ============================================================
# API 通信辅助
# ============================================================

def api_headers() -> dict:
    return {"Content-Type": "application/json"}


def api_get(path: str, timeout: int = 10) -> requests.Response:
    return requests.get(f"{BASE_URL}{path}", headers=api_headers(), timeout=timeout)


def api_patch(path: str, payload: dict = None, timeout: int = 30) -> requests.Response:
    return requests.patch(f"{BASE_URL}{path}", json=payload or {}, headers=api_headers(), timeout=timeout)


# ============================================================
# 核心逻辑
# ============================================================

def find_archival_tool_id() -> str:
    """
    遍历系统工具列表，找到代表长期记忆/归档检索的工具并返回其 ID。

    GET /v1/tools/?name=... 按名称精准搜索

    Returns:
        tool_id (str): 工具的 UUID

    Raises:
        SystemExit: 如果无法找到任何匹配的工具
    """
    print("─── 正在扫描系统工具库（Searching for Archival Memory Tool）───")

    for tool_name in TOOL_CANDIDATE_NAMES:
        try:
            resp = api_get(f"/v1/tools/?name={tool_name}", timeout=10)
            if resp.status_code == 200:
                tools = resp.json()
                if tools and len(tools) > 0:
                    tool = tools[0]
                    tool_id = tool.get("id")
                    actual_name = tool.get("name", tool_name)
                    tool_type = tool.get("tool_type", "unknown")
                    print(f"    ✅ 发现工具 → 名称: '{actual_name}' | ID: {tool_id} | 类型: {tool_type}")
                    return tool_id
                else:
                    print(f"    ⚠️ 按名称 '{tool_name}' 查询返回空列表，继续尝试下一个...")
            else:
                print(f"    ⚠️ 查询 '{tool_name}' 失败 (HTTP {resp.status_code})，继续尝试下一个...")
        except Exception as e:
            print(f"    ⚠️ 查询 '{tool_name}' 异常: {e}，继续尝试下一个...")

    # ----- 兜底：获取全部工具，用关键词匹配 -----
    print()
    print("    🔍 候选名称均未命中，尝试全量扫描 + 关键词模糊匹配...")
    try:
        resp = api_get("/v1/tools/", timeout=15)
        if resp.status_code == 200:
            all_tools = resp.json()
            # 按关键词匹配
            keywords = ["archival", "archive", "passage", "memory", "search"]
            for tool in all_tools:
                t_name = (tool.get("name") or "").lower()
                t_type = (tool.get("tool_type") or "").lower()
                combined = f"{t_name} {t_type}"
                if any(kw in combined for kw in keywords):
                    tool_id = tool.get("id")
                    print(f"    ✅ 模糊匹配命中 → 名称: '{tool.get('name')}' | ID: {tool_id}")
                    return tool_id
            print(f"    [❌] 全量扫描后仍未找到匹配的归档检索工具。")
            print(f"    系统共有 {len(all_tools)} 个可用工具：")
            for t in all_tools:
                print(f"         - {t.get('name')} (ID: {t.get('id')})")
        else:
            print(f"    [❌] 获取工具列表失败 (HTTP {resp.status_code})")
    except Exception as e:
        print(f"    [❌] 全量扫描异常: {e}")

    print()
    print("    ================================================================")
    print("    [🛠️] 需要你手动介入：")
    print("    请打开 Letta 服务端，确认工具的正确名称后，")
    print("    修改脚本顶部 TOOL_CANDIDATE_NAMES 列表，然后重新运行。")
    print("    ================================================================")
    sys.exit(1)


def get_agent_tools(agent_id: str) -> list:
    """
    获取指定 Agent 当前已绑定的工具列表。

    GET /v1/agents/{agent_id}/tools

    Returns:
        工具对象列表
    """
    try:
        resp = api_get(f"/v1/agents/{agent_id}/tools", timeout=10)
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"    ⚠️ 获取 Agent {agent_id} 工具列表失败 (HTTP {resp.status_code}): {resp.text[:200]}")
            return []
    except Exception as e:
        print(f"    ⚠️ 获取 Agent {agent_id} 工具列表异常: {e}")
        return []


def attach_tool_to_agent(agent_id: str, tool_id: str) -> bool:
    """
    将指定工具挂载到目标 Agent。

    PATCH /v1/agents/{agent_id}/tools/attach/{tool_id}

    Returns:
        True 表示挂载成功，False 表示失败
    """
    try:
        resp = api_patch(f"/v1/agents/{agent_id}/tools/attach/{tool_id}", timeout=15)
        if resp.status_code in (200, 201, 204):
            return True
        elif resp.status_code == 409 or "already" in (resp.text or "").lower():
            print(f"      ⚠️ 工具已存在，跳过。")
            return True
        else:
            print(f"      [❌] 挂载失败 (HTTP {resp.status_code}): {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"      [❌] 挂载请求异常: {e}")
        return False


def main():
    start_time = time.time()

    print()
    print("=" * 60)
    print("  🛠️  MakoTalker 三位一体工具包挂载引擎")
    print("  「补全茉子的检索之力」")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 0. 前置检查：API 连通性
    # --------------------------------------------------------
    print("─── 检查 Letta API 连通性 ───")
    try:
        resp = api_get("/v1/agents/", timeout=10)
        if resp.status_code == 200:
            print("    ✅ Letta API 连接正常")
        else:
            print(f"    ⚠️ API 返回 HTTP {resp.status_code}，继续尝试...")
        print()
    except requests.exceptions.ConnectionError:
        print(f"    [❌] 无法连接到 Letta API ({BASE_URL})")
        print("     请确保 Letta Docker 服务已启动（端口 8283）。")
        sys.exit(1)
    except Exception as e:
        print(f"    ⚠️ 连通性检查异常: {e}（继续尝试...）")
        print()

    # --------------------------------------------------------
    # 步骤一：找到归档检索工具的系统 ID
    # --------------------------------------------------------
    print("─── [步骤 1/3] 定位 Archival Memory 检索工具 ───")
    tool_id = find_archival_tool_id()
    print()

    # --------------------------------------------------------
    # 步骤二：遍历三个 Agent，逐个挂载工具
    # --------------------------------------------------------
    print("─── [步骤 2/3] 批量挂载工具 ───")
    print(f"    工具 ID : {tool_id}")
    print(f"    Agent   : {len(TARGET_AGENT_IDS)} 个")
    print()

    attach_success = 0
    attach_fail = 0
    already_have = 0

    for agent_id in TARGET_AGENT_IDS:
        label = AGENT_LABELS.get(agent_id, agent_id)
        print(f"  🔍 {label}")

        # 检查当前已有工具
        current_tools = get_agent_tools(agent_id)
        current_tool_ids = {t.get("id") for t in current_tools if t.get("id")}
        current_tool_names = {t.get("name") for t in current_tools if t.get("name")}

        if tool_id in current_tool_ids:
            print(f"      ✅ 工具已存在，无需重复挂载。")
            already_have += 1
        else:
            print(f"      📋 当前工具: {', '.join(sorted(current_tool_names)) or '(无)'}")
            print(f"      🔗 正在挂载 archical_memory_search...")

            if attach_tool_to_agent(agent_id, tool_id):
                print(f"      ✅ {label} 成功装备技能：archival_memory_search！")
                attach_success += 1
            else:
                attach_fail += 1

        # 挂载后再次验证
        updated_tools = get_agent_tools(agent_id)
        updated_names = {t.get("name") for t in updated_tools if t.get("name")}
        print(f"      📋 更新后工具列表: {', '.join(sorted(updated_names)) or '(无)'}")
        print()

    # --------------------------------------------------------
    # 步骤三：汇总报告
    # --------------------------------------------------------
    print("─── [步骤 3/3] 汇总报告 ───")
    print()
    print(f"  📊 工具挂载统计:")
    print(f"     新增挂载 : {attach_success}")
    print(f"     已存在   : {already_have}")
    print(f"     失败     : {attach_fail}")
    print()

    # --------------------------------------------------------
    # 最终输出
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  ✨ 工具包挂载完成！")
    print("=" * 60)
    print(f"  Agent 总数       : {len(TARGET_AGENT_IDS)}")
    for aid in TARGET_AGENT_IDS:
        label = AGENT_LABELS.get(aid, aid)
        print(f"    - {label}")
        print(f"      {aid}")
    print(f"  检索工具 ID     : {tool_id}")
    print(f"  总耗时          : {elapsed:.1f} 秒")
    print("=" * 60)
    print()
    print("  📖 现在三个 Agent 都已装备 archival_memory_search 工具，")
    print("  可以检索共享 Archive「mako_soul_base」中的 523 条历史羁绊了。")
    print()

    if attach_fail > 0:
        print("  ⚠️  部分挂载失败，请检查上方日志。")
        print()


if __name__ == "__main__":
    main()
