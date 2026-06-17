#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
 MakoTalker 共享数据源创建 · 历史记忆清洗导入 · 多 Agent 灵魂绑定脚本
=============================================================================
 功能：
   1. 创建全局独立 Archive "mako_soul_base"
   2. 批量清洗 ~500 个历史聊天 JSON 文件并注入该 Archive
   3. 将 Archive 同时绑定到三个 Agent（日常对话 / 视觉搭话·表人格 / 深度思考·里人格）

 架构理念：
   【Shared Archive + 多 Agent 触角】
   基于 Letta v0.16.8 的 Archive REST API（/v1/archives/），
   零 SDK 依赖，仅需 requests，与 letta_one_click_init.py 保持统一通信风格。

 幂等保护：
   - 脚本会自动记录已成功导入的文件名到 .import_progress.json
   - 重复运行仅导入新增/未成功的文件，绝不重复写入已有记忆
   - 如需要强制全部重导，删除 .import_progress.json 再运行即可

 使用方法：
   1. 确认 MEMORY_DIR 指向存放 JSON 文件的正确目录
   2. 确认 BASE_URL 与你的 Letta 服务地址一致（默认 http://127.0.0.1:8283）
   3. 在终端执行: python import_mako_shared_source.py

 注意：
   - 零外部依赖（仅需 requests 标准库）
   - 如果 Archive "mako_soul_base" 已存在，脚本会自动复用而非报错
   - 导入过程不可逆，建议先在少量文件上测试

 Author   : MakoTalker Team
 Version  : 3.0.0 — Archive REST API Architecture
=============================================================================
"""

import os
import json
import sys
import time
import hashlib

import requests

# ============================================================
# 【配置项】请根据实际情况修改以下全局变量
# ============================================================

# JSON 文件所在目录
MEMORY_DIR = r"C:\Users\kz740\Desktop\记忆回档"

# Letta REST API 基地址（与 letta_one_click_init.py 一致）
BASE_URL = "http://127.0.0.1:8283"

# 共享 Archive 名称（对应 Letta v0.16.8 的 Archive 系统）
ARCHIVE_NAME = "mako_soul_base"

# 本地进度跟踪文件（记录已成功导入的文件名 + 文件内容哈希）
# 用于幂等保护：重复运行不会重复导入
PROGRESS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".import_progress.json")

# 目标 Agent UUID 列表（依次为：日常对话 / 视觉搭话·表人格 / 深度思考·里人格）
TARGET_AGENT_IDS = [
    "agent-6181ee0f-0837-4c5e-9137-b77e71534258",  # mako_chat_agent（日常对话）
    "agent-806c0fd4-ec29-4a6b-a37d-db51f0db1465",  # mako_proactive_agent（视觉搭话·表人格）
    "agent-8d3fe1cb-d04d-4231-8aef-c4b2b13bfc6c",  # mako_think_agent（深度思考·里人格）
]

# 进度打印间隔（每成功导入 N 条打印一次）
PROGRESS_INTERVAL = 20


# ============================================================
# 幂等保护：进度跟踪
# ============================================================

def load_progress() -> dict:
    """加载本地进度文件，返回 { 文件名: 哈希 } 字典。"""
    if not os.path.isfile(PROGRESS_FILE):
        return {}
    try:
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, IOError):
        return {}


def save_progress(progress: dict):
    """持久化进度字典到本地文件。"""
    with open(PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)


def file_hash(filepath: str) -> str:
    """计算文件的 SHA256 哈希值，用于检测内容变更。"""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================
# API 通信辅助
# ============================================================

def api_headers() -> dict:
    return {"Content-Type": "application/json"}


def api_get(path: str, timeout: int = 10) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.get(url, headers=api_headers(), timeout=timeout)


def api_post(path: str, payload: dict, timeout: int = 30) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.post(url, json=payload, headers=api_headers(), timeout=timeout)


def api_patch(path: str, payload: dict = None, timeout: int = 30) -> requests.Response:
    url = f"{BASE_URL}{path}"
    return requests.patch(url, json=payload or {}, headers=api_headers(), timeout=timeout)


# ============================================================
# 辅助函数
# ============================================================

def load_json_files(directory: str):
    """遍历目标目录，收集所有 .json 文件路径（按文件名排序）。"""
    if not os.path.isdir(directory):
        print(f"[❌] 错误：目录不存在或无法访问 -> {directory}")
        sys.exit(1)

    json_files = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".json")
    ]
    json_files.sort()
    return json_files


def format_memory_block(data: dict) -> str:
    """将单条 JSON 编织为高语义长文本块。"""
    return (
        f"【历史羁绊记忆】\n"
        f"记录时间: {data['timestamp']}\n"
        f"主人（真寻）说: {data['user_input']}\n"
        f"常陆茉子（Mako）回复: {data['mako_response']}"
    )


# ============================================================
# 三大执行步骤
# ============================================================

def step1_create_archive() -> str:
    """
    步骤一：创建或获取全局 Archive "mako_soul_base"。

    POST /v1/archives/  → 创建新 Archive
    GET  /v1/archives/  → 如果已存在则按 name 检索

    Returns:
        archive_id (str)
    """
    print("─── [步骤 1/3] 创建/获取全局共享 Archive ───")
    print(f"    Archive 名称: {ARCHIVE_NAME}")

    # 尝试创建
    payload = {
        "name": ARCHIVE_NAME,
        "description": "MakoTalker 全局共享灵魂基底 — 真寻与茉子的全部羁绊记忆",
    }
    try:
        resp = api_post("/v1/archives/", payload, timeout=15)
        if resp.status_code in (200, 201):
            archive_id = resp.json().get("id")
            print(f"    ✅ Archive 创建成功 → ID: {archive_id}")
            return archive_id
        else:
            error_body = resp.text[:300]
            print(f"    ⚠️ 创建失败 (HTTP {resp.status_code}): {error_body}")
            # 继续尝试查找现有 Archive
    except requests.exceptions.ConnectionError:
        print(f"    [❌] 无法连接到 Letta API ({BASE_URL})")
        print("    请确保 Letta Docker 服务已启动。")
        sys.exit(1)
    except Exception as e:
        print(f"    ⚠️ 创建请求异常: {e}")

    # 查找已有的 Archive
    print(f"    🔍 正在查找已存在的 Archive '{ARCHIVE_NAME}'...")
    try:
        resp = api_get("/v1/archives/", timeout=10)
        if resp.status_code == 200:
            archives = resp.json()
            for arc in archives:
                if arc.get("name") == ARCHIVE_NAME:
                    archive_id = arc.get("id")
                    print(f"    ✅ 获取到现有 Archive → ID: {archive_id}")
                    return archive_id
        print(f"    [❌] 未找到 Archive '{ARCHIVE_NAME}'，且创建失败。")
        print("    请检查 Letta 服务及权限后重试。")
        sys.exit(1)
    except Exception as e:
        print(f"    [❌] 查找 Archive 失败: {e}")
        sys.exit(1)


def step2_load_memories_into_archive(archive_id: str, json_files: list):
    """
    步骤二：批量遍历、清洗 JSON 文件，将格式化后记忆注入 Archive。

    幂等行为：
      - 首次运行 → 全部导入
      - 后续运行 → 对比 .import_progress.json，跳过已导入且内容未变的文件

    POST /v1/archives/{archive_id}/passages/batch 一次性批量提交

    Args:
        archive_id: Archive ID
        json_files: JSON 文件路径列表
    """
    print()
    print("─── [步骤 2/3] 批量清洗并注入记忆 ───")
    print(f"    Archive ID: {archive_id}")
    print(f"    文件总数  : {len(json_files)}")
    print()

    # 加载历史进度
    progress = load_progress()
    if progress:
        print(f"    📋 检测到历史导入记录: {len(progress)} 条")
        print(f"    🔒 已导入且内容未变的文件将被自动跳过（幂等保护）")
        print()

    success_count = 0
    skip_count = 0
    fail_count = 0
    reimport_count = 0

    batch_buffer = []       # 累积待写入的 passage
    batch_file_map = {}     # 记录 batch 中每条 passage 对应的文件名
    batch_size = 20         # 每 20 条提交一次 batch

    def flush_batch():
        """将 batch_buffer 中的 passages 一次性提交到 API。"""
        nonlocal success_count, fail_count

        if not batch_buffer:
            return

        def individual_insert(passage_text, fname, fhash):
            """逐条插入一条 passage。"""
            nonlocal success_count, fail_count
            try:
                r = api_post(
                    f"/v1/archives/{archive_id}/passages",
                    {"text": passage_text},
                    timeout=30,
                )
                if r.status_code in (200, 201):
                    progress[fname] = fhash
                    save_progress(progress)
                    success_count += 1
                else:
                    print(f"      [❌] 逐条写入失败 ({fname}): HTTP {r.status_code}")
                    fail_count += 1
            except Exception as e2:
                print(f"      [❌] 逐条写入异常 ({fname}): {e2}")
                fail_count += 1

        try:
            resp = api_post(
                f"/v1/archives/{archive_id}/passages/batch",
                {"passages": [{"text": t} for t in batch_buffer]},
                timeout=60,
            )
            if resp.status_code in (200, 201):
                for fname, fhash in batch_file_map.items():
                    progress[fname] = fhash
                save_progress(progress)
                success_count += len(batch_buffer)
            else:
                print(f"    ⚠️ 批量提交失败 (HTTP {resp.status_code})，逐条重试...")
                for passage_text, (fname, fhash) in zip(batch_buffer, batch_file_map.items()):
                    individual_insert(passage_text, fname, fhash)
        except Exception as e:
            print(f"    [❌] 批量提交异常: {e}")
            print("    改为逐条写入...")
            for passage_text, (fname, fhash) in zip(batch_buffer, batch_file_map.items()):
                individual_insert(passage_text, fname, fhash)

        batch_buffer.clear()
        batch_file_map.clear()

    # ========== 遍历所有 JSON 文件 ==========

    for idx, filepath in enumerate(json_files, start=1):
        filename = os.path.basename(filepath)

        # 幂等检查
        current_hash = file_hash(filepath)
        if filename in progress:
            if progress[filename] == current_hash:
                skip_count += 1
                continue
            else:
                reimport_count += 1
                print(f"  [🔄] 检测到文件变更，重新导入 ({filename})")

        # 读取并解析 JSON
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
            print(f"  [⚠️] 文件读取/解析失败 ({filename}): {e}")
            fail_count += 1
            continue

        # 验证必要字段
        missing_keys = [k for k in ("timestamp", "user_input", "mako_response") if k not in data]
        if missing_keys:
            print(f"  [⚠️] 缺少必要字段 {missing_keys} ({filename})，已跳过。")
            fail_count += 1
            continue

        # 格式化记忆文本
        try:
            formatted_memory = format_memory_block(data)
        except Exception as e:
            print(f"  [⚠️] 格式化失败 ({filename}): {e}")
            fail_count += 1
            continue

        # 加入 batch
        batch_buffer.append(formatted_memory)
        batch_file_map[filename] = current_hash

        # 每满 batch_size 条提交一次
        if len(batch_buffer) >= batch_size:
            flush_batch()

            # 进度回显
            total_processed = skip_count + success_count + fail_count
            if success_count > 0 and success_count % PROGRESS_INTERVAL == 0:
                print(
                    f"  [⏳] 进度: {total_processed}/{len(json_files)}  "
                    f"新增 {success_count} 跳过 {skip_count} 失败 {fail_count}  "
                    f"| 已沉淀 {success_count} 条新回忆..."
                )

    # 提交剩余不足 batch_size 的零头
    if batch_buffer:
        flush_batch()

    # ========== 最终统计 ==========
    print()
    print(f"  📊 Archive 注入统计:")
    print(f"     本次新增导入 : {success_count}")
    print(f"     已跳过(无变更): {skip_count}")
    if reimport_count > 0:
        print(f"     重新导入(内容变更): {reimport_count}")
    print(f"     失败         : {fail_count}")
    print(f"     文件夹总计   : {len(json_files)}")
    print(f"     进度文件记录 : {len(progress)} 条")

    if success_count == 0 and skip_count > 0:
        print()
        print("  ✅ 所有记忆文件均已导入过且内容未发生变化。")
        print("     如需强制全部重导，请删除 .import_progress.json 后重新运行。")

    if fail_count > 0:
        print("  📋 提示：部分文件注入失败，请查看上方日志排查原因。")

    return success_count, fail_count, skip_count


def step3_attach_archive_to_agents(archive_id: str, agent_ids: list):
    """
    步骤三：多 Agent 灵魂绑定 —— 将 Archive 同时附加到所有目标 Agent。

    PATCH /v1/agents/{agent_id}/archives/attach/{archive_id}

    Args:
        archive_id: Archive ID
        agent_ids:  目标 Agent UUID 列表
    """
    print()
    print("─── [步骤 3/3] 多 Agent 灵魂绑定 ───")
    print(f"    Archive ID: {archive_id}")
    print(f"    目标 Agent: {len(agent_ids)} 个")
    print()

    agent_labels = {
        TARGET_AGENT_IDS[0]: "日常对话 (mako_chat_agent)",
        TARGET_AGENT_IDS[1]: "视觉搭话·表人格 (mako_proactive_agent)",
        TARGET_AGENT_IDS[2]: "深度思考·里人格 (mako_think_agent)",
    }

    success_bind = 0
    fail_bind = 0

    for agent_id in agent_ids:
        label = agent_labels.get(agent_id, agent_id)
        print(f"  🔗 正在绑定 → {label}")
        print(f"      Agent ID: {agent_id}")

        try:
            resp = api_patch(f"/v1/agents/{agent_id}/archives/attach/{archive_id}", timeout=15)
            if resp.status_code in (200, 201, 204):
                print(f"      ✅ 绑定成功！{label} 已获得「{ARCHIVE_NAME}」的调阅权限")
                success_bind += 1
            elif resp.status_code == 409 or "already" in resp.text.lower():
                print(f"      ⚠️ Archive 已绑定到此 Agent，跳过。")
                success_bind += 1
            else:
                print(f"      [❌] 绑定失败 HTTP {resp.status_code}: {resp.text[:200]}")
                fail_bind += 1
        except Exception as e:
            print(f"      [❌] 绑定请求异常: {e}")
            fail_bind += 1
        print()

    print(f"  📊 绑定统计:")
    print(f"     成功 : {success_bind}")
    print(f"     失败 : {fail_bind}")

    return success_bind, fail_bind


# ============================================================
# 主函数
# ============================================================

def main():
    start_time = time.time()

    print()
    print("=" * 60)
    print("  🌊 MakoTalker 共享 Archive · 灵魂绑定引擎")
    print("  「一份记忆，三身同频」")
    print("=" * 60)
    print()

    # --------------------------------------------------------
    # 0. 前置检查
    # --------------------------------------------------------
    if not os.path.isdir(MEMORY_DIR):
        print(f"[❌] 错误：MEMORY_DIR 目录不存在 -> {MEMORY_DIR}")
        sys.exit(1)

    if len(TARGET_AGENT_IDS) != len(set(TARGET_AGENT_IDS)):
        print("[⚠️] 警告：TARGET_AGENT_IDS 中包含重复的 UUID，请检查。")
        sys.exit(1)

    # --------------------------------------------------------
    # 1. 检查 API 连通性
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
    # 2. 扫描 JSON 文件
    # --------------------------------------------------------
    print("─── 扫描记忆文件 ───")
    json_files = load_json_files(MEMORY_DIR)
    total_files = len(json_files)
    if total_files == 0:
        print(f"[⚠️] 在 '{MEMORY_DIR}' 中未找到任何 .json 文件，程序退出。")
        sys.exit(0)
    print(f"    ✅ 发现 {total_files} 个 JSON 记忆文件")
    print()

    # --------------------------------------------------------
    # 执行三大步骤
    # --------------------------------------------------------

    # 步骤一：创建/获取 Archive
    archive_id = step1_create_archive()

    # 步骤二：批量清洗并注入记忆（幂等保护已内建）
    step2_load_memories_into_archive(archive_id, json_files)

    # 步骤三：多 Agent 灵魂绑定
    step3_attach_archive_to_agents(archive_id, TARGET_AGENT_IDS)

    # --------------------------------------------------------
    # 最终总结
    # --------------------------------------------------------
    elapsed = time.time() - start_time
    print()
    print("=" * 60)
    print("  ✨ 灵魂绑定完成！")
    print("=" * 60)
    print(f"  Archive 名称 : {ARCHIVE_NAME}")
    print(f"  Archive ID   : {archive_id}")
    print(f"  记忆文件数   : {total_files}")
    print(f"  绑定 Agent   : {len(TARGET_AGENT_IDS)} 个")
    for i, aid in enumerate(TARGET_AGENT_IDS):
        aliases = {
            0: "日常对话 (mako_chat_agent)",
            1: "视觉搭话·表人格 (mako_proactive_agent)",
            2: "深度思考·里人格 (mako_think_agent)",
        }
        print(f"               [{i+1}] {aliases.get(i, aid)}")
        print(f"                    {aid}")
    print(f"  总耗时       : {elapsed:.1f} 秒")
    print("=" * 60)
    print("  茉子的完整灵魂已沉入共享 Archive 之海。")
    print("  三个分身皆可触及这份羁绊。")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
