# MakoTalker — 茉子桌面 AI 语音宠物

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)

一个基于 PyQt5 的桌面 AI 伴侣应用。以视觉小说《千恋＊万花》中的角色 **常陆茉子（Hitachi Mako）** 为主题，集成了大语言模型对话、语音识别、语音合成、计算机视觉与长期记忆，打造一个有"人格"的桌面宠物。

---

## ✨ 功能

| 功能 | 说明 |
|------|------|
| 🗣️ **语音对话** | FunASR 实时语音识别 + Silero VAD 静音检测，支持语音唤醒与打断 |
| 🧠 **AI 人格** | DeepSeek + Letta/MemGPT 记忆框架，支持长程上下文与人格一致性 |
| 🔊 **语音合成** | GPT-SoVITS 本地 TTS（支持 normal/shy 情绪切换）+ Fish Audio 云端 TTS |
| 👀 **视觉感知** | Qwen2-VL 多模态模型实时分析屏幕截图 + 摄像头画面，理解用户当前状态 |
| 💬 **主动搭话** | 基于视觉上下文的主动问候（"盯着代码好久了呢，要休息一下吗？"） |
| 🔒 **声纹锁** | CAM++ 说话人验证，只响应注册用户的声音 |
| ⏰ **自然语言闹钟** | "提醒我明天下午三点开会" → 自动解析时间并到时提醒 |
| 🎨 **桌宠 UI** | PyQt5 可拖动窗口、气泡对话、立绘切换、动画特效 |
| 💾 **长期记忆** | Letta 自动管理对话历史与记忆归档 |

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────┐
│                    main.py（总装）                     │
│   桌宠窗口 · 播放队列 · 打断控制 · 启动检查             │
└─────────────────────────────────────────────────────┘
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │ DS_test    │  │ voice_    │  │ deep_      │
    │ DS_think   │  │ listener  │  │ thinker    │
    │ DS_proact  │  │           │  │            │
    │ (对话中枢)  │  │ (ASR/VAD) │  │ (视觉沙盒)  │
    └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
          │              │              │
    ┌─────▼─────┐  ┌─────▼─────┐  ┌─────▼─────┐
    │  Letta    │  │ FunASR    │  │ Qwen2-VL  │
    │  (记忆)    │  │ CAM++     │  │ (视觉模型)  │
    └───────────┘  └───────────┘  └───────────┘
          │
    ┌─────▼─────┐
    │ DeepSeek  │
    │  (云端)    │
    └───────────┘
```

**三套人格体系：**
- **表人格（chat_agent）** — 日常对话，轻松活泼、会捉弄人
- **里人格（think_agent）** — 深度视觉分析，不直接说话
- **搭话人格（proactive_agent）** — 基于视觉上下文生成主动台词

---

## 🖥️ 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 (12GB VRAM) | RTX 4060+ (16GB VRAM) |
| 内存 | 16 GB | 32 GB |
| 磁盘 | 20 GB 空闲 | 50 GB SSD |
| 麦克风 | 任意 | 降噪麦克风 |
| 摄像头 | 可选（视觉功能需要） | 1080p 以上 |

> **注意：** GPU 显存是关键瓶颈。Qwen2-VL（~4GB）+ GPT-SoVITS（~4GB）+ FunASR（~2GB）+ CAM++（~1GB）同时加载约需 10-12GB VRAM。

---

## 🚀 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/pwicdsummer/Mako.git
cd Mako
```

### 2. 创建虚拟环境

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# 或
.venv\Scripts\activate      # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入你的 API Key：

```ini
FISH_AUDIO_API_KEY=你的Fish_Audio_Key     # 可选，云端 TTS
FISH_AUDIO_REFERENCE_ID=你的Reference_ID   # 可选
```

### 5. 准备素材（⚠️ 重要）

由于版权原因，仓库**不包含**以下资产，请自行准备：

| 素材 | 目录 | 说明 |
|------|------|------|
| 角色立绘 | `img/` | 放入 PNG 图片，修改 `simple_chat_ui.py` 中的 `PET_IMAGE_PATH` |
| TTS 参考音频 | `gptsovit_voice_needed/` | GPT-SoVITS 需要的角色语音样本 |
| TTS 模型权重 | `gptsovit_weights/` | GPT-SoVITS 训练的模型文件 |

### 6. 注册声纹（可选，启用声纹锁）

```bash
# 1. 将你的录音（3~30秒 WAV）放入 my_voice/ 目录
# 2. 运行声纹注册
python register_me.py
```

### 7. 启动依赖服务

```bash
# 启动 Letta 服务（需先安装 Letta/MemGPT）
letta server --port 8283

# 启动 GPT-SoVITS 推理服务（需单独部署）
# 默认地址 http://127.0.0.1:9880
```

### 8. 初始化 Letta Agent（首次使用）

```bash
python letta_one_click_init.py
```

### 9. 启动！

```bash
python main.py
```

---

## ⚙️ 配置

| 文件 | 用途 |
|------|------|
| `.env` | API Key 等敏感配置（不被 git 追踪） |
| `.env.example` | 配置模板，列出所有可选环境变量 |
| `config.json` | 茉子的人设 prompt（日文角色卡） |
| `letta_agents.json` | Letta Agent UUID 映射（由 init 脚本生成） |

### 环境变量参考

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FISH_AUDIO_API_KEY` | — | Fish Audio API Key（可选，不设置则使用本地 TTS） |
| `FISH_AUDIO_REFERENCE_ID` | — | Fish Audio 声音克隆 ID |
| `LETTA_BASE_URL` | `http://127.0.0.1:8283` | Letta 服务地址 |
| `GPT_SOVITS_URL` | `http://127.0.0.1:9880` | GPT-SoVITS 服务地址 |

---

## 📦 模块速览

| 文件 | 职责 |
|------|------|
| `main.py` | 主程序入口，桌宠窗口 + 播放队列 + 打断控制 |
| `simple_chat_ui.py` | PyQt5 桌宠 UI 组件（气泡、输入弹窗、立绘渲染） |
| `voice_listener.py` | 实时麦克风监听（VAD → 声纹验证 → ASR 转写） |
| `DS_test.py` | 对话 Agent 接口（Letta chat_agent） |
| `DS_thinking.py` | 思考 Agent 接口（Letta think_agent） |
| `DS_proactive.py` | 主动搭话 Agent 接口（Letta proactive_agent） |
| `deep_thinker.py` | 视觉沙盒：Qwen2-VL + DeepSeek 10 轮 VQA 循环 |
| `letta_one_click_init.py` | Letta Agent 一键初始化/刷新脚本 |
| `alarm_parser.py` | 自然语言时间解析（"3分钟后提醒我..."） |
| `alarm_scheduler.py` | 闹钟后台调度器 |
| `fish_tts_plugin.py` | Fish Audio 云端 TTS 异步插件 |
| `mako_cv.py` | 计算机视觉工具（人脸检测、ROI 裁切等） |
| `mako_window_capture.py` | Windows 窗口精准截取 |
| `register_me.py` | 声纹注册脚本 |
| `record_speaker_ref.py` | 录音参考脚本 |
| `chat_logger.py` | 对话日志保存 |

---

## ⚠️ 注意事项

### 版权声明

本项目代码以 GPL v3 许可证开源。但以下内容**不属于开源范围**：

- 角色立绘（`img/`）—— 版权归 Yuzusoft 所有
- 角色参考音频（`gptsovit_voice_needed/`）—— 版权归原作权利方所有
- 启动视频（`launch_mv.mp4`）—— 可能包含版权素材

请仅以个人学习/研究目的使用，勿用于商业用途。如要在公开仓库中分发，请将上述素材替换为自制/公版资源。

### 隐私

- 所有对话记录、生成语音、记忆数据均存储在本地
- 云端请求仅发送给 DeepSeek API / Fish Audio API
- 不会自动上传任何个人数据到第三方

---

## 📄 License

代码部分采用 [GPL v3 License](LICENSE)。

角色形象与音频素材的版权归 Yuzusoft / 原作权利方所有。
