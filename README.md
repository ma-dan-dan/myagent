# MyAgent

MyAgent 是一个用于学习 Agentic Chat 架构的 Python 项目。当前项目提供基于 FastAPI 的多轮对话接口，并将模型调用、工具调用、会话上下文和长期记忆拆分为独立模块。

## 已实现能力

- 提供 `POST /api/v1/chat` 多轮聊天接口和简单网页入口。
- 按 `user_id + session_id` 隔离会话，持久化用户消息、助手消息、工具事件、会话摘要和模型用量。
- 通过 `ChatAgent` 实现轻量 Agent Loop：模型可请求工具，后端执行工具后将结果回填给模型，再获取最终回答。
- 通过 `ToolRegistry` 管理工具白名单；当前已接入 `search_schema` 工具。
- 使用本地 `data/schema_catalog.json` 提供表名、字段名、类型和业务说明的只读检索。
- 使用 LiteLLM 接入 OpenAI、DeepSeek、Qwen 三类模型，并保留可注入的 Fake LLM 测试方式。
- 使用滑动窗口、会话摘要、Token 估算和 `MEMORY.md` 管理会话上下文与长期项目记忆。

## 架构概览

```text
Web / API Request
        ↓
ChatService
        ↓
ContextManager → ChatAgent → LLMAdapter
                    ↓             ↓
              ToolRegistry    LiteLLM Provider
                    ↓
             SchemaSearchTool → SchemaCatalog
```

## 快速启动

在 PowerShell 中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

选择模型提供商，并在系统环境变量或当前 PowerShell 会话中配置对应 API Key。以下以 Qwen 为例：

```powershell
$env:LLM_PROVIDER = "qwen"
$env:QWEN_API_KEY = "<your-key>"
uvicorn app.main:app --reload
```

启动后访问 `http://127.0.0.1:8000/`，或调用聊天接口：

```powershell
$body = @{
  user_id = "demo-user"
  message = "查询产量相关的表"
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/chat" `
  -ContentType "application/json" `
  -Body $body
```

模型名称在 `app/agent/llm_providers.py` 的各 Provider 类中维护；API Key 不写入代码或仓库。

## 测试

必须使用项目虚拟环境运行测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

## 目录说明

```text
app/
  agent/       Agent Loop、LLM 适配器、工具注册表
  api/         FastAPI 路由
  memory/      上下文、摘要、Token 与长期记忆
  services/    聊天请求编排
  storage/     会话持久化与 Schema 元数据目录
  tools/       业务工具
data/          Schema 元数据与本地运行数据库位置
docs/          版本设计与实施文档
tests/         单元测试与 API 测试
web/           简单前端页面
```

## 版本文档

- `docs/v1_minimal-agentic-chat-design.md`：最小 Agentic Chat 设计。
- `docs/v2_llm-adapter-multi-provider.md`：多模型厂商适配。
- `docs/v3_context-token-memory-management.md`：上下文、Token 与长期记忆。

每次完成一个版本的实现后，同步更新本 README，并将代码、测试和文档一起提交到 GitHub。
