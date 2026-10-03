# Minimal Agentic Chat V1 运行说明

## 安装

在项目目录执行：

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 配置真实模型

一次运行只选择下面一段配置。Key 只在本地环境中设置，不要写入代码或提交到项目。

OpenAI：

```powershell
$env:LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
uvicorn app.main:app --reload
```

DeepSeek：

```powershell
$env:LLM_PROVIDER = "deepseek"
$env:DEEPSEEK_API_KEY = "<your-key>"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"
uvicorn app.main:app --reload
```

Qwen：

```powershell
$env:LLM_PROVIDER = "qwen"
$env:QWEN_API_KEY = "<your-key>"
$env:QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
uvicorn app.main:app --reload
```

模型名不再从环境变量读取。请在 `app/agent/llm_providers.py` 中修改对应类的 `MODEL_ENV` 常量；虽然该常量沿用了旧名称，但它现在保存的是实际模型名。例如当前 Qwen 类中的：

```python
class QwenLLMAdapter(LiteLLMAdapter):
    MODEL_ENV = "deepseek-v4.1-flash"
```

`OPENAI_BASE_URL`、`DEEPSEEK_BASE_URL` 和 `QWEN_BASE_URL` 都可以显式改为对应的 OpenAI-compatible 代理地址。旧的 DeepSeek 配置如果使用了 `OPENAI_*` 变量，需要迁移为 `LLM_PROVIDER=deepseek` 和 `DEEPSEEK_*` 变量；Qwen 同理使用 `QWEN_*`。

未配置 API Key 时，应用仍可启动，但 `POST /api/v1/chat` 返回 HTTP 503 和明确的配置错误，不会伪装成普通助手回复。

浏览器匿名身份和当前 Session 保存在 `localStorage` 中；清理浏览器站点数据会开始新的匿名会话。后端仍通过 `user_id + session_id` 做 SQLite 隔离。

## V3 上下文与记忆

原始消息始终保存在 SQLite 的 `messages` 表中，不会因压缩删除。近期上下文使用滑动窗口；较早内容写入 `session_summaries`，并保存摘要边界。摘要失败时保留旧摘要并继续当前聊天。

每轮上下文按固定系统规则、只读 `Agent.md`、整份受限长期记忆、会话摘要、近期原始消息和当前问题组装。长期记忆只使用 `.myagent/memory/chat/MEMORY.md`，每条候选一行，限制为 200 行和 25 KB；不使用 `topics/`、关键词 Top-K 或重排，并拒绝密钥、带值的认证 Token、Cookie、密码、原始聊天和未经确认的猜测。

上下文预算使用 `H = context_window - output_reserve`、`S = floor(H × 0.80)`，固定上下文最多使用 `floor(S × 0.20)`。当前用户消息始终完整保留；近期历史最多 12 条，但按 `R = S - F - U` 动态从最新消息向前选择，超出的旧消息进入滚动摘要。每轮最多执行两次摘要调用；摘要成功后再执行一次 Memory 提炼，主聊天完成后才写入 `MEMORY.md`，下一轮整体注入。

如果固定 system prompt 与当前问题本身就超过硬预算，或最多两次摘要后仍无法构造安全 Prompt，接口返回 HTTP 413，并提示缩短当前问题；这不是模型服务故障，也不会写入当前用户原始消息。

响应中的 `usage.current_turn` 和 `usage.session_total` 区分 Provider 返回的真实 `input_tokens`、`output_tokens`、`total_tokens` 与 LiteLLM `token_counter` 得到的 `estimated_context_tokens`；估算失败只降级为 UTF-8 字节估算，不影响聊天。

pytest 中的 `FakeLLM` 通过 `create_app(llm_adapter=fake)` 或直接创建 `ChatAgent(fake, registry)` 注入；浏览器真实 API 请求不会自动使用 FakeLLM。适配器测试使用 mock 的 `litellm.completion`，不会调用外部模型 API：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app
```
