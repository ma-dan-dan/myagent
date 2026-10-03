# 多提供商 LLMAdapter V1.0 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 保持现有 `ChatAgent -> LLMAdapter.complete(messages, tools)` 契约不变，参照 DA 工程的 `DeepSeekModel`、`QwenModel`、`OpenAIModel` 分层，为项目增加由 LiteLLM 驱动的 OpenAI、DeepSeek、Qwen 三种真实模型接入，并保留可注入的 FakeLLM 测试方式。

**架构：** `ChatAgent` 只依赖既有的 `LLMAdapter` Protocol，不识别模型厂商。`LLMAdapterFactory` 根据 `LLM_PROVIDER` 创建 `OpenAILLMAdapter`、`DeepSeekLLMAdapter` 或 `QwenLLMAdapter`；三者继承共享的 `LiteLLMAdapter`，仅提供厂商名称、模型名前缀、默认地址和环境变量名。共享层负责 LiteLLM 调用、工具 schema 转换、响应解析及统一异常转换。

**技术栈：** Python 3.12、FastAPI、Pydantic v2、LiteLLM、pytest、现有手写 AgentLoop。

---

## 0. 范围与边界

### 本次必须实现

1. 只支持 `openai`、`deepseek`、`qwen` 三种真实模型提供商。
2. 通过 LiteLLM 的 `litellm.completion(...)` 发送非流式请求。
3. 保留 `LLMAdapter.complete(messages, tools) -> LLMResponse`，不得改动 `ChatAgent` 的调用方式。
4. 保留现有 `LLMResponse` 的两种结果：普通文本 `message` 与单个 `tool_call`。
5. 用工厂按配置选择模型实现；`app/main.py` 只调用工厂，不再直接实例化 OpenAI SDK 适配器。
6. 所有自动化测试必须模拟 LiteLLM，不允许测试调用真实网络或要求真实 Key。
7. 在运行文档中说明三种供应商的环境变量写法，以及旧的 `OPENAI_*` 配置如何迁移到 DeepSeek/Qwen 的专属变量。

### 本次明确不做

1. 不改 `ChatAgent`、`ToolRegistry`、工具、Session、API 路由、前端和 SQLite 存储逻辑。
2. 不实现流式输出、重试、模型列表、运行时切换、模型实例缓存、MCP、Skill、JSON mode、推理过程或多模型路由。
3. 不引入 OpenAI Agents SDK；当前项目的手写 AgentLoop 继续负责“模型调用 → 工具执行 → 再调用模型”。
4. 不把 `FakeLLM` 注册为生产供应商；它仅作为测试替身，实现相同协议即可。

### 与原 DA 工程的对应关系

| DA 工程 | 本项目 V1.0 | 责任 |
| --- | --- | --- |
| `da.models.base.LLMBaseModel` | `app.agent.adapter.LLMAdapter` Protocol | 让 Agent 只依赖统一模型契约 |
| `LLMBaseModel.create_model()` | `app.agent.llm_factory.LLMAdapterFactory` | 根据 provider 创建正确实现 |
| `OpenAICompatibleModel` | `app.agent.adapter.LiteLLMAdapter` | 共享请求、工具 schema、响应解析与异常处理 |
| `DeepSeekModel` | `app.agent.llm_providers.DeepSeekLLMAdapter` | DeepSeek 专属配置 |
| `QwenModel` | `app.agent.llm_providers.QwenLLMAdapter` | Qwen 专属配置 |
| `OpenAIModel` | `app.agent.llm_providers.OpenAILLMAdapter` | OpenAI 专属配置 |
| `da.models.litellm_adapter.LiteLLMAdapter` | 本项目 `LiteLLMAdapter` | 生成 LiteLLM 模型标识并调用 LiteLLM |
| `MockLLMModel` | 测试中的 `FakeLLM` | 模拟模型决定，验证真实 AgentLoop |

注意：DA 中的 `OpenAICompatibleModel` 还包含会话、流式、重试、OpenAI Agents SDK 等复杂能力；本项目只借鉴其“共享逻辑放基类、厂商差异放子类”的分层，不照搬这些超出当前范围的能力。

## 1. 目标目录与职责

| 文件 | 操作 | 职责 |
| --- | --- | --- |
| `app/agent/adapter.py` | 修改 | 保留 `ToolSpec`、`LLMResponse`、`LLMAdapter`、模型异常；新增共享 `LiteLLMAdapter`，删除旧的直接 OpenAI SDK 调用实现 |
| `app/agent/llm_providers.py` | 新建 | 定义 `OpenAILLMAdapter`、`DeepSeekLLMAdapter`、`QwenLLMAdapter` 三个薄子类 |
| `app/agent/llm_factory.py` | 新建 | 定义不可变 `LLMAdapterConfig` 与 `LLMAdapterFactory`，集中选择提供商 |
| `app/main.py` | 修改 | 将默认模型装配改为 `LLMAdapterFactory.from_env()` |
| `requirements.txt` | 修改 | 用 `litellm` 替代项目对 `openai` 的直接依赖 |
| `tests/test_llm_adapter.py` | 修改 | 覆盖工厂、三种模型命名、请求映射、响应映射和异常，不联网 |
| `docs/minimal-agentic-chat-run.md` | 修改 | 写明新的环境变量与启动命令 |

不创建 `app/agent/models/` 多层目录。本项目当前规模较小，三个厂商类放在 `llm_providers.py` 更容易阅读；等将来提供商数量显著增长时，再拆分为独立模块。

## 2. 配置约定

`LLM_PROVIDER` 决定当前激活的提供商，值只允许为 `openai`、`deepseek`、`qwen`，忽略大小写并在内部标准化为小写。

每个提供商只读取自己的变量；避免用户把 DeepSeek Key 误配置在 `OPENAI_API_KEY` 下而难以排查：

| Provider | 必填 Key | 必填模型名 | 可选 Base URL | LiteLLM 模型名 | 默认 Base URL |
| --- | --- | --- | --- | --- | --- |
| `openai` | `OPENAI_API_KEY` | `OPENAI_MODEL` | `OPENAI_BASE_URL` | 官方地址时为 `<model>`；自定义 OpenAI 兼容地址时为 `openai/<model>` | 由 LiteLLM/OpenAI 使用官方默认地址 |
| `deepseek` | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `DEEPSEEK_BASE_URL` | `deepseek/<model>` | `https://api.deepseek.com` |
| `qwen` | `QWEN_API_KEY` | `QWEN_MODEL` | `QWEN_BASE_URL` | `dashscope/<model>` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |

迁移规则：此前若使用 DeepSeek 却把 Key、模型和地址填在 `OPENAI_*` 中，必须改为 `LLM_PROVIDER=deepseek` 加 `DEEPSEEK_*` 三个变量。不要为了兼容错误配置而让 DeepSeek 类读取 `OPENAI_*`；清晰、可定位的配置边界比隐式兼容更重要。

`LLMAdapterFactory.from_env()` 在 `LLM_PROVIDER` 缺失时默认选择 `openai`，以保持当前项目“服务可以启动、第一次真实请求时才报告缺少 Key/模型”的行为。若 `LLM_PROVIDER` 值不受支持，工厂必须在应用创建阶段抛出清晰的 `LLMConfigurationError`，列出允许值。

## 3. 核心调用链

```text
create_app()
  -> LLMAdapterFactory.from_env()
  -> 按 LLM_PROVIDER 创建某个具体 Adapter
  -> ChatAgent(adapter, registry)

用户请求
  -> ChatAgent.run(messages)
  -> adapter.complete(messages, registry.specs())
  -> LiteLLMAdapter 组织 litellm.completion(...)
  -> LiteLLM 调用 OpenAI / DeepSeek / Qwen
  -> LLMResponse(message 或 tool_call)
  -> ChatAgent 决定直接回复或执行既有 ToolRegistry
```

关键原则：提供商选择只发生在应用装配时；工具调用循环仍只发生在 `ChatAgent`。三种适配器都必须向上返回相同的 `LLMResponse`，因此 ChatAgent 不应出现 `if provider == ...`。

## 4. 实施任务

### Task 1：先写多提供商适配层的失败测试

**文件：**

- 修改：`tests/test_llm_adapter.py`
- 修改：`tests/test_chat_agent.py`（仅在导入路径或测试替身需要适配时修改；原有断言语义不得削弱）

- [ ] **Step 1：将现有的 `FakeClient` 改为 LiteLLM 调用替身。**

在 `tests/test_llm_adapter.py` 增加一个能记录关键字参数的函数替身，并用 `monkeypatch` 替换 `litellm.completion`。替身返回现有测试已使用的 `SimpleNamespace(choices=[...])` 响应，确保测试不会触网。

```python
def fake_completion(**kwargs):
    calls.append(kwargs)
    return response

monkeypatch.setattr("litellm.completion", fake_completion)
```

- [ ] **Step 2：为模型标识和默认地址写参数化失败测试。**

覆盖以下三组输入与期望：

```python
(
    "openai", "gpt-4.1-mini", None,
    "gpt-4.1-mini", None,
),
(
    "deepseek", "deepseek-chat", None,
    "deepseek/deepseek-chat", "https://api.deepseek.com",
),
(
    "qwen", "qwen-plus", None,
    "dashscope/qwen-plus", "https://dashscope.aliyuncs.com/compatible-mode/v1",
),
```

断言 `litellm.completion` 收到正确的 `model`、`api_key` 和（当非空时）`api_base`。另写一个自定义 Base URL 用例，断言它覆盖默认地址。

- [ ] **Step 3：为工厂和配置错误写失败测试。**

至少覆盖：

```python
assert isinstance(LLMAdapterFactory.create(config_for_openai), OpenAILLMAdapter)
assert isinstance(LLMAdapterFactory.create(config_for_deepseek), DeepSeekLLMAdapter)
assert isinstance(LLMAdapterFactory.create(config_for_qwen), QwenLLMAdapter)

with pytest.raises(LLMConfigurationError, match="openai, deepseek, qwen"):
    LLMAdapterFactory.create(LLMAdapterConfig(provider="unknown", api_key="k", model="m"))
```

还要测试：选中 DeepSeek 却缺少 `DEEPSEEK_API_KEY` 时，错误消息包含准确的变量名；不能笼统提示 `OPENAI_API_KEY`。

- [ ] **Step 4：保留并改写现有响应映射测试。**

普通回答仍应映射为 `LLMResponse.message("模型回复")`；函数调用仍应解析 JSON 参数，映射成 `LLMResponse.tool_call(...)`，并确认发送给 LiteLLM 的 `tools` 保持 OpenAI function-tool schema。

- [ ] **Step 5：运行失败测试，确认失败原因是实现尚未完成。**

运行：

```powershell
pytest tests/test_llm_adapter.py -q
```

预期：因 `LLMAdapterConfig`、`LLMAdapterFactory`、三种具体适配器或 LiteLLM 依赖尚不存在而失败；不得因真实网络调用失败。

### Task 2：建立稳定契约与 LiteLLM 共享层

**文件：**

- 修改：`app/agent/adapter.py`
- 修改：`requirements.txt`

- [ ] **Step 1：保留已有公共契约。**

不得修改下列名称、字段或方法签名：

```python
class ToolSpec(BaseModel): ...
class LLMResponse(BaseModel): ...
class LLMAdapter(Protocol):
    def complete(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse: ...
class LLMServiceUnavailable(RuntimeError): ...
class LLMConfigurationError(LLMServiceUnavailable): ...
```

这能确保 `ChatAgent`、现有 FakeLLM 和 API 注入测试无需理解新的厂商层。

- [ ] **Step 2：删除旧 `OpenAICompatibleAdapter` 的直接 OpenAI SDK 客户端逻辑，新增共享 `LiteLLMAdapter`。**

`LiteLLMAdapter` 必须实现 `LLMAdapter`，并至少拥有以下只读配置属性：`provider`、`model`、`api_key`、`base_url`、`litellm_model_name`。它接收已解析的配置，不直接读取环境变量。

共享层必须提供以下逻辑：

```python
def _tool_payload(self, tools: list[ToolSpec]) -> list[dict[str, Any]]: ...
def _request_kwargs(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> dict[str, Any]: ...
def _to_llm_response(self, response: Any) -> LLMResponse: ...
def complete(self, messages: list[dict[str, Any]], tools: list[ToolSpec]) -> LLMResponse: ...
```

`_request_kwargs` 的行为必须是：总是包含 `model`、`messages`、`api_key`；有 `base_url` 时使用 LiteLLM 参数名 `api_base`；有工具时加入 `tools` 和 `tool_choice="auto"`。`complete` 只能在该处调用 `litellm.completion(**request_kwargs)`。

- [ ] **Step 3：沿用已有响应校验强度。**

`_to_llm_response` 必须延续当前实现的保护：没有 `choices[0].message`、Tool 无名称、Tool 参数无法 JSON 解析、Tool 参数不是对象、普通消息内容为空，都抛出 `LLMServiceUnavailable`。调用 LiteLLM 时捕获底层异常，并以 `raise ... from exc` 保留异常链；面向用户的消息不得泄露 API Key 或底层完整错误文本。

- [ ] **Step 4：只替换直接依赖。**

将 `requirements.txt` 中的 `openai>=1.40,<4` 替换为兼容 Python 3.12 的 LiteLLM 依赖：

```text
litellm>=1,<2
```

不要修改 FastAPI、Pydantic、pytest 或 uvicorn 的版本范围。不要在业务代码中保留 `from openai import OpenAI`。

- [ ] **Step 5：运行适配器测试。**

运行：

```powershell
pytest tests/test_llm_adapter.py -q
```

预期：共享层的普通回复、工具调用、空响应和网络异常映射测试通过；提供商工厂相关测试仍可能失败，留给下一任务。

### Task 3：实现三个薄厂商类与工厂

**文件：**

- 新建：`app/agent/llm_providers.py`
- 新建：`app/agent/llm_factory.py`
- 修改：`app/agent/__init__.py`（仅在项目已有显式导出风格且确有需要时）

- [ ] **Step 1：定义不可变配置对象。**

在 `app/agent/llm_factory.py` 定义：

```python
@dataclass(frozen=True)
class LLMAdapterConfig:
    provider: str
    api_key: str | None
    model: str | None
    base_url: str | None = None
```

该对象对应 DA 的 `ModelConfig` 中当前项目真正需要的四项；不要提前加入温度、重试、思考模式、追踪等字段。

- [ ] **Step 2：定义三种具体适配器。**

在 `app/agent/llm_providers.py` 定义并继承 `LiteLLMAdapter`：

```python
class OpenAILLMAdapter(LiteLLMAdapter): ...
class DeepSeekLLMAdapter(LiteLLMAdapter): ...
class QwenLLMAdapter(LiteLLMAdapter): ...
```

每个类只通过类常量或很小的覆盖定义以下差异：

| 类 | provider | LiteLLM 前缀 | 默认地址 | Key 变量 | Model 变量 | Base URL 变量 |
| --- | --- | --- | --- | --- | --- |
| `OpenAILLMAdapter` | `openai` | `""` | `None` | `OPENAI_API_KEY` | `OPENAI_MODEL` | `OPENAI_BASE_URL` |
| `DeepSeekLLMAdapter` | `deepseek` | `"deepseek/"` | `https://api.deepseek.com` | `DEEPSEEK_API_KEY` | `DEEPSEEK_MODEL` | `DEEPSEEK_BASE_URL` |
| `QwenLLMAdapter` | `qwen` | `"dashscope/"` | `https://dashscope.aliyuncs.com/compatible-mode/v1` | `QWEN_API_KEY` | `QWEN_MODEL` | `QWEN_BASE_URL` |

模型名已经带有 `/` 时，不得重复加前缀。OpenAI 的官方地址或未设置地址时，原生模型无前缀；当 `OPENAI_BASE_URL` 是非 `api.openai.com` 的 OpenAI 兼容代理地址时，按 DA 的 `LiteLLMAdapter._get_litellm_model_name()` 规则使用 `openai/<model>`。这与 DA 的 `LiteLLMAdapter.MODEL_PREFIX_MAP` 和自定义地址处理保持一致。

- [ ] **Step 3：实现工厂。**

`LLMAdapterFactory` 必须有以下两个公开入口：

```python
class LLMAdapterFactory:
    @classmethod
    def create(cls, config: LLMAdapterConfig) -> LLMAdapter: ...

    @classmethod
    def from_env(cls) -> LLMAdapter: ...
```

`create` 将标准化后的 `config.provider` 映射到三个具体类。映射表只能在工厂中维护，禁止在 `ChatAgent` 或 `main.py` 里出现 provider 分支。

`from_env` 读取 `LLM_PROVIDER`，缺失时使用 `openai`；随后让选中的具体类读取其专属变量并构造 `LLMAdapterConfig`。API Key 或模型名缺失不在 `from_env` 中报错，而是在第一次 `complete` 时抛出 `LLMConfigurationError`，以保持当前 Web 服务可正常启动的行为。

- [ ] **Step 4：让错误指向选中的提供商。**

共享层在执行 `complete` 前验证 `api_key` 和 `model`。错误消息必须基于具体类声明的变量名，例如：

```text
缺少真实模型配置：DEEPSEEK_API_KEY, DEEPSEEK_MODEL。
```

不得把 DeepSeek/Qwen 的缺失配置错误写成 `OPENAI_API_KEY`。不支持的 `LLM_PROVIDER` 必须在 `from_env`/`create` 明确报出允许值 `openai, deepseek, qwen`。

- [ ] **Step 5：运行适配器测试。**

运行：

```powershell
pytest tests/test_llm_adapter.py -q
```

预期：全部通过，且执行中无任何真实 HTTP 请求。

### Task 4：仅修改应用装配点

**文件：**

- 修改：`app/main.py`

- [ ] **Step 1：替换默认装配代码。**

将：

```python
from app.agent.adapter import LLMAdapter, OpenAICompatibleAdapter
...
agent = ChatAgent(llm_adapter if llm_adapter is not None else OpenAICompatibleAdapter.from_env(), registry)
```

替换为：

```python
from app.agent.adapter import LLMAdapter
from app.agent.llm_factory import LLMAdapterFactory
...
agent = ChatAgent(llm_adapter if llm_adapter is not None else LLMAdapterFactory.from_env(), registry)
```

不得改变 `create_app(..., llm_adapter=...)` 的参数和优先级。测试注入的 `FakeLLM` 仍必须绕过工厂与真实模型配置。

- [ ] **Step 2：运行 API 与 Agent 回归测试。**

运行：

```powershell
pytest tests/test_chat_agent.py tests/test_chat_api.py -q
```

预期：全部通过；它们继续使用 FakeLLM，不读取环境变量，也不调用 LiteLLM 网络。

### Task 5：更新运行文档并完成全量验证

**文件：**

- 修改：`docs/minimal-agentic-chat-run.md`

- [ ] **Step 1：补充三种可复制的 PowerShell 配置示例。**

示例必须是以下结构，模型名仅为示例且不声称它一定对所有账户可用：

```powershell
# OpenAI
$env:LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "<your-key>"
$env:OPENAI_MODEL = "gpt-4.1-mini"

# DeepSeek
$env:LLM_PROVIDER = "deepseek"
$env:DEEPSEEK_API_KEY = "<your-key>"
$env:DEEPSEEK_MODEL = "deepseek-chat"
$env:DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# Qwen
$env:LLM_PROVIDER = "qwen"
$env:QWEN_API_KEY = "<your-key>"
$env:QWEN_MODEL = "qwen-plus"
```

在每段示例后说明：一次运行只能选择一段配置；Key 不要写入仓库；若通过代理，使用对应的 `*_BASE_URL` 显式覆盖默认地址。

- [ ] **Step 2：解释 FakeLLM 的位置。**

运行文档中说明：`FakeLLM` 存在于 pytest 测试中，通过 `create_app(llm_adapter=fake)` 或直接创建 `ChatAgent(fake, registry)` 注入；浏览器中的真实 API 请求不会自动使用 FakeLLM。

- [ ] **Step 3：安装并运行全量测试。**

运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
```

预期：全部测试通过，且没有测试访问真实模型服务。

- [ ] **Step 4：做边界审计。**

运行：

```powershell
rg -n "OpenAICompatibleAdapter|from openai import OpenAI|chat\.completions\.create" app tests
rg -n "LLM_PROVIDER|DEEPSEEK_API_KEY|QWEN_API_KEY|OPENAI_API_KEY" app docs tests
```

预期：第一条不再出现旧适配器、直接 OpenAI SDK 客户端调用；第二条只出现在工厂、具体提供商类、测试和运行文档中。

- [ ] **Step 5：报告结果。**

报告应包含：修改文件清单、三类模型与原 DA 文件的对应、实际 pytest 结果、没有进行真实 API 调用的证据，以及用户启动 DeepSeek 时所需的环境变量。当前目录不是 Git 仓库，不要执行 commit 命令。

## 5. 验收标准

1. `ChatAgent` 和 `ToolRegistry` 的源代码不需要理解 `openai`、`deepseek`、`qwen` 字符串。
2. `create_app(llm_adapter=fake)` 仍能正常运行既有 API 测试。
3. `LLM_PROVIDER=deepseek` 时，调用参数中的模型为 `deepseek/<DEEPSEEK_MODEL>`，默认地址为 `https://api.deepseek.com`。
4. `LLM_PROVIDER=qwen` 时，调用参数中的模型为 `dashscope/<QWEN_MODEL>`，默认地址为 DashScope 兼容地址。
5. `LLM_PROVIDER=openai` 且未设置地址时，调用参数中的模型名不添加 LiteLLM 前缀，也不传 `api_base`；使用非官方 OpenAI 兼容地址时，模型名使用 `openai/<model>`。
6. 工具 schema 和模型 Tool Call 响应仍能被现有 AgentLoop 正确处理。
7. 缺少 Key/模型时得到面向选中提供商的 `LLMConfigurationError`，不泄露密钥。
8. 不支持的 Provider 得到明确错误，而非静默回退到其他模型。
9. 全量 pytest 通过；所有测试都使用 mocked `litellm.completion` 或既有 FakeLLM。

## 6. 交给另一个 Agent 的提示词

将下面整段原样发送给实施 Agent：

```text
请在 D:\project\python\myagent 按 docs\llm-adapter-multi-provider-v1.0-plan.md 实施“多提供商 LLMAdapter V1.0”。先完整阅读该计划、当前 app\agent\adapter.py、app\main.py、tests\test_llm_adapter.py、tests\test_chat_agent.py，以及原项目 D:\project\qiuzhao\DA 中以下文件：

- da\models\base.py
- da\models\openai_compatible.py
- da\models\litellm_adapter.py
- da\models\deepseek_model.py
- da\models\qwen_model.py
- da\models\openai_model.py

目标是只改模型接入层：保留 ChatAgent 使用的 LLMAdapter.complete(messages, tools) -> LLMResponse 契约，引入 LiteLLM，并以“共享 LiteLLMAdapter + OpenAILLMAdapter / DeepSeekLLMAdapter / QwenLLMAdapter 三个薄子类 + LLMAdapterFactory”的结构实现。请严格遵循计划中的文件边界、环境变量、LiteLLM 模型前缀、默认 Base URL、异常行为和验收标准。

约束：
1. 不修改 ChatAgent、ToolRegistry、工具、Session、API 协议、前端和数据库逻辑；app/main.py 仅允许改默认 adapter 的装配入口。
2. 不实现流式、重试、模型路由、MCP、Skill、OpenAI Agents SDK 或其他计划外能力。
3. 保留所有已有注释，不删除或改写无关代码；本次不新增代码注释。
4. 不访问真实模型 API，不要求真实 API Key。所有测试必须 mock litellm.completion 或注入既有 FakeLLM。
5. 先写/更新失败测试，再实现最小代码使测试通过；不要只改实现不补测试。
6. 依赖变更后使用 .\.venv\Scripts\python.exe -m pip install -r requirements.txt，并使用 .\.venv\Scripts\python.exe -m pytest -q 验证。
7. 当前目录不是 Git 仓库，不要执行 git commit、reset、checkout 或删除无关文件。

完成后请输出：实际修改的文件、关键架构说明、pytest 的完整结果、以及 DeepSeek/OpenAI/Qwen 各自的 PowerShell 配置方式。若计划与现有代码有冲突，先以“保持 LLMAdapter 契约、最小改动、可测试”为原则处理，并清楚说明原因。
```
