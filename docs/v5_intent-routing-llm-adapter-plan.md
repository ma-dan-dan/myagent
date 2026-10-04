# V5 Intent Routing LLM Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** 为意图路由增加独立的 LLMAdapter 实例，使主聊天模型与意图识别模型可以分别配置、调用和统计，同时保持现有 LLMAdapter.complete(messages, tools) -> LLMResponse 契约和 V4 路由行为。

**Architecture:** 应用启动时保留主聊天 Adapter，并通过独立配置创建第二个意图 Adapter。INTENT_PROVIDER=llm 使用专用 Adapter，fake 和 jev 行为不变。两个 Adapter 共享现有 LiteLLM Provider 实现，但拥有独立实例、配置和指标来源。

**Tech Stack:** Python 3、dataclasses、Pydantic、FastAPI、LiteLLM、pytest、SQLite。

---

## 约束与现状

1. 先阅读 app/config.py、app/agent/adapter.py、app/agent/llm_factory.py、app/agent/llm_providers.py、app/intent/factory.py、app/intent/llm_classifier.py、app/main.py、相关测试和 README.md，保留已有注释与工作区修改。
2. 当前链路是 main.py -> IntentClassifierFactory.from_env(agent.llm) -> LLMIntentClassifier(agent.llm)；本计划改为独立的 intent_llm。
3. 禁止 git reset、git checkout、删除文件或覆盖无关改动。完成后必须把本次改动和当前工作区相关未提交改动一起提交。
4. 不实现流式、重试、模型路由、MCP、Skill、SQL 执行、向量检索、Workflow 或多 Agent。
5. 测试禁止访问真实模型或 Jev 网络；LiteLLM 使用 mock，意图 Adapter 使用 Fake/Stub。

## 配置契约

app/config.py 继续是唯一运行时环境变量读取入口。新增 INTENT_LLM_PROVIDER，允许 openai、deepseek、qwen，默认值为当前 DEFAULT_LLM_PROVIDER。意图模型复用对应 Provider 的 API Key 环境变量、代码内模型名和固定 Base URL；不新增 Base URL 环境变量。未设置 INTENT_LLM_PROVIDER 时沿用主聊天 Provider，避免现有部署突然要求另一套 Key。即使 Provider 相同，也必须创建两个不同 Adapter 对象。

建议新增接口：

~~~python
@dataclass(frozen=True)
class IntentLLMRuntimeConfig:
    provider: str
    api_key: str | None
    model: str
    base_url: str | None


def get_intent_llm_runtime_config() -> IntentLLMRuntimeConfig:
    provider = (os.getenv(INTENT_LLM_PROVIDER_ENV) or DEFAULT_LLM_PROVIDER).strip().lower()
    provider_config = LLM_PROVIDER_CONFIGS.get(provider)
    if provider_config is None:
        raise ValueError(provider)
    return IntentLLMRuntimeConfig(
        provider=provider,
        api_key=os.getenv(provider_config.api_key_env),
        model=provider_config.model,
        base_url=provider_config.base_url,
    )
~~~

API Key 不得写入源码、测试、README 或提交历史；业务模块不得直接调用 os.getenv() 或 os.environ。

## 文件边界

| 文件 | 职责 |
| --- | --- |
| app/config.py | 意图 LLM Provider 环境变量名、默认值、运行时配置和读取函数。 |
| app/intent/llm_adapter.py | 意图 Adapter 薄封装，保持 LLMAdapter 契约，复用现有 LiteLLM Provider Adapter。 |
| app/intent/llm_factory.py | 读取意图配置并构造独立 Adapter。 |
| app/intent/factory.py | INTENT_PROVIDER=llm 使用专用 Adapter，保留显式测试注入。 |
| app/intent/llm_classifier.py | 使用注入 Adapter，记录真实 Provider/Model。 |
| app/main.py | 分别装配主聊天 Adapter 与意图 Adapter，增加测试注入点。 |
| tests/test_config.py | 意图 LLM 配置测试。 |
| tests/test_intent_adapter.py | 工厂、实例隔离和 LiteLLM mock 测试。 |
| tests/test_intent_classifiers.py、tests/test_intent_api.py、tests/test_chat_api.py | 分类器、API、指标和 Fake 注入回归测试。 |
| README.md | 调用链、配置方式和 PowerShell 示例。 |

## Task 1：配置测试先行

**Files:** Modify tests/test_config.py; then implement app/config.py.

- [ ] Step 1: 写失败测试

~~~python
def test_intent_llm_runtime_config_uses_selected_provider(monkeypatch):
    monkeypatch.setenv("INTENT_LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "intent-key")
    config = get_intent_llm_runtime_config()
    assert config.provider == "deepseek"
    assert config.api_key == "intent-key"
    assert config.model == "deepseek-flash"
    assert config.base_url == "https://api.deepseek.com"


def test_intent_llm_runtime_config_defaults_to_main_provider(monkeypatch):
    monkeypatch.delenv("INTENT_LLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "intent-key")
    config = get_intent_llm_runtime_config()
    assert config.provider == "qwen"
    assert config.model == "deepseek-v4.1-flash"
~~~

- [ ] Step 2: 确认失败

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py::test_intent_llm_runtime_config_uses_selected_provider tests/test_config.py::test_intent_llm_runtime_config_defaults_to_main_provider -q
~~~

Expected: FAIL，因为函数尚未定义。

- [ ] Step 3: 最小实现并验证

在 app/config.py 增加 INTENT_LLM_PROVIDER_ENV、IntentLLMRuntimeConfig 和 get_intent_llm_runtime_config()。函数通过 LLM_PROVIDER_CONFIGS 取得模型和固定 Base URL，仅通过 api_key_env 读取 Key。运行：

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py -q
~~~

## Task 2：独立意图 Adapter

**Files:** Create app/intent/llm_adapter.py、app/intent/llm_factory.py、tests/test_intent_adapter.py。

- [ ] Step 1: 写失败测试

测试 IntentLLMAdapterFactory.from_env() 在 INTENT_LLM_PROVIDER=deepseek 时返回独立的 IntentLLMAdapter；它与 LLMAdapterFactory.create() 产生的主聊天 Adapter 不是同一对象，并暴露正确 provider/model/base_url。再 mock app.agent.adapter.litellm.completion，断言意图 Key、模型前缀和固定 Base URL，测试不得联网。

- [ ] Step 2: 最小实现

IntentLLMAdapter 必须复用 LLMAdapterFactory.create() 返回的 Provider Adapter，不得复制 LiteLLM 调用、Tool 序列化、usage 解析或异常转换逻辑。它可持有底层 Adapter 并转发 complete()，同时暴露 provider、model、base_url、litellm_model_name。IntentLLMAdapterFactory 读取 get_intent_llm_runtime_config() 后构造它；缺 Key 沿用 LLMConfigurationError，不伪造回复。

- [ ] Step 3: 验证并提交

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_intent_adapter.py -q
git add app/config.py app/intent/llm_adapter.py app/intent/llm_factory.py tests/test_config.py tests/test_intent_adapter.py
git commit -m "feat: add dedicated intent llm adapter"
~~~

## Task 3：分类器与启动装配

**Files:** Modify app/intent/factory.py、app/intent/llm_classifier.py、app/main.py、tests/test_intent_classifiers.py、tests/test_intent_api.py、tests/test_chat_api.py。

- [ ] Step 1: 写失败测试

验证 LLMIntentClassifier(intent_adapter).classify() 的 IntentDecision.provider 等于 Adapter 的真实 Provider（例如 deepseek），不再固定为 llm；无显式注入时工厂构造专用 Adapter；显式传入 Stub/Fake 时不构造真实 Adapter；create_app(llm_adapter=chat_fake, intent_llm_adapter=intent_fake) 将两个 Adapter 分别用于聊天和意图；SQLite 意图指标记录意图 Adapter 的真实 Provider/Model。

~~~python
llm = StubLLM(
    LLMResponse.message('{"intent":"chat","data_action":null,"confidence":0.95}'),
    provider="deepseek",
    model="deepseek-flash",
)
decision = LLMIntentClassifier(llm).classify("你好")
assert decision.provider == "deepseek"
assert decision.model == "deepseek-flash"
~~~

- [ ] Step 2: 确认失败

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py tests/test_intent_api.py tests/test_chat_api.py -q
~~~

- [ ] Step 3: 最小实现

LLMIntentClassifier 保持 complete(messages, tools=[])，只把 provider 元数据改为 getattr(self.llm, "provider", "llm")。IntentClassifierFactory.from_env(llm: LLMAdapter | None = None) 在显式传入时使用注入，未传入且 Provider 为 llm 时调用 IntentLLMAdapterFactory.from_env()。create_app() 增加 intent_llm_adapter: LLMAdapter | None = None；生产默认创建独立 Adapter，测试可注入 Fake。fake、jev、低置信度降级、原始决策和 routed_intent 行为保持不变；配置错误继续走现有不可用/503 路径。

- [ ] Step 4: 验证并提交

~~~powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py tests/test_intent_api.py tests/test_chat_api.py -q
git add app/intent/factory.py app/intent/llm_classifier.py app/main.py tests/test_intent_classifiers.py tests/test_intent_api.py tests/test_chat_api.py
git commit -m "feat: route intent classification through dedicated llm"
~~~

## Task 4：README、全量回归和全部提交

**Files:** Modify README.md；按测试结果修改必要测试；保留本计划文件。

- [ ] Step 1: 更新 README

~~~text
Request
  └─ IntentRouter
       └─ LLMIntentClassifier
            └─ IntentLLMAdapter → LiteLLM Provider
  └─ ChatService
       └─ ChatAgent
            └─ ChatLLMAdapter → LiteLLM Provider
~~~

说明两个 Adapter 是独立实例，INTENT_LLM_PROVIDER 可选择不同 Provider，模型和固定 Base URL 在 app/config.py 维护，Key 只通过 PowerShell/系统环境变量设置；INTENT_PROVIDER=jev 时不使用意图 LLM Adapter。示例：

~~~powershell
$env:LLM_PROVIDER = "qwen"
$env:QWEN_API_KEY = "<your-chat-key>"
$env:INTENT_PROVIDER = "llm"
$env:INTENT_LLM_PROVIDER = "deepseek"
$env:DEEPSEEK_API_KEY = "<your-intent-key>"
uvicorn app.main:app --reload
~~~

不得写入真实 Key，不得声称完成真实数据库、SQL、RAG 或在线 Jev 验证。

- [ ] Step 2: 完整验证

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
~~~

Expected: pytest 全部通过、compileall 退出码为 0、环境变量读取仅在 app/config.py、diff 无空白错误。

- [ ] Step 3: 检查并提交全部相关改动

~~~powershell
git status --short
git diff --stat
git add app/config.py app/agent/llm_providers.py app/intent app/main.py tests README.md docs/v5_intent-routing-llm-adapter-plan.md
git commit -m "feat: add dedicated intent routing llm adapter"
git status --short
git log -1 --oneline
~~~

实现 Agent 必须保留并提交当前工作区已有的相关修改，不得重置或删除；若发现无法判断的无关文件，停止并报告。最终报告必须包含提交哈希、完整测试结果、配置方式、两个 Adapter 的关系，以及没有进行真实模型/真实 Jev 网络调用的说明。

## 验收标准

- [ ] 主聊天继续使用原有 Adapter 和 Provider 配置。
- [ ] INTENT_PROVIDER=llm 使用独立意图 Adapter 实例。
- [ ] 原始 IntentDecision 保存真实意图 Provider 和模型元数据。
- [ ] fake、jev、低置信度降级和 V4 API 行为兼容。
- [ ] 缺少意图模型 Key 不伪造分类成功。
- [ ] 所有外部调用在测试中被 mock 或替换。
- [ ] pytest、compileall、git diff --check 通过。
- [ ] 实现 Agent 已提交全部相关改动。


---

## 交给另一个 Agent 的执行提示词

请在 D:\project\python\myagent 按本文件实施 V5 独立意图路由 LLM Adapter。不要只给建议，必须实际修改代码、测试和 README。

实施要求：

1. 先阅读本计划和现有实现，重点阅读 app/config.py、app/agent/adapter.py、app/agent/llm_factory.py、app/agent/llm_providers.py、app/intent/factory.py、app/intent/llm_classifier.py、app/main.py 以及相关测试。
2. 为 INTENT_PROVIDER=llm 增加独立的意图 LLMAdapter 实例。主聊天模型继续由 ChatAgent 使用，意图识别不再默认复用主聊天 Adapter。
3. 在 app/config.py 集中新增 INTENT_LLM_PROVIDER 和意图 LLM 运行时配置读取函数。意图模型复用现有 Provider 的模型名和代码内固定 Base URL，API Key 只从对应 Provider 环境变量读取。其他业务模块不得直接调用 os.getenv() 或 os.environ。
4. 新增 app/intent/llm_adapter.py 和 app/intent/llm_factory.py。必须复用现有 LiteLLM Provider Adapter 和 LLMAdapterFactory，不要复制 LiteLLM 调用、Tool 序列化、usage 解析或异常转换。
5. 修改 IntentClassifierFactory 和 app/main.py，支持生产默认创建独立意图 Adapter，并提供 Fake/Stub 注入能力。修改 LLMIntentClassifier，使 IntentDecision.provider 记录实际意图 Adapter Provider，model 记录实际意图模型。
6. 保持 fake、jev、低置信度降级、原始 decision、routed_intent、API 协议、SQLite 隔离和主聊天 Tool Loop 行为不变。
7. 必须按 TDD：先补失败测试并运行确认失败，再实现最小代码。所有 LiteLLM 和 TypeSafe 调用必须 mock，禁止真实网络和真实 API Key。
8. 更新 README 的调用链、配置项和 PowerShell 示例，不写入真实 Key，不扩展 SQL、MCP、Skill、SSE、向量库、Workflow 或多 Agent。
9. 不删除已有注释，不执行 git reset、git checkout、删除文件或回滚无关改动。当前工作区可能已有未提交修改，请先检查 git status，并保留所有相关改动。
10. 完成后运行：

~~~powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
~~~

确认测试、编译和空白检查全部通过，且运行时环境变量读取只在 app/config.py。

11. 最后必须提交全部相关改动，不要只提交自己新增的文件：

~~~powershell
git status --short
git diff --stat
git add app/config.py app/agent/llm_providers.py app/intent app/main.py tests README.md docs/v5_intent-routing-llm-adapter-plan.md
git commit -m "feat: add dedicated intent routing llm adapter"
git status --short
git log -1 --oneline
~~~

提交前不得丢弃当前已有修改。最终报告必须包含：修改文件、架构变化、测试完整结果、提交哈希、工作区是否干净，以及明确说明没有调用真实模型 API 或 Jev 在线 API。
