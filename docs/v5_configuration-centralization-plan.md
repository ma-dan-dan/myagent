# V5 配置集中管理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将项目全部运行时环境变量的名称、默认值和读取逻辑集中到 `app/config.py`，使其成为唯一配置入口，同时保持真实密钥只由系统环境变量提供。

**Architecture:** `app/config.py` 以清晰的配置分区管理项目路径、主聊天模型、意图识别和运行参数，并提供只读的配置读取函数。`LLMAdapterFactory` 与 `IntentClassifierFactory` 不再直接调用 `os.getenv()`，只使用该模块提供的配置结果；各 Provider Adapter 的环境变量名与默认模型也从该模块引用，避免重复定义。

**Tech Stack:** Python 3、dataclasses、pytest、FastAPI、LiteLLM、TypeSafe SDK。

---

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `app/config.py` | 唯一的环境变量读取入口；按类别定义配置项、默认值与 Provider 配置。 |
| `app/agent/llm_providers.py` | 从集中配置引用各模型 Provider 的环境变量名与默认模型。 |
| `app/agent/llm_factory.py` | 使用集中配置构造聊天模型 Adapter。 |
| `app/intent/factory.py` | 使用集中配置选择意图分类器和读取 Jev Key。 |
| `tests/test_config.py` | 验证默认值、环境变量覆盖和无 Key 场景。 |
| `tests/test_llm_adapter.py` | 调整为验证工厂通过集中配置读取模型配置。 |
| `tests/test_intent_classifiers.py` | 调整为验证 Jev Key 由集中配置读取。 |
| `README.md` | 补充集中配置位置和启动配置示例。 |

### Task 1: 为集中配置定义测试契约

**Files:**

- Create: `tests/test_config.py`

- [ ] **Step 1: 编写主聊天模型配置的失败测试**

```python
from app.config import get_llm_runtime_config


def test_get_llm_runtime_config_reads_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "qwen")
    monkeypatch.setenv("QWEN_API_KEY", "test-key")
    monkeypatch.setenv("QWEN_BASE_URL", "https://example.test/v1")

    config = get_llm_runtime_config()

    assert config.provider == "qwen"
    assert config.api_key == "test-key"
    assert config.model == "deepseek-v4.1-flash"
    assert config.base_url == "https://example.test/v1"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py::test_get_llm_runtime_config_reads_selected_provider -q`

Expected: FAIL，因为 `get_llm_runtime_config` 尚未定义。

- [ ] **Step 3: 编写 Jev 配置的失败测试**

```python
from app.config import get_intent_runtime_config


def test_get_intent_runtime_config_reads_jev_key(monkeypatch):
    monkeypatch.setenv("INTENT_PROVIDER", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-jev-key")

    config = get_intent_runtime_config()

    assert config.provider == "jev"
    assert config.typesafe_api_key == "test-jev-key"
```

- [ ] **Step 4: 运行测试确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py::test_get_intent_runtime_config_reads_jev_key -q`

Expected: FAIL，因为 `get_intent_runtime_config` 尚未定义。

### Task 2: 在 config.py 实现集中配置入口

**Files:**

- Modify: `app/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: 定义不可变配置数据类型和分类常量**

在 `app/config.py` 添加 `LLMProviderConfig`、`LLMRuntimeConfig` 和 `IntentRuntimeConfig` 三个 `@dataclass(frozen=True)`。按“项目路径”“主聊天模型”“意图识别”“运行参数”四个分区放置配置常量；每个分区使用 Python 合法的 `#` 单行说明，说明这些值的作用。保留既有路径常量和值不变。

`LLMProviderConfig` 必须具有下列字段：

```python
provider: str
api_key_env: str
model: str
base_url_env: str
```

`LLMRuntimeConfig` 必须具有下列字段：

```python
provider: str
api_key: str | None
model: str
base_url: str | None
```

`IntentRuntimeConfig` 必须具有下列字段：

```python
provider: str
typesafe_api_key: str | None
```

- [ ] **Step 2: 集中定义所有当前环境变量名和默认值**

在 `app/config.py` 定义：

```python
LLM_PROVIDER_ENV = "LLM_PROVIDER"
DEFAULT_LLM_PROVIDER = "openai"
INTENT_PROVIDER_ENV = "INTENT_PROVIDER"
DEFAULT_INTENT_PROVIDER = "llm"
TYPESAFE_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_MODEL = "typesafe/jev-1.13"
INTENT_MIN_CONFIDENCE = 0.70
```

再定义 `LLM_PROVIDER_CONFIGS`，包含下列保持现有行为不变的 Provider 配置：

```python
"openai": LLMProviderConfig("openai", "OPENAI_API_KEY", "gpt-4o-mini", "OPENAI_BASE_URL")
"deepseek": LLMProviderConfig("deepseek", "DEEPSEEK_API_KEY", "deepseek-flash", "DEEPSEEK_BASE_URL")
"qwen": LLMProviderConfig("qwen", "QWEN_API_KEY", "deepseek-v4.1-flash", "QWEN_BASE_URL")
```

- [ ] **Step 3: 实现唯一读取函数**

```python
def get_llm_runtime_config() -> LLMRuntimeConfig:
    provider = (os.getenv(LLM_PROVIDER_ENV) or DEFAULT_LLM_PROVIDER).strip().lower()
    provider_config = LLM_PROVIDER_CONFIGS.get(provider)
    if provider_config is None:
        raise ValueError(provider)
    return LLMRuntimeConfig(
        provider=provider,
        api_key=os.getenv(provider_config.api_key_env),
        model=provider_config.model,
        base_url=os.getenv(provider_config.base_url_env),
    )


def get_intent_runtime_config() -> IntentRuntimeConfig:
    return IntentRuntimeConfig(
        provider=(os.getenv(INTENT_PROVIDER_ENV) or DEFAULT_INTENT_PROVIDER).strip().lower(),
        typesafe_api_key=os.getenv(TYPESAFE_API_KEY_ENV),
    )
```

- [ ] **Step 4: 运行新测试确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py -q`

Expected: PASS，两个测试均通过。

### Task 3: 迁移模型和意图工厂的环境读取

**Files:**

- Modify: `app/agent/llm_providers.py`
- Modify: `app/agent/llm_factory.py`
- Modify: `app/intent/factory.py`
- Modify: `app/intent/jev_classifier.py`
- Modify: `tests/test_llm_adapter.py`
- Modify: `tests/test_intent_classifiers.py`

- [ ] **Step 1: 编写工厂不再直接读取 os.getenv 的失败测试**

在 `tests/test_llm_adapter.py` 添加：

```python
def test_factory_uses_config_runtime_values(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "key")

    adapter = LLMAdapterFactory.from_env()

    assert adapter.provider == "deepseek"
    assert adapter.model == "deepseek-flash"
```

在 `tests/test_intent_classifiers.py` 添加：

```python
def test_jev_factory_reads_key_through_central_config(monkeypatch, fake_llm):
    monkeypatch.setenv("INTENT_PROVIDER", "jev")
    monkeypatch.setenv("TYPESAFE_API_KEY", "jev-key")

    classifier = IntentClassifierFactory.from_env(fake_llm)

    assert isinstance(classifier, JevIntentClassifier)
    assert classifier.api_key == "jev-key"
```

- [ ] **Step 2: 运行新增测试确认失败**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_llm_adapter.py tests/test_intent_classifiers.py -q`

Expected: 当前代码仍可能通过部分行为测试，但 `rg -n "os.getenv" app/agent/llm_factory.py app/intent/factory.py` 显示两个工厂仍直接读取环境变量；此步骤用于确认迁移目标。

- [ ] **Step 3: 迁移生产代码**

1. `app/agent/llm_providers.py` 从 `app.config` 导入 `LLM_PROVIDER_CONFIGS`，并用对应配置初始化三个 Adapter 的 `PROVIDER`、`API_KEY_ENV`、`MODEL_ENV`、`BASE_URL_ENV`。
2. `app/agent/llm_factory.py` 删除 `import os`，从 `app.config` 导入 `get_llm_runtime_config` 与 `LLM_PROVIDER_CONFIGS`；`from_env()` 使用配置结果创建既有 `LLMAdapterConfig`。不改变 `create()` 和 Provider 白名单错误信息。
3. `app/intent/factory.py` 删除 `import os`，从 `app.config` 导入 `get_intent_runtime_config` 与 `TYPESAFE_API_KEY_ENV`；通过配置结果完成 fake、llm、jev 分支，并保留缺失 Key 时的中文错误信息。
4. `app/intent/jev_classifier.py` 从 `app.config` 导入 `JEV_MODEL`，让 `MODEL = JEV_MODEL`。

- [ ] **Step 4: 运行相关测试确认通过**

Run: `./.venv/Scripts/python.exe -m pytest tests/test_config.py tests/test_llm_adapter.py tests/test_intent_classifiers.py tests/test_intent_api.py -q`

Expected: PASS。

### Task 4: 更新使用文档并做全量验证

**Files:**

- Modify: `README.md`

- [ ] **Step 1: 更新 README 的配置说明**

在“快速启动”前新增“配置位置”小节，明确说明：

```text
所有环境变量名、默认 Provider 和默认模型均集中维护在 app/config.py。
真实 API Key 仅在系统环境变量或当前 PowerShell 会话中设置，不能写入 app/config.py、README 或 Git。
```

列出当前变量：`LLM_PROVIDER`、`OPENAI_API_KEY`、`OPENAI_BASE_URL`、`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`QWEN_API_KEY`、`QWEN_BASE_URL`、`INTENT_PROVIDER`、`TYPESAFE_API_KEY`。明确模型名称在 `app/config.py` 的 `LLM_PROVIDER_CONFIGS` 中维护。

- [ ] **Step 2: 运行环境变量读取残留检查**

Run: `rg -n "os\.getenv|os\.environ" app`

Expected: 仅 `app/config.py` 出现环境变量读取；不应在工厂或业务模块出现。

- [ ] **Step 3: 运行全量测试与编译检查**

Run: `./.venv/Scripts/python.exe -m pytest -q`

Expected: PASS，所有测试通过。

Run: `./.venv/Scripts/python.exe -m compileall -q app tests`

Expected: 退出码 0。

- [ ] **Step 4: 检查提交内容并同步 GitHub**

Run: `git diff --check; git status --short; git diff -- app/config.py app/agent/llm_factory.py app/agent/llm_providers.py app/intent/factory.py app/intent/jev_classifier.py tests README.md`

Expected: 无空白错误，变更只包含集中配置、测试和 README。

随后执行：

```powershell
git add app/config.py app/agent/llm_factory.py app/agent/llm_providers.py app/intent/factory.py app/intent/jev_classifier.py tests/test_config.py tests/test_llm_adapter.py tests/test_intent_classifiers.py README.md docs/v5_configuration-centralization-plan.md
git commit -m "refactor: centralize runtime configuration"
git push origin master
```

## 给另一个 Agent 的执行提示词

```text
请在 D:\project\python\myagent 按照 docs/v5_configuration-centralization-plan.md 实施 V5：配置集中管理。

目标：让 app/config.py 成为项目唯一的运行时环境变量读取入口。将全部已有环境变量的名称、默认 Provider、默认模型与读取逻辑集中到该文件；其他业务模块不能继续直接调用 os.getenv() 或 os.environ。

必须遵守：
1. 修改前阅读 app/config.py、app/agent/llm_factory.py、app/agent/llm_providers.py、app/intent/factory.py、app/intent/jev_classifier.py 和相关测试，沿用现有风格。
2. 不要删除、覆盖或回滚任何已有注释与无关改动。
3. 真实 API Key 绝不能写入 app/config.py、README、测试或 Git；config.py 只定义环境变量名并通过 os.getenv() 读取系统环境变量。
4. 在 app/config.py 中按“项目路径”“主聊天模型”“意图识别”“运行参数”分类，并用清晰的 Python 单行注释解释每个分区用途，使初学者能一眼理解。
5. 需要集中管理的变量至少包括：LLM_PROVIDER、OPENAI_API_KEY、OPENAI_BASE_URL、DEEPSEEK_API_KEY、DEEPSEEK_BASE_URL、QWEN_API_KEY、QWEN_BASE_URL、INTENT_PROVIDER、TYPESAFE_API_KEY。
6. 不改变当前模型默认值和既有运行行为；尤其保持当前 OpenAI、DeepSeek、Qwen、fake、llm、jev Provider 的可用性和错误提示语义。
7. LLMAdapterFactory、IntentClassifierFactory 及其他业务模块必须通过 app.config 的读取函数/配置对象获取配置，不能自行读取环境变量。
8. 为集中配置新增测试，至少覆盖：Qwen Provider 的 API Key/Base URL 读取、Jev Key 读取、默认 Provider，以及缺少 Key 时的既有错误行为。
9. 更新 README：说明所有配置项在 app/config.py 集中维护，真实 Key 只能通过系统环境变量或当前 PowerShell 会话设置；列出现有变量及用途。
10. 使用项目虚拟环境验证：
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m compileall -q app tests
    rg -n "os\\.getenv|os\\.environ" app
   最后一条检查应只在 app/config.py 中看到运行时环境变量读取。
11. 检查 git diff --check；提交信息使用：refactor: centralize runtime configuration；随后 git push origin master。
12. 最终回复请说明：改了哪些文件、集中管理了哪些配置、验证命令及结果、提交哈希与 GitHub 推送结果。

注意：这是 Python 项目；分类注释必须使用 Python 合法的 # 单行注释，不能使用会导致 Python 语法错误的 //。
```
