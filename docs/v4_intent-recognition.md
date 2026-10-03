# V4 意图识别实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 在现有 Agentic Chat 请求入口之前增加三分类意图识别，并通过统一接口支持 Fake、TypeSafe Jev 官方在线 API 和普通 LLM 三种分类器，记录可比较的延迟、Token 与分类结果。

**架构：** 新增独立的 `app/intent/` 模块。`IntentClassifier` 是唯一的分类器协议；Fake、Jev、普通 LLM 都实现该协议。`IntentRouter` 负责置信度降级，`ChatService` 负责根据路由结果继续现有聊天流程或返回当前阶段的稳定占位响应。Jev 不进入 LiteLLM，因为它是固定标签的决策 API，不是生成式 Chat Completion API。

**技术栈：** Python 3.12、FastAPI、Pydantic v2、httpx、LiteLLM、TypeSafe 官方 `typesafe-sdk`、pytest。

---

## 1. 范围与边界

### 1.1 本版本实现内容

- 三个一级意图：`chat`、`data_operation`、`nl2sql`。
- `data_operation` 预留二级操作字段：`read`、`create`、`update`、`delete`、`unknown`。
- 现阶段仅把 `read` 视作将来可继续接入的只读数据查询入口；所有 `data_operation` 和 `nl2sql` 都只返回稳定测试占位响应，不连接业务数据库、不生成 SQL、不执行 SQL。
- 三个可替换分类器：
  - `FakeIntentClassifier`：测试中使用，不发起网络请求。
  - `JevIntentClassifier`：调用 TypeSafe Jev 官方在线 API。
  - `LLMIntentClassifier`：复用现有 `LLMAdapter`，要求模型返回受约束 JSON。
- 对每次分类记录 provider、model、意图、二级操作、置信度、耗时、Token 用量、是否降级。
- 在 API 响应中返回本次 `intent_decision`；提供按用户和 provider 聚合的意图指标接口。
- 提供固定评测样本和命令行评测脚本，输出准确率、平均延迟、P50/P95 延迟与 Token 总量。

### 1.2 本版本明确不做

- DDL + Sample Value 双路 RAG。
- 真实业务数据库连接、SQL 生成、SQL 校验、SQL 执行和 Reflection Graph。
- 写操作授权、审批、审计或真实增删改。
- 前端路由页面改造；前端只需展示后端既有响应中新增的分类字段。

### 1.3 术语

```text
CHAT             普通问答、项目讨论、解释类问题。
DATA_OPERATION   用户希望获取或操作数据；本版本仅返回占位响应。
NL2SQL           用户明确要求生成、改写、解释或展示 SQL；本版本仅返回占位响应。
```

`data_action` 不是第四种一级意图，而是 `DATA_OPERATION` 的预留扩展字段。这样未来接入 CRUD 时不需要破坏三分类 API 契约。

## 2. 目标调用流程

```text
POST /api/v1/chat
  ↓
ChatService.chat(user_id, session_id, message)
  ↓
IntentRouter.route(message)
  ↓
IntentClassifier.classify(message)
  ├─ FakeIntentClassifier
  ├─ JevIntentClassifier
  └─ LLMIntentClassifier
  ↓
置信度 >= 0.70 且结果合法？
  ├─ 是：保留分类结果
  └─ 否：降级为 CHAT，标记 fallback_reason
  ↓
SessionService.record_intent_classification(...)
  ↓
intent == CHAT ?
  ├─ 是：ContextManager → ChatAgent → 既有工具调用流程
  └─ 否：写入用户消息和稳定占位助手消息，直接返回
```

`CHAT` 是保守降级目的地：低置信度或普通 LLM 返回非法 JSON 时，系统不应误进入未来的数据库或 SQL 执行分支。

## 3. Jev 官方在线 API 约定

Jev 使用 TypeSafe 的决策模型接口，只做分类，不生成自然语言。实现使用官方 Python 包 `typesafe-sdk`，由其处理官方 API 请求与响应格式。

```text
Python 包：typesafe-sdk
API Key 环境变量：TYPESAFE_API_KEY
模型常量：typesafe/jev-1.13
调用方式：TypeSafeClient().system_one(state=..., questions=...)
```

Jev 分类必须在一次请求中提出两个 `Choice` 问题：

```text
intent：chat / data_operation / nl2sql
data_action：read / create / update / delete / unknown
```

当一级意图不是 `data_operation` 时，最终写入 `IntentDecision.data_action = None`，不暴露 Jev 对无关二级问题的预测。

普通 LLM 分类器必须使用以下固定 JSON 结构，不传工具：

```json
{
  "intent": "chat",
  "data_action": null,
  "confidence": 0.95
}
```

模型输出会先经 `json.loads`，再经 Pydantic 校验；非法 JSON、未知标签、缺字段和范围外置信度都会触发 Router 的 `CHAT` 降级，不会进入数据或 SQL 占位分支。

## 4. 数据模型与接口契约

### 4.1 `app/intent/models.py`

```python
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.memory.models import TokenUsage


class IntentName(str, Enum):
    CHAT = "chat"
    DATA_OPERATION = "data_operation"
    NL2SQL = "nl2sql"


class DataAction(str, Enum):
    READ = "read"
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    UNKNOWN = "unknown"


class IntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: IntentName
    data_action: DataAction | None = None
    confidence: float = Field(ge=0, le=1)
    provider: str = Field(min_length=1, max_length=40)
    model: str = Field(min_length=1, max_length=100)
    latency_ms: float = Field(ge=0)
    usage: TokenUsage = Field(default_factory=TokenUsage)
    fallback_reason: str | None = Field(default=None, max_length=300)

    @model_validator(mode="after")
    def validate_data_action(self) -> "IntentDecision":
        if self.intent != IntentName.DATA_OPERATION:
            self.data_action = None
        return self
```

### 4.2 `app/intent/classifier.py`

```python
from typing import Protocol

from app.intent.models import IntentDecision


class IntentClassifier(Protocol):
    def classify(self, message: str) -> IntentDecision:
        """Classify one user message without mutating session state."""
```

分类器不接收 `SessionService`、`ChatAgent` 或工具注册表；它只能基于当前用户消息产生分类结果。这避免分类层反向依赖 Agent Loop。

### 4.3 配置和工厂

在 `app/config.py` 增加：

```python
DEFAULT_INTENT_PROVIDER = "llm"
INTENT_PROVIDER_ENV = "INTENT_PROVIDER"
INTENT_MIN_CONFIDENCE = 0.70
```

`IntentClassifierFactory.from_env(llm)` 读取 `INTENT_PROVIDER`，允许值是 `fake`、`jev`、`llm`。Jev 只从 `TYPESAFE_API_KEY` 环境变量读取密钥；模型名保持为 `JevIntentClassifier.MODEL = "typesafe/jev-1.13"` 的代码常量。普通 LLM 复用已经创建好的 `LLMAdapter`，不会重复创建模型适配器。

`create_app()` 新增可选参数 `intent_classifier: IntentClassifier | None = None`，优先使用注入实例，便于 API 测试选择 Fake 分类器。

## 5. 文件变更清单

| 文件 | 操作 | 责任 |
|---|---|---|
| `requirements.txt` | 修改 | 添加 `typesafe-sdk` 官方客户端依赖。 |
| `app/intent/__init__.py` | 新建 | 暴露意图模块公共类型。 |
| `app/intent/models.py` | 新建 | 三类意图、CRUD 预留动作、分类结果模型。 |
| `app/intent/classifier.py` | 新建 | 分类器协议与分类异常。 |
| `app/intent/fake_classifier.py` | 新建 | 可预测的测试分类器。 |
| `app/intent/llm_classifier.py` | 新建 | 基于现有 `LLMAdapter` 的 JSON 分类器。 |
| `app/intent/jev_classifier.py` | 新建 | TypeSafe Jev 官方在线分类器。 |
| `app/intent/factory.py` | 新建 | `fake`、`jev`、`llm` 创建和配置校验。 |
| `app/intent/router.py` | 新建 | 置信度门限和 `CHAT` 降级策略。 |
| `app/config.py` | 修改 | 分类器 provider 和置信度配置。 |
| `app/main.py` | 修改 | 创建或注入分类器并传入 `ChatService`。 |
| `app/services/chat_service.py` | 修改 | 在现有 Agent 前进行路由，并处理两类测试占位响应。 |
| `app/storage/session_service.py` | 修改 | 新建分类记录表、写入记录、聚合指标查询。 |
| `app/schemas/chat.py` | 修改 | 扩展聊天响应和指标响应模型。 |
| `app/api/chat.py` | 修改 | 添加意图指标查询接口与分类异常映射。 |
| `app/evaluation/__init__.py` | 新建 | 评测包。 |
| `app/evaluation/intent_benchmark.py` | 新建 | 固定样本评测与指标输出。 |
| `data/intent_eval_cases.json` | 新建 | 不含真实用户数据的标注样本。 |
| `tests/test_intent_classifiers.py` | 新建 | 三种分类器和 JSON 校验测试。 |
| `tests/test_intent_router.py` | 新建 | 低置信度、非法结果和异常策略测试。 |
| `tests/test_intent_api.py` | 新建 | 三路 API、持久化、指标接口测试。 |
| `tests/test_intent_benchmark.py` | 新建 | 准确率与延迟聚合测试。 |
| `README.md` | 修改 | 实现完成后补充已落地的意图识别能力、配置和评测命令。 |

## 6. 实施任务

### Task 1：建立意图模型和失败测试

**Files:**

- Create: `app/intent/__init__.py`
- Create: `app/intent/models.py`
- Create: `app/intent/classifier.py`
- Create: `tests/test_intent_classifiers.py`

- [ ] **Step 1: 写入失败测试**

```python
import pytest
from pydantic import ValidationError

from app.intent.models import DataAction, IntentDecision, IntentName


def test_non_data_intent_clears_data_action():
    result = IntentDecision(
        intent=IntentName.CHAT,
        data_action=DataAction.DELETE,
        confidence=0.9,
        provider="fake",
        model="fake-intent-v1",
        latency_ms=0,
    )

    assert result.data_action is None


def test_confidence_must_be_between_zero_and_one():
    with pytest.raises(ValidationError):
        IntentDecision(
            intent=IntentName.CHAT,
            confidence=1.1,
            provider="fake",
            model="fake-intent-v1",
            latency_ms=0,
        )
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py -q
```

Expected: FAIL，因为 `app.intent` 尚不存在。

- [ ] **Step 3: 实现 `IntentName`、`DataAction`、`IntentDecision` 和 `IntentClassifier` 协议**

严格采用第 4 节的模型代码。添加 `IntentClassificationError(RuntimeError)`，它代表分类服务不可用或配置错误，不代表低置信度。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/intent tests/test_intent_classifiers.py
git commit -m "feat: add intent classification contract"
```

### Task 2：实现 Fake 分类器和 Router 降级策略

**Files:**

- Create: `app/intent/fake_classifier.py`
- Create: `app/intent/router.py`
- Create: `tests/test_intent_router.py`

- [ ] **Step 1: 写入失败测试**

```python
from app.intent.fake_classifier import FakeIntentClassifier
from app.intent.models import IntentName
from app.intent.router import IntentRouter


def test_router_uses_classifier_result_above_threshold():
    classifier = FakeIntentClassifier.for_result("查今天的产量", "data_operation", "read", 0.95)
    result = IntentRouter(classifier, min_confidence=0.70).route("查今天的产量")

    assert result.intent is IntentName.DATA_OPERATION
    assert result.data_action.value == "read"
    assert result.fallback_reason is None


def test_router_falls_back_to_chat_for_low_confidence():
    classifier = FakeIntentClassifier.for_result("帮我看看", "nl2sql", None, 0.30)
    result = IntentRouter(classifier, min_confidence=0.70).route("帮我看看")

    assert result.intent is IntentName.CHAT
    assert result.provider == "fallback"
    assert result.fallback_reason == "low_confidence"
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_router.py -q
```

Expected: FAIL，因为 Fake 分类器和 Router 尚不存在。

- [ ] **Step 3: 最小实现**

`FakeIntentClassifier` 接收 `dict[str, IntentDecision]`；`for_result()` 仅是测试构造器。未命中的消息固定返回 `chat`、置信度 `1.0`、provider `fake`、model `fake-intent-v1`、latency `0.0`。

`IntentRouter.route()` 只捕获“结果合法但置信度不足”的情况并降级为 `CHAT`。分类器抛出 `IntentClassificationError` 时必须继续抛出，让 API 返回 503；不能伪造成功聊天响应。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py tests/test_intent_router.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/intent tests/test_intent_router.py
git commit -m "feat: add intent router fallback"
```

### Task 3：实现普通 LLM 意图分类器

**Files:**

- Create: `app/intent/llm_classifier.py`
- Modify: `tests/test_intent_classifiers.py`

- [ ] **Step 1: 写入失败测试**

```python
from app.agent.adapter import LLMResponse
from app.intent.llm_classifier import LLMIntentClassifier
from app.intent.models import IntentName


class StubLLM:
    provider = "qwen"
    model = "test-qwen"

    def complete(self, messages, tools):
        assert tools == []
        return LLMResponse.message(
            '{"intent":"nl2sql","data_action":null,"confidence":0.91}'
        )


def test_llm_classifier_parses_valid_json():
    result = LLMIntentClassifier(StubLLM()).classify("帮我生成查询产量的 SQL")

    assert result.intent is IntentName.NL2SQL
    assert result.provider == "llm"
    assert result.model == "test-qwen"
```

再补充一个非法 JSON 测试，断言抛出 `IntentClassificationError`。

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py -q
```

Expected: FAIL，因为 `LLMIntentClassifier` 尚不存在。

- [ ] **Step 3: 实现分类器**

构造一个只含 `system` 和当前 `user` 消息的列表，调用 `llm.complete(messages, tools=[])`。System Prompt 必须逐项说明三个意图边界，并要求只输出第 3 节 JSON，不要 Markdown、不输出解释。

当 `LLMResponse.kind != "message"`、内容不是 JSON 或 `IntentDecision` 校验失败时，抛出 `IntentClassificationError("普通 LLM 返回了无效的意图分类结果。")`。将 `LLMResponse.usage` 写入 `IntentDecision.usage`，并用 `time.perf_counter()` 计算真实 `latency_ms`。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/intent/llm_classifier.py tests/test_intent_classifiers.py
git commit -m "feat: add LLM intent classifier"
```

### Task 4：实现 Jev 官方在线分类器和工厂

**Files:**

- Modify: `requirements.txt`
- Create: `app/intent/jev_classifier.py`
- Create: `app/intent/factory.py`
- Modify: `app/config.py`
- Modify: `tests/test_intent_classifiers.py`

- [ ] **Step 1: 写入失败测试**

```python
import pytest

from app.intent.factory import IntentClassifierFactory
from app.intent.classifier import IntentClassificationError


def test_jev_factory_requires_typesafe_api_key(monkeypatch, stub_llm):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setenv("INTENT_PROVIDER", "jev")

    with pytest.raises(IntentClassificationError, match="TYPESAFE_API_KEY"):
        IntentClassifierFactory.from_env(stub_llm)
```

使用 monkeypatch 替换 `app.intent.jev_classifier.TypeSafeClient`，测试 `response.answers["intent"].choice`、`confidence`、`data_action` 被转换为 `IntentDecision`，确保单测不会访问真实网络。

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py -q
```

Expected: FAIL，因为 Jev 分类器和工厂尚不存在。

- [ ] **Step 3: 实现官方 Jev 接入**

在 `requirements.txt` 添加：

```text
typesafe-sdk>=0.7,<1
```

`JevIntentClassifier` 使用：

```python
from typesafe_sdk import Choice, TypeSafeClient
```

调用 `TypeSafeClient(api_key=self.api_key).system_one(...)`，其中 `state={"message": message}`；以两个 `Choice` 传入第 3 节的 `intent` 与 `data_action`。读取 `response.answers["intent"].choice` 和 `.confidence`，最终 `confidence` 使用 intent Choice 的置信度。Jev 没有本项目统一的生成 Token 账单时，`TokenUsage()` 保持全零；延迟仍必须使用 `perf_counter()` 记录。

`IntentClassifierFactory` 仅允许 `fake`、`jev`、`llm`。未知值抛出 `IntentClassificationError`，错误信息列出允许值。`JevIntentClassifier.MODEL` 固定为 `typesafe/jev-1.13`；不得把 `TYPESAFE_API_KEY` 写入日志、响应、SQLite 或测试快照。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_classifiers.py tests/test_intent_router.py -q
```

Expected: PASS，且测试中没有真实网络请求。

- [ ] **Step 5: 提交**

```powershell
git add requirements.txt app/config.py app/intent tests/test_intent_classifiers.py
git commit -m "feat: add Jev intent classifier"
```

### Task 5：接入 ChatService、响应模型和分类记录

**Files:**

- Modify: `app/main.py`
- Modify: `app/services/chat_service.py`
- Modify: `app/storage/session_service.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/api/chat.py`
- Create: `tests/test_intent_api.py`

- [ ] **Step 1: 写入失败 API 测试**

```python
def test_chat_intent_keeps_existing_agent_path(tmp_path):
    classifier = FakeIntentClassifier.for_result("你好", "chat", None, 1.0)
    client = make_client_with_intent_classifier(
        tmp_path,
        fake_llm=FakeLLM([LLMResponse.message("你好，我是助手。")]),
        intent_classifier=classifier,
    )

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "你好"})

    assert response.status_code == 200
    assert response.json()["message"] == "你好，我是助手。"
    assert response.json()["intent_decision"]["intent"] == "chat"


def test_data_operation_returns_stable_placeholder(tmp_path):
    classifier = FakeIntentClassifier.for_result("查今天产量", "data_operation", "read", 1.0)
    client = make_client_with_intent_classifier(tmp_path, fake_llm=FakeLLM([]), intent_classifier=classifier)

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "查今天产量"})

    assert response.status_code == 200
    assert response.json()["intent_decision"]["data_action"] == "read"
    assert "数据查询意图" in response.json()["message"]
```

再补充 `nl2sql` 占位响应、低置信度降级为真实 ChatAgent、分类记录写入 SQLite、按 provider 查询指标和缺失 Jev Key 返回 503 的测试。

- [ ] **Step 2: 运行失败 API 测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_api.py -q
```

Expected: FAIL，因为 API 尚未暴露 `intent_decision`。

- [ ] **Step 3: 实现持久化与路由**

在 `SessionService._initialize()` 增加 `intent_classifications` 表：

```text
id INTEGER PRIMARY KEY AUTOINCREMENT
user_id TEXT NOT NULL
session_id TEXT NOT NULL
provider TEXT NOT NULL
model TEXT NOT NULL
intent TEXT NOT NULL
data_action TEXT NULL
confidence REAL NOT NULL
latency_ms REAL NOT NULL
input_tokens INTEGER NOT NULL
output_tokens INTEGER NOT NULL
total_tokens INTEGER NOT NULL
fallback_reason TEXT NULL
created_at TEXT NOT NULL
```

新增 `record_intent_classification()` 与 `get_intent_metrics(user_id, provider)`；后者计算请求数、各意图数量、平均延迟、P50/P95 延迟、总输入和输出 Token。

`ChatResponse` 新增必填字段：

```python
intent_decision: IntentDecision
```

`ChatService.chat()` 必须先路由，再创建或确认 Session，然后写入分类记录。`CHAT` 分支保留现有 `ContextManager → ChatAgent` 逻辑。另两类分支写入原始用户消息和以下固定助手消息：

```text
DATA_OPERATION：已识别为数据查询意图，当前仅完成意图路由测试。
NL2SQL：已识别为 NL2SQL 意图，当前仅完成意图路由测试。
```

在 `app/api/chat.py` 增加：

```text
GET /api/v1/intent-metrics?user_id=<required>&provider=<optional>
```

配置错误或 Jev 服务不可用时，将 `IntentClassificationError` 映射为 503，不写入消息和分类记录。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_api.py tests/test_chat_api.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/main.py app/services/chat_service.py app/storage/session_service.py app/schemas/chat.py app/api/chat.py tests/test_intent_api.py
git commit -m "feat: route chat requests by intent"
```

### Task 6：加入可复现量化评测

**Files:**

- Create: `data/intent_eval_cases.json`
- Create: `app/evaluation/__init__.py`
- Create: `app/evaluation/intent_benchmark.py`
- Create: `tests/test_intent_benchmark.py`

- [ ] **Step 1: 写入评测样本和失败测试**

`data/intent_eval_cases.json` 至少放入 15 条不含真实用户数据的标注样本，三类意图各至少 5 条。例如：

```json
[
  {"id": "chat-1", "message": "解释一下什么是 ToolRegistry", "expected_intent": "chat"},
  {"id": "data-1", "message": "查询今天各产线的产量", "expected_intent": "data_operation"},
  {"id": "sql-1", "message": "生成查询本月订单金额的 SQL", "expected_intent": "nl2sql"}
]
```

```python
from app.evaluation.intent_benchmark import summarize_results


def test_benchmark_calculates_accuracy_and_latency_percentiles():
    report = summarize_results(
        [
            {"expected": "chat", "actual": "chat", "latency_ms": 10, "input_tokens": 1, "output_tokens": 0},
            {"expected": "nl2sql", "actual": "chat", "latency_ms": 30, "input_tokens": 2, "output_tokens": 0},
        ]
    )

    assert report["accuracy"] == 0.5
    assert report["avg_latency_ms"] == 20.0
    assert report["p95_latency_ms"] == 30.0
    assert report["total_input_tokens"] == 3
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_benchmark.py -q
```

Expected: FAIL，因为评测模块尚不存在。

- [ ] **Step 3: 实现评测命令**

`app.evaluation.intent_benchmark` 读取标注样本，顺序调用由 `IntentClassifierFactory` 创建的分类器，输出 JSON 报告。报告必须包含：`provider`、`model`、`sample_count`、`accuracy`、`avg_latency_ms`、`p50_latency_ms`、`p95_latency_ms`、`total_input_tokens`、`total_output_tokens` 和每条样本的预测结果。

PowerShell 执行方式：

```powershell
$env:INTENT_PROVIDER = "fake"
.\.venv\Scripts\python.exe -m app.evaluation.intent_benchmark
```

切换 `INTENT_PROVIDER` 为 `jev` 或 `llm` 后使用同一批样本重复执行，才能获得可比较结果。报告不得包含 API Key 或完整模型原始响应。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_benchmark.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add data/intent_eval_cases.json app/evaluation tests/test_intent_benchmark.py
git commit -m "test: add intent classifier benchmark"
```

### Task 7：全量验证、README 与 GitHub 同步

**Files:**

- Modify: `README.md`
- Verify: 全部源码与测试

- [ ] **Step 1: 执行全量测试**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected: 全部通过。

- [ ] **Step 2: 执行静态编译检查**

```powershell
.\.venv\Scripts\python.exe -m compileall app tests
```

Expected: exit code 0。

- [ ] **Step 3: 人工验证三种配置**

```powershell
$env:INTENT_PROVIDER = "fake"
.\.venv\Scripts\python.exe -m app.evaluation.intent_benchmark

$env:INTENT_PROVIDER = "llm"
.\.venv\Scripts\python.exe -m app.evaluation.intent_benchmark

$env:INTENT_PROVIDER = "jev"
$env:TYPESAFE_API_KEY = "<your-key>"
.\.venv\Scripts\python.exe -m app.evaluation.intent_benchmark
```

Fake 与普通 LLM 必须完成；Jev 仅在真实 API Key 已配置时执行。检查输出中有分类结果、延迟、Token 汇总，且没有泄露密钥。

- [ ] **Step 4: 更新 README**

只追加已验证完成的意图识别能力、`INTENT_PROVIDER` 切换方式、`TYPESAFE_API_KEY` 配置方式和评测命令。不要把 RAG、真实数据库、SQL 生成或 CRUD 执行写入 README。

- [ ] **Step 5: 提交并推送**

```powershell
git add README.md
git commit -m "docs: document intent routing"
git push
```

## 7. 验收标准

- `INTENT_PROVIDER=fake|jev|llm` 可创建对应分类器；未知值明确失败。
- Fake、Jev、普通 LLM 都输出同一 `IntentDecision` 类型。
- Jev 通过官方在线 API 调用，密钥仅从 `TYPESAFE_API_KEY` 读取。
- 普通 LLM 分类不调用任何工具，且其 JSON 输出经过 Pydantic 校验。
- 低置信度和非法分类结果只降级到 `CHAT`；配置或网络不可用返回 HTTP 503。
- `CHAT` 保持当前 Agentic Chat 与 `search_schema` 工具行为。
- `DATA_OPERATION` 和 `NL2SQL` 返回稳定占位响应，不访问业务数据库。
- 会话数据库记录每次分类的 provider、模型、意图、延迟和 Token。
- 评测命令可用同一标注集比较 Fake、Jev、普通 LLM 的准确率、延迟和 Token。
- 全量测试与 `compileall` 通过，README 仅描述已经完成的功能，且所有提交已推送至 GitHub。

## 8. 交给实施 Agent 的提示词

```text
请在 D:\project\python\myagent 实现 docs\v4_intent-recognition.md 中的 V4 意图识别计划。

必须遵守：
1. 先阅读现有代码，沿用当前 FastAPI、Pydantic、LLMAdapter、SessionService 风格；不得删除或回滚已有改动、已有注释。
2. 新增独立 app/intent 模块，定义统一 IntentClassifier 协议；Fake、TypeSafe Jev 官方在线 API、普通 LLM 三种实现必须可以通过 INTENT_PROVIDER 切换，并输出同一 IntentDecision。
3. Jev 必须使用官方 typesafe-sdk，密钥只读 TYPESAFE_API_KEY；模型名使用代码常量 typesafe/jev-1.13；不得写入或输出任何 API Key。
4. 保持三个一级意图：chat、data_operation、nl2sql。data_operation 保留 read/create/update/delete/unknown 二级字段，但当前只返回测试占位响应，绝不接入真实业务数据库、SQL 生成或写操作。
5. 普通 LLM 分类器复用现有 LLMAdapter，不使用工具，JSON 输出必须经过 json.loads 和 Pydantic 校验。
6. 低置信度或非法分类结果降级为 chat；配置错误或外部服务不可用必须返回 503，不要伪造成功结果。
7. 以测试先行方式逐 Task 执行。单元测试必须 mock Jev 客户端，不能真的访问网络。最后运行 .\.venv\Scripts\python.exe -m pytest -q 和 .\.venv\Scripts\python.exe -m compileall app tests。
8. 完成后更新 README.md，只写实际已完成和验证过的 V4 能力；不要写 RAG、真实数据库、SQL 执行或 CRUD 已完成。
9. 使用 PowerShell。每个逻辑 Task 单独提交，最终 git push 到已配置的 origin/master；最终报告提交哈希、测试结果、Jev 配置方式和任何未能验证的外部 API 情况。
```
