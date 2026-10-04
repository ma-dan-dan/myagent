# V4 意图识别修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 修复 V4 意图识别中“分类结果”和“实际路由结果”混用导致的指标归因错误，明确 Fake 的评测定位，并同步 README 架构图。

**架构：** `IntentClassifier` 只产生原始分类结果；`IntentRouter` 产生独立的路由结果。原始结果保留真实 provider、model、置信度、耗时和 Token，用于 Jev 与普通 LLM 的量化对比；路由结果仅决定本次请求进入 `chat`、`data_operation` 还是 `nl2sql` 分支。

**技术栈：** Python 3.12、FastAPI、Pydantic v2、SQLite、pytest。

---

## 1. 当前问题

### 1.1 低置信度导致模型归因丢失

当前 [app/intent/router.py](../app/intent/router.py) 在低置信度时修改了原始 `IntentDecision`：

```python
decision.model_copy(
    update={
        "intent": IntentName.CHAT,
        "data_action": None,
        "provider": "fallback",
        "fallback_reason": "low_confidence",
    }
)
```

这会造成一次真实由 Jev 或普通 LLM 发起的分类记录，写入 SQLite 时 provider 变成 `fallback`。因此：

```text
真实 Jev / LLM 分类
  ↓ 低置信度
IntentRouter 改写 provider 为 fallback
  ↓
intent_classifications.provider = fallback
  ↓
GET /api/v1/intent-metrics?provider=jev 或 provider=llm 漏计
```

这与 V4 的核心目标“比较 Jev 和普通 LLM 的延迟、Token 和分类效果”冲突。

### 1.2 Fake 不是准确率基线

`IntentClassifierFactory.from_env()` 在 `INTENT_PROVIDER=fake` 时创建默认 `FakeIntentClassifier()`。该对象对没有显式映射的消息全部返回 `chat`。

因此，对当前 15 条评测样本执行：

```powershell
$env:INTENT_PROVIDER = "fake"
.\.venv\Scripts\python.exe -m app.evaluation.intent_benchmark
```

会得到约 `33.3%` 的准确率。这不是 Fake 模型的真实性能，而是测试替身的默认行为，不能与 Jev、普通 LLM 的评测结果放在同一比较结论中。

### 1.3 README 架构图仍为 V3 链路

README 已写入 V4 的文字说明，但“架构概览”仍是：

```text
Web / API Request → ChatService → ContextManager → ChatAgent
```

它遗漏了 `IntentRouter`、三种分类器、`data_operation` 和 `nl2sql` 占位分支，读者无法从图中理解请求首先经过意图识别。

## 2. 最终设计

### 2.1 原始分类与路由结果分层

保留现有 `IntentDecision` 作为**原始分类结果**，新增 `IntentRouteResult`：

```python
class IntentRouteResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: IntentDecision
    routed_intent: IntentName
    fallback_reason: str | None = Field(default=None, max_length=300)
```

字段职责固定如下：

| 字段 | 含义 | 低置信度时是否改变 |
|---|---|---|
| `decision.intent` | 分类器原始预测 | 不改变 |
| `decision.provider` | 实际分类器来源，例如 `jev`、`llm`、`fake` | 不改变 |
| `decision.model` | 实际模型名称 | 不改变 |
| `decision.latency_ms` | 实际分类耗时 | 不改变 |
| `decision.usage` | 实际分类 Token | 不改变 |
| `routed_intent` | 后端本次实际进入的分支 | 低置信度时改为 `chat` |
| `fallback_reason` | 路由为何未采用原始预测 | 设置为 `low_confidence` 或 `invalid_result` |

例如普通 LLM 原始输出为 `nl2sql`、置信度 `0.20` 时，正确结果应是：

```json
{
  "decision": {
    "intent": "nl2sql",
    "confidence": 0.2,
    "provider": "llm",
    "model": "qwen-model",
    "latency_ms": 85.4
  },
  "routed_intent": "chat",
  "fallback_reason": "low_confidence"
}
```

### 2.2 非法结果的路由

当分类器无法产生可校验的 `IntentDecision` 时，不存在可信的原始 provider、模型、Token 或意图。此时 `IntentRouter` 返回：

```text
decision.provider = fallback
decision.model = intent-router
decision.intent = chat
routed_intent = chat
fallback_reason = invalid_result
```

这与低置信度不同：低置信度已经存在有效原始结果，必须保留来源；非法结果没有可保留的原始结果。

### 2.3 SQLite 指标语义

保留 `intent_classifications.intent` 作为原始分类意图，新增：

```text
routed_intent TEXT NOT NULL
```

新增记录的含义：

```text
intent          原始分类标签
routed_intent   实际执行分支
provider        真实分类器来源；仅非法结果为 fallback
fallback_reason 路由降级原因
```

对已有 SQLite 数据库执行兼容迁移：

```sql
ALTER TABLE intent_classifications ADD COLUMN routed_intent TEXT NOT NULL DEFAULT 'chat';
UPDATE intent_classifications
SET routed_intent = intent
WHERE routed_intent = 'chat' AND fallback_reason IS NULL;
```

迁移只用于保留旧数据可读性。旧版本已经被写成 `provider=fallback` 的低置信度记录无法恢复原始 provider；不应编造历史数据。

指标接口返回两组计数：

```json
{
  "classified_intent_counts": {"chat": 2, "nl2sql": 1},
  "routed_intent_counts": {"chat": 3}
}
```

这样可以分别观察“模型怎么判断”与“系统最终怎么处理”。

### 2.4 Fake 的正确定位

Fake 是测试替身，不是可与 Jev、普通 LLM 比较的真实模型。

评测脚本按 provider 分为两种模式：

| provider | 评测行为 | 是否可以与真实模型比较准确率 |
|---|---|---|
| `fake` | 为每条评测样本创建与 `expected_intent` 一致的固定结果，验证评测、序列化和汇总链路 | 不可以；报告标记 `benchmark_mode=fixture` |
| `jev` | 调用官方 Jev 在线 API | 可以 |
| `llm` | 调用现有普通 LLM 分类器 | 可以 |

Fake 评测报告必须显式包含：

```json
{
  "benchmark_mode": "fixture",
  "comparable_to_real_models": false
}
```

Jev 和 LLM 报告必须包含：

```json
{
  "benchmark_mode": "model",
  "comparable_to_real_models": true
}
```

## 3. 文件变更清单

| 文件 | 操作 | 责任 |
|---|---|---|
| `app/intent/models.py` | 修改 | 新增 `IntentRouteResult`。 |
| `app/intent/router.py` | 修改 | 返回 `IntentRouteResult`，不改写低置信度原始分类来源。 |
| `app/services/chat_service.py` | 修改 | 按 `routed_intent` 分支，响应同时返回原始分类和实际路由。 |
| `app/schemas/chat.py` | 修改 | 扩展 `ChatResponse`、`IntentMetrics`。 |
| `app/storage/session_service.py` | 修改 | 迁移并记录 `routed_intent`，聚合两组意图计数。 |
| `app/api/chat.py` | 修改 | 保持指标接口响应类型与新字段一致。 |
| `app/evaluation/intent_benchmark.py` | 修改 | 分离 fixture 与真实模型评测模式。 |
| `tests/test_intent_router.py` | 修改 | 验证低置信度保留 provider、模型、Token 和原始意图。 |
| `tests/test_intent_api.py` | 修改 | 验证持久化的原始意图、路由意图和 provider 指标。 |
| `tests/test_intent_benchmark.py` | 修改 | 验证 Fake 报告不是可比较模型基线。 |
| `README.md` | 修改 | 更新 V4 架构概览图。 |

## 4. 实施任务

### Task 1：先建立原始分类与路由分离的失败测试

**Files:**

- Modify: `tests/test_intent_router.py`
- Modify: `app/intent/models.py`
- Modify: `app/intent/router.py`

- [ ] **Step 1: 写入失败测试**

```python
def test_low_confidence_keeps_original_classifier_attribution():
    classifier = FakeIntentClassifier.for_result("帮我看看", "nl2sql", None, 0.30)

    result = IntentRouter(classifier, min_confidence=0.70).route("帮我看看")

    assert result.decision.intent is IntentName.NL2SQL
    assert result.decision.provider == "fake"
    assert result.decision.model == "fake-intent-v1"
    assert result.routed_intent is IntentName.CHAT
    assert result.fallback_reason == "low_confidence"
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_router.py -q
```

Expected: FAIL，因为当前 Router 返回的是被改写后的 `IntentDecision`。

- [ ] **Step 3: 实现 `IntentRouteResult` 和 Router**

`IntentRouter.route()` 的正常返回值、低置信度返回值和非法结果返回值都必须是 `IntentRouteResult`。

低置信度分支必须使用原始 `decision`，不能调用 `model_copy()` 改写 `provider`、`model`、`intent`、`latency_ms` 或 `usage`。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_router.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/intent/models.py app/intent/router.py tests/test_intent_router.py
git commit -m "fix: preserve intent classifier attribution"
```

### Task 2：修复 API 响应、持久化和指标统计

**Files:**

- Modify: `app/services/chat_service.py`
- Modify: `app/schemas/chat.py`
- Modify: `app/storage/session_service.py`
- Modify: `app/api/chat.py`
- Modify: `tests/test_intent_api.py`

- [ ] **Step 1: 写入失败测试**

```python
def test_low_confidence_is_counted_under_its_source_provider(tmp_path):
    classifier = FakeIntentClassifier.for_result("ambiguous", "nl2sql", None, 0.2)
    client = make_client(tmp_path, classifier, [LLMResponse.message("普通回答")])

    response = client.post("/api/v1/chat", json={"user_id": "alice", "message": "ambiguous"})
    metrics = client.get("/api/v1/intent-metrics", params={"user_id": "alice", "provider": "fake"})

    assert response.status_code == 200
    assert response.json()["intent_decision"]["intent"] == "nl2sql"
    assert response.json()["routed_intent"] == "chat"
    assert metrics.json()["request_count"] == 1
    assert metrics.json()["classified_intent_counts"] == {"nl2sql": 1}
    assert metrics.json()["routed_intent_counts"] == {"chat": 1}
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_api.py -q
```

Expected: FAIL，因为当前低置信度结果被写入 `provider=fallback`。

- [ ] **Step 3: 修改响应和数据库**

`ChatResponse` 保持：

```python
intent_decision: IntentDecision
routed_intent: IntentName
fallback_reason: str | None = None
```

`ChatService` 使用 `route_result.routed_intent` 决定进入 Agent 或占位分支；将 `route_result.decision` 与 `route_result.routed_intent` 一起传给 `SessionService.record_intent_classification()`。

给已有数据库执行第 2.3 节迁移。`get_intent_metrics()` 返回 `classified_intent_counts` 与 `routed_intent_counts`，不再返回含义不清的单一 `intent_counts`。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_api.py tests/test_chat_api.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/services/chat_service.py app/schemas/chat.py app/storage/session_service.py app/api/chat.py tests/test_intent_api.py
git commit -m "fix: separate classification and routing metrics"
```

### Task 3：明确 Fake 评测模式

**Files:**

- Modify: `app/evaluation/intent_benchmark.py`
- Modify: `tests/test_intent_benchmark.py`

- [ ] **Step 1: 写入失败测试**

```python
def test_fake_benchmark_is_marked_as_fixture_not_model_comparison():
    report = run_benchmark(FakeIntentClassifier(), load_cases())

    assert report["benchmark_mode"] == "fixture"
    assert report["comparable_to_real_models"] is False
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_benchmark.py -q
```

Expected: FAIL，因为当前报告没有区分 Fake 与真实模型。

- [ ] **Step 3: 实现模式分离**

当分类器是 `FakeIntentClassifier` 时，基准脚本为每一条 case 构造与 `expected_intent` 相同的 `IntentDecision`，仅用于验证评测数据读取、统计和 JSON 输出链路。报告加上：

```python
"benchmark_mode": "fixture",
"comparable_to_real_models": False,
```

Jev 和 LLM 的报告加上：

```python
"benchmark_mode": "model",
"comparable_to_real_models": True,
```

README 和评测输出必须明确：Fake 的 `accuracy=1.0` 是测试夹具预期，不是模型精度。

- [ ] **Step 4: 验证通过**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_intent_benchmark.py -q
```

Expected: PASS。

- [ ] **Step 5: 提交**

```powershell
git add app/evaluation/intent_benchmark.py tests/test_intent_benchmark.py
git commit -m "fix: distinguish fixture and model benchmarks"
```

### Task 4：更新 README 架构图和说明

**Files:**

- Modify: `README.md`

- [ ] **Step 1: 替换架构概览**

使用下图替换 README 现有的 V3 图：

```text
Web / API Request
        ↓
ChatService
        ↓
IntentRouter
  ├─ FakeIntentClassifier
  ├─ JevIntentClassifier
  └─ LLMIntentClassifier
        ↓
  ├─ chat
  │    ↓
  │  ContextManager → ChatAgent → ToolRegistry → SchemaSearchTool
  │                       ↓
  │                   LLMAdapter → LiteLLM Provider
  │
  ├─ data_operation → 当前测试占位响应
  └─ nl2sql         → 当前测试占位响应
```

- [ ] **Step 2: 修正 V4 说明**

将“低置信度或非法分类结果降级为普通聊天”改为：

```text
低置信度或非法分类结果会路由到普通聊天；有效低置信度分类仍保留其真实 provider、模型、耗时和 Token，用于指标统计。
```

把 Fake 的评测说明改为：

```text
Fake 用于测试与评测链路验证，不作为 Jev 或普通 LLM 的准确率比较基线。
```

- [ ] **Step 3: 提交**

```powershell
git add README.md
git commit -m "docs: clarify intent routing architecture"
```

### Task 5：全量验证与同步

- [ ] **Step 1: 执行测试和编译检查**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall app tests
```

Expected: 全部测试通过，两个命令 exit code 都为 0。

- [ ] **Step 2: 人工验证低置信度指标归因**

使用注入的 `FakeIntentClassifier.for_result(..., confidence=0.2)` 发起一条 API 请求。确认：

```text
GET /api/v1/intent-metrics?user_id=alice&provider=fake
request_count = 1
classified_intent_counts 中保留原始预测
routed_intent_counts 中为 chat
```

- [ ] **Step 3: 推送 GitHub**

```powershell
git push
```

## 5. 验收标准

- 低置信度 Jev 或 LLM 分类保留真实 provider、model、latency 和 Token。
- API 同时展示原始分类意图和实际路由意图。
- SQLite 同时保留原始意图和路由意图；指标按真实 provider 查询不漏计低置信度调用。
- 非法结果仍安全路由至 `chat`，但不伪造模型来源。
- Fake 评测报告明确为 fixture，不能作为真实模型准确率对比依据。
- README 架构图包含 IntentRouter、三个分类器和三条路由分支。
- `.venv` 下的 `pytest -q` 与 `compileall app tests` 均通过，修改已推送 GitHub。

## 6. 交给实施 Agent 的提示词

```text
请在 D:\project\python\myagent 按 docs\v4_intent-recognition-fix.md 实施 V4 修复。

重点目标：把“原始分类结果”与“实际路由结果”分开。低置信度时只能把 routed_intent 降级为 chat，不能把 Jev/LLM/Fake 的 provider、model、原始 intent、延迟或 Token 改成 fallback。SQLite 指标必须能按真实 provider 完整统计分类调用。

要求：
1. 先阅读现有 V4 代码和测试，不删除或回滚任何已有改动、注释或提交。
2. 采用 IntentRouteResult 或等价明确模型，保留 IntentDecision 表示原始分类。
3. 迁移现有 SQLite 的 intent_classifications，新增 routed_intent，不能破坏已有数据库启动。
4. Fake 只做测试夹具，不得把默认全 chat 的 33.3% 当作真实模型准确率。报告必须区分 fixture 与 model 模式。
5. 更新 README 架构图为 V4 实际调用链。
6. 先写失败测试，再做最小实现；使用 PowerShell。
7. 最后运行 .\.venv\Scripts\python.exe -m pytest -q 与 .\.venv\Scripts\python.exe -m compileall app tests，更新 README 后提交并 git push。
8. 最终报告提交哈希、测试结果、迁移兼容性和是否进行了真实 Jev 在线调用。不要因缺少 API Key 而伪造在线验证结果。
```
