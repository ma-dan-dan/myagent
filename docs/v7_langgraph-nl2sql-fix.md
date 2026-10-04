# V7 LangGraph NL2SQL Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 V7 LangGraph NL2SQL 的字段白名单绕过、会话上下文丢失、Reflection 反馈丢失、数据库不可用状态错误和配置开关未生效问题。

**Architecture:** 保留现有 `SchemaLinking → GenSQL → ValidateSQL → Execute → Reflection → Output` 图，不重做 LangGraph 架构。新增一个受控的 ContextPrepare 节点把现有 `ContextManager` 接入图；SQLValidator 负责最终表/列白名单，SQLExecutor 负责只读执行，Reflection 反馈必须回流到下一轮 GenSQL。ChatService 使用统一收尾逻辑记录 SQL/Reflection usage，并在主回答完成后执行已有摘要和 MEMORY.md 维护。

**Tech Stack:** Python 3、FastAPI、Pydantic、LangGraph、SQLGlot、SQLite Read-only Executor、现有 LiteLLM LLMAdapter、pytest。

---

## 本次只修复的问题

1. 拒绝 `SELECT *`、`table.*` 绕过字段白名单；允许 `COUNT(*)` 等聚合内部 wildcard，但不能返回未授权列。
2. NL2SQL 图接入现有 `ContextManager`，保留 user_id + session_id 的多轮上下文、摘要、MEMORY.md 和 Token 预算。
3. Reflection 的 `reason` 必须进入下一次 GenSQL Prompt。
4. 业务数据库连接失败、数据库文件不存在、数据库未配置时必须稳定返回 HTTP 503，不返回 200 伪装成功。
5. 明确 `NL2SQL_DATABASE_URL` 的兼容格式；支持文件路径和 `sqlite:///...`，拒绝未知数据库 scheme。
6. `NL2SQL_ENABLED` 必须真正控制功能；关闭时返回明确 503。
7. 增加默认装配的真实临时 SQLite 业务库测试，不能只测注入 Fake Graph。

## 不在本次范围

- 不实现 PostgreSQL/MySQL 驱动。
- 不增加写 SQL、DDL、Workflow、MCP、Skill、SSE、子 Agent 或模型路由。
- 不修改 V6 DDL/SampleValue 召回算法。
- 不修改现有 `LLMAdapter.complete(messages, tools)` 契约。
- 不把聊天会话 SQLite 作为业务数据库。
- 不调用真实模型、真实 Embedding API 或真实业务数据库。
- 不删除已有注释，不回滚无关改动，不执行 `git reset`、`git checkout` 或删除操作。

## 文件边界

| 文件 | 修复职责 |
| --- | --- |
| `app/config.py` | 完成 NL2SQL enabled、数据库路径/URL 和运行限制的集中读取。 |
| `app/nl2sql/models.py` | 保存 ContextManager 产物、usage、maintenance 和图状态。 |
| `app/nl2sql/prompt.py` | 让 GenSQL Prompt 接收历史上下文和 Reflection reason。 |
| `app/nl2sql/validator.py` | 拒绝投影 wildcard，保留安全聚合 wildcard，继续执行 AST/表列白名单校验。 |
| `app/nl2sql/executor.py` | 解析 SQLite 配置、区分数据库不可用和查询失败、保持只读保护。 |
| `app/nl2sql/nodes.py` | 增加 ContextPrepare 节点，传递上下文和 Reflection reason。 |
| `app/nl2sql/graph.py` | 调整节点顺序、State 和条件边，保持循环上限。 |
| `app/services/chat_service.py` | NL2SQL 使用 ContextManager 和统一 usage/摘要/Memory 收尾。 |
| `app/main.py` | 装配 config.enabled、ContextManager 和修复后的 Executor。 |
| `app/api/chat.py` | 映射 NL2SQL 配置/数据库不可用和最终执行错误。 |
| `app/schemas/chat.py` | 仅在必要时补可选的 NL2SQL 状态字段，保持兼容。 |
| `README.md` | 明确 URL/路径格式、启用开关和 V7 实际行为。 |
| `tests/test_nl2sql_validator.py` | wildcard 和白名单回归。 |
| `tests/test_nl2sql_graph.py` | ContextPrepare、Reflection 回传和上限回归。 |
| `tests/test_nl2sql_executor.py` | URL/路径解析和数据库不可用回归。 |
| `tests/test_chat_api.py` | 多轮上下文、usage、禁用和默认 wiring 回归。 |

---

## Task 1：修复投影字段白名单

**Files:**

- Modify: `app/nl2sql/validator.py`
- Modify: `tests/test_nl2sql_validator.py`

- [ ] **Step 1: 写失败测试**

新增以下测试：

```python
def test_projection_wildcard_is_rejected_by_column_allowlist():
    result = SQLValidator().validate(
        "SELECT * FROM production_output",
        {"production_output"},
        {"production_output": {"output_quantity"}},
    )
    assert result.ok is False


def test_qualified_projection_wildcard_is_rejected():
    result = SQLValidator().validate(
        "SELECT production_output.* FROM production_output",
        {"production_output"},
        {"production_output": {"output_quantity"}},
    )
    assert result.ok is False


def test_count_wildcard_is_allowed_without_returning_all_columns():
    result = SQLValidator().validate(
        "SELECT COUNT(*) FROM production_output",
        {"production_output"},
        {"production_output": {"output_quantity"}},
    )
    assert result.ok is True
```

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_validator.py -k "wildcard" -q
```

当前实现会错误地通过前两个 wildcard 测试。

- [ ] **Step 3: 最小实现**

遍历 AST 时区分两类 wildcard：

1. `SELECT *` 和 `SELECT table.*` 属于投影 wildcard，直接返回“必须明确选择授权字段”。
2. `COUNT(*)`、`COUNT(DISTINCT ...)` 等函数内部 wildcard 不代表返回所有列，可以保留。
3. 继续校验函数中的真实列引用、表别名和子查询表名。
4. 不通过字符串前缀判断 SQL 类型，继续以 SQLGlot AST 为准。

- [ ] **Step 4: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_validator.py -q
git add app/nl2sql/validator.py tests/test_nl2sql_validator.py
git commit -m "fix: enforce nl2sql column allowlist"
```

---

## Task 2：修复数据库配置和不可用错误

**Files:**

- Modify: `app/config.py`
- Modify: `app/nl2sql/executor.py`
- Modify: `app/main.py`
- Modify: `app/api/chat.py`
- Modify: `tests/test_config.py`
- Modify: `tests/test_nl2sql_executor.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 写失败测试**

覆盖以下约定：

```python
def test_sqlite_path_configuration_is_preserved(monkeypatch):
    monkeypatch.setenv("NL2SQL_DATABASE_URL", "D:/data/business.sqlite3")
    config = get_nl2sql_runtime_config()
    assert config.database_url == "D:/data/business.sqlite3"


def test_sqlite_url_is_normalized_to_a_file_path(tmp_path):
    url = f"sqlite:///{tmp_path / 'business.sqlite3'}"
    executor = ReadOnlySQLiteExecutor.from_database_url(url, 100, 30, 5)
    assert executor.database_path.name == "business.sqlite3"


def test_missing_database_file_raises_unavailable(tmp_path):
    executor = ReadOnlySQLiteExecutor(tmp_path / "missing.sqlite3", 100, 30, 5)
    with pytest.raises(SQLExecutorUnavailable):
        executor.execute("SELECT 1", {})
```

API 测试还要验证配置了不存在的业务库时返回 503，而不是 200。

- [ ] **Step 2: 最小实现**

1. 保留现有 `NL2SQL_DATABASE_URL` 环境变量名以兼容已有配置，但在 RuntimeConfig 中增加明确的 `database_path` 或等价规范化字段。
2. 支持两种输入：普通 Windows/POSIX 文件路径，以及 `sqlite:///...` URL。
3. 对非 SQLite scheme 返回配置错误，不把 URL 当作文件名。
4. 将 SQLite `connect()` 单独包在连接异常处理内：文件不存在、权限错误、打不开数据库都抛 `SQLExecutorUnavailable`。
5. 已建立连接后的 SQL 失败继续抛 `SQLExecutionError`，由 Graph 进入有限 Reflection；最终无法修复时返回稳定 503 或明确的执行失败错误，不能返回成功响应。
6. 保持只读连接、authorizer、查询进度限制、最大行数/列数和参数绑定。

- [ ] **Step 3: 让 `NL2SQL_ENABLED` 生效**

在 `app/config.py` 集中定义 `NL2SQL_ENABLED_ENV = "NL2SQL_ENABLED"`，默认值为 `true`，通过已有配置函数解析 `true/false/1/0/yes/no`。`main.py` 根据该值装配 `UnavailableNL2SQLGraphService` 或等价不可用实现；NL2SQL 请求返回明确 503“NL2SQL 功能未启用”。不得在业务模块直接读取环境变量。

- [ ] **Step 4: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_config.py tests/test_nl2sql_executor.py tests/test_chat_api.py -k "nl2sql or database or enabled" -q
git add app/config.py app/nl2sql/executor.py app/main.py app/api/chat.py tests/test_config.py tests/test_nl2sql_executor.py tests/test_chat_api.py
git commit -m "fix: harden nl2sql database availability"
```

---

## Task 3：接入 ContextManager 和多轮会话

**Files:**

- Modify: `app/nl2sql/models.py`
- Modify: `app/nl2sql/prompt.py`
- Modify: `app/nl2sql/nodes.py`
- Modify: `app/nl2sql/graph.py`
- Modify: `app/services/chat_service.py`
- Modify: `app/main.py`
- Modify: `tests/test_nl2sql_graph.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 写失败测试**

测试同一 `user_id + session_id` 连续两轮 NL2SQL：

1. 第二轮的 GenSQL Prompt 包含上一轮已持久化的摘要/近期消息。
2. 当前用户问题仍然完整存在，不能被历史窗口截断。
3. `ContextManager.prepare()` 返回的 `estimated_context_tokens` 和 `context_window` 进入 NL2SQL usage。
4. 第一轮/第二轮触发摘要时，摘要 usage 和 memory usage 写入 SQLite。
5. 主回答完成后才写 MEMORY.md；失败或 413 不写成功回答对应的 MemoryEntry。

```python
import json


def test_nl2sql_uses_same_session_context_and_usage(tmp_path):
    client, fake_llm = build_nl2sql_test_client(tmp_path, recent_message_limit=2)
    first = client.post("/api/v1/chat", json={"user_id": "u1", "message": "查询产量"})
    session_id = first.json()["session_id"]
    second = client.post(
        "/api/v1/chat",
        json={"user_id": "u1", "session_id": session_id, "message": "那按日期呢"},
    )
    assert first.status_code == 200
    assert second.status_code == 200
    assert "查询产量" in json.dumps(fake_llm.calls[-1][0], ensure_ascii=False)
```

测试代码必须在测试文件中定义完整 Fake 依赖，不引用未定义的外部 helper。

- [ ] **Step 2: 最小实现 State 扩展**

给 `NL2SQLState` 增加结构化字段：

```python
context_messages: list[dict[str, str]] = Field(default_factory=list)
estimated_context_tokens: int = 0
context_window: int = 0
maintenance: CompactionResult = Field(default_factory=CompactionResult)
```

不要把 `ContextManager`、LLM 或数据库连接对象放进 Graph State。

- [ ] **Step 3: 增加 ContextPrepareNode**

将图调整为：

```text
START
  → schema_linking
  → context_prepare
  → gen_sql
  → validate_sql
  → execute_sql
  → reflection
  → gen_sql 或 output
```

`context_prepare` 只在 SchemaLinking 状态为 `ok` 时调用：

```python
prepared = context_manager.prepare(
    state.user_id,
    state.session_id,
    state.user_message,
    extra_context=state.schema_context,
)
```

节点把 `prepared.messages`、估算 Token、窗口和 `prepared.maintenance` 的 Pydantic 数据写入 State。GenSQL/Reflection Prompt 必须使用这些上下文数据，但仍然只输出结构化 JSON。

- [ ] **Step 4: 统一 ChatService NL2SQL 收尾**

`_nl2sql_response()` 成功路径必须执行：

1. 追加 user/assistant 原始消息。
2. 记录 `chat` usage，并写入 estimated context tokens/context window。
3. 如果 `maintenance.compacted`，记录 summary/memory usage。
4. 主回答完成后调用 `long_term_memory.upsert()`。
5. 将 Graph 的 GenSQL/Reflection usage 汇总到当前 turn。

不能复用普通 ChatAgent 的 `ToolEvent`；NL2SQL 图没有 Tool Loop。

- [ ] **Step 5: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_graph.py tests/test_chat_api.py -k "context or session or usage or memory or nl2sql" -q
git add app/nl2sql/models.py app/nl2sql/prompt.py app/nl2sql/nodes.py app/nl2sql/graph.py app/services/chat_service.py app/main.py tests/test_nl2sql_graph.py tests/test_chat_api.py
git commit -m "fix: preserve nl2sql session context"
```

---

## Task 4：把 Reflection 原因反馈到下一轮 GenSQL

**Files:**

- Modify: `app/nl2sql/prompt.py`
- Modify: `app/nl2sql/nodes.py`
- Modify: `tests/test_nl2sql_graph.py`

- [ ] **Step 1: 写失败测试**

Fake LLM 第一次生成缺少日期过滤的 SQL，Reflection 返回：

```json
{"decision":"regenerate","reason":"必须增加 output_date 的时间范围过滤"}
```

断言下一次 GenSQL Prompt 包含这句 Reflection reason，并且第二次生成的 SQL 可以通过执行。

- [ ] **Step 2: 最小实现**

扩展 `build_gensql_messages`，增加 `reflection_reason: str | None` 参数，并在用户消息中加入长度受限的 Reflection 原因；当该值为空时写入“无”。

`gen_sql_node` 将 `current.reflection.reason` 传入下一轮 Prompt。只保留短原因并限制长度，不能把完整执行结果或内部异常堆进 Prompt。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_graph.py -k "reflection" -q
git add app/nl2sql/prompt.py app/nl2sql/nodes.py tests/test_nl2sql_graph.py
git commit -m "fix: feed reflection reason into sql generation"
```

---

## Task 5：失败状态、响应和完整默认 wiring 回归

**Files:**

- Modify: `app/nl2sql/graph.py`
- Modify: `app/services/chat_service.py`
- Modify: `app/api/chat.py`
- Modify: `app/schemas/chat.py`
- Modify: `tests/test_chat_api.py`
- Modify: `README.md`

- [ ] **Step 1: 写失败测试：默认装配真实临时业务库**

测试使用：

- `tmp_path / "business.sqlite3"` 创建合成表和合成行。
- `NL2SQL_DATABASE_URL` 传入普通路径或 `sqlite:///...` URL。
- Fake SchemaLinkingService 返回与业务表一致的候选。
- Fake LLM 返回 GenSQL JSON 和 Reflection JSON。
- `create_app()` 不注入 Graph，只注入必要 Fake SchemaLinkingService/IntentClassifier/LLM。

断言：

1. API 返回 200。
2. `nl2sql_status == "ok"`。
3. SQL 被自动限制到最大行数。
4. QueryResult 包含受限列/行。
5. Fake ChatAgent 没有被调用。

- [ ] **Step 2: 写失败测试：最终失败状态**

覆盖：

- Reflection 达到上限：返回明确失败，不继续调用 LLM/Executor。
- 业务数据库连接失败：503。
- SQL 不安全：422，且不调用 Executor。
- `NL2SQL_ENABLED=false`：503，且不调用 SchemaLinking/LLM/Executor。
- 两个用户相同 session_id：上下文和 Graph State 不互相泄露。

- [ ] **Step 3: 最小实现**

1. Graph 失败状态保留 `status`、`final_message` 和安全的 `error_code`，不把内部异常文本直接返回。
2. ChatService 对基础设施错误抛出领域异常，API 稳定映射 503；安全校验失败映射 422；上下文预算失败映射 413。
3. NL2SQL 响应字段保持可选，普通 chat 和 V6 RAG 响应不受影响。
4. 失败请求可以持久化用户问题和安全的助手错误提示，但不得写入完整数据库异常、连接字符串或敏感结果。

- [ ] **Step 4: 更新 README**

说明：

- `NL2SQL_DATABASE_URL` 支持的 SQLite 文件路径/URL 格式。
- 默认未配置业务数据库时返回 503。
- `NL2SQL_ENABLED=false` 时返回 503。
- SQL 只读、表列白名单、最大行数、超时、Reflection 次数限制。
- V7 测试使用 Fake LLM/Fake Executor/临时 SQLite，不代表真实业务库在线验证。

- [ ] **Step 5: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
git add app tests README.md docs/v7_langgraph-nl2sql-fix.md requirements.txt
git commit -m "fix: harden langgraph nl2sql workflow"
git status --short
git log -1 --oneline
```

## 验收标准

- `SELECT *` 和 `table.*` 不能绕过字段白名单；`COUNT(*)` 等聚合 wildcard 仍可安全使用。
- NL2SQL 使用 ContextManager 的摘要、近期消息、MEMORY.md 和 Token 预算。
- Reflection reason 会进入下一轮 GenSQL Prompt。
- SQL 生成/Reflection usage、估算上下文 Token 和真实 usage 分开记录。
- 数据库未配置、文件不存在、连接失败都返回 503；安全校验失败返回 422。
- `NL2SQL_ENABLED` 开关真实生效。
- 文件路径和 `sqlite:///...` 配置不会被当成错误的文件名。
- Graph 继续由 LangGraph 条件边控制，最多 3 次 SQL 生成、2 次 Reflection，不无限循环。
- 默认 `create_app()` wiring 有成功和失败回归测试。
- 不修改 chat/V6 RAG 的既有行为，不使用真实模型、Embedding 或业务数据库。

---

## 交给另一个 Agent 的执行提示词

请在 `D:\project\python\myagent` 按 `docs\v7_langgraph-nl2sql-fix.md` 实际修改代码，不要只给建议。先完整阅读该文档、`Agent.md`、`docs\v7_langgraph-nl2sql-plan.md`、当前 V7 实现和所有 NL2SQL/Chat/RAG 测试，然后严格按 Task 1 到 Task 5 执行。

必须遵守：

1. 先写失败测试并确认失败，再进行最小实现；每个 Task 完成后运行对应测试。
2. 不删除已有注释，不回滚已有改动，不执行 `git reset`、`git checkout` 或删除无关文件。
3. 保留 LangGraph 图结构，不改成普通 while 循环；保留 `SchemaLinking → GenSQL → ValidateSQL → Execute → Reflection → Output` 语义。
4. 修复 `SELECT *`/`table.*` 字段白名单绕过，但允许 `COUNT(*)` 等不返回全部列的聚合 wildcard。
5. 把现有 ContextManager 接入 NL2SQL 图，保持 user_id + session_id 多轮上下文、摘要、MEMORY.md、Token 预算和 usage 记录。
6. Reflection reason 必须传给下一次 GenSQL；Reflection/GenSQL 调用继续使用现有 `LLMAdapter.complete(messages, tools=[])`。
7. 数据库只允许独立只读 Executor，绝不能复用聊天会话 SQLite；数据库未配置、文件不存在或连接失败必须返回明确 503。
8. `NL2SQL_DATABASE_URL` 要支持普通 SQLite 文件路径和 `sqlite:///...`，并对未知 scheme 返回配置错误。
9. `NL2SQL_ENABLED=false` 必须真正禁用 NL2SQL 并返回 503。
10. 所有模型、Embedding 和数据库测试使用 Fake/mock/临时合成 SQLite，不调用真实模型、真实 Embedding 或真实业务数据库。
11. 使用 PowerShell 运行：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
```

12. 修改完成后提交全部改动，不要只提交新增文件：

```powershell
git add app tests README.md docs/v7_langgraph-nl2sql-fix.md requirements.txt
git commit -m "fix: harden langgraph nl2sql workflow"
git status --short
git log -1 --oneline
```

最终报告必须包含：修改文件、每个 review 问题的修复方式、Graph 节点和循环上限、上下文/usage/Memory 数据流、数据库配置格式、完整测试结果、提交哈希，并明确说明没有调用真实模型、Embedding 或业务数据库。
