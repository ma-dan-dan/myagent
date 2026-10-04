# V7 LangGraph NL2SQL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 V6 SchemaLinking 基础上增加受 LangGraph 编排的只读 NL2SQL 链路：SchemaLinking → GenSQL → ValidateSQL → Execute → Reflection → Output。

**Architecture:** `nl2sql` 意图进入独立 `NL2SQLGraphService`，图 State 保存结构化的 Schema、SQL 草稿、校验结果、执行结果和 Reflection 次数。GenSQL 与 Reflection 复用现有 `LLMAdapter.complete(messages, tools=[])`，SQL 执行由注入的只读 `SQLExecutor` 负责，不注册为通用 Agent Tool；校验失败或执行结果不符合问题时，图通过条件边回到 GenSQL，但受最大尝试次数限制。

**Tech Stack:** Python 3、FastAPI、Pydantic、LangGraph、SQLGlot、LiteLLM Adapter、SQLite/可替换只读 SQL Executor、pytest。

---

## 目标链路

```text
Web / API Request
        ↓
ChatService → IntentRouter
                    ├─ chat → ContextManager → ChatAgent
                    ├─ data_operation/read → V6 SchemaLinkingService → ChatAgent(tools=[])
                    └─ nl2sql → NL2SQLGraphService
                                   ├─ SchemaLinkingNode
                                   ├─ GenSQLNode
                                   ├─ ValidateSQLNode
                                   ├─ ExecuteSQLNode
                                   ├─ ReflectionNode
                                   └─ OutputNode
```

V7 只新增 `nl2sql` 链路，不把 `data_operation/read` 改成 SQL 查询，也不删除 V6 RAG。`data_operation/read` 继续只回答 Schema 相关问题；用户明确要求生成或展示 SQL 时才进入 NL2SQL 图。

## 范围与安全边界

1. SQL 只允许单条只读 `SELECT`/`WITH`。
2. 禁止 `INSERT`、`UPDATE`、`DELETE`、`DROP`、`ALTER`、`TRUNCATE`、多语句和未授权表。
3. SchemaLinking 返回的候选表/列是 SQL 白名单；模型不能引用候选之外的表和字段。
4. Execute 使用只读 Executor；不能使用聊天会话 SQLite 作为业务数据库。
5. SQL 执行需要显式配置业务数据库；缺少配置时返回明确 503，不能伪造结果。
6. 执行结果限制最大行数和列数，Reflection 只接收摘要或少量受控结果，不接收完整原始数据。
7. 图循环由 State 中的计数器限制，模型不能决定是否无限重试。
8. 不实现写数据库、MCP、Skill、SSE、子 Agent、向量重排、第二套模型路由或新的前端页面。
9. 所有测试使用 FakeLLM、FakeSQLExecutor、FakeEmbedding 或 mock，不调用真实模型、真实数据库或真实网络。
10. 不删除已有注释，不回滚已有改动，不执行 `git reset`、`git checkout` 或删除无关文件。

## 文件边界

| 文件 | 职责 |
| --- | --- |
| `app/config.py` | NL2SQL 开关、SQL 方言、数据库连接配置、行数/超时/尝试次数限制；唯一读取环境变量的位置。 |
| `app/nl2sql/models.py` | `NL2SQLState`、`SQLDraft`、`SQLValidationResult`、`QueryResult`、`ReflectionDecision` 等 Pydantic 模型。 |
| `app/nl2sql/prompt.py` | GenSQL 和 Reflection Prompt 构造，只接收结构化输入。 |
| `app/nl2sql/validator.py` | SQLGlot 解析、只读校验、表/列白名单和 LIMIT 校验。 |
| `app/nl2sql/executor.py` | `SQLExecutor` Protocol、只读 SQLite Executor、Fake Executor 所需接口和异常。 |
| `app/nl2sql/nodes.py` | LangGraph 节点函数；每个节点只读 State 并返回 State 更新。 |
| `app/nl2sql/graph.py` | 构建 StateGraph、条件路由、循环上限和 `NL2SQLGraphService`。 |
| `app/services/chat_service.py` | `nl2sql` 分支、会话消息和 usage 收尾；保持既有 chat/RAG 分支。 |
| `app/schemas/chat.py` | 在不破坏既有字段的前提下增加可选 NL2SQL 响应字段。 |
| `app/main.py` | 装配 NL2SQLGraphService 和可替换 Executor。 |
| `app/api/chat.py` | 仅增加 NL2SQL 领域异常的稳定 HTTP 映射。 |
| `requirements.txt` | 增加 LangGraph、SQLGlot 和 SQLite 以外所需的最小依赖。 |
| `README.md` | 更新 V7 图、配置和已验证能力，不能声称接入未验证的真实数据库。 |
| `tests/test_nl2sql_models.py` | Pydantic State/结果模型测试。 |
| `tests/test_nl2sql_validator.py` | SQL 安全校验测试。 |
| `tests/test_nl2sql_graph.py` | LangGraph 节点、循环、Reflection 和 Fake Executor 测试。 |
| `tests/test_chat_api.py` | API 主链路、503/413、session 隔离和响应兼容性测试。 |

不要修改 `LLMAdapter.complete(messages, tools)` 契约、V6 `SchemaLinkingService` 的召回职责、普通 `ChatAgent` Tool Loop 或意图分类模型接口。

---

## Task 1：集中配置和定义 NL2SQL 数据模型

**Files:**

- Modify: `app/config.py`
- Create: `app/nl2sql/__init__.py`
- Create: `app/nl2sql/models.py`
- Create: `tests/test_nl2sql_models.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: 写失败测试**

覆盖以下模型：

```python
def test_sql_draft_requires_structured_status_and_safe_fields():
    draft = SQLDraft(
        status="ok",
        sql="SELECT output_quantity FROM production_output LIMIT 100",
        tables=["production_output"],
        parameters={},
        explanation="查询产量",
    )
    assert draft.status == "ok"
    assert draft.sql is not None


def test_clarify_draft_does_not_require_sql():
    draft = SQLDraft(
        status="clarify",
        sql=None,
        tables=[],
        parameters={},
        explanation="请补充时间范围",
    )
    assert draft.sql is None


def test_graph_state_has_bounded_attempt_fields():
    state = NL2SQLState.model_validate({
        "user_id": "u1",
        "session_id": "s1",
        "user_message": "查询产量",
        "attempt": 0,
        "reflection_count": 0,
        "max_attempts": 3,
        "max_reflections": 2,
    })
    assert state.attempt == 0
```

模型约束：

- `SQLDraft.status` 只能是 `ok`、`clarify`、`reject`。
- `SQLDraft.sql` 在 `ok` 时必填，在其他状态时必须为空或被拒绝。
- `parameters` 必须是 JSON 可序列化的标量映射，不允许嵌入 SQL。
- `ReflectionDecision.decision` 只能是 `pass`、`regenerate`、`clarify`、`reject`。
- `QueryResult` 保存 `columns`、受限 `rows`、`row_count`、`truncated`、`error`，不保存连接对象。
- `NL2SQLState` 的可变字段使用 `default_factory`，不能共享列表或字典。

- [ ] **Step 2: 运行失败测试**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_models.py tests/test_config.py -q
```

预期新模型导入失败。

- [ ] **Step 3: 最小实现**

在 `app/config.py` 增加集中配置，不在其他文件读取环境变量：

```python
NL2SQL_ENABLED = True
NL2SQL_DIALECT = "sqlite"
NL2SQL_DATABASE_URL = None
NL2SQL_MAX_SQL_ATTEMPTS = 3
NL2SQL_MAX_REFLECTIONS = 2
NL2SQL_MAX_ROWS = 100
NL2SQL_MAX_COLUMNS = 30
NL2SQL_QUERY_TIMEOUT_SECONDS = 5
NL2SQL_MAX_SQL_LENGTH = 12000
```

如果项目现有 config 已有配置对象，应沿用现有读取函数并为上述常量提供 RuntimeConfig，而不是在 `app/nl2sql` 中调用 `os.getenv`。

- [ ] **Step 4: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_models.py tests/test_config.py -q
git add app/config.py app/nl2sql/__init__.py app/nl2sql/models.py tests/test_nl2sql_models.py tests/test_config.py
git commit -m "feat: add nl2sql graph state models"
```

---

## Task 2：实现 GenSQL/Reflection Prompt 和现有 LLMAdapter 适配

**Files:**

- Create: `app/nl2sql/prompt.py`
- Modify: `app/nl2sql/models.py`
- Create: `tests/test_nl2sql_prompt.py`

- [ ] **Step 1: 写失败测试**

Prompt 测试必须验证：

1. 包含用户原问题。
2. 包含 SchemaLinking 返回的表、字段、类型和受限样例。
3. 明确只允许 `SELECT/WITH`。
4. 明确不能虚构表/字段。
5. Reflection Prompt 包含 SQL、校验错误或执行摘要。
6. 不包含 API Key、Cookie、Bearer、密码。

- [ ] **Step 2: 最小实现**

定义两个纯函数：

```python
def build_gensql_messages(
    user_message: str,
    schema_context: str,
    dialect: str,
    previous_error: str | None,
) -> list[dict[str, str]]:
    system = "只允许生成单条只读 SELECT/WITH SQL；只能使用给定 Schema；信息不足时返回 clarify；只输出 JSON。"
    user = f"方言：{dialect}\nSchema：{schema_context}\n问题：{user_message}\n错误：{previous_error or '无'}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def build_reflection_messages(
    user_message: str,
    schema_context: str,
    sql: str,
    validation_error: str | None,
    execution_summary: str | None,
    dialect: str,
) -> list[dict[str, str]]:
    system = "你是只读 SQL Reflection 器，只能返回 pass、regenerate、clarify 或 reject 的 JSON。"
    user = (
        f"方言：{dialect}\nSchema：{schema_context}\n问题：{user_message}\n"
        f"SQL：{sql}\n校验错误：{validation_error or '无'}\n"
        f"执行摘要：{execution_summary or '无'}"
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
```

模型只输出 JSON，不允许输出 Markdown 代码围栏。Prompt 不要求模型输出思维过程；`explanation` 只能是面向用户的简短说明。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_prompt.py -q
git add app/nl2sql/prompt.py app/nl2sql/models.py tests/test_nl2sql_prompt.py
git commit -m "feat: add structured nl2sql prompts"
```

---

## Task 3：实现 SQLGlot 安全校验器

**Files:**

- Create: `app/nl2sql/validator.py`
- Modify: `requirements.txt`
- Create: `tests/test_nl2sql_validator.py`

- [ ] **Step 1: 写失败测试**

必须覆盖：

```python
def test_select_from_allowed_table_passes():
    result = SQLValidator(dialect="sqlite").validate(
        "SELECT output_quantity FROM production_output LIMIT 10",
        allowed_tables={"production_output"},
        allowed_columns={"production_output": {"output_quantity"}},
    )
    assert result.ok is True


@pytest.mark.parametrize("sql", [
    "DELETE FROM production_output",
    "UPDATE production_output SET output_quantity = 0",
    "DROP TABLE production_output",
    "SELECT 1; DELETE FROM production_output",
])
def test_write_or_multi_statement_sql_is_rejected(sql):
    result = SQLValidator().validate(sql, {"production_output"}, {})
    assert result.ok is False


def test_unknown_table_is_rejected():
    result = SQLValidator().validate(
        "SELECT password FROM users",
        allowed_tables={"production_output"},
        allowed_columns={},
    )
    assert result.ok is False


def test_limit_is_added_or_required():
    result = SQLValidator(max_rows=100).validate(
        "SELECT output_quantity FROM production_output",
        allowed_tables={"production_output"},
        allowed_columns={"production_output": {"output_quantity"}},
    )
    assert result.ok is True
    assert "LIMIT" in result.normalized_sql.upper()
```

- [ ] **Step 2: 最小实现**

实现 `SQLValidator.validate(sql, allowed_tables, allowed_columns) -> SQLValidationResult`。

规则：

1. 空 SQL、超长 SQL、解析失败直接失败。
2. SQLGlot 只接受一个 AST；存在多个 statement 直接失败。
3. 根节点只能是 `Select` 或等价 `WITH` 查询。
4. 遍历 AST 提取真实表名和列名，与白名单比较；别名必须正确解析。
5. 拒绝 `SELECT *` 时要使用明确配置；默认可以允许，但仍必须限制列数和返回列数。
6. 没有 LIMIT 时添加最大 LIMIT；已有 LIMIT 超过上限时压到上限。
7. 返回 `normalized_sql`、引用表、引用列和可展示错误；不能返回密码、Key 或完整数据库异常。

安装依赖时使用项目虚拟环境；必须锁定兼容版本范围，不引入 LangChain SQL Agent。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_validator.py -q
git add app/nl2sql/validator.py requirements.txt tests/test_nl2sql_validator.py
git commit -m "feat: validate read only generated sql"
```

---

## Task 4：实现只读 SQL Executor

**Files:**

- Create: `app/nl2sql/executor.py`
- Modify: `app/config.py`
- Create: `tests/test_nl2sql_executor.py`

- [ ] **Step 1: 写失败测试**

定义注入协议并测试 Fake Executor：

```python
class FakeSQLExecutor:
    def __init__(self, result: QueryResult | Exception):
        self.result = result
        self.calls = []

    def execute(self, sql: str, parameters: dict[str, object]) -> QueryResult:
        self.calls.append((sql, parameters))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result
```

SQLite Executor 测试使用临时测试数据库和合成表，覆盖：

- SELECT 能读取数据
- DML/DDL 被拒绝
- 参数原样作为参数绑定，不拼接进 SQL
- 超过最大行数时 `truncated=True`
- 执行失败转换为稳定 `SQLExecutionError`
- 未配置业务数据库时抛出明确 `SQLExecutorUnavailable`

- [ ] **Step 2: 最小实现**

定义 `SQLExecutor` Protocol，提供 `execute(sql: str, parameters: dict[str, object]) -> QueryResult`；同时实现 `ReadOnlySQLiteExecutor`，构造函数接收专用数据库路径、最大行数、最大列数和超时配置。

Executor 只接受已经通过 SQLValidator 的 SQL；仍需做第二层只读保护。使用只读连接、查询超时/进度限制、最大行数和最大列数。不得读取 `DEFAULT_DB_PATH`，业务数据库路径必须来自专用 NL2SQL 配置或构造函数注入。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_executor.py -q
git add app/nl2sql/executor.py app/config.py tests/test_nl2sql_executor.py
git commit -m "feat: add injectable readonly sql executor"
```

---

## Task 5：实现 LangGraph State 和节点

**Files:**

- Modify: `requirements.txt`
- Create: `app/nl2sql/nodes.py`
- Create: `app/nl2sql/graph.py`
- Create: `tests/test_nl2sql_graph.py`

- [ ] **Step 1: 写失败测试：成功路径**

使用 Fake LLM 和 Fake Executor 验证：

1. SchemaLinking 返回一个明确候选。
2. GenSQL 返回合法 SQL JSON。
3. ValidateSQL 通过。
4. Execute 返回合成结果。
5. Reflection 返回 `pass`。
6. Output 返回 SQL、列名、行数和结果摘要。
7. 主模型调用次数和 `tools=[]` 都可断言。

```python
def test_graph_success_path_calls_executor_and_outputs_result():
    graph = build_test_graph(
        llm=FakeLLM([LLMResponse.message(
            '{"status":"ok","sql":"SELECT output_quantity FROM production_output",'
            '"tables":["production_output"],"parameters":{},'
            '"explanation":"查询产量"}'
        ), LLMResponse.message(
            '{"decision":"pass","reason":"结果符合问题"}'
        )]),
        executor=FakeSQLExecutor(QueryResult(
            columns=["output_quantity"], rows=[[10]], row_count=1,
            truncated=False,
        )),
    )
    result = graph.invoke(make_initial_state("查询产量"))

    assert result["status"] == "ok"
    assert result["query_result"].rows == [[10]]
```

- [ ] **Step 2: 写失败测试：错误回路**

覆盖：

- SQL 语法/白名单校验失败后 Reflection 要求重新生成。
- Executor 报错后 Reflection 要求重新生成。
- Reflection `clarify` 直接 Output，不再次执行。
- 达到 `max_attempts` 后稳定失败，不再调用 GenSQL 或 Execute。
- 禁止写操作时不调用 Executor。
- 空/歧义 Schema 时不调用 GenSQL、Executor 或 Reflection。

- [ ] **Step 3: 最小实现 State**

使用 Pydantic State 或 TypedDict，至少包含：

```python
from pydantic import BaseModel, Field


class NL2SQLState(BaseModel):
    user_id: str
    session_id: str
    user_message: str
    schema_candidates: list[SchemaCandidate] = Field(default_factory=list)
    schema_context: str = ""
    sql_draft: SQLDraft | None = None
    validation: SQLValidationResult | None = None
    query_result: QueryResult | None = None
    reflection: ReflectionDecision | None = None
    validation_error: str | None = None
    execution_error: str | None = None
    attempt: int = 0
    reflection_count: int = 0
    max_attempts: int = 3
    max_reflections: int = 2
    status: str = "running"
    final_message: str | None = None
```

实际实现中若使用 `TypedDict`，必须通过节点边界的 Pydantic 模型验证输入/输出，不允许任意字典扩散。

- [ ] **Step 4: 最小实现节点职责**

实现以下函数，函数名固定，便于测试：`schema_linking_node`、`gen_sql_node`、`validate_sql_node`、`execute_sql_node`、`reflection_node`、`output_node`。每个函数接收 State 和一个显式依赖参数，返回经过 Pydantic 校验的 State 更新字典；不得在节点内部读取环境变量或创建全局模型/数据库连接。

节点规则：

- `schema_linking_node` 只调用现有 SchemaLinkingService，不调用聊天 LLM。
- `gen_sql_node` 每次只调用一次 LLM，传入 `tools=[]`，并递增 `attempt`。
- `validate_sql_node` 只做确定性校验。
- `execute_sql_node` 只调用 SQLExecutor，不让 LLM 直接触发数据库。
- `reflection_node` 每次最多一次 LLM，传入 `tools=[]`，并递增 `reflection_count`。
- `output_node` 只格式化最终用户结果，不泄漏内部错误、连接信息或模型隐式推理。

- [ ] **Step 5: 构建 StateGraph 和条件边**

使用 LangGraph 的 `StateGraph`、`START`、`END` 和 conditional edges，逻辑必须等价于：

```python
START -> schema_linking
schema_linking -> gen_sql | output
gen_sql -> validate_sql
validate_sql -> execute_sql | reflection | output
execute_sql -> reflection
reflection -> gen_sql | output
```

路由函数必须同时检查 `attempt` 和 `reflection_count`，任何上限达到都只能进入 `output`。不要使用无限 `while` 包裹图，不要使用隐式 Agent Executor。

- [ ] **Step 6: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest tests/test_nl2sql_graph.py -q
git add requirements.txt app/nl2sql/nodes.py app/nl2sql/graph.py tests/test_nl2sql_graph.py
git commit -m "feat: add langgraph nl2sql workflow"
```

---

## Task 6：接入 ChatService、API 和会话 usage

**Files:**

- Modify: `app/services/chat_service.py`
- Modify: `app/main.py`
- Modify: `app/api/chat.py`
- Modify: `app/schemas/chat.py`
- Modify: `tests/test_chat_api.py`

- [ ] **Step 1: 写失败测试**

覆盖：

1. Fake IntentClassifier 路由到 `nl2sql`。
2. ChatService 调用 NL2SQLGraphService，不调用普通 ChatAgent Tool Loop。
3. API 返回向后兼容的 ChatResponse，并包含可选 `nl2sql_status`、`sql`、`query_result` 字段。
4. 空/歧义 Schema 返回普通追问。
5. 未配置业务数据库返回 503。
6. SQL 安全校验失败返回稳定错误，不执行 SQL。
7. user_id + session_id 隔离保持不变。
8. 图的 LLM usage 汇总到当前 turn，真实 usage 与 estimated context tokens 继续分开。

- [ ] **Step 2: 最小实现**

1. `ChatService` 增加可注入的 `nl2sql_graph_service`。
2. `IntentName.NL2SQL` 分支调用图服务；原有 `chat`、V6 `data_operation/read` 和占位分支保持行为不变。
3. NL2SQL 成功或稳定失败都持久化 user/assistant 消息，但不把完整 SQL 执行结果写入 `MEMORY.md`。
4. 图服务异常转换为明确的领域异常；API 映射：
   - 配置/数据库/LLM 不可用：503
   - 上下文超预算：413
   - SQL 不安全或输入不可执行：422
5. `main.py` 默认不使用会话数据库作为业务数据库；仅当专用 NL2SQL 数据库配置存在时装配真实 SQLite Executor，否则装配 Unavailable Executor。
6. 允许测试通过构造函数注入 FakeGraph/FakeExecutor，不访问真实网络。

- [ ] **Step 3: 验证并提交**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_chat_api.py tests/test_nl2sql_graph.py -q
git add app/services/chat_service.py app/main.py app/api/chat.py app/schemas/chat.py tests/test_chat_api.py
git commit -m "feat: route nl2sql through langgraph"
```

---

## Task 7：README、架构图和完整回归

**Files:**

- Modify: `README.md`
- Modify: `docs/images/myagent-architecture.svg` 或对应架构图源文件
- Modify: `requirements.txt`
- Create/Modify: `tests/test_nl2sql_*`

- [ ] **Step 1: 更新 README**

只描述已经实现和测试过的能力：

- `nl2sql` 由 LangGraph 编排。
- 链路为 SchemaLinking → GenSQL → ValidateSQL → Execute → Reflection → Output。
- SQL 只能只读执行，且有表白名单、行数限制、超时和最大反思次数。
- `LLMAdapter` 仍是 GenSQL/Reflection 的模型接口。
- 默认没有业务数据库时，NL2SQL 返回明确不可用，不使用会话 SQLite。
- 测试使用 Fake LLM/Fake Executor，不进行真实模型和真实数据库验证。

不要写“支持生产数据库”“支持任意 SQL”“支持增删改”“在线验证通过”等未经验证的描述。

- [ ] **Step 2: 完整验证**

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
```

预期：pytest 全部通过，compileall 成功，环境变量读取仍只在 `app/config.py`，没有 diff 空白错误。

- [ ] **Step 3: 提交全部改动**

```powershell
git add app tests README.md requirements.txt docs/v7_langgraph-nl2sql-plan.md docs/images
git commit -m "feat: add langgraph nl2sql workflow"
git status --short
git log -1 --oneline
```

最终报告必须包含修改文件、Graph 节点和循环规则、SQL 安全约束、Executor 配置、完整测试结果、提交哈希，以及明确说明没有调用真实模型或真实数据库。

## 验收标准

- `nl2sql` 不再返回旧占位响应，而是进入 LangGraph。
- Graph 至少包含 SchemaLinking、GenSQL、ValidateSQL、Execute、Reflection、Output 节点。
- GenSQL 和 Reflection 复用 `LLMAdapter.complete(messages, tools=[])`。
- SQL 只能是单条只读查询，表/列必须在召回 Schema 白名单内。
- 空/歧义 Schema 不生成、不执行 SQL。
- 执行失败可以有限次 Reflection 重生成，达到上限后稳定结束。
- 禁止写 SQL、未知表、未知列、多语句时绝不调用 Executor。
- 缺少业务数据库配置时返回明确 503，不使用会话 SQLite，不伪造查询结果。
- 执行结果受最大行数、列数和超时限制。
- 原有 chat、V6 data_operation/read、会话隔离、Token/usage 和长期记忆行为不被破坏。
- 测试不访问真实模型、真实数据库或真实网络。

---

## 交给另一个 Agent 的执行提示词

请在 `D:\project\python\myagent` 按 `docs\v7_langgraph-nl2sql-plan.md` 实际实施 V7，不要只给建议。先完整阅读该计划、`Agent.md`、当前 V6 代码、V6 Fix 代码和相关测试，再严格按 Task 1 到 Task 7 执行。

必须遵守：

1. 使用 LangGraph 实现 `SchemaLinking → GenSQL → ValidateSQL → Execute → Reflection → Output` 图；不要把流程写成普通 while 循环，也不要把 SQL 执行注册为通用 Agent Tool。
2. 保留现有 `LLMAdapter.complete(messages, tools)` 契约；GenSQL 和 Reflection 必须调用 `tools=[]`，不得重新实现一套模型 Adapter。
3. 保留 V6 `data_operation/read` 的 DDL/SampleValue RAG 链路；只有 `nl2sql` 意图进入新图。
4. SQL 只允许单条只读 `SELECT/WITH`，使用 SQLGlot 做 AST 校验；未知表、未知列、写操作和多语句不得执行。
5. Execute 使用独立只读 SQLExecutor，绝不能使用聊天会话 SQLite；未配置业务数据库时返回明确 503，不能伪造结果。
6. Reflection 和 GenSQL 都必须有最大调用/循环次数；达到限制后稳定返回，不能无限重试。
7. 所有新配置集中在 `app/config.py`；业务模块不得直接调用 `os.getenv()` 或 `os.environ`。
8. 先写失败测试并确认失败，再写最小实现。所有模型、Embedding 和 SQL 执行测试使用 Fake 或 mock，不访问真实 API、真实数据库或真实网络。
9. 不删除已有注释，不回滚任何已有改动，不执行 `git reset`、`git checkout` 或删除无关文件。
10. 运行以下 PowerShell 命令：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests
rg -n "os\.getenv|os\.environ" app
git diff --check
git status --short
```

11. 完成后提交全部改动，不要只提交新增文件：

```powershell
git add app tests README.md requirements.txt docs/v7_langgraph-nl2sql-plan.md docs/images
git commit -m "feat: add langgraph nl2sql workflow"
git status --short
git log -1 --oneline
```

最终报告必须包含：修改文件、Graph 节点、Reflection 循环上限、SQL 校验规则、Executor 配置、API 行为、完整测试结果、提交哈希，并明确说明没有调用真实模型、真实 Embedding 或真实数据库。
