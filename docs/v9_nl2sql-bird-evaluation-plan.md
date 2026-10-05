# V9 BIRD Dev LangGraph NL2SQL Evaluation Plan

本文档用于指导另一位工程师实现 V9 的 NL2SQL 端到端评测。

## 目标

基于 BIRD Dev 的题目、数据库和官方 Gold SQL，评估 SchemaLinking → GenSQL → Execute → Reflection → Output 工作流。

## 工作流边界

~~~text
SchemaLinking
    ↓
GenSQL
    ↓
Execute
    ↓
Reflection
    ├── pass / clarify / reject → Output
    └── regenerate → GenSQL
~~~
只允许上述五个逻辑节点。Reflection 可以有限次数返回 GenSQL，但不能无限循环，也不能新增其他业务节点。

本版本不包含：

- 独立 ContextPrepare 节点；
- SQLGlot AST 解析或 SQL AST 白名单校验；
- 独立 ValidateSQL 工作流节点；
- Tool Loop、MCP、Skill、子 Agent、SSE、向量重排或模型路由；
- Gold SQL、Gold Tables、Gold Result 注入 Prompt、Schema Context、Reflection 输入或 LangGraph State；
- ChatAgent、API、Session、Intent、ToolRegistry 的无关修改。

不使用 AST 不代表允许任意 SQL 写入真实数据库。评测必须继续使用现有只读 Executor 的安全边界：只读连接、写操作拒绝、超时、最大行数/列数和稳定错误处理。不能为了跑分切换到读写连接。

## BIRD Dev、Gold SQL 与 Gold Tables

BIRD Dev 是题目记录和数据库文件的组合。每道题通常包括：

~~~json
{
  "question": "查询订单金额大于 100 的用户",
  "db_id": "shop",
  "SQL": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.amount > 100;"
}
~~~

db_id 用于定位数据库，例如：

~~~text
<bird-root>/dev_databases/shop/shop.sqlite
~~~

多个题目可以共享同一个数据库。数据库文件提供全部候选表和真实数据，Gold SQL 提供这道题的标准查询逻辑。

Gold SQL 只由评测器用来生成标签和参考结果。下面的 SQL 引用了 users 和 orders：

~~~sql
SELECT u.name
FROM users AS u
JOIN orders AS o ON u.id = o.user_id
WHERE o.amount > 100;
~~~

因此 Gold Tables 为：

~~~text
{users, orders}
~~~

Gold Tables 不得写入 RAG 文本，也不得作为 GenSQL 输入。若数据集未来直接提供 Gold Tables，应优先使用官方标签。若 V8 已实现 BIRD Loader 和 Gold Table 提取器，V9 必须复用，不能再创建第二套解析规则。

## 两条执行支路

### Gold SQL 参考支路

~~~text
Gold SQL + 当前题目的 SQLite 数据库
        ↓
只读 Gold Executor
        ↓
Gold Result
~~~

Gold SQL 不调用 LangGraph，也不进入 Reflection。它只由评测器旁路执行一次并缓存。Gold Result 不得进入预测流程。

### LangGraph 预测支路

~~~text
question + db_id
        ↓
SchemaLinking
        ↓
GenSQL
        ↓
Execute
        ↓
Reflection
        ├── regenerate → GenSQL
        └── pass / clarify / reject → Output
        ↓
NL2SQLState
~~~

评测器从最终 State 读取：

~~~text
sql_draft.sql
query_result
execution_error
reflection
attempt
reflection_count
final_message
status
~~~

最后比较：

~~~text
Predicted Result ↔ Gold Result
~~~

## 数据库和 RAG 索引

Gold SQL 和预测 SQL 必须在同一个 db_id 对应的数据库快照上执行：

~~~text
db_id
  ├── RAG DDL/SampleValue 索引
  ├── Gold SQL Executor
  └── LangGraph Execute Executor
~~~

不得出现 RAG 使用数据库 A、Gold SQL 使用数据库 B、预测 SQL 使用数据库 C 的情况。

沿用 V8 的 DDL + SampleValue 索引流程：

~~~text
SQLite database
  ├── DDL records
  └── safe SampleValue records
          ↓
      Qwen Embedding
          ↓
      LanceDB
        ├── ddl_documents
        └── sample_value_documents
~~~

必须使用：

~~~text
namespace = db_id
table_id = db_id + "." + table_name
~~~

问题查询时只能搜索当前题目的 namespace。Gold SQL、Gold Tables、Gold Result 不能参与索引文本或检索输入。

## 端到端评测流程

### 1. 加载题目

加载：

~~~text
question_id
question
db_id
gold_sql
database_path
~~~

支持 --limit 小样本运行和 --question-id 指定题目。缺少数据库文件或 Gold SQL 时必须明确失败，不能静默跳过。报告记录数据集版本或输入文件哈希。

### 2. 准备 Gold Result

使用只读 Gold Executor 执行：

~~~text
gold_sql → gold_result
~~~

要求：

- Gold SQL 执行失败的样本标记为 gold_execution_error，不计入正常准确率分母；
- 记录 Gold Result 是否截断；
- Gold 结果被截断时，不得伪装成完整正确标签；
- Gold Result 不发送给任何模型。

### 3. 运行 LangGraph

只把自然语言问题作为用户输入：

~~~text
question → LangGraph.invoke(user_id, session_id, question)
~~~

每个 db_id 使用绑定了该数据库的 Graph Service 或等价 Executor 工厂，不能复用错误数据库的全局 Executor。

### 4. 读取预测结果

优先读取最终 State 的 query_result，并记录：

~~~text
predicted_sql
status
execution_error
reflection_decision
attempt
reflection_count
~~~

如果最终没有可用 query_result，该题不能算执行正确，必须记录失败类别。

### 5. 比较结果

使用固定、可测试的 ResultComparator 比较：

~~~text
predicted_result ↔ gold_result
~~~

不能把 SQL 字符串完全一致作为唯一标准。

## ResultComparator 规则

### 主要判定

默认采用执行结果等价：

- 两边执行都成功；
- 结果列数一致；
- 结果行数一致；
- 行值相同；
- 对没有显式排序要求的结果默认忽略行顺序；
- 保留重复行次数，不能简单转换为 Set；
- NULL、空字符串和字符串 "NULL" 必须区分；
- 浮点值允许配置绝对误差，默认 1e-2，并在报告中记录；
- 结果截断状态不一致时不能判定为正确。

### 列名和列顺序

默认比较结果值和列数量，不要求模型生成的列别名与 Gold SQL 完全一致，以允许：

~~~sql
SELECT amount FROM orders;
~~~

和：

~~~sql
SELECT o.amount AS order_amount FROM orders AS o;
~~~

在结果列值相同的情况下被视为功能等价。

如果项目后续需要严格列语义，可以增加 strict_columns 配置，但必须把宽松模式和严格模式分开报告。

### 行顺序

默认模式参考 DA 的结果比较方式，允许忽略行顺序；这适用于没有明确 ORDER BY 的集合型查询。

对于明确要求排序的题目，可以在数据模型中保留 ordered_result 标记。V9 不使用 AST 解析来推断该标记；优先使用数据集元信息或显式 CLI 配置。无法确定时使用统一的宽松模式，并在报告中注明。

### SQL 文本指标

可以额外记录经过空白归一化后的 exact_sql_match，但它只能作为辅助诊断指标：

- 文本不同、结果相同：Execution Accuracy 可以为 true；
- 文本相同、结果不同：Execution Accuracy 必须为 false；
- 不使用 AST 进行 SQL 规范化相等判断。

## Reflection 评测

Reflection 的价值是评估失败 SQL 或错误结果后，是否能够有限次修复。

### 必须区分的指标

~~~text
initial_execution_success
initial_execution_accuracy
final_execution_accuracy
reflection_recovery
~~~

定义：

~~~text
initial_execution_success:
    第一次 GenSQL 生成的 SQL 是否执行成功

initial_execution_accuracy:
    第一次执行结果是否等价于 Gold Result

final_execution_accuracy:
    经过 Reflection 允许的重试后，最终结果是否等价

reflection_recovery:
    第一次结果错误，但 Reflection 后最终正确
~~~

例如：

~~~text
总题数：100
第一次执行正确：60
Reflection 后最终正确：72
Reflection 修复成功：12
~~~

应报告：

~~~text
initial_execution_accuracy = 60%
final_execution_accuracy = 72%
reflection_recovery = 12 / 40 = 30%
~~~

### Reflection 决策约束

允许的决策：

~~~text
pass
regenerate
clarify
reject
~~~

路由规则：

~~~text
pass       → Output
clarify    → Output
reject     → Output
regenerate → GenSQL
~~~

必须设置 max_attempts 和 max_reflections。达到任一上限后不得继续调用模型，进入 Output 并记录 max_attempts_reached 或 max_reflections_reached。

如果 Reflection 判断 SchemaLinking 选错表，当前 V9 不允许回到 SchemaLinking；它只能在已有 Schema Context 上请求 GenSQL 重新生成。该限制必须在报告中明确。

## 指标体系

### Schema Linking 指标

复用 V8：

~~~text
Recall@K
FullRecall@K
Precision@K
~~~

Gold Tables 只用于离线指标，不进入预测流程。

### NL2SQL 端到端指标

至少统计：

| 指标 | 含义 |
|---|---|
| gold_valid_rate | Gold SQL 能否在当前数据库成功执行 |
| initial_generation_valid_rate | 首次生成的 SQL 是否有有效结构并成功进入 Execute |
| initial_execution_success_rate | 首次 SQL 是否执行成功 |
| initial_execution_accuracy | 首次执行结果是否等价 |
| final_execution_accuracy | Reflection 后最终结果是否等价 |
| reflection_recovery_rate | Reflection 修复失败样本的比例 |
| clarify_rate | 最终要求用户澄清的比例 |
| reject_rate | 最终拒绝的比例 |
| max_attempt_rate | 达到生成或 Reflection 上限的比例 |
| average_latency_ms | 每题端到端耗时 |
| average_llm_calls | 每题模型调用次数 |
| average_attempts | 每题 GenSQL 次数 |
| average_reflections | 每题 Reflection 次数 |

### 关键分组分析

至少按以下维度分组：

- 数据库 db_id；
- Gold Tables 数量：单表、多表；
- DDL-only、SampleValue-only、Fusion/RRF；
- 首次成功与 Reflection 修复；
- empty/ambiguous SchemaLinking；
- 执行错误、结果错误、澄清、拒绝。

这样可以区分：

~~~text
RAG 没召回正确表
GenSQL 生成错误
Execute 执行失败
结果比较失败
Reflection 没有修复
~~~

## 报告格式

建议输出到不提交 Git 的 artifacts：

~~~text
artifacts/nl2sql-eval/
├── cases.jsonl
├── summary.json
└── summary.csv
~~~

### 单题明细

~~~json
{
  "question_id": "bird_001",
  "db_id": "shop",
  "question": "查询订单金额大于 100 的用户",
  "gold_tables": ["orders", "users"],
  "retrieved_tables": ["orders", "users", "payments"],
  "predicted_sql": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.amount > 100",
  "gold_sql": {
    "executed": true,
    "row_count": 2,
    "truncated": false
  },
  "prediction": {
    "initial_execution_success": true,
    "final_status": "ok",
    "row_count": 2,
    "truncated": false
  },
  "comparison": {
    "execution_accuracy": true,
    "exact_sql_match": false,
    "reason": "result_values_equal"
  },
  "attempts": 1,
  "reflection_count": 0,
  "latency_ms": 842
}
~~~

报告不得写入 API Key、认证 Token、Cookie、数据库密码或不必要的完整真实数据集结果。没有实际运行的 Benchmark 分数不得写入报告。

## 建议文件边界

V9 优先复用 V8 的 BIRD/RAG 评测基础设施，新增端到端评测模块：

~~~text
app/nl2sql_eval/
├── __init__.py
├── models.py              # 评测样本、Gold Result、Predicted Result、比较结果
├── reference_executor.py  # Gold SQL 只读旁路执行
├── result_comparator.py   # 结果等价比较
├── evaluator.py           # 运行 LangGraph 并汇总指标
└── report.py              # JSONL/JSON/CSV 报告

scripts/
└── evaluate_nl2sql.py     # PowerShell CLI

tests/
├── test_nl2sql_eval_comparator.py
├── test_nl2sql_eval_reference_executor.py
├── test_nl2sql_eval_workflow.py
└── test_nl2sql_eval_metrics.py
~~~

如果 V8 已实现 app/rag_eval/bird_loader.py、gold_tables.py、sqlite_catalog.py，V9 直接复用；不要再创建第二份 BIRD Loader 或第二份 Gold Table 解析逻辑。

## TDD 要求

### Comparator 测试

必须先写失败测试，再实现：

1. SQL 文本不同但结果完全相同，判定正确；
2. 行顺序不同但值和重复次数相同，宽松模式判定正确；
3. 重复行数量不同，判定错误；
4. 浮点值在 1e-2 内，判定正确；
5. 浮点值超过误差，判定错误；
6. NULL 与字符串 "NULL" 不相同；
7. 列数量不同，判定错误；
8. 一侧被截断，判定错误；
9. Gold 或预测执行失败，判定错误并返回稳定原因；
10. 空结果集在两侧都为空时可以判定正确。

### Reference Executor 测试

使用临时 SQLite：

- Gold SQL 能得到稳定结果；
- 数据库不存在时返回明确错误；
- 执行超时或错误可以分类；
- 不允许写操作；
- 不把 Gold Result 注入预测 State。

### Workflow 测试

使用 Fake LLM 或 Mock，不调用真实 API：

1. SchemaLinking 成功 → GenSQL → Execute → Reflection pass → Output；
2. 第一次 Execute 失败 → Reflection regenerate → GenSQL 第二次成功 → Output；
3. Reflection clarify → 不再 GenSQL，直接 Output；
4. Reflection reject → 不再 GenSQL，直接 Output；
5. 达到 max_attempts 或 max_reflections 后停止；
6. 最终 State 中能读取预测 SQL、预测结果和 Reflection 信息；
7. Gold SQL 只被 Reference Executor 调用，不出现在 Fake LLM 输入中。

### 回归命令

~~~powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m compileall -q app tests scripts
git diff --check
~~~

没有 BIRD 数据集或真实 Qwen Key 时，只能运行单元测试和临时 SQLite 集成测试，不得伪造 BIRD 全量结果。真实 Benchmark 运行必须在报告中说明数据集路径、样本数量、Embedding 配置来源和是否调用真实模型。

## CLI 设计

建议命令：

~~~powershell
.\\.venv\\Scripts\\python.exe scripts\\evaluate_nl2sql.py --dataset-root D:\\data\\bird --question-file D:\\data\\bird\\dev.json --database-root D:\\data\\bird\\dev_databases --index-path .\\artifacts\\nl2sql-eval\\lancedb --output-dir .\\artifacts\\nl2sql-eval\\reports --top-k 3 --max-attempts 3 --max-reflections 2 --limit 20
~~~

CLI 必须支持：

- --limit；
- --question-id；
- --top-k；
- --max-attempts；
- --max-reflections；
- --rebuild-index；
- 数据集、数据库、索引和报告路径；
- 单题失败后继续评测其余题目，并在报告中记录失败原因。

## README 更新要求

只写实际完成并验证过的能力：

- V9 支持基于 BIRD Dev 的 NL2SQL 端到端评测；
- Gold SQL 旁路执行生成 Gold Result；
- LangGraph 预测结果从 State 中读取；
- 使用 Execution Accuracy 判断功能等价；
- 统计首次执行、Reflection 修复和最终准确率；
- 支持 SQL 文本一致性作为辅助指标；
- 支持临时 SQLite/Fake LLM 的无网络测试。

不得写成：

- SQL 文本必须和 Gold SQL 一致；
- 已完成 BIRD 全量真实跑分，除非确实运行过；
- 已实现 SQLGlot AST 校验；
- 已经支持 Spider2 或其他未验证数据集；
- 已经调用真实模型，除非报告有实际运行记录。

## 验收标准

- [ ] Gold SQL 只在评测器旁路执行，不进入 Prompt 或预测 State；
- [ ] 预测 SQL 通过当前 LangGraph 工作流生成和执行；
- [ ] 最终结果从 LangGraph State 的 query_result 等字段读取；
- [ ] SQL 文本不同但执行结果等价的样本可以判定为正确；
- [ ] 结果比较处理行顺序、重复行、NULL、浮点误差、空结果和截断；
- [ ] Reflection 的 regenerate、pass、clarify、reject 分支有测试；
- [ ] Reflection 和 GenSQL 有调用次数上限；
- [ ] 报告区分首次准确率、最终准确率和 Reflection 修复率；
- [ ] RAG Schema Recall 与 NL2SQL Execution Accuracy 分开统计；
- [ ] 不新增或依赖 SQLGlot AST 校验节点；
- [ ] 不调用真实模型的测试全部通过；
- [ ] 没有 BIRD 数据集或 API Key 时不伪造真实 Benchmark 结果；
- [ ] README 只写实际完成和验证过的能力；
- [ ] 执行 Agent 提交全部 V9 改动。

## 交给另一位工程师的执行提示词

~~~text
请在 D:\\project\\python\\myagent 按 docs\\v9_nl2sql-bird-evaluation-plan.md 实现 V9：基于 BIRD Dev 的 LangGraph NL2SQL 端到端评测。

开始前完整阅读：
- docs\\v9_nl2sql-bird-evaluation-plan.md
- docs\\v8_rag-evaluation-plan.md
- app\\config.py
- app\\rag\\models.py
- app\\rag\\embedding.py
- app\\rag\\indexer.py
- app\\rag\\vector_store.py
- app\\rag\\retriever.py
- app\\rag\\service.py
- app\\nl2sql\\models.py
- app\\nl2sql\\executor.py
- app\\nl2sql\\graph.py
- app\\nl2sql\\nodes.py
- app\\nl2sql\\prompt.py
- tests\\test_nl2sql_models.py
- tests\\test_nl2sql_executor.py
- tests\\test_nl2sql_graph.py
- tests\\test_nl2sql_prompt.py
- D:\\project\\qiuzhao\\DA\\benchmark\\scripts\\schema_recall_bird.py
- D:\\project\\qiuzhao\\DA\\benchmark\\scripts\\evaluation.py

目标是评测当前固定工作流：
SchemaLinking → GenSQL → Execute → Reflection → Output

必须遵守：
1. Gold SQL 只能由评测器旁路执行，生成 Gold Result；绝不能把 Gold SQL、Gold Tables 或 Gold Result 注入 Prompt、Schema Context、Reflection 输入或 LangGraph State。
2. 预测 SQL 必须通过当前 LangGraph 生成和执行，最终从返回的 NL2SQLState/query_result 等字段读取预测结果。
3. 用执行结果等价作为主要正确性标准，不要求预测 SQL 文本与 Gold SQL 完全一致。Exact SQL Match 只能作为辅助指标。
4. 结果比较必须处理行顺序、重复行、NULL、浮点误差、空结果、列数量和截断状态；默认使用可配置的浮点绝对误差 1e-2，并在报告中记录。
5. Reflection 只允许 pass、regenerate、clarify、reject；regenerate 回到 GenSQL，其他决策进入 Output。必须有 max_attempts 和 max_reflections，禁止无限循环。
6. 本次不新增 SQLGlot AST 解析、不新增独立 ValidateSQL 节点、不改变当前五节点业务边界。保持现有只读 Executor、超时、行数/列数限制和数据库权限边界；不能为了跑分切换成读写数据库。
7. 复用 V8 的 BIRD Loader、Gold Table 提取和 RAG 索引能力；不要实现第二套不兼容的 Loader 或 Gold Table 解析逻辑。使用 db_id 作为 namespace，Gold SQL 和 Gold Tables 不得进入索引文本。
8. 按 TDD：先写失败测试并运行确认失败，再做最小实现。测试使用 Fake LLM、Mock、临时 SQLite，不调用真实 Qwen/OpenAI/DeepSeek/Jev 网络。
9. 新增职责分离的评测模块：Gold/reference executor、result comparator、workflow evaluator、metrics/report 和 PowerShell CLI。不要把 Benchmark 逻辑塞进 ChatService、API 或生产 NL2SQL 节点。
10. 至少覆盖：Gold SQL 旁路执行、预测 State 结果读取、文本不同但结果相同、行顺序差异、重复行、NULL、浮点误差、空结果、截断、Reflection 重试、clarify/reject、次数上限和 Gold 数据不泄漏。
11. 输出 cases.jsonl、summary.json、summary.csv，至少报告 Schema Recall、initial_execution_success_rate、initial_execution_accuracy、final_execution_accuracy、reflection_recovery_rate、clarify/reject/max_attempt 比例、耗时和模型调用次数。
12. BIRD 数据集、索引文件、真实结果报告和任何密钥不得提交 Git。没有 BIRD 数据集或 QWEN_API_KEY 时，只运行单元测试/临时 SQLite 测试，并明确说明没有进行真实 Benchmark 或在线 Embedding 验证，不得伪造分数。
13. 保留已有注释，不删除或回滚无关改动；使用 PowerShell；不要执行 git reset、checkout 或删除无关文件。

验证命令：
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m compileall -q app tests scripts
git diff --check

更新 README，只写实际完成和验证过的 V9 能力。

完成后提交全部 V9 改动，提交信息使用：test: add BIRD NL2SQL evaluation
最终报告必须包含：修改文件、测试完整结果、ResultComparator 规则、Gold SQL 旁路执行方式、预测 State 读取方式、首次/最终准确率、Reflection 修复率、是否运行真实 BIRD、是否调用真实 Embedding/LLM、提交哈希和 Git 推送结果。
~~~
