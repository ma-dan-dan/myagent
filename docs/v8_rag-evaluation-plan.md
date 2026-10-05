# V8 Schema RAG Benchmark Evaluation Plan

> 本文档用于指导另一个 Agent 实现 V8 的 Schema RAG 离线评测。评测对象是当前项目的 DDL + SampleValue 双路召回和 RRF 融合，不改变现有 ChatAgent、API、Session、LLMAdapter、Intent、ToolRegistry 或 LangGraph NL2SQL 主链的默认行为。

## 目标

为当前 `app/rag` 建立一套基于 BIRD Dev 的可重复 Schema RAG 评测流程，回答以下问题：

1. DDL-only 能否召回 Gold Tables；
2. SampleValue-only 能否召回 Gold Tables；
3. DDL + SampleValue 的融合是否优于单路召回；
4. 当前 RRF 是否能够把正确表排到较高位置；
5. RAG 召回质量、上下文规模和耗时是否满足后续 NL2SQL 使用要求。

本版本只评测 Schema Linking/RAG，不把 GenSQL、SQL 校验、SQL 执行正确率混入 RAG 指标。端到端 NL2SQL 评测可以作为后续版本，但不得在本版本扩展实现。

## 背景概念

### BIRD Dev

BIRD 是 Text-to-SQL Benchmark。BIRD Dev 由多道题目和对应的数据库文件组成。每道题通常包含：

```json
{
  "question": "查询订单金额大于 100 的用户",
  "db_id": "shop",
  "SQL": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id WHERE o.amount > 100;"
}
```

`db_id` 只用于定位题目对应的数据库，例如：

```text
<bird-root>/dev_databases/shop/shop.sqlite
```

它表示数据库是哪一个，不表示当前问题使用了数据库中的哪些表。多个题目可以共享同一个数据库。

### Gold SQL 与 Gold Tables

Gold SQL 是数据集官方提供的标准 SQL。上例中的 Gold SQL 引用了 `users` 和 `orders`，因此从 Gold SQL 提取出的 Gold Tables 是：

```text
{"users", "orders"}
```

Gold Tables 是 Schema RAG 评测的标签，表示该问题实际需要的基础表。数据库文件只能提供该数据库的全部候选表，不能告诉评测器当前问题真正需要哪几张表，所以需要从 Gold SQL 生成 Gold Tables；如果未来数据集直接提供 Gold Tables，可以优先使用官方标签。

## 评测范围与边界

### 本版本包含

- BIRD Dev JSON 题目加载；
- `db_id` 到本地 SQLite 文件的映射；
- 使用 SQLGlot 从 Gold SQL 提取基础表名；
- 从 SQLite 数据库生成当前 `RagSchemaRecord` 所需的 DDL 和脱敏 SampleValue；
- 使用当前 Qwen Embedding Adapter 和 LanceDB 建立本地索引；
- DDL-only、SampleValue-only、Fusion/RRF 三组召回；
- Top-1、Top-3、Top-5、Top-10 对比；
- Recall、Full Recall、Precision、空结果、歧义结果、误召回数量和耗时统计；
- JSONL 明细报告和 JSON/CSV 汇总报告；
- Fake Embedding/Fake Retriever 单元测试，不调用真实模型网络；
- README 中的离线评测说明。

### 本版本禁止

- 不修改默认 RRF 公式或默认检索行为；
- 不将 Gold SQL、Gold Tables 或答案表名写入 RAG 查询文本或索引文本；
- 不实现 SQL 生成、SQL 执行、SQL 结果评测或反思策略；
- 不新增 API、网页入口、Agent Tool、MCP、Skill、向量重排模型或多模型路由；
- 不把 BIRD 数据集、数据库文件、真实 API Key 或生成报告中的敏感数据提交到 Git；
- 不修改 ChatAgent、ChatService 的正常调用链；
- 不把数据库全部表数量误命名为 Gold Recall。

## 现有 RAG 与评测入口

实现前必须阅读并沿用：

- `app/config.py`
- `app/rag/models.py`
- `app/rag/embedding.py`
- `app/rag/indexer.py`
- `app/rag/vector_store.py`
- `app/rag/retriever.py`
- `app/rag/context_packer.py`
- `app/rag/service.py`
- `tests/test_rag_models.py`
- `tests/test_rag_components.py`
- `tests/test_rag_fix.py`
- `D:\project\qiuzhao\DA\benchmark\scripts\schema_recall_bird.py`
- `D:\project\qiuzhao\DA\benchmark\scripts\schema_recall_spider2.py`

DA 的参考结论：

- BIRD 脚本从每道题的 Gold SQL 提取目标表；
- Schema 和 Value 分别检索后，再计算单路和 union 命中；
- Spider2 使用预先生成的 Gold Tables 文件；
- `union_match_tables_score` 才是 Gold Table 命中比例；不能把 `召回表数量 / 数据库总表数量` 当成 Gold Recall。

## 目标架构

```text
BIRD Dev question record
  ├── question --------------------------┐
  ├── db_id → SQLite database            │
  └── Gold SQL → Gold Tables             │
                                           │
SQLite database                            │
  ├── DDL records                         │
  └── safe SampleValue records            │
          ↓                               │
      Qwen Embedding → LanceDB            │
          ├── ddl_documents               │
          └── sample_value_documents      │
                                           │
question → query embedding                │
          ├── DDL-only                    │
          ├── SampleValue-only             │
          └── DDL + SampleValue + RRF      │
                    ↓                     │
              retrieved table set ────────┘
                    ↓
              compare with Gold Tables
                    ↓
          per-case + aggregate report
```

索引建立是离线阶段。每道题实时需要向量化的是自然语言问题；DDL 和 SampleValue 文本不应在每道题执行时重复建索引。

## 数据隔离要求

### Namespace

BIRD Dev 中多个数据库可能存在同名表，因此每个 `db_id` 必须使用独立 namespace：

```text
namespace = db_id
table_id = db_id + "." + table_name
```

评测时只能在当前题目对应的 namespace 内检索：

```text
retriever.search(question, namespace=db_id, limit=k)
```

禁止把所有数据库使用同一个 `default` namespace 后再跨数据库检索。

### 数据泄漏

- Gold SQL 只用于生成评测标签；
- Gold Tables 不得进入 DDL、SampleValue、查询文本或 rerank 输入；
- 题目的标准答案不能作为 Embedding 文本；
- RAG 索引只能来自数据库结构和允许保存的脱敏样例值。

## 文件边界

优先新增独立评测模块，避免把 Benchmark 逻辑塞进 `app/rag` 的业务检索组件：

```text
app/rag_eval/
├── __init__.py
├── models.py          # 题目、Gold Tables、召回结果、指标报告模型
├── bird_loader.py     # BIRD JSON 与 SQLite 路径加载
├── gold_tables.py     # Gold SQL → Gold Tables
├── sqlite_catalog.py  # SQLite DDL/SampleValue → RagSchemaRecord
├── evaluator.py       # 三组召回和指标计算
└── report.py          # JSONL/JSON/CSV 报告输出

scripts/
└── evaluate_rag.py    # PowerShell 可调用的离线评测 CLI

tests/
├── test_rag_eval_gold_tables.py
├── test_rag_eval_bird_loader.py
├── test_rag_eval_sqlite_catalog.py
└── test_rag_eval_metrics.py
```

如当前代码结构更适合放在 `app/benchmark/rag_eval`，可以采用等价目录，但必须保持 Loader、Gold Label、Catalog、Evaluator、Report 职责分离。

允许对 `app/rag/retriever.py` 做最小兼容改动，以便评测器选择召回来源：

- 默认 `search(query, namespace, limit)` 的行为、默认参数和返回模型必须保持不变；
- 可以增加明确的 `sources` 参数或独立的 source-specific 方法；
- DDL-only、SampleValue-only 和 Fusion 必须使用相同的 query、namespace 和 Top-K 规则；
- 不得改变默认 RRF 的 `rrf_k=60`、相似度阈值和歧义判断。

## Task 1：评测数据模型与 BIRD Loader

### 目标

先定义可测试的数据模型，能够把 BIRD 题目和数据库路径转换成稳定对象。

### 建议模型

```python
class BirdCase(BaseModel):
    question_id: str
    db_id: str
    question: str
    gold_sql: str
    database_path: Path


class GoldTableSet(BaseModel):
    question_id: str
    db_id: str
    tables: list[str]
```

### Loader 要求

- 支持 BIRD Dev JSON 为列表的常见格式；
- 明确读取题目字段的大小写兼容映射，例如 `SQL`/`sql`；
- 用 `db_id` 拼接 `<dataset_root>/dev_databases/<db_id>/<db_id>.sqlite`；
- 数据库文件不存在时抛出清晰错误，不能静默跳过；
- 支持 `question_id` 缺失时使用稳定的记录序号，但报告中必须标记来源；
- 支持 `--limit` 和 `--question-id` 过滤，便于先跑小样本；
- 不读取、不打印 API Key。

### TDD 验收

先写测试覆盖：

1. 正常加载一个 BIRD JSON 记录；
2. `db_id` 正确映射到 SQLite 路径；
3. 缺少数据库文件时给出稳定异常；
4. 题目过滤只保留指定记录；
5. 不把 Gold SQL 改写为查询文本。

## Task 2：Gold SQL 提取 Gold Tables

### 目标

使用已安装的 SQLGlot 从 Gold SQL 提取基础表名，作为评测标签。

### 提取规则

- 使用 SQLite 方言解析 BIRD Gold SQL；
- 遍历 `exp.Table` 节点；
- 排除 CTE 名称、子查询别名和列别名；
- 统一去除引用符并做大小写不敏感归一化；
- 对同一表去重并保持稳定排序；
- 保留数据库/Schema/表的限定名时，必须和 `RagSchemaRecord.table_id` 的规范一致；
- 解析失败必须返回明确的 `GoldTableExtractionError`，不能把空集合当成成功标签。

示例：

```sql
WITH recent AS (
    SELECT * FROM orders
)
SELECT u.name
FROM users AS u
JOIN recent AS r ON r.user_id = u.id;
```

结果应为：

```text
{"users", "orders"}
```

### TDD 验收

必须覆盖：

- 单表 SELECT；
- JOIN 多表；
- CTE 不把 CTE 名称当成基础表；
- 子查询；
- 反引号、双引号表名；
- 重复表引用去重；
- 非法 SQL 稳定失败。

## Task 3：SQLite Catalog 转换为 RAG 记录

### 目标

将 BIRD SQLite 数据库转换为当前 RAG Indexer 使用的 `RagSchemaRecord`，不改变现有索引格式。

### DDL

从 `sqlite_master` 读取真实建表 SQL，生成：

```python
RagSchemaRecord(
    namespace=db_id,
    table_id=f"{db_id}.{table_name}",
    table_name=table_name,
    ddl=ddl,
    columns=[...],
)
```

系统表、SQLite 内部对象和视图默认不纳入第一版表集合，除非 CLI 明确打开视图选项。

### SampleValue

对每个普通字段读取少量样例值，填入 `RagColumnRecord.sample_values`。必须：

- 使用 SQLite 只读连接；
- 只允许使用从 `sqlite_master`/`PRAGMA table_info` 获得的标识符，并安全引用表名和列名；
- 使用固定的小样本上限，例如每列 3 个值；
- 复用 `SchemaIndexer._safe_values` 的敏感字段和值过滤；
- 不把整行数据、密码、Token、Cookie、Bearer 凭据或认证信息写入索引；
- 没有安全样例值时保留 DDL 字段，但不生成 SampleValue 文档。

不要在 Catalog 层调用 Chat LLM。索引向量统一由现有 `QwenEmbeddingAdapter` 提供。

### TDD 验收

使用临时 SQLite 文件覆盖：

1. 表和字段能生成正确 `RagSchemaRecord`；
2. DDL 来源正确；
3. SampleValue 按列采样；
4. 敏感列不进入 SampleValue；
5. 多个数据库使用不同 namespace；
6. SQLite 连接不会写入测试数据库。

## Task 4：三组召回实验

### 召回分支

每道题、每个 K 必须使用同一个问题向量、同一个 namespace 和同一个数据库索引，分别运行：

```text
ddl_only:
    只搜索 ddl_documents

sample_value_only:
    只搜索 sample_value_documents

fusion_rrf:
    同时搜索两张表，按现有 RRF 逻辑合并
```

当前 RRF 规则保持：

```text
score += 1 / (60 + rank)
```

`60` 是当前固定 `rrf_k`，不是 Top-K，也不是从数据集中计算出的数字。评测报告必须记录实际使用的 `rrf_k`、`min_hit_score` 和 `ambiguity_margin`。

### K 值

第一版至少测试：

```text
K = 1, 3, 5, 10
```

如果默认业务链路只能返回 Top-3，仍然必须保留 Top-1/Top-5/Top-10 的离线实验，以观察召回曲线。业务默认行为不因为评测而改成 Top-10。

### 当前项目的最小兼容方案

优先为 `SchemaRetriever` 增加 source 选择能力，默认仍为 DDL + SampleValue；或者提供等价的只读评测入口。禁止复制一套与生产逻辑不一致的 RRF 算法。

## Task 5：指标计算

对每道题定义：

```text
G = Gold Tables
R_K = 某一分支最终 Top-K 表集合
```

### 核心指标

```text
Recall@K = |G ∩ R_K| / |G|

FullRecall@K = 1 if G ⊆ R_K else 0

Precision@K = |G ∩ R_K| / |R_K|
```

当 `G` 为空时必须报错或单独计为无效样本，不能把分母置为 1 伪造满分。

对全量样本按题目做宏平均，并额外提供 micro 汇总。报告至少包含：

- 每个 branch 的 Recall@1/3/5/10；
- 每个 branch 的 FullRecall@1/3/5/10；
- 每个 branch 的 Precision@1/3/5/10；
- empty rate；
- ambiguous rate；
- 平均误召回表数量；
- 平均召回耗时；
- 每个数据库的分组统计；
- 题目级失败样本列表。

`database_coverage = len(R_K) / 当前数据库总表数` 可以作为辅助指标，但必须和 Gold Recall 分开命名，不得混用。

### 结果示例

```json
{
  "question_id": "bird_001",
  "db_id": "shop",
  "gold_tables": ["orders", "users"],
  "branches": {
    "ddl_only": {
      "top_k": 3,
      "retrieved_tables": ["users", "products", "payments"],
      "matched_tables": ["users"],
      "recall": 0.5,
      "full_recall": false
    },
    "sample_value_only": {
      "top_k": 3,
      "retrieved_tables": ["orders", "payments", "products"],
      "matched_tables": ["orders"],
      "recall": 0.5,
      "full_recall": false
    },
    "fusion_rrf": {
      "top_k": 3,
      "retrieved_tables": ["orders", "users", "payments"],
      "matched_tables": ["orders", "users"],
      "recall": 1.0,
      "full_recall": true
    }
  }
}
```

## Task 6：报告与 CLI

建议 CLI：

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --dataset-root D:\data\bird `
  --question-file D:\data\bird\dev.json `
  --database-root D:\data\bird\dev_databases `
  --index-path .\artifacts\rag-eval\lancedb `
  --top-k 1 3 5 10 `
  --output-dir .\artifacts\rag-eval\reports `
  --limit 20
```

CLI 要求：

- `--dataset-root`、`--question-file`、`--database-root`、`--output-dir` 必须显式传入或由明确的参数对象提供；
- 不从业务模块直接读取 `os.getenv`；
- `--limit` 和 `--question-id` 支持小样本调试；
- `--rebuild-index` 明确控制是否覆盖评测索引；
- 报告中记录数据集路径的逻辑标识和哈希/版本信息，不写入敏感本机绝对路径也可以；
- 默认不把报告写入 Git 跟踪目录，建议输出到 `artifacts/`，并在 `.gitignore` 中确认不会提交数据库和报告。

建议输出：

```text
artifacts/rag-eval/reports/
├── cases.jsonl
├── summary.json
└── summary.csv
```

## Task 7：TDD 测试与回归

### 单元测试

必须使用临时目录、临时 SQLite 和 Fake Embedding/Fake Retriever，不能访问 Qwen、OpenAI、DeepSeek、Jev 或任何真实网络服务。

至少覆盖：

1. Gold SQL 单表、多表、JOIN、CTE、子查询和非法 SQL；
2. BIRD Loader 的 `db_id` 路径映射和缺失文件错误；
3. SQLite Catalog 的 DDL、列信息、SampleValue 和敏感字段过滤；
4. DDL-only、SampleValue-only、Fusion 三个分支都能产生表集合；
5. RRF 分数和生产 `SchemaRetriever` 一致；
6. Recall@K、FullRecall@K、Precision@K 的边界值；
7. Gold Tables 为空时不会产生虚假满分；
8. 空结果和 ambiguous 结果会在报告中稳定记录；
9. 多 namespace 不会把不同数据库的同名表混在一起；
10. 评测模块不会改变默认 ChatService/API 行为。

### 验证命令

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests scripts
git diff --check
```

如果没有 BIRD 数据集或 Qwen Key，不得伪造完整 Benchmark 结果。此时只能运行单元测试，并在报告中明确标记真实 Benchmark 未运行。拿到合法的 `QWEN_API_KEY` 后，才能执行真实 Embedding 评测；不得在报告中输出 Key。

## README 更新要求

只记录真实完成并验证过的内容：

- V8 新增 BIRD Dev Schema RAG 离线评测；
- Gold SQL → Gold Tables 的标签生成；
- DDL-only、SampleValue-only、Fusion/RRF 对照；
- Recall@K、FullRecall@K、Precision@K 和报告位置；
- BIRD 数据集需要用户自行下载并通过 CLI 参数提供；
- 真实 Benchmark 需要配置 Qwen Embedding Key，单元测试不需要 Key。

不得声称已经完成：

- BIRD 全量真实跑分；
- 端到端 SQL 执行准确率；
- RAG 自动调参；
- Spider2 生产级适配；
- 没有实际运行过的在线模型或数据集结果。

## 验收标准

- [ ] BIRD Loader 能从题目记录定位 SQLite 数据库；
- [ ] Gold SQL 能稳定提取 Gold Tables，CTE 不被误当成表；
- [ ] 数据库 DDL/SampleValue 以正确 namespace 写入当前 LanceDB 两张索引表；
- [ ] 同一问题能够分别执行 DDL-only、SampleValue-only 和 Fusion/RRF；
- [ ] RRF 默认 `rrf_k=60`、相似度阈值和默认生产行为保持不变；
- [ ] 报告区分 Gold Recall 和 database coverage；
- [ ] 至少输出 Recall@1/3/5/10、FullRecall@1/3/5/10 和 Precision@1/3/5/10；
- [ ] 所有测试不调用真实模型网络；
- [ ] 没有 BIRD 数据集或 API Key 时不伪造 Benchmark 结果；
- [ ] `python -m pytest -q`、`compileall` 和 `git diff --check` 通过；
- [ ] README 只写实际完成的 V8 能力；
- [ ] 执行 Agent 提交全部 V8 改动。

## 交给另一个 Agent 的执行提示词

```text
请在 D:\project\python\myagent 按 docs\v8_rag-evaluation-plan.md 实现 V8 Schema RAG Benchmark Evaluation。

开始前完整阅读：
- docs\v8_rag-evaluation-plan.md
- app\config.py
- app\rag\models.py
- app\rag\embedding.py
- app\rag\indexer.py
- app\rag\vector_store.py
- app\rag\retriever.py
- app\rag\context_packer.py
- app\rag\service.py
- tests\test_rag_models.py
- tests\test_rag_components.py
- tests\test_rag_fix.py
- D:\project\qiuzhao\DA\benchmark\scripts\schema_recall_bird.py
- D:\project\qiuzhao\DA\benchmark\scripts\schema_recall_spider2.py

只实现 Schema RAG 离线评测，不修改 ChatAgent、ChatService、API、Session、LLMAdapter、Intent、ToolRegistry 或 LangGraph NL2SQL 主链的默认行为。不要实现 SQL 生成、SQL 执行、MCP、Skill、SSE、向量重排模型、多 Agent 或多模型路由。

必须按 TDD：先写失败测试并运行确认失败，再做最小实现。所有单元测试使用 Fake Embedding/Fake Retriever/临时 SQLite，不调用真实模型网络。只有在用户明确提供合法 QWEN_API_KEY 并要求运行真实 Benchmark 时，才允许执行真实 Qwen Embedding；绝不输出或提交 Key。

实现要求：
1. 新增独立 rag_eval 模块，分离 BIRD Loader、Gold SQL → Gold Tables、SQLite Catalog、Evaluator、Report；
2. 使用 SQLGlot 从 Gold SQL 提取基础表，排除 CTE 名称、别名和重复表；解析失败必须明确失败，不能把空集合当成功；
3. 从每个 BIRD SQLite 数据库读取 DDL 和受限 SampleValue，使用 db_id 作为 namespace、db_id.table_name 作为 table_id；禁止跨数据库混检；
4. 使用当前 Qwen Embedding Adapter 和 LanceDB 的 ddl_documents/sample_value_documents；评测索引不得把 Gold SQL 或 Gold Tables 写入文本；
5. 支持 DDL-only、SampleValue-only、Fusion/RRF 三个分支，以及 Top-1/3/5/10；默认生产 RRF 的 rrf_k=60、相似度阈值和歧义行为不得改变；
6. 计算 Recall@K、FullRecall@K、Precision@K、empty rate、ambiguous rate、误召回数量、耗时，并区分 Gold Recall 与 database coverage；
7. 增加可用 PowerShell 调用的 scripts\evaluate_rag.py，支持 dataset/question/database/index/output 路径、limit、question-id、top-k 和 rebuild-index；不在业务模块直接读取环境变量；
8. 输出 cases.jsonl、summary.json、summary.csv，报告不能包含 API Key；BIRD 数据集和生成索引/报告不得提交 Git；
9. 更新 README，只写实际完成和验证过的 V8 能力，不伪造真实 Benchmark 分数；
10. 保留已有注释，不删除或回滚无关改动。当前目录按项目约束处理，不执行 reset、checkout 或删除无关文件。

完成后必须使用 PowerShell 运行：
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m compileall -q app tests scripts
git diff --check

如果没有 BIRD 数据集或 QWEN_API_KEY，只报告单元测试结果，并明确说明未进行真实 Benchmark/在线 Embedding 验证，不要伪造结果。

完成后提交全部 V8 改动，提交信息使用：feat: add schema rag benchmark evaluation
最终报告包含：修改文件、测试完整结果、CLI 用法、是否运行真实 BIRD、是否调用真实 Qwen Embedding、Gold Table 提取方式、三组召回指标定义、提交哈希和推送结果。
```
