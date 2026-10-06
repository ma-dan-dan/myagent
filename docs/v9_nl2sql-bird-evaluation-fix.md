# V9 NL2SQL BIRD 评测修复实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** 修复 V9 评测实现与既定五节点 LangGraph 工作流之间的偏差，并确保浮点容差和 DDL/SampleValue 索引完整性真正参与评测。

**Architecture:** 预测链路严格保持 SchemaLinking → GenSQL → Execute → Reflection → Output。Gold SQL 仍只在评测器旁路执行，预测结果仍从 LangGraph State 读取；本修复不新增 AST 校验，不把 Gold 信息注入预测链路。索引复用按当前评测分支需要的所有 source namespace 判断，结果比较器由 CLI 显式注入用户配置的浮点绝对误差。

**Tech Stack:** Python、LangGraph、Pydantic、SQLite、LanceDB、pytest、PowerShell。

---

## 修复范围与禁止事项

本次只修复以下问题：

1. 生产 NL2SQL 图和 V9 元数据必须与五节点计划一致。
2. --float-abs-tolerance 必须真正影响 ResultComparator 的执行结果判定。
3. RAG 索引复用必须检查当前分支需要的全部 source namespace，不能只检查 DDL。
4. 增加回归测试、更新 README 中与 V9 实际工作流相关的描述，并提交全部改动。

禁止扩展到：

- SQLGlot AST 解析、SQL AST 白名单或独立 ValidateSQL 节点；
- ContextPrepare 业务节点；
- SQL 写操作、非只读 Executor、真实数据库业务功能；
- Tool、MCP、Skill、SSE、向量重排、模型路由、子 Agent；
- ChatService、API、Session、Intent 和既有 V8 业务逻辑的无关修改。

保留现有 ReadOnlySQLiteExecutor 的只读连接、写操作拒绝、超时、最大行数/列数和稳定错误处理。即使移除工作流中的 AST 校验，也不能绕过 Executor 安全边界。

不要删除已有注释，不要回滚其他 Agent 的改动，不要执行 git reset、git checkout 或删除无关文件。不要调用真实模型 API、真实 Qwen Embedding API 或真实 BIRD 全量评测。

## 文件边界

- Modify: app/nl2sql/graph.py：仅调整 LangGraph 节点注册、路由和 Execute 前置条件，使其成为五节点图。
- Modify: app/nl2sql/nodes.py：让 Execute 直接使用经 Pydantic 校验后的 SQLDraft.sql，依靠现有只读 Executor 返回执行错误。
- Modify: app/nl2sql_eval/factory.py：不再为评测图注入或依赖 SQLValidator；按当前 sources 检查完整索引 namespace。
- Modify: scripts/evaluate_nl2sql.py：注入 ResultComparator，并把 workflow 元数据改为五节点字符串。
- Modify: scripts/evaluate_rag.py：索引复用检查当前构建所需的 DDL 和 SampleValue namespace。
- Modify: app/nl2sql_eval/evaluator.py：只在确有必要时调整首次执行指标定义，保持 Gold 旁路、结果比较和报告字段契约。
- Modify: tests/test_nl2sql_graph.py、tests/test_nl2sql_eval_workflow.py：覆盖五节点图和执行失败后的 Reflection。
- Modify: tests/test_nl2sql_eval_comparator.py 或 tests/test_nl2sql_eval_cli.py：覆盖自定义浮点容差确实影响评测判定。
- Add or modify: 与索引完整性最贴近的现有 RAG/评测测试文件。
- Modify: README.md：只写实际验证过的五节点 V9 评测能力。

除非现有导入链路确实要求，否则不要删除 app/nl2sql/validator.py、context_prepare_node 或历史模型字段；它们可以保留为未接入本次五节点图的旧代码，但生产图和评测图不能调用它们。

---

### Task 1: 先锁定五节点工作流并补失败测试

Files:
- Modify: tests/test_nl2sql_graph.py
- Modify: tests/test_nl2sql_eval_workflow.py
- Modify: app/nl2sql/graph.py
- Modify: app/nl2sql/nodes.py

- [ ] Step 1: 写五节点拓扑失败测试。

在 tests/test_nl2sql_graph.py 增加测试，使用现有 Fake Schema、Fake LLM 和 Fake Executor 创建图，检查编译后的图节点集合：

    def test_graph_contains_only_the_five_nl2sql_workflow_nodes():
        graph = build_graph(
            FakeLLM([]),
            FakeSQLExecutor(QueryResult()),
        )

        nodes = set(graph.graph.get_graph().nodes)

        assert {"schema_linking", "gen_sql", "execute_sql", "reflection", "output"} <= nodes
        assert "context_prepare" not in nodes
        assert "validate_sql" not in nodes

如果 LangGraph 版本返回包含 __start__/__end__ 的集合，只断言五个业务节点存在且两个禁用节点不存在。

- [ ] Step 2: 写不依赖 AST 校验的失败测试。

把原来要求坏 SQL 在 AST 校验阶段被拦截的测试改成执行器边界测试：Fake Executor 对错误 SQL 抛出 SQLExecutionError，图应进入 Reflection，而不是依赖 SQLValidator。

测试至少覆盖：

    def test_execution_error_is_sent_to_reflection_without_validate_node():
        llm = FakeLLM(
            [
                LLMResponse.message(
                    '{"status":"ok","sql":"SELECT missing FROM production_output",'
                    '"tables":["production_output"],"parameters":{},"explanation":"首次生成"}'
                ),
                LLMResponse.message('{"decision":"regenerate","reason":"执行失败"}'),
                LLMResponse.message(
                    '{"status":"ok","sql":"SELECT output_quantity FROM production_output",'
                    '"tables":["production_output"],"parameters":{},"explanation":"修复"}'
                ),
                LLMResponse.message('{"decision":"pass","reason":"结果正确"}'),
            ]
        )
        executor = FakeSQLExecutor(SQLExecutionError("no such column: missing"))

        result = build_graph(llm, executor).invoke("u1", "s1", "查询产量")

        assert result.status == "ok"
        assert len(llm.calls) == 4

实际测试可以沿用现有 Fake Executor，但必须证明坏 SQL 没有被 AST 节点静默拦截，且仍由只读 Executor/Reflection 处理。

- [ ] Step 3: 运行失败测试确认基线问题。

运行：

    .\.venv\Scripts\python.exe -m pytest -q tests/test_nl2sql_graph.py tests/test_nl2sql_eval_workflow.py

预期：拓扑测试因仍存在 context_prepare/validate_sql 失败；更新后的执行失败测试因当前 Execute 仍要求 validation 通过而失败。不得跳过这一步。

- [ ] Step 4: 最小修改 LangGraph 为五节点。

在 app/nl2sql/graph.py 中：

1. 从 _build_graph 的 add_node 和边中移除 context_prepare、validate_sql。
2. 将 schema_linking 成功分支直接连接到 gen_sql。
3. 将 gen_sql 直接连接到 execute_sql。
4. 保留 execute_sql → reflection、Reflection 的 regenerate → gen_sql 和其他决策到 output。
5. 不在图中调用 ContextManager 或 SQLValidator。
6. 保留 max_attempts、max_reflections 和现有终止条件。
7. 不修改 ChatService、API、Session 或生产请求协议。

如果现有构造函数或启动装配仍传入历史 validator 参数，为保持最小改动可以保留兼容参数，但不得保存为执行依赖、不得被调用、不得参与图路由。不要为了兼容参数重新注册 ValidateSQL 节点。

- [ ] Step 5: 最小修改 Execute 节点。

在 app/nl2sql/nodes.py 中，Execute 的前置条件只检查有效 SQLDraft：

    if not current.sql_draft or current.sql_draft.status != "ok" or not current.sql_draft.sql:
        return update

    try:
        result = executor.execute(current.sql_draft.sql, current.sql_draft.parameters)
    except SQLExecutorUnavailable:
        raise
    except SQLExecutionError as exc:
        update["execution_error"] = str(exc)
        return NL2SQLState.model_validate(update).model_dump()

    update["query_result"] = result.model_dump()
    return NL2SQLState.model_validate(update).model_dump()

保留 Pydantic 对 SQLDraft 的 JSON/字段校验，不新增 SQL 解析器或字符串白名单。

- [ ] Step 6: 调整首次执行指标语义并让测试通过。

没有 ValidateSQL 节点后：

- initial_generation_valid 表示首次 GenSQL 返回合法 SQLDraft、有非空 SQL，并进入 Execute；
- initial_execution_success 表示首次 Execute 没有错误；
- initial_execution_accuracy 表示首次执行结果与 Gold Result 等价；
- 执行失败不能伪装为生成失败；
- malformed JSON、Pydantic 校验失败或没有 SQL 时才算首次生成无效。

更新 _initial_cycle 和相关测试，确保执行失败后仍能进入 Reflection，且没有再次引入 SQLValidator。

- [ ] Step 7: 运行 Task 1 测试。

    .\.venv\Scripts\python.exe -m pytest -q tests/test_nl2sql_graph.py tests/test_nl2sql_eval_workflow.py

预期：通过，并且测试中没有 Fake LLM、Gold SQL 或 Gold Result 被注入预测 Prompt。

- [ ] Step 8: 提交 Task 1。

    git add app/nl2sql/graph.py app/nl2sql/nodes.py tests/test_nl2sql_graph.py tests/test_nl2sql_eval_workflow.py
    git commit -m "fix: align V9 workflow with five nodes"

---

### Task 2: 修复浮点容差从 CLI 到 Comparator 的传递

Files:
- Modify: scripts/evaluate_nl2sql.py
- Test: tests/test_nl2sql_eval_comparator.py 或 tests/test_nl2sql_eval_cli.py

- [ ] Step 1: 写失败测试。

增加一个不调用外部 API 的测试：

    def test_configured_float_tolerance_changes_execution_accuracy():
        predicted = QueryResult(columns=["value"], rows=[[1.05]], row_count=1, truncated=False)
        gold = QueryResult(columns=["value"], rows=[[1.0]], row_count=1, truncated=False)

        strict = ResultComparator(float_abs_tolerance=0.01)
        relaxed = ResultComparator(float_abs_tolerance=0.1)

        assert strict.compare(predicted, gold).execution_accuracy is False
        assert relaxed.compare(predicted, gold).execution_accuracy is True

另外增加 CLI 组装层测试，验证 args.float_abs_tolerance 被用于创建 ResultComparator，不能只验证报告 JSON 中的字段。若不便直接运行 CLI，可抽取只负责构造 NL2SQLWorkflowEvaluator 的小函数并断言 comparator.float_abs_tolerance。

- [ ] Step 2: 运行测试确认当前 CLI 组装未覆盖该行为。

    .\.venv\Scripts\python.exe -m pytest -q tests/test_nl2sql_eval_comparator.py tests/test_nl2sql_eval_cli.py

预期：新增的 CLI 传递测试在实现前失败，说明报告字段和实际比较逻辑目前可能脱节。

- [ ] Step 3: 注入配置后的 Comparator。

在 scripts/evaluate_nl2sql.py 创建 NL2SQLWorkflowEvaluator 时增加：

    comparator=ResultComparator(
        float_abs_tolerance=args.float_abs_tolerance
    ),

保持 write_workflow_reports 的 float_abs_tolerance 参数，使实际判定和报告元数据使用同一个值。不要修改默认值 1e-2、行顺序、重复行、NULL、截断和列数规则。

- [ ] Step 4: 运行测试确认通过。

    .\.venv\Scripts\python.exe -m pytest -q tests/test_nl2sql_eval_comparator.py tests/test_nl2sql_eval_cli.py

- [ ] Step 5: 提交 Task 2。

    git add scripts/evaluate_nl2sql.py tests/test_nl2sql_eval_comparator.py tests/test_nl2sql_eval_cli.py
    git commit -m "fix: wire NL2SQL evaluation tolerance"

---

### Task 3: 修复 DDL/SampleValue 索引完整性检查

Files:
- Modify: scripts/evaluate_rag.py
- Modify: app/nl2sql_eval/factory.py
- Test: 现有 RAG/评测测试文件或新增 tests/test_nl2sql_eval_factory.py

- [ ] Step 1: 写索引缺层测试。

用 Fake/Mock LanceVectorStore 和临时 SQLite 覆盖：

| 当前分支 | DDL | SampleValue | 预期 |
|---|---:|---:|---|
| fusion_rrf | 有 | 有 | 复用索引 |
| fusion_rrf | 有 | 无 | 重建索引 |
| ddl_only | 有 | 无 | 可复用 DDL |
| sample_value_only | 无 | 有 | 可复用 SampleValue |
| sample_value_only | 有 | 无 | 重建 SampleValue |

断言缺少当前分支所需 namespace 时会调用 SchemaIndexer.index，而不是直接设置 service._indexed = True。

示例纯逻辑测试：

    assert required_namespaces_ready(
        store, "shop", (RagSourceType.DDL, RagSourceType.SAMPLE_VALUE)
    ) is False
    assert required_namespaces_ready(
        store, "shop", (RagSourceType.DDL,)
    ) is True

如果不新增公共 helper，直接测试 factory/script 中的等价逻辑即可；不要引入第二个索引存储实现。

- [ ] Step 2: 运行测试确认当前实现会错误复用索引。

    .\.venv\Scripts\python.exe -m pytest -q tests/test_rag_eval_cli.py tests/test_nl2sql_eval_factory.py

预期：DDL 有、SampleValue 无、fusion_rrf 场景在修复前失败。

- [ ] Step 3: 按 source 集合修复索引就绪判断。

所有索引就绪判断必须等价于：

    required_sources = sources or (
        RagSourceType.DDL,
        RagSourceType.SAMPLE_VALUE,
    )
    index_ready = all(
        store.has_namespace(source, db_id)
        for source in required_sources
    )

scripts/evaluate_rag.py 构建的是双路索引，因此必须确认 DDL 和 SampleValue 两个 namespace 都存在。

app/nl2sql_eval/factory.py 使用当前 self.sources；None 表示 SchemaLinkingService 的默认双路召回，不能只检查 DDL。只在 index_ready 为真时设置服务已索引，否则让现有 lazy indexing 走 SchemaIndexer.index。

保留 --rebuild-index 的强制重建行为。不要删除旧索引文件，不要自动清空用户目录。

- [ ] Step 4: 运行测试确认通过。

    .\.venv\Scripts\python.exe -m pytest -q tests/test_rag_eval_cli.py tests/test_nl2sql_eval_factory.py

- [ ] Step 5: 提交 Task 3。

    git add scripts/evaluate_rag.py app/nl2sql_eval/factory.py tests/test_rag_eval_cli.py tests/test_nl2sql_eval_factory.py
    git commit -m "fix: validate both RAG index sources"

---

### Task 4: 修正 V9 元数据、README 和完整回归

Files:
- Modify: scripts/evaluate_nl2sql.py
- Modify: README.md
- Test: tests/test_nl2sql_eval_cli.py

- [ ] Step 1: 先写元数据失败测试。

增加对评测元数据构造的测试，要求 workflow 字段严格为：

    SchemaLinking -> GenSQL -> Execute -> Reflection -> Output

并断言不包含 ContextPrepare、ValidateSQL 或 SQLGlot。

- [ ] Step 2: 修正脚本元数据。

将 scripts/evaluate_nl2sql.py 中的 workflow 字段改为五节点字符串。报告中可以继续记录 rag_branch、max_attempts、max_reflections 和浮点容差，但不得宣称不存在的节点。

- [ ] Step 3: 更新 README。

只保留实际验证过的内容：

- V9 使用 BIRD Dev 题目和数据库进行离线评测；
- Gold SQL 只旁路执行生成参考结果；
- 预测流程是五节点 LangGraph：SchemaLinking、GenSQL、Execute、Reflection、Output；
- 主要指标是结果集等价的 Execution Accuracy；
- 报告区分首次执行、最终执行和 Reflection 修复；
- 测试使用 Fake LLM、Mock 和临时 SQLite，不调用真实模型。

明确不要写：

- 已实现 SQLGlot AST 校验；
- 已完成 BIRD 全量真实跑分；
- 已调用真实 Qwen/OpenAI/DeepSeek；
- SQL 文本必须与 Gold SQL 完全一致。

- [ ] Step 4: 运行完整验证。

    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m compileall -q app tests scripts
    git diff --check
    git status --short

预期：所有测试通过、编译无输出、git diff --check 无输出；没有 BIRD 数据集或 API Key 时不得伪造真实 Benchmark 分数。

- [ ] Step 5: 提交全部修复并推送。

    git add app tests scripts README.md docs/v9_nl2sql-bird-evaluation-fix.md
    git commit -m "fix: align V9 NL2SQL evaluation"
    git push origin master

如果工作区中存在不属于 V9 的用户改动，不要覆盖或回滚；应先停止提交并在报告中说明。提交前确认没有 BIRD 数据、LanceDB 评测产物、API Key、Token 或临时数据库被加入 Git。

## 验收标准

- [ ] 生产 LangGraph 的业务节点只有 schema_linking、gen_sql、execute_sql、reflection、output。
- [ ] 生产图不调用 ContextPrepare 或 SQLValidator，不依赖 SQLGlot AST 校验。
- [ ] 执行错误仍由只读 Executor 捕获，并可进入 Reflection regenerate。
- [ ] Reflection 的 pass、clarify、reject 和次数上限行为没有回归。
- [ ] Gold SQL、Gold Tables、Gold Result 没有进入预测 Prompt、Schema Context、Reflection 输入或 State。
- [ ] --float-abs-tolerance 的实际值被注入 ResultComparator，并由测试证明会影响判定。
- [ ] RAG 双路评测不会在 SampleValue namespace 缺失时错误复用只有 DDL 的索引。
- [ ] DDL-only 分支不被强制要求 SampleValue，SampleValue-only 分支不被强制要求 DDL。
- [ ] V9 元数据和 README 都只描述五节点工作流。
- [ ] pytest、compileall 和 git diff --check 全部通过。
- [ ] 真实 BIRD、Embedding、LLM 是否运行在报告中如实说明。
- [ ] 所有 V9 修复已提交并推送到 origin/master。

## 给另一位工程师的执行提示词

    请在 D:\project\python\myagent 按 docs\v9_nl2sql-bird-evaluation-fix.md 执行 V9 修复。

    开始前完整阅读：
    - docs\v9_nl2sql-bird-evaluation-fix.md
    - docs\v9_nl2sql-bird-evaluation-plan.md
    - app\nl2sql\graph.py
    - app\nl2sql\nodes.py
    - app\nl2sql\models.py
    - app\nl2sql\executor.py
    - app\nl2sql_eval\factory.py
    - app\nl2sql_eval\evaluator.py
    - app\nl2sql_eval\result_comparator.py
    - scripts\evaluate_nl2sql.py
    - scripts\evaluate_rag.py
    - tests\test_nl2sql_graph.py
    - tests\test_nl2sql_eval_workflow.py
    - tests\test_nl2sql_eval_comparator.py
    - tests\test_nl2sql_eval_cli.py
    - tests\test_rag_eval_cli.py

    本次只修复三类问题：
    1. 将实际 LangGraph 对齐为 SchemaLinking -> GenSQL -> Execute -> Reflection -> Output；
    2. 让 --float-abs-tolerance 真正传入 ResultComparator；
    3. 让 RAG 索引复用按当前 source 集合检查全部 namespace，不能只检查 DDL。

    严格要求：
    - 不新增或恢复 ContextPrepare、ValidateSQL、SQLGlot AST 校验节点；
    - 不把 SQLValidator、Gold SQL、Gold Tables 或 Gold Result 放入预测链路；
    - 执行安全仍由现有只读 Executor、超时、最大行数/列数和写操作拒绝保证；
    - Gold SQL 只能在评测器旁路执行；
    - 结果等价比较规则保持不变，只修复浮点容差实际传递；
    - 使用 Fake LLM、Mock、临时 SQLite，绝不调用真实模型或 Embedding 网络；
    - 先写失败测试，确认失败后再实现；
    - 不删除已有注释，不回滚无关改动，不执行 git reset、checkout 或删除无关文件；
    - 不提交 BIRD 数据、LanceDB 评测产物、临时数据库、密钥或 Token。

    验证命令：
    .\.venv\Scripts\python.exe -m pytest -q
    .\.venv\Scripts\python.exe -m compileall -q app tests scripts
    git diff --check

    完成后：
    1. 更新 README，只写实际验证过的五节点 V9 能力；
    2. 报告修改文件、五节点图结构、索引完整性策略、浮点容差传递、测试完整结果；
    3. 明确是否运行了真实 BIRD、真实 Embedding 或真实 LLM；没有运行就如实写未运行；
    4. 提交全部改动，提交信息使用：fix: align V9 NL2SQL evaluation；
    5. 执行 git push origin master，并报告提交哈希和推送结果。
