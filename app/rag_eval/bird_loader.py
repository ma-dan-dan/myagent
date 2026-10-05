from __future__ import annotations

import json
from pathlib import Path

from app.rag_eval.models import BirdCase


class BirdDatasetError(ValueError):
    pass


def load_bird_cases(
    question_file: str | Path,
    database_root: str | Path,
    *,
    limit: int | None = None,
    question_ids: set[str] | None = None,
) -> list[BirdCase]:
    question_path = Path(question_file)
    root = Path(database_root)
    try:
        payload = json.loads(question_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BirdDatasetError(f"无法读取 BIRD 题目文件：{question_path}") from exc
    if not isinstance(payload, list):
        raise BirdDatasetError("BIRD 题目文件必须是 JSON 数组。")
    if limit is not None and limit < 1:
        raise BirdDatasetError("limit 必须大于 0。")
    cases: list[BirdCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BirdDatasetError(f"第 {index} 条 BIRD 题目不是对象。")
        db_id = _required(item, ("db_id", "database_id"), index)
        question = _required(item, ("question", "Question"), index)
        gold_sql = _required(item, ("SQL", "sql"), index)
        raw_question_id = item.get("question_id", item.get("id"))
        question_id = str(raw_question_id) if raw_question_id is not None else f"record-{index}"
        if question_ids and question_id not in question_ids:
            continue
        database_path = root / db_id / f"{db_id}.sqlite"
        if not database_path.is_file():
            raise BirdDatasetError(f"BIRD 数据库不存在：{db_id}，期望路径：{database_path}")
        cases.append(
            BirdCase(
                question_id=question_id,
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                database_path=database_path,
                question_id_source="dataset" if raw_question_id is not None else "generated",
            )
        )
        if limit is not None and len(cases) >= limit:
            break
    return cases


def _required(item: dict, names: tuple[str, ...], index: int) -> str:
    for name in names:
        value = item.get(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise BirdDatasetError(f"第 {index} 条 BIRD 题目缺少字段：{' / '.join(names)}")
