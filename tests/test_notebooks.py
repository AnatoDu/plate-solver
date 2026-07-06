"""Смок ноутбуков: case-файлы, на которые они ссылаются, валидны.

Ноутбуки в CI не исполняются (без новых dev-зависимостей); гейт — JSON
корректен и каждый упомянутый case-файл проходит валидатор Problem.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]
_NBS = sorted((_ROOT / "notebooks").glob("*.ipynb"))


@pytest.mark.parametrize("nb_path", _NBS, ids=lambda p: p.stem)
def test_notebook_case_files_validate(nb_path):
    nb = json.loads(nb_path.read_text(encoding="utf-8"))
    text = "".join("".join(c.get("source", [])) for c in nb["cells"])
    cases = set(re.findall(r"cases/[\w/]+\.toml", text))
    for case in cases:
        problem = Problem.from_toml(_ROOT / case)     # CaseError ⇒ красный тест
        assert problem.geometry.kind
    if nb_path.stem != "01_circle_api":               # 01 — чистый API без case
        assert cases, f"{nb_path.name}: ожидалась ссылка на case-файл"
