"""Примеры фабрики в docs/API.md исполняемы (не протухают молча).

Извлекаются все python-блоки раздела о фабрике (`analytic_auto`) и
выполняются как есть: пример, переставший работать, роняет ворота.
"""

from __future__ import annotations

import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]


def _factory_blocks() -> list[str]:
    text = (_ROOT / "docs" / "API.md").read_text(encoding="utf-8")
    start = text.index("`analytic_auto` — ФАБРИКА")
    end = text.index("- `ladder`", start)
    section = text[start:end]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def test_factory_examples_execute():
    blocks = _factory_blocks()
    assert len(blocks) == 3, "ожидались три копируемых примера фабрики"
    for code in blocks:
        exec(compile(code, "<docs/API.md>", "exec"), {})  # noqa: S102
