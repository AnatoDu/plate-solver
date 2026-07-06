"""Хеш-ворота эталонного отчёта: правки reference запрещены.

results/reference/reference_v0.3.md заморожен SHA-256. Любое изменение
чисел = красный тест, а НЕ повод поправить хеш: обновление отчёта — только
осознанным коммитом (перегенерация scripts/run_reference.py) с обоснованием
в CHANGELOG и новой строкой хеша здесь.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "results" / "reference" / "reference_v0.3.md"

#: заморожено при миграции golden → reference (freeze v0.3.0)
_SHA256 = "38fa5c8231e1348412e1ec7d682e02529ec0dd7906c6090d732b340d51e80af4"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.3.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")
