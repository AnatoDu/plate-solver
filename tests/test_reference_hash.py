"""Хеш-ворота эталонного отчёта: правки reference запрещены.

results/reference/reference_v0.4.md заморожен SHA-256. Любое изменение
чисел = красный тест, а НЕ повод поправить хеш: обновление отчёта — только
осознанным коммитом (перегенерация scripts/run_reference.py) с обоснованием
в CHANGELOG и новой строкой хеша здесь. Исторический reference_v0.3.md
(freeze v0.3.0) заморожен отдельно и в v0.4 не тронут.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "results" / "reference" / "reference_v0.4.md"

#: заморожено при релизе v0.4.0 (лестница включает нелинейные karman-ступени)
_SHA256 = "f337783fba4ce4b8867549ad688266835f5508a623ee3cd5457c04fa87ac4185"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.4.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")
