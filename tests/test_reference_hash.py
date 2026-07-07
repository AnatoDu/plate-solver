"""Хеш-ворота эталонного отчёта: правки reference запрещены.

results/reference/reference_v0.5.md заморожен SHA-256. Любое изменение
чисел = красный тест, а НЕ повод поправить хеш: обновление отчёта — только
осознанным коммитом (перегенерация scripts/run_reference.py) с обоснованием
в CHANGELOG и новой строкой хеша здесь. Исторические reference_v0.3.md /
reference_v0.4.md заморожены отдельно и в v0.5 не тронуты.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "results" / "reference" / "reference_v0.5.md"

#: заморожено при релизе v0.5.0 (лестница включает ступени Кармана и полной КТН)
_SHA256 = "f61548de71679eb4ccbac9cf300c7f9ba7ff657b56e756bc7ccda5db4aff3270"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.5.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")
