"""Хеш-ворота эталонного отчёта: правки reference запрещены.

results/reference/reference_v0.6.md заморожен SHA-256. Любое изменение
чисел = красный тест, а НЕ повод поправить хеш: обновление отчёта — только
осознанным коммитом (перегенерация scripts/run_reference.py) с обоснованием
в CHANGELOG и новой строкой хеша здесь. Исторические reference_v0.3.md /
reference_v0.4.md / reference_v0.5.md заморожены отдельно и в v0.6 не тронуты.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "results" / "reference" / "reference_v0.6.md"

#: заморожено при релизе v0.6.0 (числа золотой серии преемственны с v0.5;
#: изменилась лишь версия в заголовке отчёта)
_SHA256 = "4b17dd53a86ebdedd4af0bd801f8f3a260917faf3e470543fc9cdcb9d3d01e90"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.6.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")
