"""Хеш-ворота эталонного отчёта: правки reference запрещены.

results/reference/reference_v0.8.md заморожен SHA-256. Любое изменение
чисел = красный тест, а НЕ повод поправить хеш: обновление отчёта — только
осознанным коммитом (перегенерация scripts/run_reference.py) с обоснованием
в CHANGELOG и новой строкой хеша здесь. Исторические reference_v0.3.md …
reference_v0.6.md заморожены отдельно и в v0.8 НЕ тронуты (протокол §0:
при смене чисел заводится НОВЫЙ файл отчёта).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_REF = _ROOT / "results" / "reference" / "reference_v0.8.md"

#: заморожено при релизе v0.8.0. Числа КЛАССИЧЕСКОГО тракта преемственны с
#: v0.6 (GoldenConfig не тронут); изменились числа уточнённой теории —
#: исправлена размерность члена реакции лицевого условия (снят множитель D),
#: нормировка шага МОР приведена к лицевому оператору, критерий останова
#: итерации Кармана — к истинной невязке. Обоснование — CHANGELOG [0.8.0].
_SHA256 = "949d959d047adfbc85d7460d403f8c897bcc11f54dfa60b998b1dcfbbf17e77e"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.8.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")
