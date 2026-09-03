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
_REF_CSV = _ROOT / "results" / "reference" / "reference_v0.8.csv"

#: заморожено при релизе v0.8.0. Числа КЛАССИЧЕСКОГО тракта преемственны с
#: v0.6 (GoldenConfig не тронут); изменились числа уточнённой теории —
#: исправлена размерность члена реакции лицевого условия (снят множитель D),
#: нормировка шага МОР приведена к лицевому оператору, критерий останова
#: итерации Кармана — к истинной невязке. Обоснование — CHANGELOG [0.8.0].
_SHA256 = "949d959d047adfbc85d7460d403f8c897bcc11f54dfa60b998b1dcfbbf17e77e"

#: CSV-спутник отчёта (те же числа в машинном виде) — под теми же воротами:
#: README описывает пару «md + csv» как замороженную, но csv до v0.8.0 не
#: проверялся ничем и мог разойтись с отчётом незамеченным (аудит 0.8.0).
_SHA256_CSV = "aed3ed18a29c4619292d35cec61425046792eaa41e3cffd1dff6e4e29da18cea"


def test_reference_report_frozen():
    digest = hashlib.sha256(_REF.read_bytes()).hexdigest()
    assert digest == _SHA256, (
        "results/reference/reference_v0.8.md изменён; обновление эталонного "
        "отчёта — только осознанным коммитом с обоснованием в CHANGELOG "
        f"(получен sha256 {digest})")


def test_reference_csv_frozen():
    """CSV-спутник отчёта заморожен теми же воротами, что и markdown."""
    digest = hashlib.sha256(_REF_CSV.read_bytes()).hexdigest()
    assert digest == _SHA256_CSV, (
        "results/reference/reference_v0.8.csv изменён; он содержит ТЕ ЖЕ числа, "
        "что и отчёт, и обновляется только вместе с ним "
        f"(получен sha256 {digest})")
