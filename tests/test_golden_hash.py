"""Хеш-ворота golden (фаза 2, P0): правки golden в этой фазе запрещены безусловно.

Замороженный снимок ``results/golden/golden_results.md`` (состояние v0.1.0)
защищён SHA-256; живой ``golden_results.md`` в корне обязан совпадать со
снимком байт-в-байт. Красный тест здесь означает: кто-то изменил числа
golden — это остановка работы, а не повод поправить хеш (TODO_PHASE2, п. 5).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SNAPSHOT = _ROOT / "results" / "golden" / "golden_results.md"
_LIVE = _ROOT / "golden_results.md"

# SHA-256 снимка golden на момент v0.1.0 (104 ворот, контакт 8000 итераций).
GOLDEN_SHA256 = "7a646bd87a7896c6429bfc53cdbfce2da209f2bc0f18a2a1aef21402406e3803"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_golden_snapshot_frozen():
    """ВОРОТА: снимок results/golden/golden_results.md не изменён (SHA-256)."""
    assert _SNAPSHOT.is_file(), "снимок golden отсутствует — восстановите из git"
    assert _sha256(_SNAPSHOT) == GOLDEN_SHA256, (
        "results/golden/golden_results.md изменён; правки golden в фазе 2 "
        "запрещены безусловно (TODO_PHASE2, «красные линии»)"
    )


def test_live_golden_matches_snapshot():
    """ВОРОТА: живой golden_results.md байт-в-байт равен замороженному снимку."""
    assert _LIVE.is_file(), "golden_results.md в корне отсутствует"
    assert _LIVE.read_bytes() == _SNAPSHOT.read_bytes(), (
        "корневой golden_results.md разошёлся со снимком results/golden/ — "
        "вероятно, перегенерирован run_golden.py; верните git checkout -- "
        "golden_results.md (правки golden в фазе 2 запрещены)"
    )
