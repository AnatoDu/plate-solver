"""Детерминизм эталонного прогона: повторный прогон ≡ закоммиченному.

run_reference.py пишет отчёт БЕЗ провенанса (git-хеш/дата — в спутнике
provenance.json вне хеша), поэтому повторный прогон обязан дать файл,
идентичный замороженному в репозитории. Это одновременно и детерминизм,
и актуальность отчёта. Полный прогон (золотая серия + лестница) — маркеры
big + fem.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.big
@pytest.mark.fem
def test_reference_rerun_identical(tmp_path):
    committed = (_ROOT / "results" / "reference" / "reference_v0.6.md"
                 ).read_bytes()
    # скрипт пишет в фиксированный путь: сохраняем и восстанавливаем файлы
    ref_dir = _ROOT / "results" / "reference"
    backup = {f.name: f.read_bytes() for f in ref_dir.glob("*")}
    try:
        out = subprocess.run(
            [sys.executable, str(_ROOT / "scripts" / "run_reference.py")],
            cwd=_ROOT, capture_output=True, text=True, timeout=3600)
        assert out.returncode == 0, out.stderr[-2000:]
        regenerated = (ref_dir / "reference_v0.6.md").read_bytes()
    finally:
        for name, data in backup.items():
            (ref_dir / name).write_bytes(data)
    assert regenerated == committed, (
        "повторный run_reference.py дал ИНОЙ отчёт — недетерминизм или "
        "рассинхрон с закоммиченным reference_v0.6.md")
