"""Реестр = ворота: параметризованный pytest по cases/ci/*.toml.

Каждый ci-случай гоняется через plate-verify (exit 0). Новый файл в
cases/ci/ автоматически становится тестом. Тяжёлые (Q=1024) ступени —
cases/ladder/ с маркером big в своих тестах.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plate_solver.cli import main_verify

_CI = Path(__file__).resolve().parents[1] / "cases" / "ci"


@pytest.mark.parametrize("case", sorted(_CI.glob("*.toml")), ids=lambda p: p.stem)
def test_ci_case_verifies(case, tmp_path):
    """ВОРОТА: plate-verify по ci-случаю завершается кодом 0."""
    assert main_verify([str(case), "--out", str(tmp_path)]) == 0
