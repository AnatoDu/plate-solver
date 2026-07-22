"""Абсолютный эталон контактной задачи (v0.6.3): круг + мягкий шарнир + основание.

Сертифицированное осесимметричное решение Кирхгофа (analytic_auto.
axisym_contact_solution) подключено в resolver — контакт круга гейтится
АБСОЛЮТНЫМ эталоном, а не только инвариантами. Гейт — прогиб ВНЕ контактной
зоны (``w_max = Δ`` тривиален). Прочий контакт эталона не имеет (ворота
инвариантов) — резолвер отклоняет `analytic` понятной ошибкой.
"""

from __future__ import annotations

import pytest

from plate_solver import dispatch
from plate_solver.problem import CaseError, Problem
from plate_solver.references import resolve_reference, verify_result

_D = 2.1e6 * 0.06**3 / (12 * (1 - 0.3**2))
_W_FREE0 = 3 * 4.0 * 1.0**4 / (64 * _D)                   # прогиб центра свободного круга


def _case(bc="soft_hinge", theory="classic", ref="analytic", gap="2.0e-3"):
    return f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "{bc}"
[load]
type = "uniform"
q0 = 4.0
[model]
theory = "{theory}"
h = 0.06
[contact]
enabled = true
gap = {gap}
max_iter = 2000
tol = 1e-8
[discretization]
p = 8
Q = 128
grid_n = 16
[verify]
reference = "{ref}"
tol = 2e-2
"""


def _solve(tmp_path, **kw):
    p = tmp_path / "case.toml"
    p.write_text(_case(**kw), encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


def test_certified_contact_reference_gates(tmp_path):
    """Круг+шарнир+классика+основание — прогиб вне зоны совпал с эталоном Кирхгофа."""
    res = _solve(tmp_path)
    rep = verify_result(res)
    assert rep.ok                                        # гейт пройден
    (row,) = rep.rows
    assert row.gated and row.rel < 2e-2                  # абсолютный эталон, не инвариант
    assert "Кирхгоф" in row.name


def test_certified_contact_gap_out_of_range_rejected(tmp_path):
    """Зазор вне (0, w_free(0)) — эталон не существует, понятный отказ."""
    p = tmp_path / "case.toml"
    p.write_text(_case(gap=f"{2 * _W_FREE0:.6e}"), encoding="utf-8")   # Δ > w_free(0)
    with pytest.raises(CaseError, match="w_free"):
        resolve_reference(Problem.from_toml(str(p)))


@pytest.mark.parametrize("bc,theory", [("clamped", "classic"), ("soft_hinge", "ktn_full")])
def test_unsupported_contact_analytic_rejected(tmp_path, bc, theory):
    """Прочий контакт (не круг+шарнир+классика) + analytic — понятный отказ."""
    p = tmp_path / "case.toml"
    p.write_text(_case(bc=bc, theory=theory), encoding="utf-8")
    with pytest.raises(CaseError, match="инвариант"):
        resolve_reference(Problem.from_toml(str(p)))
