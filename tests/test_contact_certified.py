"""Абсолютный эталон контактной задачи: круг + основание, кромка soft_hinge/clamped.

Сертифицированное осесимметричное решение Кирхгофа (analytic_auto.
axisym_contact_solution) подключено в resolver — контакт круга гейтится
АБСОЛЮТНЫМ эталоном, а не только инвариантами. Гейт — прогиб ВНЕ контактной
зоны (``w_max = Δ`` тривиален). Покрытие (v0.6.4): теория ``classic`` × кромка
``soft_hinge``|``clamped`` (защемление ЖЁСТЧЕ: ``w_free(0)=q₀a⁴/64D``).

``ktn_linear`` в контакте эталона НЕ имеет: КТН-поправка входит в лицевое
условие Синьорини (реакция гонится прогибом нижней лицевой), поэтому базовое
поле отклоняется от кирхгофовского с ростом толщины — классический эталон
годился бы лишь в тонком пределе (ложно-положителен). ``ktn_full`` и прочий
контакт — тоже ворота инвариантов; резолвер отклоняет `analytic` понятной
ошибкой.
"""

from __future__ import annotations

import pytest

from plate_solver import dispatch
from plate_solver.problem import CaseError, Problem
from plate_solver.references import resolve_reference, verify_result

_D = 2.1e6 * 0.06**3 / (12 * (1 - 0.3**2))
_W_FREE0_SOFT = 3 * 4.0 * 1.0**4 / (64 * _D)             # прогиб центра свободного круга (шарнир)

# зазор, заведомо лежащий в (0, w_free(0)) для каждой кромки (защемление ЖЁСТЧЕ)
_GAP = {"soft_hinge": "2.0e-3", "clamped": "0.7e-3"}


def _case(bc="soft_hinge", theory="classic", ref="analytic", gap=None):
    gap = _GAP[bc] if gap is None else gap
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
p = 10
Q = 160
grid_n = 16
[verify]
reference = "{ref}"
tol = 2e-2
"""


def _solve(tmp_path, **kw):
    p = tmp_path / "case.toml"
    p.write_text(_case(**kw), encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


@pytest.mark.parametrize("bc", ["soft_hinge", "clamped"])
def test_certified_contact_matrix_gates(tmp_path, bc):
    """Круг+основание (classic), обе кромки: прогиб вне зоны совпал с эталоном Кирхгофа."""
    res = _solve(tmp_path, bc=bc, theory="classic")
    rep = verify_result(res)
    assert rep.ok                                        # гейт пройден
    (row,) = rep.rows
    assert row.gated and row.rel < 2e-2                  # абсолютный эталон, не инвариант
    assert "Кирхгоф" in row.name and bc in row.name


def test_certified_contact_gap_out_of_range_rejected(tmp_path):
    """Зазор вне (0, w_free(0)) — эталон не существует, понятный отказ."""
    p = tmp_path / "case.toml"
    p.write_text(_case(gap=f"{2 * _W_FREE0_SOFT:.6e}"), encoding="utf-8")   # Δ > w_free(0)
    with pytest.raises(CaseError, match="w_free"):
        resolve_reference(Problem.from_toml(str(p)))


def test_clamped_gap_valid_only_for_softer_hinge(tmp_path):
    """Зазор 2e-3 годен для шарнира, но ВЫШЕ w_free(0) защемлённого круга — отказ."""
    p = tmp_path / "case.toml"
    p.write_text(_case(bc="clamped", gap="2.0e-3"), encoding="utf-8")
    with pytest.raises(CaseError, match="w_free"):
        resolve_reference(Problem.from_toml(str(p)))


@pytest.mark.parametrize("bc,theory", [("soft_hinge", "ktn_linear"),
                                       ("clamped", "ktn_linear"),
                                       ("soft_hinge", "ktn_full"),
                                       ("clamped", "ktn_full")])
def test_unsupported_contact_analytic_rejected(tmp_path, bc, theory):
    """ktn_linear/ktn_full в контакте эталона не имеют — analytic отклонён (инварианты).

    Для ``ktn_linear`` классический эталон был бы ложно-положителен лишь в тонком
    пределе (реакция гонится лицевым прогибом, отклонение растёт с толщиной).
    """
    p = tmp_path / "case.toml"
    p.write_text(_case(bc=bc, theory=theory), encoding="utf-8")
    with pytest.raises(CaseError, match="инвариант"):
        resolve_reference(Problem.from_toml(str(p)))
