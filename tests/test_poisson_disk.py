"""Верификация решателя Пуассона на диске (тест-ворота шага).

Эталон: −Δv = 1 на единичном круге, v|∂Ω = 0  ⇒  v = (1 − x² − y²)/4
(analytic.disk_poisson_unit). Точное решение лежит в структуре ``v = ω·Φ``
(здесь ω = (1−x²−y²)/2, значит Φ ≡ 1/2), поэтому при ТОЧНОМ интегрировании
решение Ритца воспроизводится почти идеально — остаточная погрешность задаётся
СТУПЕНЧАТОЙ маской квадратуры (~O(1/Q), NOTES.md §4). Отсюда: для <0.1% нужен
большой Q при умеренном p.

Ворота: отн. L²-погрешность < 0.1 % (умеренный p, большой Q) и монотонно падает
как с ростом p, так и с ростом Q.
"""

from __future__ import annotations

import numpy as np

from plate_solver import analytic, geometry
from plate_solver import basis as B
from plate_solver import quadrature as quad
from plate_solver.poisson import PoissonSolver

DOM = geometry.make_circle(1.0)
# Независимая (плотная) эталонная квадратура для оценки L²-нормы погрешности.
_REF = quad.interior_nodes(DOM, 256)
_VEX = analytic.disk_poisson_unit(_REF.x, _REF.y)
_DENOM = float(np.sqrt((_REF.w * _VEX**2).sum()))


def _rel_l2(p: int, Q: int) -> float:
    """Отн. L²-погрешность решения −Δv=1 на диске для (p, Q)."""
    sol = PoissonSolver(DOM, B.ChebyshevBasis(p, DOM.bbox), quad.interior_nodes(DOM, Q))
    c = sol.solve(np.ones(sol.quad.x.size))
    vnum = sol.evaluate(c, _REF.x, _REF.y)
    return float(np.sqrt((_REF.w * (vnum - _VEX) ** 2).sum())) / _DENOM


def test_disk_poisson_unit_reference():
    # Эталон самосогласован: центр = 1/4, граница = 0.
    assert analytic.disk_poisson_unit(0.0, 0.0) == 0.25
    assert abs(float(analytic.disk_poisson_unit(1.0, 0.0))) < 1e-15
    assert abs(float(analytic.disk_poisson_unit(0.6, 0.8))) < 1e-15  # точка на r=1


def test_A_symmetric_positive_definite():
    sol = PoissonSolver(DOM, B.ChebyshevBasis(6, DOM.bbox), quad.interior_nodes(DOM, 128))
    assert np.allclose(sol.A, sol.A.T)
    assert np.linalg.eigvalsh(sol.A).min() > 0.0   # положительно определена


def test_solution_sign_and_center():
    # Контроль знака/масштаба: v > 0 внутри, v(0,0) ≈ 1/4 (в пределах 1 %).
    sol = PoissonSolver(DOM, B.ChebyshevBasis(4, DOM.bbox), quad.interior_nodes(DOM, 256))
    c = sol.solve(np.ones(sol.quad.x.size))
    v0 = float(sol.evaluate(c, 0.0, 0.0))
    assert v0 > 0.0
    assert abs(v0 - 0.25) / 0.25 < 1e-2


def test_gate_poisson_disk_l2_below_0p1pct():
    """ГЛАВНЫЕ ВОРОТА: отн. L²-погрешность < 0.1 % при умеренном p и большом Q."""
    assert _rel_l2(p=4, Q=1280) < 1.0e-3


def test_error_decreases_with_p():
    # При фиксированном Q погрешность падает с ростом p.
    errs = [_rel_l2(p, 160) for p in (2, 6, 12)]
    assert errs[0] > errs[1] > errs[2], errs


def test_error_decreases_with_Q():
    # При фиксированном p погрешность падает с ростом Q (~O(1/Q)).
    errs = [_rel_l2(4, Q) for Q in (64, 128, 256)]
    assert errs[0] > errs[1] > errs[2], errs


# --------------------------------------------------------------------------- #
#  Кэш матриц структуры: та же арифметика ⇒ результат равен БИТ-В-БИТ
# --------------------------------------------------------------------------- #
def test_cache_fields_bitwise_identity():
    """Кэш не меняет ни A, ни коэффициентов, ни полей — равенство бит-в-бит.

    Обоснование: psiW = ψ·diag(w) и Φ, ω кэшируются как есть; b = psiW@f и
    v = ω·(Φᵀc) повторяют порядок округления прежних assemble_load/evaluate.
    """
    basis6 = B.ChebyshevBasis(6, DOM.bbox)
    qn = quad.interior_nodes(DOM, 96)
    f = 1.0 + 0.3 * qn.x - 0.2 * qn.y**2           # неоднородная правая часть
    on = PoissonSolver(DOM, basis6, qn, cache_fields=True)
    off = PoissonSolver(DOM, basis6, qn, cache_fields=False)
    assert on.cache_fields and not off.cache_fields
    assert np.array_equal(on.A, off.A)
    c_on, c_off = on.solve(f), off.solve(f)
    assert np.array_equal(c_on, c_off)
    # evaluate_at_quad ≡ evaluate в узлах квадратуры (кэш-ветка и фолбэк)
    assert np.array_equal(on.evaluate_at_quad(c_on), on.evaluate(c_on, qn.x, qn.y))
    assert np.array_equal(off.evaluate_at_quad(c_off), off.evaluate(c_off, qn.x, qn.y))


def test_cache_fields_auto_threshold(monkeypatch):
    """Автовыбор: кэш включён при N·M ≤ CACHE_NM_MAX и выключен выше порога."""
    import plate_solver.poisson as P

    basis4 = B.ChebyshevBasis(4, DOM.bbox)
    qn = quad.interior_nodes(DOM, 64)
    assert PoissonSolver(DOM, basis4, qn).cache_fields           # N·M мало ⇒ кэш
    monkeypatch.setattr(P, "CACHE_NM_MAX", 10)                   # имитация «большой» квадратуры
    sol = PoissonSolver(DOM, basis4, qn)
    assert not sol.cache_fields                                  # авто-выключение
    c = sol.solve(np.ones(qn.x.size))
    assert np.array_equal(sol.evaluate_at_quad(c), sol.evaluate(c, qn.x, qn.y))
