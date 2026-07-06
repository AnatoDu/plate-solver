"""Верификация изгиба на круге (тест-ворота шага; эталонная таблица).

Реализованная модель — «мягкий шарнир» (M=0 на ∂Ω). Её ТОЧНОЕ решение на круге:
    w(ρ) = q (a² − ρ²)(3a² − ρ²) / (64 D)              (analytic.circular_plate_soft_hinge)
— это формула (4.2) с множителем (5+ν)/(1+ν) → 3, т.е. (4.2) при ν=1.

Поэтому верификация двухстрочная:
  • ЧИСЛЕННЫЙ метод ↔ аналитика мягкого шарнира — ошибка мала (квадратурный
    остаток ~O(1/Q)), падает с Q; базис должен быть достаточным (p ≥ 2);
  • аналитика мягкого шарнира ↔ Кирхгоф (4.2) — модельная погрешность
    1 − 3(1+ν)/(5+ν) ≈ 26.4 % при ν=0.3 (∝ (1−ν)·кривизна; NOTES.md §8).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import analytic, geometry
from plate_solver import quadrature as quad
from plate_solver.config import Config
from plate_solver.plate import PlateBending

A, NU, QLOAD = 1.0, 0.3, 4.0
DOM = geometry.make_circle(A)
_REF = quad.interior_nodes(DOM, 300)
_RHO = np.hypot(_REF.x, _REF.y)


def _cfg(p: int, Q: int) -> Config:
    return Config(a=A, q0=QLOAD, nu=NU, h=1.0, E=2.1e6, p=p, Q=Q)


def _solve(p: int, Q: int):
    cfg = _cfg(p, Q)
    pb = PlateBending.from_config(DOM, cfg)
    _, cw = pb.solve_uniform(QLOAD)
    return pb, cw, cfg.D


def _wmax_err(p: int, Q: int) -> float:
    pb, cw, D = _solve(p, Q)
    w0 = float(pb.deflection(cw, 0.0, 0.0))
    w_soft = analytic.circular_plate_soft_hinge_wmax(QLOAD, A, D)
    return abs(w0 - w_soft) / w_soft


def _field_l2(p: int, Q: int) -> float:
    pb, cw, D = _solve(p, Q)
    wn = pb.deflection(cw, _REF.x, _REF.y)
    we = analytic.circular_plate_soft_hinge(_RHO, QLOAD, A, D)
    return float(np.sqrt((_REF.w * (wn - we) ** 2).sum()) / np.sqrt((_REF.w * we**2).sum()))


def test_soft_hinge_reference_and_model_ratio():
    D = _cfg(2, 128).D
    w_soft0 = analytic.circular_plate_soft_hinge(0.0, QLOAD, A, D)
    assert w_soft0 == pytest.approx(analytic.circular_plate_soft_hinge_wmax(QLOAD, A, D))
    assert analytic.circular_plate_soft_hinge(A, QLOAD, A, D) == pytest.approx(0.0)
    # Документированное отношение к (4.2): w_soft / w_SS = 3(1+ν)/(5+ν).
    w_ss0 = analytic.circular_plate_simply_supported(0.0, QLOAD, A, NU, D)
    assert w_soft0 / w_ss0 == pytest.approx(3 * (1 + NU) / (5 + NU))


def test_sign_and_max_at_center():
    pb, cw, _ = _solve(6, 256)
    w0 = float(pb.deflection(cw, 0.0, 0.0))
    assert w0 > 0.0                                   # q>0 ⇒ w>0 (NOTES.md §0)
    for x, y in [(0.5, 0.0), (0.0, 0.5), (0.3, 0.4)]:  # максимум — в центре
        assert w0 >= float(pb.deflection(cw, x, y))


@pytest.mark.big
def test_gate_plate_circle_wmax_against_soft_hinge():
    """ГЛАВНЫЕ ВОРОТА: численный w_max ↔ аналитика мягкого шарнира < 0.1 %."""
    assert _wmax_err(6, 1280) < 1.0e-3
    assert _field_l2(6, 1280) < 2.0e-3


def test_basis_must_resolve_solution():
    # Решение — степени 2; p=1 его не представляет ⇒ ошибка велика, p=6 — мала.
    assert _wmax_err(1, 256) > 10.0 * _wmax_err(6, 256)


def test_error_decreases_with_Q():
    # Квадратурный остаток падает с ростом Q (~O(1/Q)).
    errs = [_field_l2(6, Q) for Q in (128, 256, 512)]
    assert errs[0] > errs[1] > errs[2], errs


@pytest.mark.big
def test_model_gap_vs_kirchhoff_documented():
    # Вторая строка Таблицы 4.1: численный ≈ мягкий, и ~26.4 % ниже (4.2).
    pb, cw, D = _solve(6, 1024)
    w0 = float(pb.deflection(cw, 0.0, 0.0))
    w_ss = float(analytic.circular_plate_simply_supported(0.0, QLOAD, A, NU, D))
    rel_gap = abs(w0 - w_ss) / w_ss
    assert rel_gap == pytest.approx(1 - 3 * (1 + NU) / (5 + NU), abs=2e-3)
