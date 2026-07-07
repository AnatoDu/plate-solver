r"""Ворота нелинейного контакта МОР+КТН (contact_nl.py, N3 v0.6.0, §4, §7).

Независимых эталонов у «нелинейный контакт + КТН» нет → проверяем ВЫРОЖДЕНИЕМ
(точным по единой модели §3) и физикой:

* R1 (контакт выкл): зазор → ∞ ⇒ ``r ≡ 0`` ⇒ свободное решение КТН v0.5;
* R3 (уточнение выкл): полная КТН при ``refinement_scale=0`` = кармановский
  контакт (тот же путь, машинная точность);
* физика: контакт ограничивает прогиб лицевой до зазора, реакция ``r > 0`` в
  зоне; подпись КТН — зона/пик реакции при ``ktn_full`` отличаются от ``karman``.

Вложенная схема — ЭТАЛОН корректности (§4.2), дорогая (МОР × полный нелинейный
решатель); тяжёлые случаи — маркер ``big``. Совмещённая (быстрая, T7) — веха N4.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import theory
from plate_solver.config import Config
from plate_solver.contact_nl import NonlinearContactMOR
from plate_solver.geometry import make_circle
from plate_solver.ktn_solver import KTNSolver

_DOM = make_circle(1.0)


def _cfg(q0=0.01, Q=64, beta=1.5, max_iter=2500, tol=3e-4):
    return Config(E=1.0, h=0.2, nu=0.3, a=1.0, q0=q0, p=8, Q=Q,
                  n_load_steps=3, karman_tol=1e-6, karman_max_iter=100,
                  beta=beta, max_iter=max_iter, tol=tol)


def _solver(cfg, name="ktn_full"):
    return KTNSolver.from_theory_name(_DOM, cfg, name, bc_type="clamped",
                                      inplane_bc="immovable")


def test_r1_contact_off_reduces_to_free_ktn():
    """R1: зазор → ∞ ⇒ нет контакта ⇒ r≡0, w = свободное решение КТН (машинно)."""
    cfg = _cfg()
    s = _solver(cfg)
    free = s.solve_uniform()
    res = NonlinearContactMOR(s, cfg, gap=100.0, scheme="nested").solve()
    assert res.r_max == 0.0 and res.n_contact == 0
    assert abs(res.w_max - free.w_max) / free.w_max < 1e-6


def test_face_condition_curvature_scales_with_theory():
    """Лицевое условие u_c = w + c_curv·Δw: c_curv=0 для karman, >0 для ktn_full (§4.1)."""
    assert theory.karman().face_curv_coeff == 0.0
    assert theory.classic().face_curv_coeff == 0.0
    assert theory.ktn_full(0.3, 0.2).face_curv_coeff != 0.0


@pytest.mark.big
def test_ktn_contact_limits_face_deflection():
    """Физика: контакт ограничивает ПРОГИБ до зазора; реакция r>0 в зоне."""
    cfg = _cfg()
    s = _solver(cfg)
    gap = 0.5 * s.solve_uniform().w_max
    res = NonlinearContactMOR(s, cfg, gap=gap, scheme="nested").solve()
    assert res.r_max > 0.0 and res.n_contact > 0        # контакт состоялся
    # лицевой прогиб в зоне контакта не превышает зазор (сверх допуска МОР)
    assert np.max(res.u_c_nodes[res.contact_mask]) <= gap * 1.02


@pytest.mark.big
def test_r3_refinement_zero_equals_karman_contact():
    """R3: полная КТН при refinement→0 = кармановский контакт (машинная точность)."""
    cfg = _cfg()
    gap = 0.5 * _solver(cfg, "karman").solve_uniform().w_max
    # karman-контакт
    rk = NonlinearContactMOR(_solver(cfg, "karman"), cfg, gap=gap, scheme="nested").solve()
    # ktn_full с уточнением, снятым до нуля (α=0) — тот же путь, что karman
    p0 = theory.ktn_full(0.3, cfg.h).with_refinement_scale(0.0)
    s0 = KTNSolver.from_config(_DOM, cfg, p0, bc_type="clamped", inplane_bc="immovable")
    r0 = NonlinearContactMOR(s0, cfg, gap=gap, scheme="nested").solve()
    assert abs(r0.w_max - rk.w_max) / rk.w_max < 1e-10
    assert r0.n_contact == rk.n_contact


@pytest.mark.big
def test_ktn_signature_in_contact():
    """Подпись КТН: зона/пик реакции ktn_full отличаются от karman (сглаживание §6.1)."""
    cfg = _cfg()
    gap = 0.5 * _solver(cfg, "karman").solve_uniform().w_max
    rk = NonlinearContactMOR(_solver(cfg, "karman"), cfg, gap=gap, scheme="nested").solve()
    rf = NonlinearContactMOR(_solver(cfg, "ktn_full"), cfg, gap=gap, scheme="nested").solve()
    # лицевая кривизна КТН меняет контактную границу ⇒ пик и/или число узлов иные
    assert (rf.n_contact != rk.n_contact) or (abs(rf.r_max - rk.r_max) > 1e-3 * rk.r_max)
