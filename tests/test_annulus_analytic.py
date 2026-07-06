"""Аналитика кольца и 1D-Ритц на [b, a].

Ворота: (i) sympy-подстановка w(r) в D·ΔΔw = q — невязка нулевая;
(ii) краевые условия выполняются до 1e-12; (iii) 1D-Ритц на кольце
против аналитики — rel < 1e-8 при p = 16 (ln r аналитичен на [b, a], b > 0).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import analytic
from plate_solver.config import Config
from plate_solver.radial import RadialClampedAnnulus, solve_radial_soft_hinge_annulus

A, B, Q0, NU = 1.0, 0.4, 4.0, 0.3
D = Config(q0=Q0).D                                   # жёсткость дефолтной физики


# --------------------------------------------------------------------------- #
#  (i) подстановка в уравнение: D·ΔΔw − q ≡ 0 (символьно)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bc", analytic.ANNULUS_BCS)
def test_annulus_satisfies_biharmonic(bc):
    import sympy as sp

    r = sp.symbols("r", positive=True)
    C = analytic._annulus_coeffs(A, B, Q0, D, bc, NU)
    L = sp.log(r / A)
    w = Q0 * r**4 / (64 * D) + C[0] + C[1] * r**2 + C[2] * L + C[3] * r**2 * L

    def lap(f):
        return sp.diff(f, r, 2) + sp.diff(f, r) / r

    residual = sp.simplify(D * lap(lap(w)) - Q0)
    # Однородные члены бигармоничны структурно; float-коэффициенты оставляют
    # крошки ~1e-16/r⁴ — проверяем машинный нуль по точкам кольца.
    if residual != 0:
        f = sp.lambdify(r, residual, "numpy")
        pts = np.linspace(B, A, 101)
        assert float(np.max(np.abs(f(pts)))) < 1e-10 * Q0


# --------------------------------------------------------------------------- #
#  (ii) краевые условия до 1e-12 (обе окружности)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("bc", analytic.ANNULUS_BCS)
def test_annulus_boundary_conditions(bc):
    import sympy as sp

    r = sp.symbols("r", positive=True)
    C = analytic._annulus_coeffs(A, B, Q0, D, bc, NU)
    L = sp.log(r / A)
    w = Q0 * r**4 / (64 * D) + C[0] + C[1] * r**2 + C[2] * L + C[3] * r**2 * L
    wp = sp.diff(w, r)
    lap = sp.diff(w, r, 2) + wp / r
    mr = sp.diff(w, r, 2) + NU * wp / r               # M_r = 0 ⇔ w'' + ν w'/r = 0
    second = {"clamped": wp, "soft": lap, "true_ss": mr}[bc]

    scale = analytic.annulus_uniform_wmax(A, B, Q0, D, bc, NU)
    for r0 in (A, B):
        assert abs(float(w.subs(r, r0))) < 1e-12 * scale
    second_scale = max(abs(float(second.subs(r, 0.5 * (A + B)))), scale)
    for r0 in (A, B):
        assert abs(float(second.subs(r, r0))) < 1e-12 * second_scale


def test_annulus_soft_vs_true_ss_model_gap():
    """soft ≠ true_ss при ν ≠ 1 (обе границы криволинейны) — материал model_gap."""
    w_soft = analytic.annulus_uniform_wmax(A, B, Q0, D, "soft", NU)
    w_ss = analytic.annulus_uniform_wmax(A, B, Q0, D, "true_ss", NU)
    assert abs(w_soft - w_ss) / w_ss > 0.05           # разрыв модельный, не численный


# --------------------------------------------------------------------------- #
#  (iii) ворота: 1D-Ритц на [b, a] против аналитики, rel < 1e-8 (p = 16)
# --------------------------------------------------------------------------- #
def test_gate_radial_annulus_clamped_vs_analytic():
    solver = RadialClampedAnnulus(A, B, D, p=16, nq=400)
    solver.solve(Q0, NU)
    r = np.linspace(B, A, 501)
    w_1d = solver.deflection(r)
    w_ex = analytic.annulus_uniform(r, A, B, Q0, D, "clamped", NU)
    rel = float(np.max(np.abs(w_1d - w_ex)) / np.max(np.abs(w_ex)))
    assert rel < 1e-8, rel


def test_gate_radial_annulus_soft_vs_analytic():
    rp, cw = solve_radial_soft_hinge_annulus(A, B, D, Q0, p=16, nq=400)
    r = np.linspace(B, A, 501)
    w_1d = rp.deflection(cw, r)
    w_ex = analytic.annulus_uniform(r, A, B, Q0, D, "soft", NU)
    rel = float(np.max(np.abs(w_1d - w_ex)) / np.max(np.abs(w_ex)))
    assert rel < 1e-8, rel


def test_radial_annulus_validates_radii():
    with pytest.raises(ValueError, match="0 < b < a"):
        RadialClampedAnnulus(1.0, 1.5, D)
    with pytest.raises(ValueError, match="0 < b < a"):
        analytic.annulus_uniform(0.5, 1.0, 0.0, Q0, D)
