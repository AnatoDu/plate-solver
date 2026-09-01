r"""Ворота ортотропии классической теории — [model.orthotropy] (v0.7.0).

Эталоны:

* СИЛЬНЫЙ: SSSS ортотропный прямоугольник под равномерной q — ряд Навье
  (Лехницкий «Анизотропные пластинки»; Тимошенко–Войновский-Кригер гл. 11):
  ``w_mn ~ 1/[mn·(D11(m/a)⁴ + 2(D12+2D66)(m/a)²(n/b)² + D22(n/b)⁴)]``;
  измерено rel(w) 1.7e-10 при p=14 (спектральная сходимость);
* MMS: полиномиальное манифактурное решение ЛЕЖИТ в пространстве Ритца
  mixed-all-clamped ⇒ машинная точность (измерено 1.6e-14);
* изотропная редукция D11=D22=D, D12=νD, D66=(1−ν)D/2 — почленно
  тождественна полной форме NOTES §20 (прямоугольник ~6e-11; круг —
  квадратурный след нуль-лагранжиана маски, убывает с Q);
* собственные задачи: ω_mn и N_cr Лехницкого/Лейссы (rel ≤ 2e-4).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.ladder import navier_uniform_center, navier_uniform_center_ortho
from plate_solver.problem import CaseError, Problem

_A, _B, _Q0 = 2.0, 1.2, 1.0
_DM = (2.0, 0.5, 1.0, 0.4)                      # D11, D12, D22, D66


def _ortho_model(Dm=_DM):
    return {"theory": "classic", "h": 1.0,
            "orthotropy": {"D11": Dm[0], "D12": Dm[1], "D22": Dm[2],
                           "D66": Dm[3]}}


def _ssss_case(p=14, Q=64, Dm=_DM, a=_A, b=_B):
    return {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": a, "y1": 0.0, "y2": b},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]},
        "load": {"type": "uniform", "q0": _Q0},
        "model": _ortho_model(Dm),
        "discretization": {"p": p, "Q": Q, "grid_n": 32},
        "verify": {"reference": "none"},
    }


def test_navier_ortho_series_reduces_to_isotropic():
    """Изотропная редукция ряда: D_mn = D·(m²/a²+n²/b²)² алгебраически."""
    D, nu = 1.7, 0.3
    wi, _ = navier_uniform_center_ortho(_A, _B, (D, nu * D, D, (1 - nu) * D / 2),
                                        _Q0, n_terms=50)
    wu, _ = navier_uniform_center(_A, _B, D, _Q0)
    assert abs(wi - wu) / wu < 1e-12


def test_ssss_rect_ortho_navier():
    """SSSS ортотроп против ряда Лехницкого: w (1.7e-10) и Mx (5.0e-8)."""
    w_ref, mx_ref = navier_uniform_center_ortho(_A, _B, _DM, _Q0)
    res = dispatch.solve(Problem.from_dict(_ssss_case()))
    w_c = float(res._plate.deflection(res._c, np.array([_A / 2]),
                                      np.array([_B / 2]))[0])
    mx_c, _, _ = res._plate.moments_at(res._c, np.array([_A / 2]),
                                       np.array([_B / 2]))
    assert abs(w_c - w_ref) / w_ref < 1e-8
    assert abs(float(mx_c[0]) - mx_ref) / abs(mx_ref) < 1e-6


def test_clamped_ortho_mms():
    """MMS: w_ex = (x²−1)²(y²−0.36)² ∈ пространству Ритца ⇒ машинно.

    q = D11·w_xxxx + 2(D12+2D66)·w_xxyy + D22·w_yyyy — полином, подаётся
    нагрузкой-выражением (expr); измерено 1.6e-14 уже при p=4.
    """
    import sympy as sp

    x, y = sp.symbols("x y")
    w_ex = (x**2 - 1) ** 2 * (y**2 - sp.Rational(36, 100)) ** 2
    D11, D12, D22, D66 = _DM
    q = (D11 * sp.diff(w_ex, x, 4)
         + 2 * (D12 + 2 * D66) * sp.diff(w_ex, x, 2, y, 2)
         + D22 * sp.diff(w_ex, y, 4))
    d = {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0,
                     "y1": -0.6, "y2": 0.6},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "clamped"}, {"side": "x2", "type": "clamped"},
            {"side": "y1", "type": "clamped"}, {"side": "y2", "type": "clamped"}]},
        "load": {"type": "expr", "q0": 1.0, "expr": str(sp.expand(q))},
        "model": _ortho_model(),
        "discretization": {"p": 6, "Q": 32, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(d))
    xs = np.array([0.0, 0.3, -0.5])
    ys = np.array([0.0, -0.2, 0.3])
    w_num = np.asarray(res._plate.deflection(res._c, xs, ys), float)
    f_ex = sp.lambdify((x, y), w_ex, "numpy")
    w_exact = np.asarray(f_ex(xs, ys), float)
    assert np.max(np.abs(w_num - w_exact)) / np.max(np.abs(w_exact)) < 1e-12


def test_clamped_ortho_iso_reduction_rect():
    """Изотропная редукция орто-пути защемления (прямоугольник): ~6e-11."""
    D, nu = 1.7, 0.3
    base = {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": _A, "y1": 0.0,
                     "y2": _B},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": _ortho_model((D, nu * D, D, (1 - nu) * D / 2)),
        "discretization": {"p": 12, "Q": 64, "grid_n": 32},
        "verify": {"reference": "none"},
    }
    iso = copy.deepcopy(base)
    iso["model"] = {"theory": "classic", "E": 12 * (1 - nu**2) * D, "nu": nu,
                    "h": 1.0}
    r_o = dispatch.solve(Problem.from_dict(base))
    r_i = dispatch.solve(Problem.from_dict(iso))
    assert abs(r_o.w_max - r_i.w_max) / r_i.w_max < 1e-9


def test_clamped_ortho_iso_reduction_circle():
    """Круг: редукция сходится КВАДРАТУРНО (след нуль-лагранжиана маски)."""
    D, nu = 1.7, 0.3
    rel = {}
    for Q in (64, 128):
        base = {
            "geometry": {"kind": "circle", "a": 1.0},
            "bc": {"type": "clamped"},
            "load": {"type": "uniform", "q0": 4.0},
            "model": _ortho_model((D, nu * D, D, (1 - nu) * D / 2)),
            "discretization": {"p": 10, "Q": Q, "grid_n": 32},
            "verify": {"reference": "none"},
        }
        iso = copy.deepcopy(base)
        iso["model"] = {"theory": "classic", "E": 12 * (1 - nu**2) * D,
                        "nu": nu, "h": 1.0}
        r_o = dispatch.solve(Problem.from_dict(base))
        r_i = dispatch.solve(Problem.from_dict(iso))
        rel[Q] = abs(r_o.w_max - r_i.w_max) / r_i.w_max
    assert rel[128] < rel[64]                      # квадратурная природа
    assert rel[128] < 1e-5                         # измерено ~9e-7


def test_clamped_ortho_symmetry():
    """x↔y при D11↔D22 на квадрате — тот же w_max (измерено 1.2e-16)."""
    sq = {"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0}
    d1 = _ssss_case(p=10, Q=48, Dm=(2.0, 0.5, 1.0, 0.4), a=1.0, b=1.0)
    d1["geometry"] = sq
    d2 = _ssss_case(p=10, Q=48, Dm=(1.0, 0.5, 2.0, 0.4), a=1.0, b=1.0)
    d2["geometry"] = sq
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-12


def _omega_mn(m, n, a, b, Dm, rho_h=1.0):
    D11, D12, D22, D66 = Dm
    Dmn = (D11 * (m / a) ** 4
           + 2.0 * (D12 + 2.0 * D66) * (m / a) ** 2 * (n / b) ** 2
           + D22 * (n / b) ** 4)
    return np.pi**2 * np.sqrt(Dmn / rho_h)


def test_eigen_ortho_vibration_ssss():
    """Частоты SSSS ортотропа: ω_mn = π²√(D_mn/ρh) (rel ≤ 1e-4)."""
    d = _ssss_case(p=12, Q=64)
    del d["load"]
    d["bc"] = {"type": "soft_hinge"}
    d["eigen"] = {"kind": "vibration", "n_modes": 3}
    res = dispatch.solve(Problem.from_dict(d))
    exact = np.sort([_omega_mn(m, n, _A, _B, _DM)
                     for m in range(1, 5) for n in range(1, 5)])[:3]
    for w_num, w_ex in zip(res.eigen.values, exact, strict=False):
        assert abs(w_num - w_ex) / w_ex < 1e-4


def test_eigen_ortho_buckling_ssss():
    """Одноосное выпучивание SSSS: N_cr Лехницкого/Лейссы (rel ≤ 2e-4)."""
    d = _ssss_case(p=12, Q=64)
    del d["load"]
    d["bc"] = {"type": "soft_hinge"}
    d["eigen"] = {"kind": "buckling", "n_modes": 1}
    res = dispatch.solve(Problem.from_dict(d))
    D11, D12, D22, D66 = _DM
    lam_exact = min(
        np.pi**2 / _B**2 * (D11 * (m * _B / _A) ** 2 + 2.0 * (D12 + 2.0 * D66)
                            + D22 * (_A / (m * _B)) ** 2)
        for m in range(1, 8))
    assert abs(res.eigen.values[0] - lam_exact) / lam_exact < 2e-4


def test_ortho_engineering_constants():
    """Инженерный набор = прямой D-ввод (конверсия в to_config, машинно)."""
    D11, D12, D22, D66 = _DM
    h = 1.0
    # обратная конверсия: k = 1 − ν_xy·ν_yx; ν_xy = D12/D22 ... из формул:
    # D12 = ν_xy·Ey·h³/(12k), D22 = Ey·h³/(12k) ⇒ ν_xy = D12/D22
    nu_xy = D12 / D22
    # D11/D22 = Ex/Ey; k = 1 − ν_xy²·Ey/Ex
    ratio = D11 / D22
    k = 1.0 - nu_xy**2 / ratio
    Ey = D22 * 12.0 * k / h**3
    Ex = ratio * Ey
    Gxy = D66 * 12.0 / h**3
    d1 = _ssss_case(p=10, Q=48)
    d2 = _ssss_case(p=10, Q=48)
    d2["model"] = {"theory": "classic", "h": h,
                   "orthotropy": {"Ex": Ex, "Ey": Ey, "nu_xy": nu_xy,
                                  "Gxy": Gxy}}
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-12


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(bc={"type": "soft_hinge"}), "bc.type"),
    (lambda d: d["model"].update(theory="karman"), "model.orthotropy"),
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5}),
     "contact.enabled"),
    (lambda d: d.update(verify={"reference": "analytic"}), "verify.reference"),
    (lambda d: d["model"].update(E=1e6), "model.E"),
    (lambda d: d["model"]["orthotropy"].update(D12=2.0), "D12"),
    (lambda d: d["model"]["orthotropy"].update(Ex=1.0), "один набор"),
])
def test_ortho_guards(mutate, match):
    d = _ssss_case(p=8, Q=32)
    d["bc"] = {"type": "clamped"}
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_ortho_mixed_free_side_rejected():
    d = _ssss_case(p=8, Q=32)
    d["bc"]["sides"][2] = {"side": "y1", "type": "free"}
    with pytest.raises(CaseError, match="free"):
        Problem.from_dict(d)


def test_ortho_winkler_mms():
    """Ортотропия + Винклер (v0.7.0): MMS машинно — свёртка +k_w∫ψψ
    конститутив-независима и складывается с S_ortho по построению (2.8e-15)."""
    import sympy as sp

    x, y = sp.symbols("x y")
    kw = 300.0
    w_ex = (x**2 - 1) ** 2 * (y**2 - sp.Rational(36, 100)) ** 2
    D11, D12, D22, D66 = _DM
    q = (D11 * sp.diff(w_ex, x, 4)
         + 2 * (D12 + 2 * D66) * sp.diff(w_ex, x, 2, y, 2)
         + D22 * sp.diff(w_ex, y, 4) + kw * w_ex)
    d = {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0,
                     "y1": -0.6, "y2": 0.6},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "clamped"}, {"side": "x2", "type": "clamped"},
            {"side": "y1", "type": "clamped"}, {"side": "y2", "type": "clamped"}]},
        "load": {"type": "expr", "q0": 1.0, "expr": str(sp.expand(q))},
        "model": {**_ortho_model(), "winkler": kw},
        "discretization": {"p": 6, "Q": 32, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(d))
    xs = np.array([0.0, 0.3, -0.5])
    ys = np.array([0.0, -0.2, 0.3])
    w_num = np.asarray(res._plate.deflection(res._c, xs, ys), float)
    f_ex = sp.lambdify((x, y), w_ex, "numpy")
    w_exact = np.asarray(f_ex(xs, ys), float)
    assert np.max(np.abs(w_num - w_exact)) / np.max(np.abs(w_exact)) < 1e-12


def test_eigen_ortho_winkler_shift():
    """Частоты SSSS ортотропа на основании: ω² = π⁴·D_mn + k_w (rel ≤ 1e-4)."""
    kw = 300.0
    d = _ssss_case(p=12, Q=64)
    del d["load"]
    d["bc"] = {"type": "soft_hinge"}
    d["model"]["winkler"] = kw
    d["eigen"] = {"kind": "vibration", "n_modes": 3}
    res = dispatch.solve(Problem.from_dict(d))
    exact = np.sort([np.sqrt(_omega_mn(m, n, _A, _B, _DM) ** 2 + kw)
                     for m in range(1, 5) for n in range(1, 5)])[:3]
    for w_num, w_ex in zip(res.eigen.values, exact, strict=False):
        assert abs(w_num - w_ex) / w_ex < 1e-4


def test_ortho_supports_sherman_morrison():
    """Опоры на ортотропном операторе: тождество Шермана–Моррисона машинно.

    R = k·w_free(P)/(1 + k·G_N(P, P)) оператор-независимо (измерено 2.6e-8 —
    обратный ход той же факторизации).
    """
    from plate_solver.clamped import ClampedPlate
    from plate_solver.config import Config
    from plate_solver.geometry import make_circle

    P = (0.3, 0.2)
    k = 1e4 * 192307.7
    Dm = (2e5, 5e4, 1e5, 4e4)
    dom = make_circle(1.0)
    free = ClampedPlate.from_config(dom, Config(p=12, Q=64, ortho_D=Dm))
    c_free = free.solve_uniform(4.0)
    w_free_P = float(free.deflection(c_free, np.array([P[0]]),
                                     np.array([P[1]]))[0])
    psi_P = free.structure_at(np.array([P[0]]), np.array([P[1]]))[:, 0]
    G_N = float(psi_P @ free.solve_from_b(psi_P))
    R_pred = k * w_free_P / (1.0 + k * G_N)
    sup = ClampedPlate.from_config(
        dom, Config(p=12, Q=64, ortho_D=Dm, supports_points=(P,),
                    supports_stiffness=k))
    c_s = sup.solve_uniform(4.0)
    R = k * float(sup.deflection(c_s, np.array([P[0]]), np.array([P[1]]))[0])
    assert abs(R - R_pred) / R_pred < 1e-5


def test_eigen_ortho_supports_monotone():
    """Опора поднимает частоты ортотропа (Курант–Фишер оператор-независим)."""
    d0 = _ssss_case(p=10, Q=48)
    del d0["load"]
    d0["bc"] = {"type": "soft_hinge"}
    d0["eigen"] = {"kind": "vibration", "n_modes": 4}
    d1 = copy.deepcopy(d0)
    d1["supports"] = {"points": [[0.7, 0.5]], "stiffness": 1e5 * 192307.7}
    e0 = dispatch.solve(Problem.from_dict(d0)).eigen.values
    e1 = dispatch.solve(Problem.from_dict(d1)).eigen.values
    assert all(v1 >= v0 * (1.0 - 1e-12)
               for v0, v1 in zip(e0, e1, strict=False))


def _karman_ortho_case(q0, *, ortho="eng", method="picard", bc="clamped"):
    E, nu = 2.1e6, 0.3
    g_iso = E / (2 * (1 + nu))
    d = {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": _A, "y1": 0.0,
                     "y2": _B},
        "bc": {"type": bc},
        "load": {"type": "uniform", "q0": q0},
        "model": {"theory": "karman", "h": 0.05, "karman_method": method},
        "discretization": {"p": 10, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    if ortho == "eng":
        d["model"]["orthotropy"] = {"Ex": 2 * E, "Ey": E, "nu_xy": nu,
                                    "Gxy": 0.8 * g_iso}
    elif ortho == "iso-eng":
        d["model"]["orthotropy"] = {"Ex": E, "Ey": E, "nu_xy": nu,
                                    "Gxy": g_iso}
    else:
        d["model"]["E"] = E
        d["model"]["nu"] = nu
    return d


def test_karman_ortho_linear_limit():
    """ОРТОТРОПНЫЙ КАРМАН (v0.7.0), R1: малое q = классическая ортотропия.

    Мембранный закон N = A·ε из инженерного набора; при q → 0 усилия N → 0 и
    остаётся ортотропный изгиб, сертифицированный рядом Лехницкого
    (измерено: бит-точно).
    """
    rk = dispatch.solve(Problem.from_dict(_karman_ortho_case(1e-6)))
    d_cl = _karman_ortho_case(1e-6)
    d_cl["model"] = {"theory": "classic", "h": 0.05,
                     "orthotropy": d_cl["model"]["orthotropy"]}
    rc = dispatch.solve(Problem.from_dict(d_cl))
    assert abs(rk.w_max - rc.w_max) / rc.w_max < 1e-10


def test_karman_ortho_iso_reduction_nonlinear():
    """R2: изотропный инженерный набор = стандартный Карман В НЕЛИНЕЙНОМ
    режиме (A_iso ≡ C·D_ps почленно; измерено 1.4e-16 при w/w_lin − 1 ≈ 1%)."""
    r1 = dispatch.solve(Problem.from_dict(
        _karman_ortho_case(30.0, ortho="iso-eng")))
    r2 = dispatch.solve(Problem.from_dict(_karman_ortho_case(30.0, ortho=None)))
    assert abs(r1.w_max - r1.w_max_classic) / r1.w_max_classic > 1e-3
    assert abs(r1.w_max - r2.w_max) / r2.w_max < 1e-12


def test_karman_ortho_hardening_and_newton():
    """R3 + R4: знак мембранного ужесточения; Пикар = Ньютон (согласованный
    касательный с ортотропной ∂N/∂c; измерено 6.1e-10)."""
    rp = dispatch.solve(Problem.from_dict(_karman_ortho_case(60.0)))
    rn = dispatch.solve(Problem.from_dict(
        _karman_ortho_case(60.0, method="newton")))
    assert rp.w_max < rp.w_max_classic
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-8


def test_karman_ortho_guards():
    """Карман+ортотропия: D-набор, movable и mixed — отказ."""
    d = _karman_ortho_case(1.0)
    d["model"]["orthotropy"] = {"D11": 2.0, "D12": 0.5, "D22": 1.0,
                                "D66": 0.4}
    with pytest.raises(CaseError, match="model.orthotropy"):
        Problem.from_dict(d)
    d2 = _karman_ortho_case(1.0)
    d2["model"]["inplane_bc"] = "movable"
    with pytest.raises(CaseError, match="model.inplane_bc"):
        Problem.from_dict(d2)
    d3 = _karman_ortho_case(1.0)
    d3["bc"] = {"type": "mixed", "sides": [
        {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
        {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]}
    with pytest.raises(CaseError, match="bc.type"):
        Problem.from_dict(d3)
