r"""Ворота переменной толщины — [model] h_expr (v0.7.0).

Эталон — MMS: сильная форма пластины переменной жёсткости
``q = (D(w_xx+νw_yy))_xx + (D(w_yy+νw_xx))_yy + 2((1−ν)D·w_xy)_xy``
(Эйлер–Лагранж полного функционала; форма оператора — Тимошенко–В.-Кригер,
гл. о пластинах непостоянной толщины). КЛЮЧЕВОЙ факт: при D ≠ const гауссов
член НЕ нуль-лагранжиан — сборка обязана идти ПОЛНОЙ билинейной формой.

Манифактурное решение строится на ∏-омеге (x²−1)(y²−0.36), а ω пакета —
НОРМИРОВАННАЯ R-функция (иное пространство) ⇒ ворота НЕ машинные, а
СПЕКТРАЛЬНЫЕ: измерено rel 6.6e-5 (p=8) → 2.9e-6 (p=12) → 1.8e-7 (p=16) →
8.2e-9 (p=20). Карман — редукционный сертификат (линейный предел 8.7e-16,
Пикар↔Ньютон 3.9e-10, знак мембранного ужесточения).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.problem import CaseError, Problem

_E, _NU = 2.1e6, 0.3
_HE = "0.5*(1 + 0.3*x + 0.2*y)"


def _mms_rect():
    """Манифактурная пара (expr-нагрузка q, точное w) на [−1,1]×[−0.6,0.6]."""
    import sympy as sp

    x, y = sp.symbols("x y")
    om = (x**2 - 1) * (y**2 - sp.Rational(36, 100))
    w_m = (om**2 * (1 + x / 2)).expand()
    h_e = sp.Rational(1, 2) * (1 + sp.Rational(3, 10) * x
                               + sp.Rational(1, 5) * y)
    D_e = _E * h_e**3 / (12 * (1 - sp.Rational(3, 10)**2))
    mx = -(D_e * (sp.diff(w_m, x, 2) + _NU * sp.diff(w_m, y, 2)))
    my = -(D_e * (sp.diff(w_m, y, 2) + _NU * sp.diff(w_m, x, 2)))
    mxy = -(1 - _NU) * D_e * sp.diff(w_m, x, y)
    q = (sp.diff(-mx, x, 2) + sp.diff(-my, y, 2)
         + 2 * sp.diff(-mxy, x, y)).expand()
    return str(q), sp.lambdify((x, y), w_m, "numpy")


def _rect_case(p, Q, *, q_expr, model=None):
    return {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0,
                     "y1": -0.6, "y2": 0.6},
        "bc": {"type": "clamped"},
        "load": {"type": "expr", "q0": 1.0, "expr": q_expr},
        "model": model or {"theory": "classic", "E": _E, "nu": _NU,
                           "h_expr": _HE},
        "discretization": {"p": p, "Q": Q, "grid_n": 16},
        "verify": {"reference": "none"},
    }


def _err(res, f_ex):
    xs = np.array([0.0, 0.4, -0.5])
    ys = np.array([0.0, -0.2, 0.3])
    w_num = np.asarray(res._plate.deflection(res._c, xs, ys), float)
    w_ex = np.asarray(f_ex(xs, ys), float)
    return float(np.max(np.abs(w_num - w_ex)) / np.max(np.abs(w_ex)))


def test_varh_mms_spectral():
    """MMS: спектральная сходимость к манифактурному решению.

    Измерено 6.6e-5 (p=8) → 1.8e-7 (p=16), отношение 2.7e-3 — ворота
    err(16) < 1e-6 и err(16) < 0.1·err(8) ловят потерю сходимости с запасом.
    """
    q_expr, f_ex = _mms_rect()
    e8 = _err(dispatch.solve(Problem.from_dict(
        _rect_case(8, 48, q_expr=q_expr))), f_ex)
    e16 = _err(dispatch.solve(Problem.from_dict(
        _rect_case(16, 80, q_expr=q_expr))), f_ex)
    assert e16 < 1e-6
    assert e16 < 0.1 * e8


def test_varh_mms_circle_floor():
    """Круг: пол маскированной квадратуры (общий с const-D), убывает с Q."""
    import sympy as sp

    x, y = sp.symbols("x y")
    w_m = ((1 - x**2 - y**2) ** 2 * (1 + x / 2)).expand()
    h_e = sp.Rational(1, 2) * (1 + sp.Rational(3, 10) * x)
    D_e = _E * h_e**3 / (12 * (1 - sp.Rational(3, 10)**2))
    mx = -(D_e * (sp.diff(w_m, x, 2) + _NU * sp.diff(w_m, y, 2)))
    my = -(D_e * (sp.diff(w_m, y, 2) + _NU * sp.diff(w_m, x, 2)))
    mxy = -(1 - _NU) * D_e * sp.diff(w_m, x, y)
    q = (sp.diff(-mx, x, 2) + sp.diff(-my, y, 2)
         + 2 * sp.diff(-mxy, x, y)).expand()
    f_ex = sp.lambdify((x, y), w_m, "numpy")
    err = {}
    for Q in (64, 128):
        d = {
            "geometry": {"kind": "circle", "a": 1.0},
            "bc": {"type": "clamped"},
            "load": {"type": "expr", "q0": 1.0, "expr": str(q)},
            "model": {"theory": "classic", "E": _E, "nu": _NU,
                      "h_expr": "0.5*(1 + 0.3*x)"},
            "discretization": {"p": 10, "Q": Q, "grid_n": 16},
            "verify": {"reference": "none"},
        }
        err[Q] = _err(dispatch.solve(Problem.from_dict(d)), f_ex)
    assert err[128] < err[64]                      # квадратурная природа пола
    assert err[128] < 3e-2                         # измерено ~8e-3


def test_varh_const_reduction():
    """h_expr="1.0" = скалярная h=1.0: полная форма vs ∫ΔψΔψ (~5e-10)."""
    d1 = _rect_case(8, 48, q_expr="1.0",
                    model={"theory": "classic", "E": _E, "nu": _NU,
                           "h_expr": "1.0"})
    d1["load"] = {"type": "uniform", "q0": 4.0}
    d2 = copy.deepcopy(d1)
    d2["model"] = {"theory": "classic", "E": _E, "nu": _NU, "h": 1.0}
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r2.w_max < 1e-8


def test_varh_monotone_bracketing():
    """Вилка монотонности энергии: h1 < h(x,y) < h2 ⇒ w(h2) < w(поле) < w(h1)."""
    def w_of(model_h):
        d = _rect_case(8, 48, q_expr="1.0",
                       model={"theory": "classic", "E": _E, "nu": _NU,
                              **model_h})
        d["load"] = {"type": "uniform", "q0": 4.0}
        return dispatch.solve(Problem.from_dict(d)).w_max

    w_thick = w_of({"h": 0.75})
    w_field = w_of({"h_expr": "0.625 + 0.125*x"})   # 0.5 ≤ h ≤ 0.75
    w_thin = w_of({"h": 0.5})
    assert w_thick < w_field < w_thin


def test_varh_karman_linear_limit():
    """Карман+h_expr при малой нагрузке = классика+h_expr (измерено 9e-16)."""
    dk = _rect_case(8, 48, q_expr="1.0")
    dk["load"] = {"type": "uniform", "q0": 1e-3}
    dk["model"] = {"theory": "karman", "E": _E, "nu": _NU, "h_expr": _HE}
    dc = copy.deepcopy(dk)
    dc["model"]["theory"] = "classic"
    rk = dispatch.solve(Problem.from_dict(dk))
    rc = dispatch.solve(Problem.from_dict(dc))
    assert abs(rk.w_max - rc.w_max) / rc.w_max < 1e-8


def test_varh_karman_hardening_and_newton():
    """Умеренная нагрузка: мембранное ужесточение (знак) + Пикар = Ньютон."""
    dp = _rect_case(8, 48, q_expr="1.0")
    dp["load"] = {"type": "uniform", "q0": 50.0}
    dp["model"] = {"theory": "karman", "E": _E, "nu": _NU, "h_expr": _HE,
                   "karman_method": "picard"}
    dn = copy.deepcopy(dp)
    dn["model"]["karman_method"] = "newton"
    rp = dispatch.solve(Problem.from_dict(dp))
    rn = dispatch.solve(Problem.from_dict(dn))
    assert rp.w_max < rp.w_max_classic             # ужесточение — строгий знак
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-8   # измерено 3.9e-10


def test_varh_moments_local_D():
    """Моменты — с ЛОКАЛЬНОЙ D(x, y): MMS-сверка Mx в точке (спектрально)."""
    import sympy as sp

    x, y = sp.symbols("x y")
    q_expr, _ = _mms_rect()
    om = (x**2 - 1) * (y**2 - sp.Rational(36, 100))
    w_m = (om**2 * (1 + x / 2)).expand()
    h_e = sp.Rational(1, 2) * (1 + sp.Rational(3, 10) * x
                               + sp.Rational(1, 5) * y)
    D_e = _E * h_e**3 / (12 * (1 - sp.Rational(3, 10)**2))
    mx_ex = sp.lambdify((x, y), -(D_e * (sp.diff(w_m, x, 2)
                                         + _NU * sp.diff(w_m, y, 2))), "numpy")
    res = dispatch.solve(Problem.from_dict(_rect_case(16, 80, q_expr=q_expr)))
    Mx, My, Mxy = res.moments_on_grid()
    i, j = 8, 8                                    # внутренняя точка сетки
    ref = float(mx_ex(res.Xg[i, j], res.Yg[i, j]))
    assert abs(Mx[i, j] - ref) / abs(ref) < 1e-4


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["model"].update(theory="ktn_full"), "model.theory"),
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d.update(bc={"type": "soft_hinge"}), "bc.type"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5}),
     "contact.enabled"),
    (lambda d: d["model"].update(theory="karman", inplane_bc="movable"),
     "model.inplane_bc"),
    (lambda d: d["model"].update(h=1.0), "model.h"),
    (lambda d: d["load"].update(thermal_moment=3.0), "thermal_moment"),
    (lambda d: d.update(supports={"points": [[0.0, 0.0]], "stiffness": 1e3}),
     "supports.points"),
    (lambda d: d["model"].update(h_expr="z + 1"), "h_expr"),
])
def test_varh_guards(mutate, match):
    d = _rect_case(6, 32, q_expr="1.0")
    d["load"] = {"type": "uniform", "q0": 4.0}
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_varh_nonpositive_rejected():
    """h ≤ 0 где-то на области — CaseError с точкой минимума."""
    d = _rect_case(6, 32, q_expr="1.0")
    d["load"] = {"type": "uniform", "q0": 4.0}
    d["model"]["h_expr"] = "0.5 + x"               # ≤ 0 при x ≤ −0.5
    with pytest.raises(CaseError, match="h_expr"):
        dispatch.solve(Problem.from_dict(d))


def test_varh_eigen_rejected():
    d = _rect_case(6, 32, q_expr="1.0")
    del d["load"]
    d["eigen"] = {"kind": "vibration", "n_modes": 3}
    with pytest.raises(CaseError, match="h_expr"):
        Problem.from_dict(d)


def test_varh_winkler_mms():
    """h_expr + Винклер: MMS сильной формы с +k_w·w (композиция свёрток).

    D0·(S_D/D0 + (k_w/D0)·∫ψψ) = ∫D(x,y)[…] + k_w∫ψψ — нормировка D0
    сокращается по построению; ворота спектральные (как основной MMS).
    """
    import sympy as sp

    x, y = sp.symbols("x y")
    kw = 300.0
    om = (x**2 - 1) * (y**2 - sp.Rational(36, 100))
    w_m = (om**2 * (1 + x / 2)).expand()
    h_e = sp.Rational(1, 2) * (1 + sp.Rational(3, 10) * x
                               + sp.Rational(1, 5) * y)
    D_e = _E * h_e**3 / (12 * (1 - sp.Rational(3, 10)**2))
    mx = -(D_e * (sp.diff(w_m, x, 2) + _NU * sp.diff(w_m, y, 2)))
    my = -(D_e * (sp.diff(w_m, y, 2) + _NU * sp.diff(w_m, x, 2)))
    mxy = -(1 - _NU) * D_e * sp.diff(w_m, x, y)
    q = (sp.diff(-mx, x, 2) + sp.diff(-my, y, 2)
         + 2 * sp.diff(-mxy, x, y) + kw * w_m).expand()
    f_ex = sp.lambdify((x, y), w_m, "numpy")
    d = _rect_case(16, 80, q_expr=str(q))
    d["model"]["winkler"] = kw
    e16 = _err(dispatch.solve(Problem.from_dict(d)), f_ex)
    assert e16 < 1e-6
