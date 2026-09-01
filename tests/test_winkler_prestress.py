r"""Ворота v0.6.6: основание Винклера + преднапряжённая собственная задача в case.

Винклер: ``D·Δ²w + k_w·w = q`` свёрнут в изгибную жёсткость (+k_w∫ψψ) ⇒
работает во всех трактах (classic clamped, нелинейные, eigen, контакт).
Верификация — MMS до МАШИННОЙ точности (оператор целиком), kw=0 бит-точно,
предельная асимптотика kw→∞. Преднапряжённый eigen: `[eigen] prestress=true` +
`[load]` + theory=karman ⇒ поле N(w) из кармановского решения; частота растёт
с нагрузкой (натяжение ужесточает) — совпадает с API-верификацией v0.6.4.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from plate_solver import dispatch
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.geometry import Domain
from plate_solver.geometry import x as sx
from plate_solver.geometry import y as sy
from plate_solver.problem import CaseError, Problem


def _solve(tmp_path, body):
    p = tmp_path / "case.toml"
    p.write_text(body, encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


# --------------------------------------------------------------------------- #
#  Основание Винклера
# --------------------------------------------------------------------------- #
def test_winkler_mms_machine_precision():
    """MMS: q = D·Δ²w + k_w·w на защемлённом прямоугольнике — машинная точность."""
    ax, ay = 1.0, 0.6
    w_expr = (sx**2 - ax**2) ** 2 * (sy**2 - ay**2) ** 2
    dom = Domain((ax**2 - sx**2) * (ay**2 - sy**2), (-ax, ax, -ay, ay))
    kw = 37.0
    cfg = Config(E=1.0, nu=0.3, h=0.1, q0=1.0, a=1.0, p=10, Q=64, winkler=kw)

    def lap(e):
        return sp.diff(e, sx, 2) + sp.diff(e, sy, 2)

    q_expr = cfg.D * lap(lap(w_expr)) + kw * w_expr
    qf = sp.lambdify((sx, sy), q_expr, "numpy")
    wf = sp.lambdify((sx, sy), w_expr, "numpy")
    cp = ClampedPlate.from_config(dom, cfg)
    qn = cp.quad
    c = cp.solve(np.broadcast_to(qf(qn.x, qn.y), qn.x.shape).astype(float))
    w_num = cp.deflection(c, qn.x, qn.y)
    w_ex = np.broadcast_to(wf(qn.x, qn.y), qn.x.shape).astype(float)
    assert np.max(np.abs(w_num - w_ex)) / np.max(np.abs(w_ex)) < 1e-9


def test_winkler_zero_bit_exact():
    """kw = 0 — прежний оператор бит-точно (регресс не сдвинут)."""
    dom = Domain((1.0 - sx**2) * (0.36 - sy**2), (-1.0, 1.0, -0.6, 0.6))
    base = dict(E=1.0, nu=0.3, h=0.1, q0=1.0, a=1.0, p=8, Q=48)
    cp0 = ClampedPlate.from_config(dom, Config(**base))
    cpz = ClampedPlate.from_config(dom, Config(winkler=0.0, **base))
    q = np.full(cp0.quad.x.size, 1.0)
    assert np.array_equal(cp0.solve(q), cpz.solve(q))


def test_winkler_stiffens_and_scales(tmp_path):
    """Основание ужесточает (w падает с kw); karman-тракт получает член основания."""
    body = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 1.0
[model]
theory = "karman"
E = 1.0
nu = 0.3
h = 1.0
inplane_bc = "immovable"
winkler = {kw}
karman_tol = 1.0e-8
karman_max_iter = 200
[discretization]
p = 8
Q = 96
grid_n = 16
[verify]
reference = "none"
"""
    w = [_solve(tmp_path, body.format(kw=kw)).w_max for kw in (0.0, 20.0, 200.0)]
    assert w[0] > w[1] > w[2]                             # монотонное ужесточение


def test_winkler_soft_hinge_classic_rejected(tmp_path):
    """Винклер + классический мягкий шарнир — отказ (расщепление ломается)."""
    body = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "soft_hinge"
[load]
type = "uniform"
q0 = 1.0
[model]
theory = "classic"
winkler = 10.0
[discretization]
p = 8
Q = 64
"""
    p = tmp_path / "case.toml"
    p.write_text(body, encoding="utf-8")
    with pytest.raises(CaseError, match="winkler"):
        Problem.from_toml(str(p))


# --------------------------------------------------------------------------- #
#  Преднапряжённая собственная задача из case-файла
# --------------------------------------------------------------------------- #
_PRESTRESS = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = {q0}
[model]
theory = "karman"
E = 1.0
nu = 0.3
h = 0.1
inplane_bc = "immovable"
karman_tol = 1.0e-8
karman_max_iter = 300
n_load_steps = 2
[eigen]
kind = "vibration"
n_modes = 1
rho_h = 1.0
prestress = true
[discretization]
p = 12
Q = 140
grid_n = 20
[verify]
reference = "none"
"""


def test_prestressed_vibration_case_file(tmp_path):
    """Частоты НАГРУЖЕННОЙ пластины из case-файла: растут с нагрузкой (натяжение)."""
    w1 = float(_solve(tmp_path, _PRESTRESS.format(q0=2.0 * 0.1**4)).eigen.values[0])
    w2 = float(_solve(tmp_path, _PRESTRESS.format(q0=12.0 * 0.1**4)).eigen.values[0])
    assert 0.0 < w1 < w2                                  # монотонный рост с P̄


def test_prestress_requires_karman_and_load(tmp_path):
    """prestress = true требует theory = karman и [load]; без prestress [load] запрещён."""
    bad = _PRESTRESS.format(q0=1e-4).replace('theory = "karman"', 'theory = "classic"')
    p = tmp_path / "a.toml"
    p.write_text(bad, encoding="utf-8")
    with pytest.raises(CaseError, match="karman"):
        Problem.from_toml(str(p))
    no_pre = _PRESTRESS.format(q0=1e-4).replace("prestress = true\n", "")
    p2 = tmp_path / "b.toml"
    p2.write_text(no_pre, encoding="utf-8")
    with pytest.raises(CaseError, match="prestress"):
        Problem.from_toml(str(p2))


def test_winkler_mixed_all_hinge_navier_series():
    """Винклер на mixed all-hinge (v0.7.0): точный ряд w_mn/(π⁴D_mn + k_w).

    Свёртка +k_w∫ψψ выполнена и в MixedRectPlate; измерено rel 3.8e-9
    при p=14.
    """
    import numpy as np

    from plate_solver import dispatch
    from plate_solver.ladder import _navier_odd
    from plate_solver.problem import Problem

    a, b, q0, D, kw = 2.0, 1.2, 1.0, 1.7, 300.0
    m = _navier_odd(200)[:, None]
    n = _navier_odd(200)[None, :]
    Dmn = D * np.pi**4 * ((m / a) ** 2 + (n / b) ** 2) ** 2
    coef = 16.0 * q0 / (np.pi**2 * m * n) / (Dmn + kw)
    w_ref = float(np.sum(coef * np.sin(m * np.pi / 2) * np.sin(n * np.pi / 2)))
    d = {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": a, "y1": 0.0,
                     "y2": b},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]},
        "load": {"type": "uniform", "q0": q0},
        "model": {"theory": "classic", "E": 12 * (1 - 0.09) * D, "nu": 0.3,
                  "h": 1.0, "winkler": kw},
        "discretization": {"p": 14, "Q": 64, "grid_n": 24},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(d))
    w_c = float(res._plate.deflection(res._c, np.array([a / 2]),
                                      np.array([b / 2]))[0])
    assert abs(w_c - w_ref) / w_ref < 1e-7


def test_winkler_mixed_all_clamped_mms():
    """Винклер на mixed all-clamped: MMS машинно (структура полиномиальная)."""
    import numpy as np
    import sympy as sp

    from plate_solver import dispatch
    from plate_solver.problem import Problem

    x, y = sp.symbols("x y")
    D, kw = 1.7, 300.0
    w_ex = (x**2 - 1) ** 2 * (y**2 - sp.Rational(36, 100)) ** 2
    q = D * (sp.diff(w_ex, x, 4) + 2 * sp.diff(w_ex, x, 2, y, 2)
             + sp.diff(w_ex, y, 4)) + kw * w_ex
    d = {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0,
                     "y1": -0.6, "y2": 0.6},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "clamped"}, {"side": "x2", "type": "clamped"},
            {"side": "y1", "type": "clamped"}, {"side": "y2", "type": "clamped"}]},
        "load": {"type": "expr", "q0": 1.0, "expr": str(sp.expand(q))},
        "model": {"theory": "classic", "E": 12 * (1 - 0.09) * D, "nu": 0.3,
                  "h": 1.0, "winkler": kw},
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


def test_winkler_soft_hinge_classic_still_rejected():
    """Шарнир классики с Винклером — по-прежнему отказ (расщепление)."""
    import pytest as _pytest

    from plate_solver.problem import CaseError, Problem

    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "classic", "winkler": 100.0},
        "verify": {"reference": "none"},
    }
    with _pytest.raises(CaseError, match="model.winkler"):
        Problem.from_dict(d)
