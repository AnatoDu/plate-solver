r"""Ворота линейной (погонной) нагрузки — [load] type="line" (v0.7.0).

Эталоны:

* ряд Навье для линии ``x = ξ`` на SS-прямоугольнике (``ladder.navier_line``,
  сходимость ряда ~4e-13) — измеренная сходимость Ритца АЛГЕБРАИЧЕСКАЯ
  ~p⁻²·⁸ (скачок перерезывающей [Q_n] = −P ⇒ w ∈ H^{7/2−ε}):
  rel 8.5e-3 (p=8) → 2.8e-3 (p=12) → 1.3e-3 (p=16) → 4.0e-4 (p=24);
* предел полосы: линия = предел полосы ширины 2ε с q0 = P/(2ε), скорость
  O(ε²) (измерено 1.99–2.00);
* линейный предел Кармана (редукция N→0 точна по построению, ~1.5e-10).

Бетти-взаимность НЕ используется как ворота: она тождественна по симметрии
матрицы Ритца (b_gᵀA⁻¹b_l ≡ b_lᵀA⁻¹b_g) и ничего не верифицирует.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.ladder import navier_line_center
from plate_solver.problem import CaseError, Problem

_A, _B = 2.0, 1.2
_D1 = {"theory": "classic", "E": 12 * (1 - 0.3**2), "nu": 0.3, "h": 1.0}  # D = 1


def _line_case(p=16, *, bc="soft_hinge", model=None, p0=(1.0, 0.0),
               p1=(1.0, _B), intensity=1.0, geometry=None):
    return {
        "geometry": geometry or {"kind": "rectangle", "x1": 0.0, "x2": _A,
                                 "y1": 0.0, "y2": _B},
        "bc": {"type": bc},
        "load": {"type": "line", "p0": list(p0), "p1": list(p1),
                 "intensity": intensity},
        "model": dict(model if model is not None else _D1),
        "discretization": {"p": p, "Q": 64, "grid_n": 32},
        "verify": {"reference": "none"},
    }


def _center_w(res, x=_A / 2, y=_B / 2):
    return float(res._plate.deflection(res._c, np.array([x]), np.array([y]))[0])


def test_navier_line_ss():
    """SS-прямоугольник, полная линия x = 1: Ритц → ряд Навье (rel ≤ 1e-3).

    Измерено 4.02e-4 при p=24 (алгебраическая сходимость ~p⁻²·⁸) — запас 2.5×.
    """
    res = dispatch.solve(Problem.from_dict(_line_case(p=24)))
    w_ref = navier_line_center(_A, _B, 1.0, 1.0, 1.0)
    assert abs(_center_w(res) - w_ref) / w_ref < 1e-3


def test_line_p_convergence():
    """Алгебраическая сходимость не потеряна: err(p=28) < 0.35·err(p=16)."""
    w_ref = navier_line_center(_A, _B, 1.0, 1.0, 1.0)
    err = {}
    for p in (16, 28):
        res = dispatch.solve(Problem.from_dict(_line_case(p=p)))
        err[p] = abs(_center_w(res) - w_ref) / w_ref
    assert err[28] < 0.35 * err[16]                    # измерено отношение 0.20


def test_clamped_strip_limit():
    """Линия = предел полосы ширины 2ε с q0 = P/(2ε): скорость O(ε²).

    Полоса интегрируется ТОЧНОЙ тензорной квадратурой (structure_at + 1D
    Гаусс), оба решения — solve_from_b одного оператора; измерено
    d(ε): 5.13e-3 (0.05) → 1.29e-3 (0.025), скорость 1.99.
    """
    from plate_solver.clamped import ClampedPlate
    from plate_solver.config import Config
    from plate_solver.geometry import make_rectangle

    dom = make_rectangle(0.0, _A, 0.0, _B)
    cfg = Config(E=12 * (1 - 0.3**2), nu=0.3, h=1.0, p=16, Q=64)
    plate = ClampedPlate.from_config(dom, cfg)

    tx, wx = np.polynomial.legendre.leggauss(32)
    ty, wy = np.polynomial.legendre.leggauss(96)
    ys = 0.5 * _B * (ty + 1.0)
    wy_s = 0.5 * _B * wy

    def strip_b(eps):
        xs = 1.0 + eps * tx
        wx_s = eps * wx
        XX, YY = np.meshgrid(xs, ys, indexing="ij")
        psi = plate.structure_at(XX.ravel(), YY.ravel())
        wgt = np.outer(wx_s, wy_s).ravel() / (2.0 * eps)   # q0 = P/(2ε)
        return psi @ wgt

    b_line = dispatch._line_load_vector(
        Problem.from_dict(_line_case(bc="clamped")).load, dom, plate)
    w_line = float(plate.deflection(plate.solve_from_b(b_line),
                                    np.array([1.0]), np.array([_B / 2]))[0])
    d = {}
    for eps in (0.05, 0.025):
        w_strip = float(plate.deflection(plate.solve_from_b(strip_b(eps)),
                                         np.array([1.0]), np.array([_B / 2]))[0])
        d[eps] = abs(w_strip - w_line) / abs(w_line)
    assert d[0.025] < d[0.05] / 3.0                    # скорость ~O(ε²)
    assert d[0.025] < 2.6e-3                           # измерено 1.29e-3


def test_karman_line_linear_limit():
    """Карман с малой линией = классика (редукция N→0; измерено 1.5e-10)."""
    mk = dict(_D1)
    mk["theory"] = "karman"
    rk = dispatch.solve(Problem.from_dict(
        _line_case(bc="clamped", model=mk, intensity=1e-3)))
    rc = dispatch.solve(Problem.from_dict(
        _line_case(bc="clamped", intensity=1e-3)))
    assert abs(_center_w(rk) - _center_w(rc)) / abs(_center_w(rc)) < 1e-8


def test_partial_segment_symmetry():
    """Частичный отрезок: зеркальная симметрия поля + p0↔p1 бит-точно."""
    sq = {"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0}
    d1 = _line_case(bc="clamped", geometry=sq,
                    p0=(0.5, 0.3), p1=(0.5, 0.7))
    r1 = dispatch.solve(Problem.from_dict(d1))
    d2 = copy.deepcopy(d1)
    d2["load"]["p0"], d2["load"]["p1"] = d2["load"]["p1"], d2["load"]["p0"]
    r2 = dispatch.solve(Problem.from_dict(d2))
    # p0↔p1: те же узлы Гаусса в обратном порядке — суммирование в ином
    # порядке даёт расхождение последних битов (НЕ бит-точно), физика та же
    m0 = np.isfinite(r1.w_grid)
    assert np.max(np.abs(r1.w_grid[m0] - r2.w_grid[m0])) \
        <= 1e-12 * np.max(np.abs(r1.w_grid[m0]))

    w = r1.w_grid
    m = np.isfinite(w) & np.isfinite(w[:, ::-1])
    assert np.max(np.abs(w[m] - w[:, ::-1][m])) / np.nanmax(np.abs(w)) < 1e-10


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["load"].update(p1=[3.0, 0.6]), "выходит за область"),
    (lambda d: d["load"].update(p0=[0.0, 0.0], p1=[0.0, 1.2]), "на границе"),
    (lambda d: d["load"].update(p1=d["load"]["p0"]), "load.p1"),
    (lambda d: d["load"].update(intensity=0.0), "load.intensity"),
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d["model"].update(theory="ktn_full"), "model.theory"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5}),
     "contact.enabled"),
    (lambda d: d.update(verify={"reference": "analytic"}), "verify.reference"),
])
def test_line_guards(mutate, match):
    d = _line_case()
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_line_mixed_rejected():
    d = _line_case()
    d["bc"] = {"type": "mixed", "sides": [
        {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
        {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]}
    with pytest.raises(CaseError, match="bc.type"):
        Problem.from_dict(d)


def test_line_plate2_rejected():
    """line у ВТОРОЙ пластины пары — отказ (валидатор смотрит и
    plate2.load, не только первую пластину)."""
    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "classic"},
        "contact": {"enabled": True, "target": "plate2", "gap": 0.01},
        "plate2": {"bc": {"type": "soft_hinge"},
                   "load": {"type": "line", "p0": [0.0, -0.5],
                            "p1": [0.0, 0.5], "intensity": 1.0}},
        "verify": {"reference": "none"},
    }
    with pytest.raises(CaseError, match="plate2.load.type"):
        Problem.from_dict(d)


def test_line_through_hole_rejected():
    """Хорда кольца через отверстие — отказ (плотная ω-выборка вдоль отрезка)."""
    d = _line_case(bc="clamped",
                   geometry={"kind": "annulus", "a": 1.0, "b": 0.4},
                   p0=(-0.9, 0.0), p1=(0.9, 0.0))
    with pytest.raises(CaseError, match="выходит за область"):
        dispatch.solve(Problem.from_dict(d))


def test_line_ortho_iso_reduction():
    """line + [model.orthotropy]: изотропная редукция = изотропная линия.

    Оператор сертифицирован ортотропными воротами, функционал — рядом Навье
    линии; их сочетание проверяется редукцией (той же полной формой).
    """
    D, nu = 1.7, 0.3
    d_o = _line_case(bc="clamped", model={
        "theory": "classic", "h": 1.0,
        "orthotropy": {"D11": D, "D12": nu * D, "D22": D,
                       "D66": (1 - nu) * D / 2}})
    d_i = _line_case(bc="clamped", model={
        "theory": "classic", "E": 12 * (1 - nu**2) * D, "nu": nu, "h": 1.0})
    r_o = dispatch.solve(Problem.from_dict(d_o))
    r_i = dispatch.solve(Problem.from_dict(d_i))
    assert abs(_center_w(r_o) - _center_w(r_i)) / abs(_center_w(r_i)) < 1e-9


def test_line_save_fields(tmp_path):
    """Сохранение полей при line — не падает, npz полон."""
    res = dispatch.solve(Problem.from_dict(_line_case(p=10)))
    res.save(tmp_path)
    z = np.load(tmp_path / "fields.npz")
    assert "w" in z and "Mx" in z and "Qx" in z
    assert (tmp_path / "result.json").is_file()
