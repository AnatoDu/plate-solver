r"""Ворота нагрузки выражением — [load] type="expr" (v0.7.0, exprfield.py).

Верификация:

* редукция: expr="1.0" ≡ uniform БИТ-ТОЧНО на всех четырёх теориях
  (константа даёт тот же массив нагрузки, тракт детерминирован);
* машинный сертификат: q = sin(πx/a)·sin(πy/b) на SS-прямоугольнике против
  ТОЧНОГО одночленного решения Навье (ladder.rect_sin_exact) — спектральная
  сходимость до машинного плато (измерено 5.7e-15 при p=16);
* эквивалентность гауссиане: expr-гауссиана против type="gaussian"
  (та же математика, разные деревья выражений — расходятся последними битами);
* символьный Δq (член КТН −h_*²Δq) против рукописной аналитической формулы;
* безопасность: токен-ограда отклоняет инъекции ДО sympy.parse_expr
  (parse_expr — это eval; ограду НЕ ослаблять);
* существующие охранные отказы накрывают expr (нелинейный контакт,
  преднапряжённая собственная задача).
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.ladder import rect_sin_wmax
from plate_solver.problem import CaseError, Problem

_BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"theory": "classic"},
    "discretization": {"p": 10, "Q": 48, "grid_n": 32},
    "verify": {"reference": "none"},
}


def _case(**over):
    d = copy.deepcopy(_BASE)
    for k, v in over.items():
        d[k] = v
    return d


@pytest.mark.parametrize("theory", ["classic", "karman", "ktn_linear", "ktn_full"])
def test_expr_uniform_bit_exact(theory):
    """expr="1.0" ≡ uniform бит-точно: q0·1.0 даёт тот же массив нагрузки."""
    d1 = _case(model={"theory": theory})
    d2 = _case(model={"theory": theory},
               load={"type": "expr", "expr": "1.0", "q0": 4.0})
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert r1.w_max == r2.w_max
    # вне Ω сетка несёт NaN — сравнение с equal_nan (внутри — бит-точно)
    assert np.array_equal(r1.w_grid, r2.w_grid, equal_nan=True)


def test_expr_sin_vs_navier_exact():
    """sin-нагрузка через ПОЛНЫЙ case-тракт = точное решение Навье (машинно).

    Измерено: rel 5.7e-15 при p=16/Q=48 (машинное плато спектральной
    сходимости); допуск 1e-12 — запас >2 порядков БЕЗ ослабления сути
    (машинный сертификат, как Леви 2.2e-11).
    """
    a, b = 1.0, 1.2
    d = {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": a, "y1": 0.0, "y2": b},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]},
        "load": {"type": "expr", "expr": f"sin(pi*x/{a})*sin(pi*y/{b})",
                 "q0": 4.0},
        "model": {"theory": "classic"},
        "discretization": {"p": 16, "Q": 48, "grid_n": 32},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(d))
    w_c = float(res._plate.deflection(res._c,
                                      np.array([[a / 2]]), np.array([[b / 2]]))[0, 0])
    w_ex = rect_sin_wmax(a, b, res.config.D) * 4.0
    assert abs(w_c - w_ex) / w_ex < 1e-12


def test_expr_matches_gaussian():
    """expr-гауссиана = type="gaussian" (одна математика, разные деревья)."""
    x0, y0, sigma = 0.2, -0.1, 0.3
    d1 = _case(load={"type": "gaussian", "q0": 4.0, "x0": x0, "y0": y0,
                     "sigma": sigma})
    d2 = _case(load={"type": "expr", "q0": 4.0,
                     "expr": f"exp(-((x-{x0})**2+(y-({y0}))**2)/(2*{sigma}**2))"})
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    # разные деревья выражений: расхождение — последние биты округления
    # (измерено 1.1e-13 по w_max при |Δw| ~ 1e-20 абсолютных)
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 5e-13


def test_expr_ktn_full_symbolic_lap_q():
    """Символьный Δq expr-гауссианы = рукописная формула Δq = q·(r²−2σ²)/σ⁴.

    Прямое сравнение векторов Δq в узлах квадратуры (измерено 1.3e-15) и
    сквозной ktn_full: полная КТН с членом −h_*²Δq от expr = от gaussian.
    """
    x0, y0, sigma = 0.2, -0.1, 0.3
    d1 = _case(model={"theory": "ktn_full"},
               load={"type": "gaussian", "q0": 4.0, "x0": x0, "y0": y0,
                     "sigma": sigma})
    d2 = _case(model={"theory": "ktn_full"},
               load={"type": "expr", "q0": 4.0,
                     "expr": f"exp(-((x-{x0})**2+(y-({y0}))**2)/(2*{sigma}**2))"})
    p1, p2 = Problem.from_dict(d1), Problem.from_dict(d2)
    r1 = dispatch.solve(p1)
    lap1 = dispatch._load_laplacian(p1.load, r1._plate.quad)
    lap2 = dispatch._load_laplacian(p2.load, r1._plate.quad)
    assert lap1 is not None and lap2 is not None
    assert np.max(np.abs(lap1 - lap2)) / np.max(np.abs(lap1)) < 1e-13
    r2 = dispatch.solve(p2)
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-12


@pytest.mark.parametrize("bad", [
    "__import__('os').system('true')",
    "().__class__",
    "x[0]",
    "lambda: 1",
    "'abc'",
    "1;2",
    "z + x",
    "sin.func",
    "",
    "1j*x",
    "x = 1",
    "open(x)",
    "1/0",
    "sqrt(-1)",
])
def test_expr_rejected(bad):
    """Инъекции/кривой синтаксис — CaseError на РАЗБОРЕ case (fail-fast)."""
    with pytest.raises(CaseError, match="load.expr"):
        Problem.from_dict(_case(load={"type": "expr", "expr": bad, "q0": 1.0}))


def test_expr_scientific_notation_accepted():
    """Научная нотация в числах (2.5e-3) — НЕ ложный отказ (токен NUMBER)."""
    d = _case(load={"type": "expr", "expr": "2.5e-3*x + 1E0 + .5", "q0": 1.0})
    res = dispatch.solve(Problem.from_dict(d))
    assert res.w_max > 0.0


def test_expr_missing_q0_rejected():
    with pytest.raises(CaseError, match="load.q0"):
        Problem.from_dict(_case(load={"type": "expr", "expr": "1.0"}))


def test_expr_nonlinear_contact_force_rejected():
    """expr + СИЛОВОЙ нелинейный контакт → CaseError (поле нагрузки допущено
    только в позиционном основании, v0.7.0 — tests/test_contact_field_load)."""
    d = _case(model={"theory": "ktn_full"},
              load={"type": "expr", "expr": "1.0", "q0": 4.0},
              contact={"enabled": True, "force": 1.0})
    with pytest.raises(CaseError, match="load.type"):
        Problem.from_dict(d)


def test_expr_eigen_prestress_rejected():
    """expr + [eigen] prestress → CaseError (преднапряжение — под uniform)."""
    d = _case(model={"theory": "karman"},
              load={"type": "expr", "expr": "1.0", "q0": 4.0})
    d["eigen"] = {"kind": "vibration", "n_modes": 3, "prestress": True}
    with pytest.raises(CaseError, match="load.type"):
        Problem.from_dict(d)


@pytest.mark.parametrize("expr", ["abs(x)", "min(x, y)", "max(x, 0.0)"])
def test_expr_nonsmooth_ktn_full_graceful(expr):
    """Негладкий expr + ktn_full: член Δq честно опущен с warning.

    field_laplacian отклоняет негладкость ДО lambdify (иначе lambdify
    падал бы на DiracDelta/производных негладких функций).
    """
    d = _case(model={"theory": "ktn_full"},
              load={"type": "expr", "q0": 4.0, "expr": expr})
    d["discretization"] = {"p": 8, "Q": 32, "grid_n": 16}
    res = dispatch.solve(Problem.from_dict(d))
    assert res.w_max > 0.0
    assert any("Δq опущен" in w for w in res.warnings)


@pytest.mark.parametrize("bad", [
    "9**9**9",                       # степенная башня (DoS разбора)
    "9**100000",                     # гигантский целый показатель
    "123456789012345",               # целый литерал длиннее 12 цифр
])
def test_expr_resource_fence(bad):
    """Ресурсная ограда: DoS-строки отклоняются БЫСТРО, до sympy."""
    import time

    t0 = time.time()
    with pytest.raises(CaseError, match="load.expr"):
        Problem.from_dict(_case(load={"type": "expr", "expr": bad, "q0": 1.0}))
    assert time.time() - t0 < 1.0


def test_expr_verify_reference_rejected():
    """expr + verify.reference ≠ none — отказ (реестр эталонов не знает expr)."""
    d = _case(load={"type": "expr", "expr": "1.0", "q0": 4.0})
    d["verify"] = {"reference": "analytic"}
    with pytest.raises(CaseError, match="verify.reference"):
        Problem.from_dict(d)


def test_expr_plate2_load():
    """expr-нагрузка ВТОРОЙ пластины пары: константа = uniform бит-точно."""
    base = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "classic"},
        "contact": {"enabled": True, "target": "plate2", "gap": 0.01},
        "plate2": {"bc": {"type": "soft_hinge"},
                   "load": {"type": "uniform", "q0": 1.0}},
        "discretization": {"p": 8, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    d2 = copy.deepcopy(base)
    d2["plate2"]["load"] = {"type": "expr", "expr": "1.0", "q0": 1.0}
    r1 = dispatch.solve(Problem.from_dict(base))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert r1.w_max == r2.w_max
    d3 = copy.deepcopy(base)
    d3["plate2"]["load"] = {"type": "expr", "expr": "1.0 + 0.5*x", "q0": 1.0}
    r3 = dispatch.solve(Problem.from_dict(d3))
    assert np.isfinite(r3.w_max) and r3.w_max > 0.0


def test_expr_save_fields(tmp_path):
    """Сохранение полей при expr: q_top — само поле q0·g."""
    d = _case(load={"type": "expr", "q0": 4.0,
                    "expr": "exp(-(x**2+y**2)/0.5)"})
    res = dispatch.solve(Problem.from_dict(d))
    res.save(tmp_path)
    z = np.load(tmp_path / "fields.npz")
    assert "w" in z and "sx_top" in z
    q_top, _ = res._q_faces_on_grid()
    i = np.unravel_index(np.nanargmax(np.where(np.isfinite(res.w_grid),
                                               1.0, np.nan)), res.Xg.shape)
    x0, y0 = res.Xg[i], res.Yg[i]
    assert abs(q_top[i] - 4.0 * np.exp(-(x0**2 + y0**2) / 0.5)) < 1e-12
