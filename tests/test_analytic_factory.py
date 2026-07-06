"""Фабрика аналитических эталонов (F4): сертификаты, тождества, ограда.

F4.2 — самосертификация: эталон существует только с проверенным
сертификатом (PDE + КУ; ряды — контроль остатка). F4.3 — согласование
с ручными функциями на пересечении областей действия (1e-12).
F4.4 — новые ladder-ступени (big) и отказы резолвера вне ограды.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import sympy as sp

from plate_solver import analytic
from plate_solver.analytic_auto import (
    FactoryError,
    axisym_solution,
    levy_solution,
    navier_solution,
    strip_solution,
)

_ROOT = Path(__file__).resolve().parents[1]
D, NU, Q0, A, B, P = 100.0, 0.3, 4.0, 1.0, 0.4, 5.0


# --------------------------------------------------------------------------- #
#  F4.3: тождества с ручными решениями (1e-12)
# --------------------------------------------------------------------------- #
def test_identity_circle_uniform():
    """Круг clamped/soft uniform ≡ замкнутые формулы."""
    s = axisym_solution(a=A, bc_outer="clamped", D=D, nu=NU, q_coeffs=(Q0,))
    r = np.linspace(0.0, 0.95, 8)
    ref = Q0 * (A**2 - r**2) ** 2 / (64 * D)
    assert np.allclose(s.w(r, 0 * r), ref, rtol=1e-12, atol=1e-15)
    s2 = axisym_solution(a=A, bc_outer="soft", D=D, nu=NU, q_coeffs=(Q0,))
    # мягкий шарнир пакета: предел ν → 1 формулы Тимошенко (NOTES §8)
    ref2 = Q0 * (A**2 - r**2) * (3 * A**2 - r**2) / (64 * D)
    assert np.allclose(s2.w(r, 0 * r), ref2, rtol=1e-12, atol=1e-15)


def test_identity_circle_point():
    """Круг, сила в центре ≡ analytic.circle_point_{clamped,soft}."""
    s = axisym_solution(a=A, bc_outer="clamped", D=D, nu=NU, P=P)
    r = np.linspace(0.05, 0.95, 7)
    ref = analytic.circle_point_clamped(r, A, P, D)
    assert np.allclose(s.w(r, 0 * r), np.asarray(ref, float), rtol=1e-12)
    s2 = axisym_solution(a=A, bc_outer="soft", D=D, nu=NU, P=P)
    ref2 = analytic.circle_point_soft(r, A, P, D)
    assert np.allclose(s2.w(r, 0 * r), np.asarray(ref2, float), rtol=1e-12)


@pytest.mark.parametrize("bc", ["clamped", "soft", "true_ss"])
def test_identity_annulus_uniform(bc):
    """Кольцо (обе кромки один тип) ≡ analytic.annulus_uniform (4×4)."""
    s = axisym_solution(a=A, b=B, bc_outer=bc, bc_inner=bc, D=D, nu=NU,
                        q_coeffs=(Q0,))
    r = np.linspace(B + 1e-3, A - 1e-3, 9)
    ref = analytic.annulus_uniform(r, A, B, Q0, D, nu=NU, bc=bc)
    scale = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(s.w(r, 0 * r) - ref))) <= 1e-12 * scale


def test_identity_navier_levy_uniform():
    """Навье/Леви uniform ≡ ручным рядам (в центре и вне центра)."""
    sn = navier_solution(x1=0, x2=1, y1=0, y2=1, D=D,
                         load={"type": "uniform", "q0": Q0})
    for x, y in ((0.5, 0.5), (0.3, 0.7)):
        ref = float(analytic.navier_rect_uniform(x, y, 0, 1, 0, 1, Q0, D))
        assert float(sn.w(x, y)) == pytest.approx(ref, rel=1e-10)
    sl = levy_solution(x1=0, x2=1, y1=0, y2=1, D=D, q0=Q0,
                       bc_y1="clamped", bc_y2="clamped")
    ref = float(analytic.levy_rect_uniform(0.5, 0.5, 0, 1, 0, 1, Q0, D))
    assert float(sl.w(0.5, 0.5)) == pytest.approx(ref, rel=1e-9)


def test_identity_strip_closed_forms():
    """Полоса HH/CC uniform ≡ классическим прогибам балки-полосы."""
    st = strip_solution(x1=0.0, x2=1.0, bc_left="hinge", bc_right="hinge",
                        D=D, q_coeffs=(Q0,))
    assert float(st.w(0.5)) == pytest.approx(5 * Q0 / (384 * D), rel=1e-12)
    sc = strip_solution(x1=0.0, x2=1.0, bc_left="clamped", bc_right="clamped",
                        D=D, q_coeffs=(Q0,))
    assert float(sc.w(0.5)) == pytest.approx(Q0 / (384 * D), rel=1e-12)


# --------------------------------------------------------------------------- #
#  F4.2: самосертификация и новое покрытие
# --------------------------------------------------------------------------- #
def test_certificates_present_and_small():
    """Сертификат заполнен и мал; PDE осесимметрии — символьная подстановка."""
    s = axisym_solution(a=A, b=B, bc_outer="clamped", bc_inner="soft",
                        D=D, nu=NU, q_coeffs=(Q0, 0.0, 1.5))  # q = q0 + 1.5r²
    assert all(abs(v) <= 1e-10 for v in s.certificate.values())
    # независимая проверка PDE полинома: D·ΔΔw − q = 0 численно на сетке
    r = sp.symbols("r", positive=True)
    w_expr = s.meta["w_expr"]
    lap = lambda f: sp.diff(f, r, 2) + sp.diff(f, r) / r     # noqa: E731
    resid = sp.lambdify(r, D * lap(lap(w_expr)) - (Q0 + 1.5 * r**2))
    rr = np.linspace(B + 0.05, A - 0.05, 11)
    assert float(np.max(np.abs(resid(rr)))) <= 1e-9 * Q0


def test_mixed_edge_annulus_new_coverage():
    """Кольцо clamped-снаружи / soft-изнутри: физика между однородными."""
    both_soft = axisym_solution(a=A, b=B, bc_outer="soft", bc_inner="soft",
                                D=D, nu=NU, q_coeffs=(Q0,))
    both_cl = axisym_solution(a=A, b=B, bc_outer="clamped", bc_inner="clamped",
                              D=D, nu=NU, q_coeffs=(Q0,))
    mixed = axisym_solution(a=A, b=B, bc_outer="clamped", bc_inner="soft",
                            D=D, nu=NU, q_coeffs=(Q0,))
    r = np.linspace(B + 1e-3, A - 1e-3, 200)
    w_s = float(np.max(np.abs(both_soft.w(r, 0 * r))))
    w_c = float(np.max(np.abs(both_cl.w(r, 0 * r))))
    w_m = float(np.max(np.abs(mixed.w(r, 0 * r))))
    assert w_c < w_m < w_s                        # жёстче ⇒ меньше прогиб


def test_navier_point_reciprocity():
    """Точечная сила: взаимность Максвелла w(x₁; ξ₂) = w(x₂; ξ₁)."""
    s12 = navier_solution(x1=0, x2=1, y1=0, y2=1, D=D, tol=1e-6,
                          load={"type": "point", "P": P, "x0": 0.4, "y0": 0.6})
    s21 = navier_solution(x1=0, x2=1, y1=0, y2=1, D=D, tol=1e-6,
                          load={"type": "point", "P": P, "x0": 0.7, "y0": 0.3})
    w_a = float(s12.w(0.7, 0.3))
    w_b = float(s21.w(0.4, 0.6))
    assert w_a == pytest.approx(w_b, rel=1e-6)


def test_levy_asym_reflection_and_monotonicity():
    """S(C,H): отражение y → −y меняет пары местами; жёсткость монотонна."""
    sch = levy_solution(x1=0, x2=1, y1=0, y2=1, D=D, q0=Q0,
                        bc_y1="clamped", bc_y2="hinge")
    shc = levy_solution(x1=0, x2=1, y1=0, y2=1, D=D, q0=Q0,
                        bc_y1="hinge", bc_y2="clamped")
    for x, y in ((0.3, 0.25), (0.5, 0.7), (0.62, 0.4)):
        assert float(sch.w(x, y)) == pytest.approx(float(shc.w(x, 1 - y)),
                                                   rel=1e-12)
    ssss = navier_solution(x1=0, x2=1, y1=0, y2=1, D=D,
                           load={"type": "uniform", "q0": Q0})
    scsc = levy_solution(x1=0, x2=1, y1=0, y2=1, D=D, q0=Q0,
                         bc_y1="clamped", bc_y2="clamped")
    mid = float(sch.w(0.5, 0.5))
    assert float(scsc.w(0.5, 0.5)) < mid < float(ssss.w(0.5, 0.5))


def test_strip_polynomial_load_certified():
    """Полоса с q(x) = q0 + 2x + 3x³: сертификат и физический контроль."""
    st = strip_solution(x1=0.0, x2=1.0, bc_left="clamped", bc_right="hinge",
                        D=D, q_coeffs=(Q0, 2.0, 0.0, 3.0))
    assert all(abs(v) <= 1e-10 for v in st.certificate.values())
    xs = np.linspace(0.01, 0.99, 50)
    w = np.asarray(st.w(xs), float)
    assert np.all(w > 0.0)                         # прогиб вниз всюду


# --------------------------------------------------------------------------- #
#  Ограда: отказы
# --------------------------------------------------------------------------- #
def test_factory_fence_errors():
    with pytest.raises(FactoryError, match="кольцо"):
        axisym_solution(a=A, b=1.5, bc_outer="soft", bc_inner="soft",
                        D=D, nu=NU, q_coeffs=(Q0,))
    with pytest.raises(FactoryError, match="точечная сила"):
        axisym_solution(a=A, b=B, bc_outer="soft", bc_inner="soft",
                        D=D, nu=NU, P=P)
    with pytest.raises(FactoryError, match="КУ кромки"):
        axisym_solution(a=A, bc_outer="free", D=D, nu=NU, q_coeffs=(Q0,))
    with pytest.raises(FactoryError, match="Леви"):
        levy_solution(x1=0, x2=1, y1=0, y2=1, D=D, q0=Q0,
                      bc_y1="free", bc_y2="hinge")
    with pytest.raises(FactoryError, match="полосы"):
        strip_solution(x1=0, x2=1, bc_left="hinge", bc_right="soft",
                       D=D, q_coeffs=(Q0,))


def test_resolver_new_coverage_and_refusals():
    """Резолвер: patch/point на прямоугольнике работают; вне ограды — отказ."""
    from plate_solver.problem import CaseError, Problem
    from plate_solver.references import resolve_reference

    base = {"geometry": {"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                         "y1": 0.0, "y2": 1.0},
            "bc": {"type": "soft_hinge"},
            "model": {"h": 0.06},
            "discretization": {"p": 6, "Q": 64, "grid_n": 8},
            "verify": {"reference": "analytic", "tol": 1.0}}
    ok = dict(base, load={"type": "patch", "q0": 4.0,
                          "zone": {"kind": "rectangle", "x1": 0.2, "x2": 0.5,
                                   "y1": 0.2, "y2": 0.5}})
    refs = resolve_reference(Problem.from_dict(ok))
    assert refs and refs[0].point is not None
    # круглая зона патча — вне ограды Навье
    bad = dict(base, load={"type": "patch", "q0": 4.0,
                           "zone": {"kind": "circle", "a": 0.2}})
    with pytest.raises(CaseError, match="прямоугольн"):
        resolve_reference(Problem.from_dict(bad))
    # patch на L-форме — по-прежнему отказ
    bad2 = {"geometry": {"kind": "L", "side": 1.0, "cut": 0.5},
            "bc": {"type": "soft_hinge"},
            "load": {"type": "patch", "q0": 4.0,
                     "zone": {"kind": "rectangle", "x1": 0.1, "x2": 0.3,
                              "y1": 0.1, "y2": 0.3}},
            "model": {"h": 0.06},
            "discretization": {"p": 6, "Q": 64, "grid_n": 8},
            "verify": {"reference": "analytic", "tol": 1.0}}
    with pytest.raises(CaseError):
        resolve_reference(Problem.from_dict(bad2))


# --------------------------------------------------------------------------- #
#  F4.4: ladder-ступени нового покрытия (боевые параметры)
# --------------------------------------------------------------------------- #
@pytest.mark.big
@pytest.mark.parametrize("case", ["rect_navier_patch", "rect_navier_point",
                                  "rect_levy_asym"])
def test_ladder_new_coverage(case):
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem
    from plate_solver.references import verify_result

    res = solve(Problem.from_toml(_ROOT / "cases" / "ladder" / f"{case}.toml"))
    rep = verify_result(res)
    assert rep.ok, "\n" + rep.table()


# --------------------------------------------------------------------------- #
#  F4.7: осесимметричный контактный эталон (круг + плоское основание)
# --------------------------------------------------------------------------- #
def test_contact_reference_certificate_and_green_limit():
    """Сертификат конструкции; предел Δ → w_free: c → 0, полная кольцевая
    сила → 8πD(w_free − Δ)/a² (асимптотика функции Грина) → 0."""
    from plate_solver.analytic_auto import axisym_contact_solution
    from plate_solver.config import Config

    cfg = Config(q0=Q0, h=0.06, p=8, Q=64)
    Dv = cfg.D
    w_free0 = 3 * Q0 * A**4 / (64 * Dv)
    prev_c = np.inf
    for frac in (0.5, 0.7, 0.8, 0.9):
        ref = axisym_contact_solution(a=A, D=Dv, q0=Q0, gap=frac * w_free0)
        assert all(abs(v) <= 1e-8 for v in ref.certificate.values())
        assert ref.meta["P_ring"] >= 0.0
        assert ref.meta["c"] < prev_c                  # зона монотонно тает
        prev_c = ref.meta["c"]
    # асимптотика Грина в глубоком пределе
    ref9 = axisym_contact_solution(a=A, D=Dv, q0=Q0, gap=0.9 * w_free0)
    green = 8 * np.pi * Dv * w_free0 * 0.1 / A**2
    assert ref9.meta["ring_force_total"] == pytest.approx(green, rel=1e-3)
    # вне контакта — отказ ограды
    with pytest.raises(FactoryError, match="w_free"):
        axisym_contact_solution(a=A, D=Dv, q0=Q0, gap=1.1 * w_free0)


def test_contact_reference_vs_mor_gates():
    """Ворота F4.7: МОР против замкнутого эталона — w_max и полная сила.

    Потолок 1e-2 выдержан; заморожено «факт × 3» (p=8, Q=128, 4000 итер.:
    rel_w = 3.88e-3 → 1.2e-2; rel_F = 3.06e-3 → 9.2e-3). Радиус зоны по
    УЗЛОВОЙ реакции НЕ гейтится: при контакте «плато + кольцевой δ-слой»
    узловое распределение реакции дискретно неединственно (ядро проекции,
    M ≫ N — тот же эффект, что у пары пластин); инварианты дискретной
    задачи — прогиб и интеграл силы, они и гейтятся.
    """
    from plate_solver import geometry
    from plate_solver.analytic_auto import axisym_contact_solution
    from plate_solver.config import Config
    from plate_solver.contact import solve_contact

    cfg = Config(q0=Q0, h=0.06, p=8, Q=128, beta=1.0, max_iter=4000,
                 tol=1e-12)
    Dv = cfg.D
    w_free0 = 3 * Q0 * A**4 / (64 * Dv)
    gap = 0.5 * w_free0
    ref = axisym_contact_solution(a=A, D=Dv, q0=Q0, gap=gap)
    cfg = Config(q0=Q0, h=0.06, p=8, Q=128, beta=1.0, max_iter=4000,
                 tol=1e-12, Delta=gap)
    res = solve_contact(cfg, geometry.make_circle(A))
    qd = res.plate.quad
    F_mor = float(np.sum(qd.w * res.r_nodes))
    F_ref = float(np.pi * ref.meta["c"] ** 2 * Q0 + ref.meta["ring_force_total"])
    w_max = float(np.max(np.abs(res.w_nodes)))
    assert abs(w_max - gap) / gap <= 1.2e-2
    assert abs(F_mor - F_ref) / F_ref <= 9.2e-3
    assert float(res.r_nodes.min()) >= 0.0
