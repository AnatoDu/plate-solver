r"""Верификация полной КТН на МЯГКОМ ШАРНИРЕ (v0.6.3, §3.5).

Отличие шарнира от защемления — ОДИН граничный член слабой формы
``−h_ψ²∮ L·∂ψ/∂n ds`` (формула Грина при ``v = 0`` на ∂Ω). Здесь — весь доступный
арсенал перепроверки:

* **Тождество Грина** — граничный член ВЕРЕН и НЕОБХОДИМ: дискретно
  ``∫ v ΔL = ∫ L Δv − ∮ L ∂ψ/∂n`` держится до точности квадратуры, без члена — ~50 %;
* **∂ψ/∂n = 0 при защемлении** — почему защемлению граничный член не нужен;
* **редукции** — члены-выкл → Карман (бит-точно), ``h → 0`` → Кирхгоф;
* **сходимость по p** — спектральная стабилизация;
* **стыковка с теориями** — шарнир податливее защемления; малая нагрузка →
  линейный Кирхгоф; непрерывный морфинг Карман↔КТН (``refinement_scale``);
* **контакт на шарнире** — редукция R1, инварианты, лицевая подпись.

Независимого литературного эталона полной КТН нет (как и отложенный gate R3 для
защемления); краевая модель §3.5 предполагается стандартной (``w = 0``, ``M_n = 0``).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from plate_solver import theory
from plate_solver.config import Config
from plate_solver.contact_nl import NonlinearContactMOR
from plate_solver.geometry import (
    make_annulus,
    make_circle,
    make_ellipse,
    make_L,
    make_rectangle,
    x,
    y,
)
from plate_solver.ktn_full import KTNPlate, _boundary_quad, _star_boundary_quad
from plate_solver.ktn_solver import KTNSolver
from plate_solver.membrane import KarmanPlate
from plate_solver.quadrature import interior_nodes


def _cfg(h=0.2, q0=0.02, p=10, Q=140):
    return Config(E=1.0, nu=0.3, h=h, q0=q0, p=p, Q=Q,
                  n_load_steps=3, karman_tol=1e-9, karman_max_iter=200,
                  karman_relax=1.0, beta=1.5, max_iter=3000, tol=3e-4)


# --------------------------------------------------------------------------- #
#  1. Тождество Грина: граничный член ВЕРЕН и НЕОБХОДИМ
# --------------------------------------------------------------------------- #
def _green_identity_residual(dom, Q=300):
    r"""Относительная невязка ``∫vΔL − (∫LΔv − ∮L∂v/∂n)`` (с членом и без)."""
    q = interior_nodes(dom, Q)
    Xb, Yb, ds, nx, ny = _boundary_quad(dom)               # звёздная ИЛИ контурная
    Nx, Ny, Nxy = 1.3, 0.7, 0.4
    h = 1 + 0.5 * x - 0.3 * y + 0.25 * x**2 - 0.15 * y**2 + 0.2 * x * y
    u = (0.8 - 0.4 * x + 0.6 * y - 0.2 * x**2 + 0.3 * y**2 + 0.35 * x * y
         - 0.1 * x**3 + 0.15 * y**3 + 0.12 * x**4 - 0.08 * y**4
         + 0.05 * x**2 * y**2 + 0.09 * x**3 * y - 0.06 * x * y**3)
    v = dom.omega_expr * h                                # v = 0 на ∂Ω
    L = Nx * sp.diff(u, x, 2) + 2 * Nxy * sp.diff(u, x, y) + Ny * sp.diff(u, y, 2)
    lap = lambda e: sp.diff(e, x, 2) + sp.diff(e, y, 2)   # noqa: E731
    f = lambda e: sp.lambdify((x, y), e, "numpy")         # noqa: E731
    LHS = float(np.sum(q.w * f(v)(q.x, q.y) * f(lap(L))(q.x, q.y)))
    RHS_vol = float(np.sum(q.w * f(L)(q.x, q.y) * f(lap(v))(q.x, q.y)))
    dvdn = f(sp.diff(v, x))(Xb, Yb) * nx + f(sp.diff(v, y))(Xb, Yb) * ny
    RHS_bnd = float(np.sum(ds * f(L)(Xb, Yb) * dvdn))
    scale = abs(LHS) + abs(RHS_vol) + abs(RHS_bnd)
    return abs(LHS - (RHS_vol - RHS_bnd)) / scale, abs(LHS - RHS_vol) / scale


@pytest.mark.parametrize("dom", [make_circle(1.0), make_ellipse(1.5, 1.0),
                                 make_rectangle(-1.0, 1.0, -0.7, 0.7),
                                 make_annulus(1.0, 0.4),
                                 make_L(1.0, 0.5)])
def test_green_identity_boundary_term_correct_and_necessary(dom):
    """Граничный член верен (с ним ~ точность квадратуры) и НЕОБХОДИМ (без него ~50%).

    Проверено на круге, эллипсе, прямоугольнике (выпуклые углы), кольце
    (МНОГОСВЯЗНАЯ, контурная квадратура) и L-ФОРМЕ (входящий угол — ТОЧНАЯ
    пореберная полигонная квадратура, v0.6.5; контурная там давала ~8 %).
    Тождество — с ГЛАДКИМИ полями; сингулярность структуры ω¹ у реентрантного
    угла — отдельный вопрос (NOTES §9), из-за которого ktn_full+soft_hinge на L
    остаётся закрытым валидатором.
    """
    rel_with, rel_without = _green_identity_residual(dom)
    assert rel_with < 5e-3                                # держится до точности квадратуры
    assert rel_without > 0.1                              # без члена — грубая ошибка
    assert rel_without > 50 * rel_with                   # член решает


def test_boundary_normal_derivative_zero_for_clamped():
    """Защемлению граничный член не нужен: ∂ψ/∂n = 0 на ∂Ω для структуры ω² (Лейбниц)."""
    dom = make_circle(1.0)
    ktn = KTNPlate.from_config(dom, _cfg(), bc_type="clamped")
    Xb, Yb, _ds, nx, ny = _star_boundary_quad(dom, 1000)
    from plate_solver.membrane import _w_structure
    _, psx, psy, *_ = _w_structure(dom, ktn.basis, Xb, Yb, 2)   # power = 2 (защемление)
    dpsidn = psx * nx + psy * ny
    assert np.max(np.abs(dpsidn)) < 1e-9


# --------------------------------------------------------------------------- #
#  2. Редукции
# --------------------------------------------------------------------------- #
def test_reduction_terms_off_equals_karman():
    """КТН-члены выкл ⇒ мягкий шарнир КТН ТОЖДЕСТВЕНЕН Карману (по построению)."""
    dom, cfg = make_circle(1.0), _cfg()
    ktn = KTNPlate.from_config(dom, cfg, bc_type="soft_hinge", include_ktn_terms=False)
    kar = KarmanPlate.from_config(dom, cfg, bc_type="soft_hinge")
    q = np.full(ktn.quad.x.size, cfg.q0)
    assert abs(ktn.solve(q).w_max - kar.solve(q).w_max) / kar.solve(q).w_max < 1e-12


def test_thin_limit_correction_shrinks():
    """Асимптотика h → 0: КТН-поправка к Карману гаснет (~(h/L)²)."""
    dom = make_circle(1.0)

    def corr(h):
        cfg = _cfg(h)
        ktn = KTNPlate.from_config(dom, cfg, bc_type="soft_hinge")
        kar = KarmanPlate.from_config(dom, cfg, bc_type="soft_hinge")
        q = np.full(ktn.quad.x.size, cfg.q0)
        return abs(ktn.solve(q).w_max - kar.solve(q).w_max) / kar.solve(q).w_max

    assert corr(0.05) < corr(0.2) < corr(0.4)


# --------------------------------------------------------------------------- #
#  3. Сходимость по p (спектральная стабилизация)
# --------------------------------------------------------------------------- #
def test_p_convergence():
    """Решение стабилизируется при росте p (спектральная сходимость)."""
    dom = make_circle(1.0)
    ws = []
    for p in (6, 9, 12):
        cfg = _cfg(p=p)
        ktn = KTNPlate.from_config(dom, cfg, bc_type="soft_hinge")
        ws.append(ktn.solve(np.full(ktn.quad.x.size, cfg.q0)).w_max)
    assert abs(ws[2] - ws[1]) < 0.3 * abs(ws[1] - ws[0]) + 1e-12   # убывающие приращения


# --------------------------------------------------------------------------- #
#  4. Стыковка с другими теориями
# --------------------------------------------------------------------------- #
def test_soft_hinge_more_compliant_than_clamped():
    """Физический порядок: мягкий шарнир гнётся сильнее защемления (как у Кирхгофа)."""
    dom, cfg = make_ellipse(1.5, 1.0), _cfg()
    sh = KTNPlate.from_config(dom, cfg, bc_type="soft_hinge")
    cl = KTNPlate.from_config(dom, cfg, bc_type="clamped")
    q = np.full(sh.quad.x.size, cfg.q0)
    assert sh.solve(q).w_max > cl.solve(q).w_max


def test_small_load_reduces_to_kirchhoff():
    """Малая нагрузка ⇒ мембранный член → 0 ⇒ КТН-шарнир ≈ линейный Кирхгоф.

    Сравнение с линейным пределом ТОЙ ЖЕ структуры (``w_max_classic``, N = 0):
    так исключается модельная погрешность мягкого шарнира между расщеплением
    (``PlateBending``) и структурным Ритцем (NOTES §8).
    """
    dom = make_circle(1.0)
    cfg = _cfg(q0=1e-6)                                   # почти линейный режим
    res = KTNPlate.from_config(dom, cfg, bc_type="soft_hinge").solve(
        np.full(interior_nodes(dom, cfg.Q).x.size, cfg.q0))
    assert abs(res.w_max - res.w_max_classic) / res.w_max_classic < 5e-3


def test_refinement_morph_karman_to_ktn():
    """Непрерывный морфинг: refinement_scale α=0 → Карман, α=1 → полная КТН (шарнир)."""
    dom, cfg = make_circle(1.0), _cfg()
    q = np.full(interior_nodes(dom, cfg.Q).x.size, cfg.q0)
    base = theory.ktn_full(cfg.nu, cfg.h)
    kar = KarmanPlate.from_config(dom, cfg, bc_type="soft_hinge").solve(q).w_max
    w0 = KTNSolver.from_config(dom, cfg, base.with_refinement_scale(0.0),
                               bc_type="soft_hinge").solve(q).w_max
    w1 = KTNSolver.from_config(dom, cfg, base.with_refinement_scale(1.0),
                               bc_type="soft_hinge").solve(q).w_max
    assert abs(w0 - kar) / kar < 1e-10                    # α=0 — точно Карман
    assert abs(w1 - kar) / kar > 1e-4                     # α=1 — заметная КТН-поправка


# --------------------------------------------------------------------------- #
#  5. Контакт на мягком шарнире (МОР+КТН)
# --------------------------------------------------------------------------- #
def _contact_cfg():
    # Лёгкая постановка контакта: сильное мембранное ужесточение мягкого шарнира
    # требует gain = linear (верхняя грань ‖G‖, гарантия сжатия теоремы 4).
    return Config(E=1.0, nu=0.3, h=0.2, q0=0.01, p=6, Q=48, n_load_steps=2,
                  karman_tol=1e-6, karman_max_iter=80, karman_relax=1.0,
                  beta=1.5, max_iter=2000, tol=3e-4)


@pytest.mark.big
def test_contact_soft_hinge_engages_and_face():
    """Контакт по КТН на шарнире (эллипс): инварианты + лицевое условие Синьорини."""
    dom, cfg = make_ellipse(1.5, 1.0), _contact_cfg()
    s = KTNSolver.from_theory_name(dom, cfg, "ktn_full", bc_type="soft_hinge")
    free = s.solve(np.full(s.quad.x.size, cfg.q0))
    mor = NonlinearContactMOR(s, cfg, gap=0.5 * float(np.max(np.abs(free.w_nodes))),
                              scheme="merged", gain_mode="linear")
    r = mor.solve()
    assert (r.r_nodes >= -1e-9).all() and r.r_max > 0.0 and r.n_contact > 0
    assert np.max(np.abs(r.u_c_nodes - r.w_nodes)) > 0.0  # лицевая ≠ срединной


def test_contact_soft_hinge_r1_reduction():
    """R1 на шарнире: зазор → ∞ ⇒ реакция ≡ 0, прогиб = свободному."""
    dom, cfg = make_circle(1.0), _contact_cfg()
    s = KTNSolver.from_theory_name(dom, cfg, "ktn_full", bc_type="soft_hinge")
    free = s.solve(np.full(s.quad.x.size, cfg.q0))
    r = NonlinearContactMOR(s, cfg, gap=100.0, scheme="nested", gain_mode="linear").solve()
    assert r.r_max == 0.0 and r.n_contact == 0
    assert abs(r.w_max - free.w_max) / free.w_max < 1e-6


def test_boundary_quad_dispatch_star_vs_contour():
    """Диспетчер ∂Ω: круг → звёздная (быстрая); кольцо → контурная (многосвязная)."""
    ann = make_annulus(1.0, 0.4)
    # низкоуровневая звёздная квадратура отвергает кольцо (центр в дырке)
    with pytest.raises(ValueError, match="звёздная"):
        _star_boundary_quad(ann)
    # диспетчер справляется: контурная квадратура ловит оба контура ω=0
    Xb, _Yb, ds, _nx, _ny = _boundary_quad(ann)
    assert Xb.size > 0
    assert abs(ds.sum() - 2 * np.pi * (1.0 + 0.4)) / (2 * np.pi * 1.4) < 5e-3


# --------------------------------------------------------------------------- #
#  Полигонная квадратура ∂Ω (v0.6.5): сертификация §3.5 на выпуклых углах
# --------------------------------------------------------------------------- #
def test_polygon_quadrature_certifies_rectangle_boundary_term():
    """§3.5 на прямоугольнике: полигонная квадратура = звёздная (выпуклые углы сходятся).

    ТОЧНАЯ пореберная квадратура кладёт узлы Гаусса сколь угодно близко к углам —
    совпадение со звёздной (др. распределение узлов) сертифицирует, что вклад
    §3.5 у ВЫПУКЛЫХ углов сходится (у РЕЕНТРАНТНОГО угла L — расходится, поэтому
    ktn_full+soft_hinge на L закрыт валидатором; NOTES §9).
    """
    cfg = _cfg(h=0.1, q0=6.0 * 0.1**4, p=8, Q=96)
    dom_star = make_rectangle(-1.0, 1.0, -0.6, 0.6)
    r_star = KTNPlate.from_config(dom_star, cfg, bc_type="soft_hinge",
                                  inplane_bc="immovable").solve_uniform()
    dom_poly = make_rectangle(-1.0, 1.0, -0.6, 0.6)
    dom_poly.polygon = ((-1.0, -0.6), (1.0, -0.6), (1.0, 0.6), (-1.0, 0.6))
    r_poly = KTNPlate.from_config(dom_poly, cfg, bc_type="soft_hinge",
                                  inplane_bc="immovable").solve_uniform()
    assert abs(r_star.w_max - r_poly.w_max) / r_star.w_max < 1e-6


def test_polygon_quadrature_stable_under_refinement_on_rectangle():
    """§3.5 (прямоугольник) стабилен при измельчении пореберной квадратуры.

    Узлы n/edge = 16 → 96 приближаются к углам в 36 раз — эффект §3.5 не дрейфует
    (7 значащих цифр в прототипе; здесь ворота 1e-5). Контраст: у входящего угла
    L дрейф ~5 % — доказательство, что проблема L в СЛАБОЙ ФОРМЕ, не в квадратуре.
    """
    import plate_solver.ktn_full as kf

    cfg = _cfg(h=0.1, q0=6.0 * 0.1**4, p=8, Q=96)
    orig = kf._polygon_boundary_quad
    vals = []
    try:
        for n in (16, 96):
            dom = make_rectangle(-1.0, 1.0, -0.6, 0.6)
            dom.polygon = ((-1.0, -0.6), (1.0, -0.6), (1.0, 0.6), (-1.0, 0.6))
            kf._polygon_boundary_quad = (
                lambda poly, n_per_edge=n, eps=1e-8: orig(poly, n_per_edge, eps))
            vals.append(KTNPlate.from_config(dom, cfg, bc_type="soft_hinge",
                                             inplane_bc="immovable").solve_uniform().w_max)
    finally:
        kf._polygon_boundary_quad = orig
    assert abs(vals[1] - vals[0]) / vals[0] < 1e-5
