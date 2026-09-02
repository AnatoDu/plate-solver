r"""Ворота лицевых величин первым классом (faces.py, веха N1 v0.5.0).

Проверяется: тождество параметров толщины §3.2; канонические значения при
ν=0.3; соответствие устаревшим именам ``ktn.py`` (h_ψ²/h_*²/h_c²) — чтобы
линейные лицевые величины ``ktn_linear`` считались ЧИСЛО-В-ЧИСЛО (Gate R5);
мембранный вклад лицевых напряжений.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import faces
from plate_solver.config import Config
from plate_solver.faces import FaceParams
from plate_solver.ktn import KTNParams


def test_thickness_identity():
    """§3.2: h_c² = h_ψ² − h_*² (assert в __post_init__ + явная проверка)."""
    fp = FaceParams(E=2.1e6, nu=0.3, h=0.1)         # __post_init__ не упал
    assert fp.h_c_sq == pytest.approx(fp.h_psi_sq - fp.h_star_sq)


def test_canonical_values_nu_030():
    """Канонические коэффициенты при ν=0.3 (прил. A): 0.2381 / 0.1845 / 0.0536 · h²."""
    fp = FaceParams(E=1.0, nu=0.3, h=1.0)
    assert fp.h_psi_sq == pytest.approx(0.238095, abs=1e-5)
    assert fp.h_star_sq == pytest.approx(0.184524, abs=1e-5)
    assert fp.h_c_sq == pytest.approx(0.053571, abs=1e-5)


def test_naming_correspondence_to_ktn():
    """Соответствие устаревших имён ktn.py канону §3.2 (историческая путаница §12)."""
    fp = FaceParams(E=2.1e6, nu=0.28, h=0.07)
    kp = KTNParams(E=2.1e6, nu=0.28, h=0.07)
    assert fp.h_psi_sq == pytest.approx(kp.h_psi2)     # h_ψ²
    assert fp.h_c_sq == pytest.approx(kp.h_star2)      # ktn "h_star2" = h_c²!
    assert fp.h_star_sq == pytest.approx(kp.h_z2)      # ktn "h_z2" = h_*²
    assert fp.c_curv == pytest.approx(kp.c_curv)       # коэффициент при Δw совпал


def test_gate_r5_face_deflection_matches_ktn_linear():
    """Gate R5 (страж слоёв): лицевой прогиб faces.py = ktn_linear число-в-число.

    Утверждение по построению тавтологично (обёртка вызывает оборачиваемую
    функцию), но нужно как СТРАЖ: если слои разойдутся — например, ``faces.py``
    заведёт собственную сборку, — ``ktn_linear`` молча сдвинет регресс.
    Содержательная проверка самой формулы — в тесте ниже.
    """
    fp = FaceParams(E=2.1e6, nu=0.3, h=0.1)
    kp = KTNParams(E=2.1e6, nu=0.3, h=0.1)
    rng = np.random.default_rng(0)
    w = rng.standard_normal(20)
    lap = rng.standard_normal(20)
    q0, r = 4.0, np.abs(rng.standard_normal(20))
    assert np.allclose(fp.face_deflection(w, lap, q0, r, surface="bottom"),
                       kp.contact_displacement(w, lap, q0, r), rtol=0, atol=0)
    assert np.allclose(fp.mid_corrected(w, lap, q0, r),
                       kp.corrected_deflection(w, lap, q0, r), rtol=0, atol=0)
    # верхняя грань — по канону §21.1 совпадает со срединной
    assert np.allclose(fp.face_deflection(w, lap, q0, r, surface="top"), w)


@pytest.mark.parametrize("nu,h,E", [(0.3, 1.0, 1.0), (0.28, 1.0, 1.0),
                                    (0.15, 0.8, 2.0)])
def test_face_deflection_equals_independent_formula_9(nu, h, E):
    r"""Лицевой прогиб = НЕЗАВИСИМО собранная формула (9) — коэффициенты и адресация.

    .. math:: u_c = w + c_{curv}\,\Delta w - \kappa_q q^+ - \kappa_r r

    Коэффициенты выписаны ЗАМКНУТЫМИ формулами (не вызовом сборки кода):

    .. math::
        c_{curv} = h_c^2 - h_*^2 = \frac{(3\nu-2)h^2}{12(1-\nu)},\quad
        \kappa_q = \frac{(1+\nu)(2-4\nu-3\nu^2)h}{16E(1-\nu)},\quad
        \kappa_r = \frac{3(1+\nu)(2-4\nu+\nu^2)h}{16E(1-\nu)}.

    Параметры подобраны так, чтобы все три слагаемых были ОДНОГО порядка
    (E = h = 1): иначе q- и r-члены тонут в ``w`` и проверка знаков теряет
    силу. Контроль не-вакуумности — перестановка ``κ_q ↔ κ_r`` и смена знака
    любого слагаемого обязаны ломать совпадение.
    """
    fp = FaceParams(E=E, nu=nu, h=h)
    rng = np.random.default_rng(7)
    w = rng.standard_normal(24)
    lap = rng.standard_normal(24)
    q_n = 0.7
    r = np.abs(rng.standard_normal(24))

    c_curv = (3.0 * nu - 2.0) * h**2 / (12.0 * (1.0 - nu))
    kappa_q = (1.0 + nu) * (2.0 - 4.0 * nu - 3.0 * nu**2) * h / (16.0 * E * (1.0 - nu))
    kappa_r = 3.0 * (1.0 + nu) * (2.0 - 4.0 * nu + nu**2) * h / (16.0 * E * (1.0 - nu))
    assert fp.c_curv == pytest.approx(c_curv, rel=1e-13)
    assert fp.kappa_q == pytest.approx(kappa_q, rel=1e-12)
    assert fp.kappa_r == pytest.approx(kappa_r, rel=1e-12)
    assert kappa_q > 0.0 and kappa_r > 0.0 and c_curv < 0.0   # знаки податливостей

    u_c = fp.face_deflection(w, lap, q_n, r, surface="bottom")
    formula_9 = w + c_curv * lap - kappa_q * q_n - kappa_r * r
    assert np.allclose(u_c, formula_9, rtol=1e-12, atol=0.0)

    # каждое слагаемое ощутимо: смена знака / перестановка коэффициентов ловится
    for wrong in (w + c_curv * lap + kappa_q * q_n - kappa_r * r,
                  w + c_curv * lap - kappa_q * q_n + kappa_r * r,
                  w - c_curv * lap - kappa_q * q_n - kappa_r * r,
                  w + c_curv * lap - kappa_r * q_n - kappa_q * r):
        assert not np.allclose(u_c, wrong, rtol=1e-9, atol=0.0)


def test_from_config():
    fp = FaceParams.from_config(Config(E=1.0, nu=0.25, h=0.2))
    assert (fp.E, fp.nu, fp.h) == (1.0, 0.25, 0.2)


def test_introspection_with_length():
    """Интроспекция §6.3: h/L и порядок (h/L)² при заданном L."""
    fp = FaceParams(E=1.0, nu=0.3, h=0.1)
    d = fp.introspection(length=1.0)
    assert set(d) >= {"h_psi_sq", "h_star_sq", "h_c_sq", "c_curv", "h_over_L", "order_h2_L2"}
    assert d["h_over_L"] == pytest.approx(0.1)
    assert d["order_h2_L2"] == pytest.approx(0.01)
    assert "h_over_L" not in fp.introspection()      # без L — только параметры


def test_membrane_face_stress():
    """Мембранный вклад лицевых напряжений σ = N/h (обе грани одинаково)."""
    m = faces.membrane_face_stress(np.array([2.0]), np.array([3.0]), np.array([1.0]), h=0.5)
    assert m["sx_m"][0] == pytest.approx(4.0) and m["sy_m"][0] == pytest.approx(6.0)


def test_face_stresses_adds_membrane():
    """face_stresses без N = чистый изгиб (ktn.stresses_faces); с N — плюс N/h."""
    Mx = np.array([1.0])
    s0 = faces.face_stresses(Mx, Mx, Mx, h=0.5, nu=0.3)
    s1 = faces.face_stresses(Mx, Mx, Mx, h=0.5, nu=0.3,
                             Nx=np.array([2.0]), Ny=np.array([2.0]), Nxy=np.array([0.0]))
    assert s1["sx_top"][0] == pytest.approx(s0["sx_top"][0] + 2.0 / 0.5)
    assert s1["sx_bot"][0] == pytest.approx(s0["sx_bot"][0] + 2.0 / 0.5)


def test_result_thickness_params_introspection():
    """Result.thickness_params (§6.3): интроспекция из результата любой теории."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_dict({
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "ktn_linear", "h": 0.1},
        "discretization": {"p": 8, "Q": 64, "grid_n": 40},
    }))
    tp = res.thickness_params()
    fp = FaceParams(E=res.config.E, nu=res.config.nu, h=0.1)
    assert tp["h_psi_sq"] == pytest.approx(fp.h_psi_sq)
    assert tp["h_over_L"] == pytest.approx(0.1, rel=1e-2)   # круг a=1 ⇒ L≈1
