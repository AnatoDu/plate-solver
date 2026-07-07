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
    """Gate R5: лицевой прогиб faces.py = ktn_linear (contact_displacement) число-в-число."""
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
