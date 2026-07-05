"""Напряжения на лицевых поверхностях (фаза 3, трек B; NOTES §19).

B0: редукция полного вида (11) при T=0, b=0 — sympy-тождество.
т1: 1D-полоса — знаки фиксируются вручную посчитанным решением.
т2: тождество кода в центре защемлённого круга (1e-12).
т3: Mxy(центр) = 0 для осесимметрии. B1: (Mx, My, Mxy) против
sympy-аналитики для решения, лежащего в структуре (машинная точность).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from plate_solver import geometry
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.ktn import stresses_faces
from plate_solver.ladder import bending_moments, bending_moments_full


def test_b0_full_formula_reduces_to_kernel():
    """B0: полный вид (с T, b) при T = 0 редуцируется к ядру (11)."""
    T, M, b, h, nu, qn = sp.symbols("T M b h nu q_n", real=True, positive=False)
    for sign in (+1, -1):
        full = T / h + sign * 6 * (M + b * T) / h**2 + nu / (1 - nu) * qn
        kernel = sign * 6 * M / h**2 + nu / (1 - nu) * qn
        assert sp.simplify(full.subs(T, 0) - kernel) == 0


def test_b3_t1_strip_signs_fixed_by_hand():
    r"""т1: цилиндрический изгиб полосы — знаки против ручного 1D-решения.

    Шарнирная полоса [0, L], q = const: w(x) = q x (L³ − 2Lx² + x³)/(24D),
    M(x) = −D w'' = q x (L − x)/2 ≥ 0. По таблице §19 (ось z вниз):
    низ (z=+h/2) растянут: σ = +6M/h² > 0; верх сжат: −6M/h² (+ обжатие).
    """
    x = sp.symbols("x", positive=True)
    L, q0, D, h, nu = 1.0, 4.0, 100.0, 0.06, 0.3
    w = q0 * x * (L**3 - 2 * L * x**2 + x**3) / (24 * D)
    M = -D * sp.diff(w, x, 2)
    assert sp.simplify(M - q0 * x * (L - x) / 2) == 0     # ручное M(x)
    xs = np.linspace(0.05, 0.95, 7)
    Mv = np.array([float(M.subs(x, xi)) for xi in xs])
    s = stresses_faces(Mx=Mv, My=nu * Mv, Mxy=np.zeros_like(Mv),
                       h=h, nu=nu, q_top=q0, q_bottom=0.0)
    manual_bot = +6.0 * Mv / h**2                          # низ: растяжение
    manual_top = -6.0 * Mv / h**2 + nu / (1 - nu) * q0     # верх: сжатие + обжатие
    assert np.allclose(s["sx_bot"], manual_bot, rtol=0, atol=1e-12 * np.max(manual_bot))
    assert np.allclose(s["sx_top"], manual_top, rtol=0, atol=1e-12 * np.max(np.abs(manual_top)))
    assert np.all(s["sx_bot"] > 0.0)                       # физика: низ растянут
    assert np.all(s["sx_top"] < 0.0)                       # верх сжат (обжатие мало́)


def test_b3_t2_clamped_circle_center_identity():
    """т2: σr в центре круга через stresses_faces ≡ 6M_центр/h² (+обжатие), 1e-12."""
    a, q0 = 1.0, 4.0
    cfg = Config(q0=q0, h=0.06, p=8, Q=128)
    cp = ClampedPlate.from_config(geometry.make_circle(a), cfg)
    c = cp.solve_uniform(q0)
    Mx, My = bending_moments(cp.domain, cp.basis, c, 2, cfg.D, cfg.nu,
                             np.array([0.0]), np.array([0.0]))
    s = stresses_faces(Mx, My, np.zeros(1), h=cfg.h, nu=cfg.nu,
                       q_top=q0, q_bottom=0.0)
    manual_bot = 6.0 * float(Mx[0]) / cfg.h**2
    manual_top = -6.0 * float(Mx[0]) / cfg.h**2 + cfg.nu / (1 - cfg.nu) * q0
    assert float(s["sx_bot"][0]) == pytest.approx(manual_bot, rel=1e-12)
    assert float(s["sx_top"][0]) == pytest.approx(manual_top, rel=1e-12)
    # физика: центр — низ растянут (M > 0 в центре при q > 0)
    assert float(Mx[0]) > 0.0 and float(s["sx_bot"][0]) > 0.0


def test_b3_t3_axisymmetric_mxy_zero_at_center():
    """т3: осесимметричный случай — Mxy(центр) = 0 (симметрия)."""
    a, q0 = 1.0, 4.0
    cfg = Config(q0=q0, h=0.06, p=8, Q=128)
    cp = ClampedPlate.from_config(geometry.make_circle(a), cfg)
    c = cp.solve_uniform(q0)
    Mx, My, Mxy = bending_moments_full(cp.domain, cp.basis, c, 2, cfg.D, cfg.nu,
                                       np.array([0.0]), np.array([0.0]))
    assert abs(float(Mxy[0])) <= 1e-10 * abs(float(Mx[0]))
    assert float(My[0]) == pytest.approx(float(Mx[0]), rel=1e-8)   # изотропия центра


def test_b1_moments_full_vs_sympy_structure():
    """B1: (Mx, My, Mxy) для решения В СТРУКТУРЕ — против sympy (машинно).

    Прямоугольник с ПОЛИНОМИАЛЬНОЙ ω (домен = bbox, квадратура точна,
    NOTES §14): w = ω²·1 при c = e₀; сравнение всех трёх моментов с
    символьными производными ω² в 12 внутренних точках.
    """
    from plate_solver.basis import ChebyshevBasis
    from plate_solver.geometry import Domain
    from plate_solver.geometry import x as sx
    from plate_solver.geometry import y as sy

    ax, ay, D, nu = 1.0, 0.5, 100.0, 0.3
    om = (ax**2 - sx**2) * (ay**2 - sy**2)
    dom = Domain(om, (-ax, ax, -ay, ay))
    basis = ChebyshevBasis(4, dom.bbox)
    c = np.zeros(basis.N)
    c[0] = 1.0                                        # w = ω²
    w_expr = om**2
    rng = np.random.default_rng(3)
    X = rng.uniform(-0.9 * ax, 0.9 * ax, 12)
    Y = rng.uniform(-0.9 * ay, 0.9 * ay, 12)
    Mx, My, Mxy = bending_moments_full(dom, basis, c, 2, D, nu, X, Y)
    fx = sp.lambdify((sx, sy), -D * (sp.diff(w_expr, sx, 2) + nu * sp.diff(w_expr, sy, 2)))
    fy = sp.lambdify((sx, sy), -D * (sp.diff(w_expr, sy, 2) + nu * sp.diff(w_expr, sx, 2)))
    fxy = sp.lambdify((sx, sy), -D * (1 - nu) * sp.diff(w_expr, sx, sy))
    scale = float(np.max(np.abs(fx(X, Y))))
    assert float(np.max(np.abs(Mx - fx(X, Y)))) <= 1e-11 * scale
    assert float(np.max(np.abs(My - fy(X, Y)))) <= 1e-11 * scale
    assert float(np.max(np.abs(Mxy - fxy(X, Y)))) <= 1e-11 * scale


def test_b3_t4_contact_stress_profile_regression(tmp_path):
    """т4: σy_bot под штампом получает +ν/(1−ν)·r — регресс-снимок профиля."""
    import json
    from pathlib import Path

    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    root = Path(__file__).resolve().parents[1]
    base = json.loads((root / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["lshape_stamp_stress_profile"]
    res = solve(Problem.from_toml(root / "cases" / "ci" / "lshape_stamp.toml"))
    Mx, My, Mxy = res.moments_on_grid()
    q_top, q_bot = res._q_faces_on_grid()
    s = stresses_faces(Mx, My, Mxy, h=res.config.h, nu=res.config.nu,
                       q_top=q_top, q_bottom=q_bot)
    ys = res.Yg[:, 0]
    j = int(np.argmin(np.abs(ys - b["y"])))
    prof = s["sy_bot"][j, :]
    keep = np.isfinite(prof)
    got = prof[keep]
    assert len(got) == len(b["sy_bot"])
    # Допуск кросс-платформенный: ci-протокол МОР (200 итер., недосошедший)
    # чувствителен к ULP BLAS (Linux/OpenBLAS против Accelerate) — на CI
    # расхождение до ~1.5e-3 в точках зоны; регресс гейтит ФОРМУ профиля
    # и скачок обжатия, а не 5-й знак (побитовые регрессы — big, локально).
    assert np.allclose(got, b["sy_bot"], rtol=1e-2, atol=1e-2)
    # обжатие реально присутствует: q_bot > 0 в зоне на этой линии
    assert float(np.max(q_bot[j, :])) > 0.0


def test_b2_fields_npz_and_replot(tmp_path):
    """B2: fields.npz полон (схема 1) и viz.replot воспроизводит фигуры."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    from plate_solver import viz
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_toml(Path(__file__).resolve().parents[1]
                                  / "cases" / "ci" / "lshape_stamp.toml"))
    res.save(tmp_path)
    data = np.load(tmp_path / "fields.npz")
    need = {"fields_schema", "x", "y", "w", "Mx", "My", "Mxy", "sx_top", "sx_bot",
            "sy_top", "sy_bot", "txy_top", "txy_bot", "r", "zone", "problem_json"}
    assert need <= set(data.files)
    assert int(data["fields_schema"]) == 1
    paths = viz.replot(tmp_path, formats=("png", "pdf"))
    names = {p.name for p in paths}
    assert {"w_surface.png", "stress_faces.png", "reaction.png"} <= names
    for p in paths:
        if p.suffix == ".png":
            assert p.stat().st_size > 10_000              # непустая картинка
        else:
            assert p.read_bytes()[:4] == b"%PDF"           # валидный pdf


def test_d4_cli_figures_smoke(tmp_path, monkeypatch):
    """D4: смок фигур через CLI на Q=64 — png > 10 КБ, pdf валиден."""
    import matplotlib

    matplotlib.use("Agg")
    from pathlib import Path

    from plate_solver.cli import main

    case = Path(__file__).resolve().parents[1] / "cases" / "ci" / "rect_mms.toml"
    out = tmp_path / "figs"
    assert main([str(case), "--out", str(out), "--figures",
                 "--fig-format", "png,pdf"]) == 0
    pngs = list(out.glob("*_w_surface.png"))
    pdfs = list(out.glob("*_w_surface.pdf"))
    assert pngs and pdfs
    assert pngs[0].stat().st_size > 10_000
    assert pdfs[0].read_bytes()[:4] == b"%PDF"
    assert (out / "fields.npz").is_file()
