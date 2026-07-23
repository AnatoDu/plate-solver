r"""MMS-сертификат ПОЛНОЙ КТН при замороженных усилиях (fixed-N, §3.4/§3.5, v0.6.4).

Заморозка мембранных усилий ``N`` линеаризует внеплоскостной оператор КТН, и по
изготовленному ``w`` строится нагрузка, обращающая сильную форму

    D Δ²w − (I − h_ψ²Δ) L(w) = (I − h_*²Δ) q,   L(w) = N_x w_xx + 2 N_xy w_xy + N_y w_yy.

Линейный решатель (`KTNPlate.solve_fixed_forces`) на прямоугольнике с
полиномиальной ω воспроизводит ``w_MMS`` до МАШИННОЙ точности — единый сертификат
ВСЕХ членов сборки. Отдельно — ПОЛОЖИТЕЛЬНЫЕ и ОТРИЦАТЕЛЬНЫЕ проверки: член B
(``h_ψ²``) и член A (``h_*²``) доказываются и присутствием, и правильностью
(рассогласованная нагрузка НЕ воспроизводит решение).

Обоснование точности: ``w_MMS = ω²·1`` лежит в структуре защемлённого решателя,
а все подынтегральные выражения полиномиальны ⇒ квадратура точна ⇒ погрешность
только машинная (NOTES §14). Круг (кривая граница) даёт остаток маски ~1/Q.
"""

from __future__ import annotations

import numpy as np
import sympy as sp

from plate_solver.config import Config
from plate_solver.geometry import Domain
from plate_solver.geometry import x as sx
from plate_solver.geometry import y as sy
from plate_solver.ktn_full import KTNPlate
from plate_solver.ladder import mms_ktn_load_and_exact


def _rect_setup(h=0.1, p=10, Q=64):
    """Защемлённый прямоугольник, полиномиальная ω; каноническое замороженное N̄."""
    ax, ay = 1.0, 0.6
    w_expr = (sx**2 - ax**2) ** 2 * (sy**2 - ay**2) ** 2
    omega = (ax**2 - sx**2) * (ay**2 - sy**2)
    dom = Domain(omega, (-ax, ax, -ay, ay))
    cfg = Config(E=1.0, nu=0.3, h=h, q0=1.0, a=1.0, p=p, Q=Q)
    plate = KTNPlate.from_config(dom, cfg, bc_type="clamped", include_ktn_terms=True)
    n_ref = cfg.D / (2 * ax) ** 2
    N = (0.8 * n_ref, 0.5 * n_ref, -0.3 * n_ref)
    return plate, cfg, w_expr, N


def _rel(plate, c, w_func):
    qn = plate.quad
    w_ex = w_func(qn.x, qn.y)
    return np.max(np.abs(plate.deflection(c, qn.x, qn.y) - w_ex)) / np.max(np.abs(w_ex))


def test_ktn_mms_full_operator_machine_precision():
    """Полный оператор (изгиб+геом.+член B+член A) воспроизводит w_MMS машинно."""
    plate, cfg, w_expr, N = _rect_setup()
    qf, lqf, wf = mms_ktn_load_and_exact(w_expr, cfg.D, N,
                                         plate._h_psi_sq, plate._h_star_sq)
    qn = plate.quad
    c = plate.solve_fixed_forces(qf(qn.x, qn.y), *N, lap_q=lqf(qn.x, qn.y))
    assert _rel(plate, c, wf) < 1e-10


def test_ktn_mms_member_b_present_and_correct():
    """Член B (h_ψ²ΔL): корректная нагрузка — машинно; кармановская — рассогласована."""
    plate, cfg, w_expr, N = _rect_setup()
    qn = plate.quad
    # правильная нагрузка (с членом B) — оператор с членом B воспроизводит w
    qf, _, wf = mms_ktn_load_and_exact(w_expr, cfg.D, N, plate._h_psi_sq, 0.0)
    c_ok = plate.solve_fixed_forces(qf(qn.x, qn.y), *N)
    assert _rel(plate, c_ok, wf) < 1e-10
    # НАГРУЗКА БЕЗ члена B (h_ψ²→0), тот же оператор (член B присутствует) — НЕ сойдётся
    qf0, _, _ = mms_ktn_load_and_exact(w_expr, cfg.D, N, 0.0, 0.0)
    c_bad = plate.solve_fixed_forces(qf0(qn.x, qn.y), *N)
    # рассогласование ≫ машинного пола (1e-10) ⇒ член B реально в сборке; его физ.
    # величина ∝ h_ψ² мала, но на 5+ порядков выше точного воспроизведения
    assert _rel(plate, c_bad, wf) > 1e-5


def test_ktn_mms_member_a_present_and_correct():
    """Член A (−h_*²Δq): с Δq — машинно; без Δq (член A выключен) — рассогласовано."""
    plate, cfg, w_expr, N = _rect_setup()
    qn = plate.quad
    qf, lqf, wf = mms_ktn_load_and_exact(w_expr, cfg.D, N,
                                         plate._h_psi_sq, plate._h_star_sq)
    q_nodes = qf(qn.x, qn.y)
    c_ok = plate.solve_fixed_forces(q_nodes, *N, lap_q=lqf(qn.x, qn.y))
    assert _rel(plate, c_ok, wf) < 1e-10
    c_bad = plate.solve_fixed_forces(q_nodes, *N, lap_q=None)   # член A выключен
    assert _rel(plate, c_bad, wf) > 1e-5               # член A реально смещает нагрузку (≫ 1e-10)


def test_ktn_mms_karman_reduction():
    """Выключение членов КТН ⇒ кармановский оператор — тоже машинно (Gate R1-стиль)."""
    ax, ay = 1.0, 0.6
    w_expr = (sx**2 - ax**2) ** 2 * (sy**2 - ay**2) ** 2
    omega = (ax**2 - sx**2) * (ay**2 - sy**2)
    dom = Domain(omega, (-ax, ax, -ay, ay))
    cfg = Config(E=1.0, nu=0.3, h=0.1, q0=1.0, a=1.0, p=10, Q=64)
    plate = KTNPlate.from_config(dom, cfg, bc_type="clamped", include_ktn_terms=False)
    n_ref = cfg.D / (2 * ax) ** 2
    N = (0.8 * n_ref, 0.5 * n_ref, -0.3 * n_ref)
    qf, _, wf = mms_ktn_load_and_exact(w_expr, cfg.D, N, 0.0, 0.0)  # h_ψ²=0 ⇒ Карман
    qn = plate.quad
    c = plate.solve_fixed_forces(qf(qn.x, qn.y), *N)
    assert _rel(plate, c, wf) < 1e-10


def test_ktn_mms_screened_poisson_inversion_exact():
    """Обращение (I − h_*²Δ)q = S точно на полиномиальном S (ряд Неймана обрывается)."""
    ax, ay = 1.0, 0.6
    w_expr = (sx**2 - ax**2) ** 2 * (sy**2 - ay**2) ** 2
    cfg = Config(E=1.0, nu=0.3, h=0.1, q0=1.0, a=1.0, p=10, Q=64)
    hss = 0.05
    N = (0.7, 0.4, -0.25)
    qf, lqf, _ = mms_ktn_load_and_exact(w_expr, cfg.D, N, 0.02, hss)
    # проверяем тождество (q − h_*²Δq)(x,y) = S(x,y) в наборе точек
    L = (N[0] * sp.diff(w_expr, sx, 2) + 2 * N[2] * sp.diff(w_expr, sx, sy)
         + N[1] * sp.diff(w_expr, sy, 2))

    def lap(e):
        return sp.diff(e, sx, 2) + sp.diff(e, sy, 2)

    S = sp.lambdify((sx, sy), cfg.D * lap(lap(w_expr)) - L + 0.02 * lap(L), "numpy")
    xs = np.linspace(-0.9, 0.9, 7)
    X, Y = np.meshgrid(xs, xs * 0.6)
    lhs = qf(X, Y) - hss * lqf(X, Y)
    assert np.max(np.abs(lhs - S(X, Y))) < 1e-9 * (np.max(np.abs(S(X, Y))) + 1.0)
