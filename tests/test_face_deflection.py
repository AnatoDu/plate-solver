"""Прогиб лицевых поверхностей (NOTES §21): вывод, сверка канонов, тождества.

РЕШЕНИЕ АВТОРА (канон пакета): контактная кинематика — КТН (NOTES §21.1,
формулы кода); 3D-восстановление (§21.2) — независимая диагностика.
т1–т5 — символика восстановления (sympy): профиль σ_z, коэффициенты,
тождество кривизны с кинематикой КТН (подтверждает поверхность и знаки
канона), изменение толщины, редукция 2D→1D. т6 — ПОСТОЯННЫЕ ворота на
ОБА вывода и их точные разности Δb_q, Δb_r (не ослаблять). т7 — тождество
пути решателя: ручной пересчёт по канону 21.1 ≡ ktn.py (1e-12);
классика: смещение ≡ w.
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

E, nu, h = sp.symbols("E nu h", positive=True)
q, r = sp.symbols("q r", real=True)
z, s = sp.symbols("z s", real=True)
lapw = sp.Symbol("Delta_w", real=True)

D = E * h**3 / (12 * (1 - nu**2))
H_ST2 = nu * h**2 / (8 * (1 - nu))               # h_*²
H_PS2 = h**2 / (6 * (1 - nu))                    # h_Ψ²
H_Z2 = H_PS2 - H_ST2


def _sigma_z():
    """Профиль σ_z(z) по §21: равновесие + параболический τ + лицевые ГУ."""
    g = sp.integrate(sp.Rational(6) * (h**2 / 4 - s**2) / h**3, (s, -h / 2, z))
    return -q + (q - r) * g


def _faces():
    """(w_bot − w_mid, w_top − w_mid) интегрированием ε_z."""
    sxsy = 12 * (-D * (1 + nu) * lapw) * z / h**3
    eps_z = (_sigma_z() - nu * sxsy) / E
    bot = sp.expand(sp.integrate(eps_z, (z, 0, h / 2)))
    top = sp.expand(sp.integrate(eps_z, (z, 0, -h / 2)))
    return bot, top


def test_t1_sigma_z_profile_faces_and_equilibrium():
    """т1: σ_z(−h/2) = −q⁺, σ_z(+h/2) = −q⁻; полный перенос нагрузки q̃."""
    sz = _sigma_z()
    assert sp.simplify(sz.subs(z, -h / 2) + q) == 0
    assert sp.simplify(sz.subs(z, h / 2) + r) == 0
    # интеграл ∂σ_z/∂z по толщине = q̃ = q − r (равновесие столбика)
    assert sp.simplify(sz.subs(z, h / 2) - sz.subs(z, -h / 2) - (q - r)) == 0


def test_t2_face_coefficients_match_notes21():
    """т2: коэффициенты §21 — h_*²·Δw и ±h(a·q⁺ + b·q⁻)/(32E)."""
    bot, top = _faces()
    assert sp.simplify(bot.coeff(lapw) - H_ST2) == 0
    assert sp.simplify(top.coeff(lapw) - H_ST2) == 0
    assert sp.simplify(bot.coeff(q) + 3 * h / (32 * E)) == 0
    assert sp.simplify(bot.coeff(r) + 13 * h / (32 * E)) == 0
    assert sp.simplify(top.coeff(q) - 13 * h / (32 * E)) == 0
    assert sp.simplify(top.coeff(r) - 3 * h / (32 * E)) == 0


def test_t3_curvature_identity_with_ktn():
    """т3: 2h_*² − h_Ψ² ≡ h_*² − h_z² (обжатие лицевой + сдвиг срединной)."""
    assert sp.simplify((2 * H_ST2 - H_PS2) - (H_ST2 - H_Z2)) == 0


def test_t4_thickness_change_sign():
    """т4: dh = −h(q⁺ + q⁻)/(2E) — давление с любой стороны СЖИМАЕТ (dh < 0)."""
    bot, top = _faces()
    dh = sp.simplify(bot - top)
    assert sp.simplify(dh + h * (q + r) / (2 * E)) == 0
    val = dh.subs({q: 4.0, r: 2.0, h: 0.06, E: 100.0, nu: sp.Rational(3, 10)})
    assert float(val) < 0.0


def test_t5_reduction_2d_to_1d():
    """т5: цилиндрический изгиб (∂/∂y=0, M_y = νM_x) — те же коэффициенты.

    В 1D: M_x + M_y = (1+ν)M_x = −D(1+ν)w″ — подстановка Δw → w″ не меняет
    вид §21; проверяем через независимую 1D-сборку σ_x + σ_y.
    """
    wxx = sp.Symbol("w_xx", real=True)
    Mx = -D * wxx                                   # 1D: w_yy = 0
    sxsy_1d = 12 * (Mx + nu * Mx) * z / h**3
    eps_1d = (_sigma_z() - nu * sxsy_1d) / E
    bot1d = sp.expand(sp.integrate(eps_1d, (z, 0, h / 2)))
    bot2d, _ = _faces()
    assert sp.simplify(bot1d - bot2d.subs(lapw, wxx)) == 0


def test_t6_freeze_divergence_with_ktn_kinematics():
    """т6: разности q,r-членов восстановления (§21.2) и канона КТН (§21.1).

    Точные символьные Δb_q, Δb_r — ПОСТОЯННЫЕ ворота на оба вывода
    (решение автора: канон — КТН; восстановление — диагностика в пределах
    O(h²)-точности теории). Изменение ЛЮБОЙ из сторон роняет тест.
    """
    mu_ = E / (2 * (1 + nu))
    lamb = nu * E / ((1 + nu) * (1 - 2 * nu))
    cq_c = -(h / (8 * (lamb + 2 * mu_)) - H_ST2 / (mu_ * h) + H_ST2 * H_Z2 / D)
    cr_cD = -(3 * h / (8 * (lamb + 2 * mu_)) + H_ST2 / (mu_ * h) - H_ST2 * H_Z2 / D) * D
    bot, _ = _faces()
    dq = sp.simplify(bot.coeff(q) - cq_c)
    dr = sp.simplify(bot.coeff(r) - cr_cD)
    assert dq != 0 and dr != 0                      # расхождение реально
    # точные значения разностей — заморожены
    dq_ref = h * (6 * nu**3 + 14 * nu**2 + nu - 1) / (32 * E * (nu - 1))
    dr_ref = h * (E * h**3 * nu**2 - 4 * E * h**3 * nu + 2 * E * h**3
                  - 26 * nu**2 + 52 * nu - 26) / (64 * E * (nu - 1) ** 2)
    assert sp.simplify(dq - dq_ref) == 0
    assert sp.simplify(dr - dr_ref) == 0


def test_t7_solver_path_identity_and_classic():
    """т7: ручной пересчёт u_c по первоисточнику ≡ ktn.py (1e-12);
    классика — смещение контакта ≡ w (ContactMOR._contact_disp)."""
    from plate_solver import geometry
    from plate_solver.config import Config
    from plate_solver.contact import solve_contact
    from plate_solver.ktn import KTNParams
    from plate_solver.plate import PlateBending

    cfg = Config(q0=4.0, h=0.2, p=6, Q=64, Delta=1e-4, max_iter=200, beta=1.0)
    dom = geometry.make_circle(1.0)
    kp = KTNParams.from_config(cfg)
    res = solve_contact(cfg, dom, ktn=kp)
    # восстановление (w, Δw) протоколом решателя на финальной реакции
    plate = PlateBending.from_config(dom, cfg)
    st = plate.solve(cfg.q0 - res.r_nodes)
    w = plate.w_at_quad(st)
    lap_w = plate.lap_w_at_quad(st)
    E_, nu_, h_ = cfg.E, cfg.nu, cfg.h
    hl2 = nu_ * h_**2 / (8 * (1 - nu_))
    hp2 = h_**2 / (6 * (1 - nu_))
    hz2 = hp2 - hl2
    mu_ = E_ / (2 * (1 + nu_))
    lamb = nu_ * E_ / ((1 + nu_) * (1 - 2 * nu_))
    Dv = kp.D
    manual = (w + (2 * hl2 - hp2) * lap_w
              - (h_ / (8 * (lamb + 2 * mu_)) - hl2 / (mu_ * h_) + hl2 * hz2 / Dv) * cfg.q0
              - (3 * h_ / (8 * (lamb + 2 * mu_)) + hl2 / (mu_ * h_) - hl2 * hz2 / Dv)
              * Dv * res.r_nodes)
    solver_path = kp.contact_displacement(w, lap_w, cfg.q0, res.r_nodes)
    scale = float(np.max(np.abs(solver_path)))
    assert float(np.max(np.abs(manual - solver_path))) <= 1e-12 * scale
    # классика: смещение контакта — ровно w (без каких-либо поправок)
    from plate_solver.contact import ContactMOR

    mor_c = ContactMOR(plate, cfg)
    st_c = plate.solve(np.full(plate.quad.x.size, cfg.q0))
    w_c = plate.w_at_quad(st_c)
    disp_c = mor_c._contact_disp(st_c, w_c, np.zeros_like(w_c))
    assert disp_c is w_c or np.array_equal(disp_c, w_c)


def test_t7b_alias_w_face_bottom():
    """Имя w_face_bottom — документированный синоним contact_displacement."""
    from plate_solver.ktn import KTNParams

    kp = KTNParams(E=100.0, nu=0.3, h=0.1)
    w = np.array([1.0, 2.0])
    lw = np.array([0.1, -0.2])
    rr = np.array([0.0, 3.0])
    assert np.array_equal(kp.w_face_bottom(w, lw, 4.0, rr),
                          kp.contact_displacement(w, lw, 4.0, rr))


def test_pair_ktn_rejected_with_meaning(tmp_path):
    """ktn + plate2 отклоняется с пояснением про срединные плоскости."""
    from plate_solver.problem import CaseError, Problem

    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "ktn", "h": 0.06},
        "contact": {"enabled": True, "target": "plate2", "gap": 0.0},
        "plate2": {"geometry": {"kind": "circle", "a": 1.0},
                   "bc": {"type": "soft_hinge"},
                   "load": {"type": "uniform", "q0": 1.0}},
        "discretization": {"p": 6, "Q": 64, "grid_n": 16},
    }
    with pytest.raises(CaseError, match="срединн"):
        Problem.from_dict(d)


# --------------------------------------------------------------------------- #
#  Поля поверхностей и σ второй пластины
# --------------------------------------------------------------------------- #
from pathlib import Path  # noqa: E402

_ROOT = Path(__file__).resolve().parents[1]


def _ktn_stamp_problem():
    """ci lshape_stamp с theory = ktn, h = 0.1.

    При h = 0.2 обжимные члены u_c превышают прогиб — касания нет.
    """
    import tomllib

    from plate_solver.problem import Problem

    d = tomllib.loads((_ROOT / "cases" / "ci" / "lshape_stamp.toml")
                      .read_text(encoding="utf-8"))
    d["model"] = {"theory": "ktn", "h": 0.1}
    d.pop("output", None)
    return Problem.from_dict(d)


def test_t8_classic_faces_identical(tmp_path):
    """Ворота полей поверхностей: classic ⇒ w_top ≡ w_mid ≡ w_bot (1e-14) и dh ≡ 0."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_toml(_ROOT / "cases" / "ci" / "lshape_stamp.toml"))
    w_top, w_bot, dh = res.faces_on_grid()
    inside = np.isfinite(res.w_grid)
    scale = float(np.nanmax(np.abs(res.w_grid)))
    assert float(np.max(np.abs(w_top[inside] - res.w_grid[inside]))) <= 1e-14 * scale
    assert float(np.max(np.abs(w_bot[inside] - res.w_grid[inside]))) <= 1e-14 * scale
    assert float(np.max(np.abs(dh[inside]))) == 0.0


def test_t9_ktn_dh_sign_and_profile_regression():
    """Ворота полей поверхностей (ktn, L-серия; КАНОН 21.1): dh = u_c − w < 0 в зоне
    (контактирующая лицевая идёт к основанию); пик |dh| — в зоне;
    профиль dh через зону — регресс в baselines."""
    import json

    from plate_solver.dispatch import solve

    res = solve(_ktn_stamp_problem())
    w_top, w_bot, dh = res.faces_on_grid()
    zone = res.contact.contact_zone
    assert zone.any()
    assert float(np.nanmax(dh[zone])) < 0.0            # сжатие всюду в зоне
    # пик |dh| — глубоко в зоне (устойчиво к кромочному звону платформ)
    inside = np.isfinite(dh)
    dh_abs = np.where(inside, np.abs(dh), -np.inf)
    i_peak = np.unravel_index(int(np.argmax(dh_abs)), dh.shape)
    assert zone[i_peak]
    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["lshape_ktn_dh_profile"]
    ys = res.Yg[:, 0]
    j = int(np.argmin(np.abs(ys - b["y"])))
    prof = dh[j, :]
    keep = np.isfinite(prof)
    got = prof[keep]
    ref = np.asarray(b["dh"], float)
    assert len(got) == len(ref)
    # Кросс-платформенность: dh = u_c − w — разность близких величин; у
    # КРОМКИ зоны недосошедший ci-МОР (200 итер.) «звенит» на другом BLAS
    # вплоть до знака отдельных малых точек (в зоне расхождения ≤ 1e-8).
    # Гейтим пик, ГЛУБОКУЮ зону (|dh| ≥ 0.3 пика) поточечно и масштаб
    # кромочного звона; полный профиль в снимке — информационно.
    peak = float(np.max(np.abs(ref)))
    assert float(np.min(got)) == pytest.approx(float(np.min(ref)), rel=1e-2)
    deep = np.abs(ref) >= 0.3 * peak
    assert deep.sum() >= 3
    assert np.allclose(got[deep], ref[deep], rtol=1e-2, atol=1e-2 * peak)
    assert float(np.max(np.abs(got[~deep]))) <= 0.5 * peak


def test_pair_fields_second_plate_canon(tmp_path):
    """npz пары — обе шестёрки σ; канон §19 у второй: q⁺₂ = r, q⁻₂ = 0."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_toml(_ROOT / "cases" / "ci" / "two_plates_ring.toml"))
    res.save_fields(tmp_path / "fields.npz")
    data = np.load(tmp_path / "fields.npz")
    need2 = {"Mx2", "My2", "Mxy2", "sx_top2", "sx_bot2", "sy_top2", "sy_bot2",
             "txy_top2", "txy_bot2"}
    assert need2 <= set(data.files)
    h2, nu2 = float(data["h"]), float(data["nu"])      # у пары h/ν общие (Config)
    k = 6.0 / h2**2
    c = nu2 / (1.0 - nu2)
    r_fld = data["r"]
    in2 = np.isfinite(data["My2"])
    # верх второй получает реакцию: sy_top2 = −6My2/h² + ν/(1−ν)·r
    lhs = data["sy_top2"][in2]
    rhs = -k * data["My2"][in2] + c * r_fld[in2]
    scale = float(np.max(np.abs(rhs))) or 1.0
    assert float(np.max(np.abs(lhs - rhs))) <= 1e-12 * scale
    # низ второй свободен: sy_bot2 = +6My2/h² (обжатия нет)
    lhs_b = data["sy_bot2"][in2]
    rhs_b = +k * data["My2"][in2]
    assert float(np.max(np.abs(lhs_b - rhs_b))) <= 1e-12 * scale
    # обжатие реально присутствует на верхней грани второй в зоне
    assert float(np.max(r_fld)) > 0.0
