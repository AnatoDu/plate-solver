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

    v0.8.0: канон приведён к опубликованной формуле (9) (``κ_r·r`` без множителя D),
    поэтому ОБА члена реакции теперь имеют размерность податливости [м/Па] и
    их разность ОДНОРОДНА (раньше Δb_r складывала ``E·h³·ν²`` с числами —
    признак самой размерной ошибки).
    """
    mu_ = E / (2 * (1 + nu))
    lamb = nu * E / ((1 + nu) * (1 - 2 * nu))
    cq_c = -(h / (8 * (lamb + 2 * mu_)) - H_ST2 / (mu_ * h) + H_ST2 * H_Z2 / D)
    cr_c = -(3 * h / (8 * (lamb + 2 * mu_)) + H_ST2 / (mu_ * h) - H_ST2 * H_Z2 / D)
    bot, _ = _faces()
    dq = sp.simplify(bot.coeff(q) - cq_c)
    dr = sp.simplify(bot.coeff(r) - cr_c)
    assert dq != 0 and dr != 0                      # расхождение реально
    # точные значения разностей — заморожены
    dq_ref = h * (6 * nu**3 + 14 * nu**2 + nu - 1) / (32 * E * (nu - 1))
    dr_ref = h * (-6 * nu**3 + 18 * nu**2 - nu + 1) / (32 * E * (nu - 1))
    assert sp.simplify(dq - dq_ref) == 0
    assert sp.simplify(dr - dr_ref) == 0
    # ОДНОРОДНОСТЬ: обе разности — чистые податливости ∝ h/E, множитель
    # зависит ТОЛЬКО от ν (до исправления Δb_r содержала слагаемое ∝ E·h⁴).
    assert sp.simplify(dr / (h / E)).free_symbols == {nu}
    assert sp.simplify(dq / (h / E)).free_symbols == {nu}


def test_t6b_kappa_r_matches_published_formula():
    r"""т6б: ``κ_r`` канона ≡ опубликованная формула (9) (символьно).

    .. math:: \kappa_r = \frac{3(1+\nu)(2-4\nu+\nu^2)h}{16E(1-\nu)}

    Ворота на размерность и на источник: κ_r — податливость [м/Па], поэтому
    ``κ_r·r`` есть длина (в отличие от прежнего ``κ_r·D·r`` ~ Па·м⁴).
    """
    from plate_solver.ktn import KTNParams

    mu_ = E / (2 * (1 + nu))
    lamb = nu * E / ((1 + nu) * (1 - 2 * nu))
    kappa_r = 3 * h / (8 * (lamb + 2 * mu_)) + H_ST2 / (mu_ * h) - H_ST2 * H_Z2 / D
    published = 3 * (1 + nu) * (2 - 4 * nu + nu**2) * h / (16 * E * (1 - nu))
    assert sp.simplify(kappa_r - published) == 0
    # и то же численно в коде
    kp = KTNParams(E=2.1e6, nu=0.3, h=0.06)
    num = float(published.subs({E: 2.1e6, nu: sp.Rational(3, 10), h: sp.Rational(6, 100)}))
    assert kp.kappa_r == pytest.approx(num, rel=1e-14)
    assert kp.kappa_r > 0.0 and kp.kappa_q > 0.0


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
              * res.r_nodes)
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
        "model": {"theory": "ktn_linear", "h": 0.06},
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
    d["model"] = {"theory": "ktn_linear", "h": 0.1}
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


def test_t9_ktn_dh_decomposition_and_profile_regression():
    """Ворота полей поверхностей (ktn, L-серия; КАНОН 21.1, v0.8.0).

    ``dh = u_c − w = c_curv·Δw − κ_q·q⁺ − κ_r·r`` — сумма КРИВИЗНОГО и
    ОБЖИМНОГО вкладов. Физически инвариантно (не зависит от толщины и режима):
    обжимный вклад СТРОГО ОТРИЦАТЕЛЕН всюду, где есть давление (пластина
    сжимается по толщине, лицевая идёт к срединной), а знак самого ``dh``
    задаётся кривизным членом. До v0.8.0 обжимный член был раздут множителем
    ``D`` (в 41.5 раза при E = 2.1e6, h = 0.06) и подавлял кривизный — отсюда
    прежнее утверждение «dh < 0 всюду в зоне».

    Профиль ``dh`` через зону — регресс в ``cases/baselines.json``.
    """
    import json

    from plate_solver.dispatch import solve
    from plate_solver.ktn import KTNParams

    prob = _ktn_stamp_problem()
    cfg = prob.to_config()
    res = solve(prob)
    w_top, w_bot, dh = res.faces_on_grid()
    zone = res.contact.contact_zone
    assert zone.any()

    # (1) обжимный вклад строго отрицателен в зоне (r > 0 и q0 > 0)
    kp = KTNParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
    assert kp.kappa_q > 0.0 and kp.kappa_r > 0.0
    r_grid = np.nan_to_num(res.contact.r_grid)
    compress = -(kp.kappa_q * cfg.q0 + kp.kappa_r * r_grid)
    assert float(compress[zone].max()) < 0.0
    # (2) разложение точно: dh − обжимный = кривизный = c_curv·Δw
    curv = dh - compress
    assert np.isfinite(curv[zone]).all()
    # (3) масштаб поправки В ЗОНЕ — O(h²/L²) от прогиба, а не сам прогиб
    #     (у входящего угла Δw сингулярна — там |dh| растёт, зона от этого свободна)
    scale = float(np.nanmax(np.abs(res.w_grid)))
    assert float(np.nanmax(np.abs(dh[zone]))) < 0.2 * scale

    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    b = base["lshape_ktn_dh_profile"]
    ys = res.Yg[:, 0]
    j = int(np.argmin(np.abs(ys - b["y"])))
    prof = dh[j, :]
    keep = np.isfinite(prof)
    got = prof[keep]
    ref = np.asarray(b["dh"], float)
    assert len(got) == len(ref)
    # Кросс-платформенность: dh — разность близких величин; у КРОМКИ зоны
    # недосошедший ci-МОР (200 итер.) «звенит» на другом BLAS. Гейтим пик и
    # ГЛУБОКУЮ часть профиля (|dh| ≥ 0.3 пика) поточечно.
    peak = float(np.max(np.abs(ref)))
    assert float(np.max(np.abs(got))) == pytest.approx(peak, rel=2e-2)
    deep = np.abs(ref) >= 0.3 * peak
    assert deep.sum() >= 3
    assert np.allclose(got[deep], ref[deep], rtol=2e-2, atol=2e-2 * peak)


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
    c = -nu2 / (1.0 - nu2)                             # обжатие СЖИМАЕТ (v0.8.0)
    r_fld = data["r"]
    in2 = np.isfinite(data["My2"])
    # верх второй получает реакцию: sy_top2 = −6My2/h² − ν/(1−ν)·r
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
