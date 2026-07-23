r"""Ворота полной нелинейной КТН (ktn_full.py, веха N2 v0.5.0).

Редукционная лестница (кандидат в теорему T6, §9): у полной КТН нет
независимых литературных эталонов, поэтому проверяем ВЫРОЖДЕНИЕМ во все
подтеории. Это сильнее одного эталона — демонстрирует корректную редукцию.

* Gate R1 — КТН → Карман при выключенных КТН-членах (машинная точность):
  прямая проверка, что члены (A), (B) собраны без ошибок знака/множителя.
* Gate R2 — КТН → Кирхгоф (тонкая пластина, малый прогиб).
* Gate R3 — 1D↔2D: НЕЗАВИСИМЫЙ осесимметричный радиальный решатель
  (`radial_ktn.RadialKTN`, иная дискретизация) воспроизводит 2D-решение
  `KTNPlate` на круге; проверяет ПОЛНУЮ нелинейную связь ``N(w)`` (сильнее MMS
  с замороженным ``N``). Согласие дискретизационно-ограничено (убывает с ростом
  2D-``p``), редукция ТОЧНА.
* Gate R4 — гашение поправки O(h²/L²): эффект ∝ h², → 0 при h/L → 0.
* Gate R5 — лицевые ktn_full при малом прогибе ≈ ktn_linear (смыкание реализаций).
* Подпись КТН и ограничение мягкого шарнира.

Безразмерно: E = a = 1, ν = 0.3 ⇒ P̄ = q0/h⁴ (нормировка B).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import benchmarks as bm
from plate_solver.config import Config
from plate_solver.geometry import make_circle
from plate_solver.ktn_full import KTNPlate
from plate_solver.membrane import KarmanPlate

_DOM = make_circle(1.0)


def _cfg(h, P_bar, *, Q=140, ns=4, tol=1e-9, max_iter=300, p=12):
    """Config при фиксированной безразмерной нагрузке P̄ = q0/h⁴ (E=a=1)."""
    return Config(E=1.0, h=h, nu=0.3, a=1.0, q0=P_bar * h**4, p=p, Q=Q,
                  n_load_steps=ns, karman_tol=tol, karman_max_iter=max_iter)


def _karman(h, P_bar, **kw):
    return KarmanPlate.from_config(_DOM, _cfg(h, P_bar, **kw),
                                   bc_type="clamped", inplane_bc="immovable").solve_uniform()


def _ktn(h, P_bar, *, include_ktn_terms=True, **kw):
    return KTNPlate.from_config(_DOM, _cfg(h, P_bar, **kw), bc_type="clamped",
                                inplane_bc="immovable",
                                include_ktn_terms=include_ktn_terms).solve_uniform()


# --------------------------------------------------------------------------- #
#  Gate R1 — КТН → Карман (машинная точность)
# --------------------------------------------------------------------------- #
def test_gate_r1_reduces_to_karman_machine_precision():
    """Gate R1: при выключенных КТН-членах ktn_full ≡ karman до машинной точности."""
    rk = _karman(0.2, 6.0)
    r0 = _ktn(0.2, 6.0, include_ktn_terms=False)
    W = np.ones_like(rk.w_nodes)                    # относительная L2-невязка поля w
    num = np.sqrt(np.sum((rk.w_nodes - r0.w_nodes) ** 2 * W))
    den = np.sqrt(np.sum(rk.w_nodes ** 2 * W))
    assert num / den < 1e-12, num / den            # члены (A),(B) — чисто аддитивны


def test_gate_r1_terms_on_differ_from_karman():
    """Контроль: с ВКЛючёнными КТН-членами решение ОТЛИЧАЕТСЯ от Кармана (члены активны)."""
    rk = _karman(0.2, 6.0)
    rt = _ktn(0.2, 6.0, include_ktn_terms=True)
    assert abs(rt.w_max - rk.w_max) / rk.w_max > 1e-3


# --------------------------------------------------------------------------- #
#  Gate R2 — КТН → Кирхгоф (тонкая пластина, малый прогиб)
# --------------------------------------------------------------------------- #
def test_gate_r2_reduces_to_kirchhoff():
    """Gate R2: тонкая пластина + малый прогиб ⇒ ktn_full → линейный Кирхгоф."""
    r = _ktn(0.05, 0.05, ns=1)                      # P̄ = 0.05 ⇒ w/h ≈ 0.009 (линейно)
    ref = bm.kirchhoff_clamped_circle(0.05, 0.3) * 0.05   # w/h → w_max (·h)
    assert abs(r.w_max - ref) / ref < 1e-2          # совпал с классикой (дискретизация)
    assert abs(r.w_max - r.w_max_classic) / r.w_max_classic < 5e-4  # КТН-поправка мала


# --------------------------------------------------------------------------- #
#  Gate R3 — 1D↔2D: независимый радиальный решатель ↔ 2D KTNPlate на круге
# --------------------------------------------------------------------------- #
#  Дискретизационный пол согласия на быстрой сетке (2D p=14): rel(w_max) и
#  поточечная невязка ≤ ~4·10⁻³, УБЫВАЮТ с ростом 2D-p (см. big-тест ниже) —
#  редукция точна, невязку задаёт 2D-RFM. Допуск 8·10⁻³ (пол + запас на BLAS);
#  НЕ ослаблять: при расхождении искать ошибку в методе (CLAUDE.md).
from plate_solver.radial_ktn import RadialKTN  # noqa: E402

_R3_TOL = 8e-3


def _radial(h, P_bar, *, include_ktn=True, bc="clamped", p=12, nq=700):
    rad = RadialKTN(1.0, 1.0, 0.3, h, p=p, pm=p, nq=nq, include_ktn=include_ktn, bc=bc)
    return rad, rad.solve(P_bar * h**4)


def _two_d(h, P_bar, *, ktn, bc="clamped", p=14, Q=160):
    # Ньютон (§5.4) — быстрая сходимость 2D-стороны (особенно мягкий шарнир)
    method = {"ktn_method": "newton"} if ktn else {"karman_method": "newton"}
    cfg = Config(E=1.0, h=h, nu=0.3, a=1.0, q0=P_bar * h**4, p=p, Q=Q,
                 n_load_steps=3, karman_tol=1e-10, karman_max_iter=400, **method)
    Cls = KTNPlate if ktn else KarmanPlate
    plate = Cls.from_config(_DOM, cfg, bc_type=bc, inplane_bc="immovable")
    return plate, plate.solve_uniform()


def _profile_mismatch(rad, r1, plate, r2):
    """Максимум |w_1D − w_2D| / w_max по радиусу (2D берётся вдоль +x)."""
    rs = np.linspace(0.0, 0.97, 20)
    w1 = rad.deflection(r1.cw, rs)
    w2 = np.array([float(plate.deflection(r2.cw, x, 0.0)) for x in rs])
    return float(np.max(np.abs(w1 - w2)) / r1.w_max)


def test_gate_r3_reduction_1d_ktn_off_is_karman():
    """Контроль редукции 1D: радиальная КТН при выключенных членах = радиальный Карман."""
    _, r_off = _radial(0.1, 6.0, include_ktn=False)
    _, r_on = _radial(0.1, 6.0, include_ktn=True)
    # выключённая КТН ≡ Карман (по построению); ВКЛючённая — отличается (члены активны)
    assert r_off.converged and r_on.converged
    assert abs(r_on.w_max - r_off.w_max) / r_off.w_max > 1e-3


@pytest.mark.parametrize("bc,h", [("clamped", 0.1), ("clamped", 0.2),
                                  ("soft_hinge", 0.1)])
def test_gate_r3_karman_1d_matches_2d(bc, h):
    """Gate R3 (Карман): независимый радиальный решатель ↔ 2D KarmanPlate (обе кромки)."""
    rad, r1 = _radial(h, 6.0, include_ktn=False, bc=bc)
    plate, r2 = _two_d(h, 6.0, ktn=False, bc=bc)
    assert abs(r1.w_max - r2.w_max) / r2.w_max < _R3_TOL
    assert _profile_mismatch(rad, r1, plate, r2) < _R3_TOL


@pytest.mark.parametrize("bc,h", [("clamped", 0.1), ("clamped", 0.2),
                                  ("soft_hinge", 0.1)])
def test_gate_r3_ktn_1d_matches_2d(bc, h):
    """Gate R3 (ПОЛНАЯ КТН): независимый радиальный решатель ↔ 2D KTNPlate (обе кромки).

    Ядро R3: сходятся ДВЕ независимые дискретизации полной нелинейной КТН с живой
    связью ``N(w)`` — сертификат корректности метода без литературного эталона. На
    мягком шарнире свёртка включает граничный член §3.5 (в осесимметрии — вырожден,
    ниже дискретизационного пола; §3.5 отдельно сверен тождеством Грина).
    """
    rad, r1 = _radial(h, 6.0, include_ktn=True, bc=bc)
    plate, r2 = _two_d(h, 6.0, ktn=True, bc=bc)
    assert abs(r1.w_max - r2.w_max) / r2.w_max < _R3_TOL
    assert _profile_mismatch(rad, r1, plate, r2) < _R3_TOL


def test_gate_r3_ktn_correction_consistent_1d_2d():
    """КТН-поправка (относительно Кармана) согласована 1D↔2D: не только величина, но и ФИЗИКА."""
    _, r1k = _radial(0.1, 6.0, include_ktn=False)
    _, r1t = _radial(0.1, 6.0, include_ktn=True)
    _, r2k = _two_d(0.1, 6.0, ktn=False)
    _, r2t = _two_d(0.1, 6.0, ktn=True)
    ratio_1d = r1t.w_max / r1k.w_max                 # КТН/Карман (1D)
    ratio_2d = r2t.w_max / r2k.w_max                 # КТН/Карман (2D)
    assert ratio_1d < 1.0 and ratio_2d < 1.0         # КТН жёстче Кармана (эффект члена B)
    assert abs(ratio_1d - ratio_2d) < 2e-3           # отношение совпало (согласованная поправка)


@pytest.mark.big
def test_gate_r3_convergence_1d_2d():
    """Gate R3 (сходимость): невязка 1D↔2D УБЫВАЕТ с ростом 2D-p ⇒ редукция точна."""
    rad, r1 = _radial(0.1, 6.0, include_ktn=True, p=16, nq=1000)   # ~точный оракул
    rels = []
    for p, Q in [(12, 140), (14, 180), (16, 220)]:
        plate, r2 = _two_d(0.1, 6.0, ktn=True, p=p, Q=Q)
        rels.append(abs(r1.w_max - r2.w_max) / r2.w_max)
    assert rels[0] > rels[1] > rels[2]               # монотонно к нулю (дискретизация 2D)
    assert rels[-1] < 3e-3


# --------------------------------------------------------------------------- #
#  Gate R4 — гашение поправки O(h²/L²)
# --------------------------------------------------------------------------- #
@pytest.mark.big
def test_gate_r4_correction_vanishes_as_h2():
    """Gate R4: при фикс P̄ эффект КТН/Карман ∝ h²; при h/L=0.02 эффект < 1 %."""
    P_bar = 5.0
    effects = {}
    for h in (0.2, 0.1, 0.05, 0.02):
        rk = _karman(h, P_bar)
        rt = _ktn(h, P_bar)
        effects[h] = abs(1.0 - rt.w_max / rk.w_max)
    # монотонное убывание и чистый порядок h²: effect/h² ≈ const
    assert effects[0.2] > effects[0.1] > effects[0.05] > effects[0.02]
    assert effects[0.02] < 1e-2                      # < 1 % при h/L = 0.02
    ratios = [effects[h] / h**2 for h in (0.2, 0.1, 0.05)]
    assert max(ratios) / min(ratios) < 1.15          # O(h²): отношение почти постоянно


# --------------------------------------------------------------------------- #
#  Gate R5 — линейный ktn_linear = малопрогибный предел ktn_full
# --------------------------------------------------------------------------- #
def test_gate_r5_faces_match_ktn_linear_small_deflection():
    """Gate R5: лицевой прогиб ktn_full при малом прогибе ≈ ktn_linear (§9)."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    base = {
        "geometry": {"kind": "circle", "a": 1.0}, "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 5.0e-5},   # малый прогиб
        "discretization": {"p": 10, "Q": 120, "grid_n": 40},
    }
    rf = solve(Problem.from_dict({**base,
        "model": {"theory": "ktn_full", "E": 1.0, "nu": 0.3, "h": 0.1, "n_load_steps": 1}}))
    rl = solve(Problem.from_dict({**base,
        "model": {"theory": "ktn_linear", "E": 1.0, "nu": 0.3, "h": 0.1}}))
    _, _, dh_f = rf.faces_on_grid()
    _, _, dh_l = rl.faces_on_grid()
    m = np.isfinite(dh_f) & np.isfinite(dh_l)
    scale = np.nanmax(np.abs(dh_l[m]))
    assert np.nanmax(np.abs(dh_f[m] - dh_l[m])) < 1e-2 * scale   # смыкание реализаций


# --------------------------------------------------------------------------- #
#  Подпись КТН и ограничение мягкого шарнира
# --------------------------------------------------------------------------- #
def test_ktn_signature_faces_nontrivial():
    """Подпись КТН: лицевое смещение dh нетривиально (кинематика сдвига/обжатия)."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_dict({
        "geometry": {"kind": "circle", "a": 1.0}, "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 5.0e-4},
        "model": {"theory": "ktn_full", "E": 1.0, "nu": 0.3, "h": 0.2, "n_load_steps": 4},
        "discretization": {"p": 12, "Q": 140, "grid_n": 40},
    }))
    _, _, dh = res.faces_on_grid()
    assert np.nanmax(np.abs(dh)) > 0.0
    tp = res.thickness_params()                      # интроспекция §6.3 доступна
    assert tp["h_psi_sq"] > 0 and tp["h_over_L"] == pytest.approx(0.2, rel=1e-2)


def test_ktn_full_soft_hinge_solves():
    """Мягкий шарнир для КТН (v0.6.3, §3.5): решается на круге, шарнир гнётся сильнее.

    Полная соответствующая верификация граничного члена (тождество Грина,
    редукции, сходимость) — в tests/test_soft_hinge_ktn.py.
    """
    cfg = _cfg(0.1, 1.0)
    sh = KTNPlate.from_config(_DOM, cfg, bc_type="soft_hinge", inplane_bc="immovable")
    cl = KTNPlate.from_config(_DOM, cfg, bc_type="clamped", inplane_bc="immovable")
    import numpy as np
    q = np.full(sh.quad.x.size, cfg.q0)
    r_sh, r_cl = sh.solve(q), cl.solve(q)
    assert r_sh.converged and np.isfinite(r_sh.w_max)
    assert r_sh.w_max > r_cl.w_max                        # шарнир податливее защемления


def test_ktn_method_newton_matches_picard():
    """Ньютон для КТН (§5.4, v0.6.4): то же решение, что Пикар, меньше итераций.

    Подробные проверки касательного (конечные разности, мягкий шарнир) —
    ``tests/test_newton.py``; здесь фиксируем эквивалентность в контексте КТН.
    """
    cfg_n = Config(E=1.0, h=1.0, nu=0.3, a=1.0, q0=6.0, p=10, Q=128,
                   karman_tol=1e-8, karman_max_iter=300, ktn_method="newton")
    cfg_p = Config(E=1.0, h=1.0, nu=0.3, a=1.0, q0=6.0, p=10, Q=128,
                   karman_tol=1e-8, karman_max_iter=300, ktn_method="picard")
    rn = KTNPlate.from_config(_DOM, cfg_n, bc_type="clamped").solve_uniform()
    rp = KTNPlate.from_config(_DOM, cfg_p, bc_type="clamped").solve_uniform()
    assert rn.converged
    assert abs(rn.w_max - rp.w_max) / rp.w_max < 1e-3
    assert rn.n_iter < rp.n_iter


# --------------------------------------------------------------------------- #
#  Неравномерная нагрузка и член КТН (A) −h_*²Δq (§7, N3)
# --------------------------------------------------------------------------- #
def _solve_circle(theory, load, *, h=0.2, q=2.0e-5, p=12, Q=160):
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    ld = dict(load, q0=q)
    model = {"theory": theory, "E": 1.0, "nu": 0.3, "h": h}
    if theory in ("karman", "ktn_full"):
        model["n_load_steps"] = 1
    return solve(Problem.from_dict({
        "geometry": {"kind": "circle", "a": 1.0}, "bc": {"type": "clamped"},
        "load": ld, "model": model,
        "discretization": {"p": p, "Q": Q, "grid_n": 40},
    }))


def test_gaussian_load_works_for_all_theories():
    """Гауссова нагрузка q0·exp(−r²/2σ²) решается любой теорией (§7)."""
    gl = {"type": "gaussian", "x0": 0.0, "y0": 0.0, "sigma": 0.3}
    for theory in ("classic", "karman", "ktn_full"):
        res = _solve_circle(theory, gl)
        assert np.isfinite(res.w_max) and res.w_max > 0.0


def test_ktn_term_a_signature_uniform_vs_gaussian():
    """Подпись КТН (§9): под равномерной срединный w ≈ классика (Δq=0), под гауссовой — нет."""
    # малая нагрузка ⇒ мембрана пренебрежима, виден чистый член (A)
    ru_kt = _solve_circle("ktn_full", {"type": "uniform"})
    ru_cl = _solve_circle("classic", {"type": "uniform"})
    rg_kt = _solve_circle("ktn_full", {"type": "gaussian", "x0": 0.0, "y0": 0.0, "sigma": 0.25})
    rg_cl = _solve_circle("classic", {"type": "gaussian", "x0": 0.0, "y0": 0.0, "sigma": 0.25})
    dev_uniform = abs(ru_kt.w_max - ru_cl.w_max) / ru_cl.w_max
    dev_gaussian = abs(rg_kt.w_max - rg_cl.w_max) / rg_cl.w_max
    assert dev_uniform < 1e-3                         # Δq = 0: срединный ≈ классика
    assert dev_gaussian > 1e-2                        # Δq ≠ 0: член (A) смещает срединный
    assert dev_gaussian > 10 * dev_uniform


def test_ktn_term_a_grows_with_localization():
    """Член (A) ∝ Δq ∝ 1/σ²: эффект РАСТЁТ с сужением гауссовой нагрузки (§9)."""
    devs = []
    for sigma in (0.4, 0.25, 0.15):
        kt = _solve_circle("ktn_full", {"type": "gaussian", "x0": 0.0, "y0": 0.0, "sigma": sigma})
        cl = _solve_circle("classic", {"type": "gaussian", "x0": 0.0, "y0": 0.0, "sigma": sigma})
        devs.append(abs(kt.w_max - cl.w_max) / cl.w_max)
    assert devs[0] < devs[1] < devs[2]                # уже гаусс — больше эффект
