r"""Ворота полной нелинейной КТН (ktn_full.py, веха N2 v0.5.0).

Редукционная лестница (кандидат в теорему T6, §9): у полной КТН нет
независимых литературных эталонов, поэтому проверяем ВЫРОЖДЕНИЕМ во все
подтеории. Это сильнее одного эталона — демонстрирует корректную редукцию.

* Gate R1 — КТН → Карман при выключенных КТН-членах (машинная точность):
  прямая проверка, что члены (A), (B) собраны без ошибок знака/множителя.
* Gate R2 — КТН → Кирхгоф (тонкая пластина, малый прогиб).
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


def test_ktn_full_soft_hinge_not_implemented():
    """Мягкий шарнир для КТН — задел §3.5: явный NotImplementedError."""
    with pytest.raises(NotImplementedError, match="ЗАЩЕМЛЕНИИ"):
        KTNPlate.from_config(_DOM, _cfg(0.1, 1.0), bc_type="soft_hinge",
                             inplane_bc="immovable")
