r"""Ворота единой параметрической модели теорий (theory.py, ktn_solver.py; N1 v0.6.0).

Проверяется, что ОДИН решатель :class:`KTNSolver` с управляемыми параметрами
(:class:`TheoryParams`) воспроизводит четыре теории ПО ПОСТРОЕНИЮ (тот же путь
исполнения) — усиление теоремы T6 (§16). Ворота R6 (§7): у каждого пресета своя
числовая проверка; существующие тесты v0.5.0 НЕ трогаются (§3.6 — отдельный
прогон полного набора это подтверждает).
"""

from __future__ import annotations

import pytest

from plate_solver import benchmarks as bm
from plate_solver import theory
from plate_solver.config import Config
from plate_solver.geometry import make_circle
from plate_solver.ktn_full import KTNPlate
from plate_solver.ktn_solver import KTNSolver
from plate_solver.membrane import KarmanPlate
from plate_solver.theory import TheoryParams

_DOM = make_circle(1.0)


# --------------------------------------------------------------------------- #
#  TheoryParams: решётка 2×2, тождество, морфинг
# --------------------------------------------------------------------------- #
def test_preset_grid_2x2():
    """Решётка §3.3: оси нелинейности (membrane) × уточнения (h_ψ²,h_*²)."""
    c, k = theory.classic(), theory.karman()
    kl, kf = theory.ktn_linear(0.3, 0.2), theory.ktn_full(0.3, 0.2)
    assert (c.membrane, c.refined) == (False, False)     # classic
    assert (k.membrane, k.refined) == (True, False)      # karman
    assert (kl.membrane, kl.refined) == (False, True)    # ktn_linear
    assert (kf.membrane, kf.refined) == (True, True)     # ktn_full


def test_thickness_identity_and_physical_values():
    """h_c² = h_ψ² − h_*² (assert); физические значения при ν=0.3 (прил. A)."""
    kf = theory.ktn_full(0.3, 1.0)
    assert kf.h_c_sq == pytest.approx(kf.h_psi_sq - kf.h_star_sq)
    assert kf.h_psi_sq == pytest.approx(0.238095, abs=1e-5)
    assert kf.h_star_sq == pytest.approx(0.184524, abs=1e-5)
    assert kf.h_c_sq == pytest.approx(0.053571, abs=1e-5)


def test_refinement_scale_morphs_full_to_karman():
    """α=1 — полная КТН; α=0 — Карман (уточнение снято); промежуточное — между."""
    kf = theory.ktn_full(0.3, 0.2)
    assert kf.with_refinement_scale(0.0).refined is False       # α=0 ⇒ Карман
    half = kf.with_refinement_scale(0.5)
    assert half.h_psi_sq == pytest.approx(0.5 * kf.h_psi_sq)
    assert half.h_star_sq == pytest.approx(0.5 * kf.h_star_sq)
    with pytest.raises(ValueError):
        kf.with_refinement_scale(1.5)


def test_from_preset_alias_and_names():
    """from_preset: имена пресетов + алиас ktn→ktn_linear."""
    assert theory.from_preset("ktn", 0.3, 0.2) == theory.ktn_linear(0.3, 0.2)
    with pytest.raises(ValueError):
        theory.from_preset("mindlin", 0.3, 0.2)


def test_solve_ktn_terms_flag():
    """КТН-члены (A),(B) активны в решении ТОЛЬКО в нелинейном режиме (§3.4)."""
    assert theory.ktn_full(0.3, 0.2).solve_ktn_terms is True
    assert theory.karman().solve_ktn_terms is False          # нет уточнения
    assert theory.ktn_linear(0.3, 0.2).solve_ktn_terms is False  # нет нелинейности
    assert theory.classic().solve_ktn_terms is False


# --------------------------------------------------------------------------- #
#  KTNSolver воспроизводит пресеты ПО ПОСТРОЕНИЮ (ворота R6, §3.6)
# --------------------------------------------------------------------------- #
def _cfg(h=0.2, q0=0.008, ns=4):
    return Config(E=1.0, h=h, nu=0.3, a=1.0, q0=q0, p=12, Q=140,
                  n_load_steps=ns, karman_tol=1e-9, karman_max_iter=300)


def test_solver_reproduces_karman_machine_precision():
    """R6: KTNSolver('karman') = KarmanPlate до машинной точности (тот же путь)."""
    cfg = _cfg()
    rk = KarmanPlate.from_config(_DOM, cfg, bc_type="clamped",
                                 inplane_bc="immovable").solve_uniform()
    solver = KTNSolver.from_theory_name(_DOM, cfg, "karman", bc_type="clamped",
                                        inplane_bc="immovable")
    sk = solver.solve_uniform()
    assert abs(sk.w_max - rk.w_max) / rk.w_max < 1e-12
    assert solver.params.h_psi_sq == 0.0 and solver.params.h_star_sq == 0.0  # не подцепил уточнение


def test_solver_reproduces_ktn_full_machine_precision():
    """R6: KTNSolver('ktn_full') = KTNPlate до машинной точности."""
    cfg = _cfg()
    rt = KTNPlate.from_config(_DOM, cfg, bc_type="clamped",
                              inplane_bc="immovable").solve_uniform()
    solver = KTNSolver.from_theory_name(_DOM, cfg, "ktn_full", bc_type="clamped",
                                        inplane_bc="immovable")
    st = solver.solve_uniform()
    assert abs(st.w_max - rt.w_max) / rt.w_max < 1e-12
    assert solver.params.compression is True and solver.params.refined is True


def test_solver_classic_reduces_to_kirchhoff():
    """R6: KTNSolver('classic') = линейный Кирхгоф (мембрана выкл ⇒ N≡0)."""
    cfg = _cfg(q0=1e-4, ns=1)
    sc = KTNSolver.from_theory_name(_DOM, cfg, "classic", bc_type="clamped",
                                    inplane_bc="immovable").solve_uniform()
    ref = bm.kirchhoff_clamped_circle(1e-4 / 0.2**4, 0.3) * 0.2
    assert abs(sc.w_max - ref) / ref < 1e-2
    assert sc.converged


def test_morphing_monotone_karman_to_ktn_full():
    """Морфинг α: непрерывный монотонный переход Карман (α=0) → полная КТН (α=1)."""
    cfg = _cfg()
    kf = theory.ktn_full(0.3, 0.2)
    ws = []
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        s = KTNSolver.from_config(_DOM, cfg, kf.with_refinement_scale(a),
                                  bc_type="clamped", inplane_bc="immovable").solve_uniform()
        ws.append(s.w_max)
    # монотонное убывание (КТН-регуляризация ужестчает): α=0 ≥ … ≥ α=1
    assert all(ws[i] > ws[i + 1] for i in range(len(ws) - 1))
    rk = KarmanPlate.from_config(_DOM, cfg, bc_type="clamped",
                                 inplane_bc="immovable").solve_uniform()
    rt = KTNPlate.from_config(_DOM, cfg, bc_type="clamped",
                              inplane_bc="immovable").solve_uniform()
    assert abs(ws[0] - rk.w_max) / rk.w_max < 1e-12       # α=0 == Карман
    assert abs(ws[-1] - rt.w_max) / rt.w_max < 1e-12      # α=1 == полная КТН


def test_advanced_theory_params_direct():
    """Продвинутый режим: явный TheoryParams (промежуточная точка решётки, §3.5)."""
    # промежуток: нелинейность вкл, только сдвиг h_ψ² (без обжатия h_*²)
    p = TheoryParams(membrane=True, h_psi_sq=0.01, h_star_sq=0.0,
                     compression=False, shear_field=False)
    s = KTNSolver.from_config(_DOM, _cfg(), p, bc_type="clamped",
                              inplane_bc="immovable").solve_uniform()
    assert s.converged and s.w_max > 0.0
