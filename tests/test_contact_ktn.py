r"""Ворота нелинейного контакта МОР+КТН (contact_nl.py, N3 v0.6.0, §4, §7).

Независимых эталонов у «нелинейный контакт + КТН» нет → проверяем ВЫРОЖДЕНИЕМ
(точным по единой модели §3) и физикой:

* R1 (контакт выкл): зазор → ∞ ⇒ ``r ≡ 0`` ⇒ свободное решение КТН v0.5;
* R3 (уточнение выкл): полная КТН при ``refinement_scale=0`` = кармановский
  контакт (тот же путь, машинная точность);
* физика: контакт ограничивает прогиб лицевой до зазора, реакция ``r > 0`` в
  зоне; подпись КТН — зона/пик реакции при ``ktn_full`` отличаются от ``karman``.

Вложенная схема — ЭТАЛОН корректности (§4.2), дорогая (МОР × полный нелинейный
решатель); тяжёлые случаи — маркер ``big``. Совмещённая (быстрая, T7) — веха N4.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from plate_solver import theory
from plate_solver.config import Config
from plate_solver.contact_nl import NonlinearContactMOR, NonlinearTwoPlateMOR
from plate_solver.geometry import make_circle
from plate_solver.ktn_solver import KTNSolver

_DOM = make_circle(1.0)


def _cfg(q0=0.01, Q=64, beta=1.5, max_iter=2500, tol=3e-4):
    return Config(E=1.0, h=0.2, nu=0.3, a=1.0, q0=q0, p=8, Q=Q,
                  n_load_steps=3, karman_tol=1e-6, karman_max_iter=100,
                  beta=beta, max_iter=max_iter, tol=tol)


def _solver(cfg, name="ktn_full"):
    return KTNSolver.from_theory_name(_DOM, cfg, name, bc_type="clamped",
                                      inplane_bc="immovable")


def test_r1_contact_off_reduces_to_free_ktn():
    """R1: зазор → ∞ ⇒ нет контакта ⇒ r≡0, w = свободное решение КТН (машинно)."""
    cfg = _cfg()
    s = _solver(cfg)
    free = s.solve_uniform()
    res = NonlinearContactMOR(s, cfg, gap=100.0, scheme="nested").solve()
    assert res.r_max == 0.0 and res.n_contact == 0
    assert abs(res.w_max - free.w_max) / free.w_max < 1e-6


def test_face_condition_curvature_scales_with_theory():
    """Лицевое условие u_c = w + c_curv·Δw: c_curv=0 для karman, >0 для ktn_full (§4.1)."""
    assert theory.karman().face_curv_coeff == 0.0
    assert theory.classic().face_curv_coeff == 0.0
    assert theory.ktn_full(0.3, 0.2).face_curv_coeff != 0.0


def test_face_curv_coeff_is_physical_for_refined_theories():
    """Лицевой коэффициент кривизны уточнённых теорий = ФИЗИЧЕСКИЙ h_c²−h_*².

    Тождество §3.2 ``h_c² = h_ψ² − h_*²`` требует физического ``h_ψ²`` и в
    пресете ``ktn_linear`` (в решении он не активен, но производный
    ``face_curv_coeff`` иначе вырождается в ``−2h_*²`` — в 2.8 раза больше
    физического, и контакт «щупает» преувеличенную лицевую поверхность).
    Единый источник — :class:`~plate_solver.faces.FaceParams` (== KTNParams).
    """
    from plate_solver.faces import FaceParams

    c_phys = FaceParams(E=1.0, nu=0.3, h=0.2).c_curv
    assert theory.ktn_linear(0.3, 0.2).face_curv_coeff == pytest.approx(c_phys)
    assert theory.ktn_full(0.3, 0.2).face_curv_coeff == pytest.approx(c_phys)


def test_gain_mode_linear_normalizes_by_linear_compliance():
    """gain_mode='linear': β_eff нормируется ЛИНЕЙНОЙ податливостью w_lin/q0.

    Обоснование (теорема 4): при неподвижной кромке ``K_geo(N) ⪰ 0`` ⇒
    ``w_nl ≤ w_lin`` ⇒ линейная податливость — верхняя грань ``‖G‖`` вдоль
    всего пути МОР ⇒ ``β_eff·‖G‖ ≤ β < 2`` равномерно. Секущая занижает
    ``‖G‖`` в ``w_lin/w_free`` раз (расходимость при сильном ужесточении).
    """
    cfg = _cfg(Q=48)
    s = _solver(cfg)
    sec = NonlinearContactMOR(s, cfg, gap=100.0, scheme="merged")
    lin = NonlinearContactMOR(s, cfg, gap=100.0, scheme="merged", gain_mode="linear")
    assert lin.gain == pytest.approx(sec._free.w_max_classic / cfg.q0)
    assert lin.gain >= sec.gain * (1.0 - 1e-12)         # w_lin ≥ w_nl (K_geo ⪰ 0)
    assert lin.beta_eff <= sec.beta_eff                 # шаг осторожнее
    with pytest.raises(ValueError, match="gain_mode"):
        NonlinearContactMOR(s, cfg, gap=1.0, gain_mode="quadratic")


@pytest.mark.big
def test_ktn_contact_limits_face_deflection():
    """Физика: контакт ограничивает ПРОГИБ до зазора; реакция r>0 в зоне."""
    cfg = _cfg()
    s = _solver(cfg)
    gap = 0.5 * s.solve_uniform().w_max
    res = NonlinearContactMOR(s, cfg, gap=gap, scheme="nested").solve()
    assert res.r_max > 0.0 and res.n_contact > 0        # контакт состоялся
    # лицевой прогиб в зоне контакта не превышает зазор (сверх допуска МОР)
    assert np.max(res.u_c_nodes[res.contact_mask]) <= gap * 1.02


@pytest.mark.big
def test_r3_refinement_zero_equals_karman_contact():
    """R3: полная КТН при refinement→0 = кармановский контакт (машинная точность)."""
    cfg = _cfg()
    gap = 0.5 * _solver(cfg, "karman").solve_uniform().w_max
    # karman-контакт
    rk = NonlinearContactMOR(_solver(cfg, "karman"), cfg, gap=gap, scheme="nested").solve()
    # ktn_full с уточнением, снятым до нуля (α=0) — тот же путь, что karman
    p0 = theory.ktn_full(0.3, cfg.h).with_refinement_scale(0.0)
    s0 = KTNSolver.from_config(_DOM, cfg, p0, bc_type="clamped", inplane_bc="immovable")
    r0 = NonlinearContactMOR(s0, cfg, gap=gap, scheme="nested").solve()
    assert abs(r0.w_max - rk.w_max) / rk.w_max < 1e-10
    assert r0.n_contact == rk.n_contact


@pytest.mark.big
def test_r4_merged_equals_nested():
    """R4 (§4.2, §7): совмещённая схема совпадает с вложенной (эталон) до допуска."""
    cfg = _cfg()
    gap = 0.5 * _solver(cfg).solve_uniform().w_max
    rn = NonlinearContactMOR(_solver(cfg), cfg, gap=gap, scheme="nested").solve()
    rm = NonlinearContactMOR(_solver(cfg), cfg, gap=gap, scheme="merged").solve()
    assert rm.converged                                  # совмещённый сходится (T7)
    assert abs(rm.w_max - rn.w_max) / rn.w_max < 5e-3
    assert abs(rm.r_max - rn.r_max) / rn.r_max < 5e-3
    assert rm.n_contact == rn.n_contact


def test_merged_scheme_default_and_faster():
    """Совмещённая схема — по умолчанию (Config.contact_scheme='merged')."""
    assert Config().contact_scheme == "merged"


@pytest.mark.big
def test_ktn_signature_in_contact():
    """Подпись КТН: зона/пик реакции ktn_full отличаются от karman (сглаживание §6.1)."""
    cfg = _cfg()
    gap = 0.5 * _solver(cfg, "karman").solve_uniform().w_max
    rk = NonlinearContactMOR(_solver(cfg, "karman"), cfg, gap=gap, scheme="nested").solve()
    rf = NonlinearContactMOR(_solver(cfg, "ktn_full"), cfg, gap=gap, scheme="nested").solve()
    # лицевая кривизна КТН меняет контактную границу ⇒ пик и/или число узлов иные
    assert (rf.n_contact != rk.n_contact) or (abs(rf.r_max - rk.r_max) > 1e-3 * rk.r_max)


@pytest.mark.big
def test_gain_mode_linear_stabilizes_under_strong_stiffening():
    """Сильное мембранное ужесточение (w_lin/w_nl ≈ 10): секущая нормировка рвёт
    условие сжатия МОР (эффективная β ≈ 16 ≫ 2) — итерация болтается, реакция
    ДИКАЯ; линейная нормировка стабилизирует контакт к плато w ≈ z с ФИЗИЧЕСКОЙ
    реакцией (секущая даёт r_max в разы больше при том же зазоре)."""
    cfg = Config(E=1.0, h=0.2, nu=0.3, a=1.0, q0=0.4, p=8, Q=48,
                 n_load_steps=3, karman_tol=1e-8, karman_max_iter=200,
                 karman_relax=0.5, beta=1.5, max_iter=6000, tol=1e-7)
    s = _solver(cfg, "karman")
    gap = 0.15 * s.solve_uniform().w_max
    bad = NonlinearContactMOR(s, cfg, gap=gap, scheme="merged").solve()
    good = NonlinearContactMOR(s, cfg, gap=gap, scheme="merged",
                               gain_mode="linear").solve()
    assert good.w_max <= 1.05 * gap                     # линейная: плато держит зазор
    assert bad.r_max > 1.5 * good.r_max                 # секущая: дикая реакция (не сжатие)
    assert bad.residual_history[-1] > 10.0 * good.residual_history[-1]  # болтается сильнее


# -- штамп: контакт с профилем препятствия z(x,y) (§9.2, N7) ------------- #
def test_stamp_profile_reduces_to_scalar():
    """Редукция штампа: постоянное поле зазора ≡ скалярный зазор (машинно)."""
    cfg = _cfg()
    s = _solver(cfg)
    z0 = 0.45 * s.solve_uniform().w_max
    r_scalar = NonlinearContactMOR(s, cfg, gap=z0, scheme="merged").solve()
    r_array = NonlinearContactMOR(s, cfg, gap=np.full(s.quad.x.size, z0),
                                  scheme="merged").solve()
    assert abs(r_array.w_max - r_scalar.w_max) / r_scalar.w_max < 1e-12
    assert r_array.n_contact == r_scalar.n_contact and r_array.r_max == r_scalar.r_max


@pytest.mark.big
def test_stamp_profile_localizes_contact():
    """Штамп: индентор z(x,y) с низшей точкой вне центра стягивает зону контакта к ней (§9.2)."""
    cfg = _cfg()
    s = _solver(cfg)
    q = s.quad
    w_free = s.solve_uniform().w_max
    xc = 0.45                                           # низшая точка индентора смещена по x
    z0 = 0.45 * w_free
    gap_prof = z0 + 3.0 * w_free * ((q.x - xc) ** 2 + q.y ** 2)
    rp = NonlinearContactMOR(s, cfg, gap=gap_prof, scheme="merged").solve()
    rf = NonlinearContactMOR(s, cfg, gap=z0, scheme="merged").solve()   # плоское для сравнения
    assert rp.n_contact > 0 and rf.n_contact > 0
    # центроид зоны штампа сдвинут к индентору (плоский зазор даёт зону у центра)
    cx = lambda res: float(np.sum(q.w * res.contact_mask * q.x)   # noqa: E731
                           / np.sum(q.w * res.contact_mask))
    assert cx(rp) > cx(rf) + 0.1                        # штамп локализует зону к xc
    assert rp.peak_xy[0] > 0.0                          # пик реакции — со стороны индентора
    assert rp.n_contact < rf.n_contact                 # зона уже плоского случая


# -- взаимный контакт двух пластин поверх КТН (§9.2, N7) ----------------- #
def _two_solvers(cfg, e2=None):
    s1 = _solver(cfg)
    cfg2 = cfg if e2 is None else replace(cfg, E=e2)
    s2 = _solver(cfg2)
    return s1, s2


def test_two_plate_requires_shared_quad():
    """Две пластины: разные квадратуры (разный Q) недопустимы — реакция поузельна."""
    cfg = _cfg()
    s1 = _solver(cfg)
    s2 = KTNSolver.from_theory_name(_DOM, _cfg(Q=48), "ktn_full",
                                    bc_type="clamped", inplane_bc="immovable")
    with pytest.raises(ValueError, match="квадратура"):
        NonlinearTwoPlateMOR(s1, s2, cfg, gap=0.1)


@pytest.mark.big
def test_two_plate_contact_off_free():
    """Большой зазор ⇒ r≡0, обе пластины дают свободные решения КТН."""
    cfg = _cfg()
    s1, s2 = _two_solvers(cfg)
    res = NonlinearTwoPlateMOR(s1, s2, cfg, gap=100.0, f2=0.0).solve()
    assert res.r_max == 0.0 and res.n_contact == 0
    assert abs(res.w1_max - s1.solve_uniform().w_max) / s1.solve_uniform().w_max < 1e-6


@pytest.mark.big
def test_two_plate_rigid_reduces_to_single():
    """Редукция: жёсткая 2-я пластина (u_c2→0) ⇒ односторонний контакт одной пластины."""
    cfg = _cfg(max_iter=1500)
    s1, s2 = _two_solvers(cfg, e2=1000.0)               # 2-я в 1000× жёстче
    gap = 0.5 * s1.solve_uniform().w_max
    single = NonlinearContactMOR(s1, cfg, gap=gap, scheme="merged").solve()
    two = NonlinearTwoPlateMOR(s1, s2, cfg, gap=gap, f2=0.0).solve()
    assert two.w2_max / two.w1_max < 5e-3               # 2-я почти неподвижна
    assert abs(two.w1_max - single.w_max) / single.w_max < 5e-3   # свелось к одиночному
    assert two.n_contact == single.n_contact


@pytest.mark.big
def test_two_plate_both_deformable_share_reaction():
    """Обе деформируемы: реакция размягчает 1-ю и сдвигает 2-ю (совместное решение)."""
    cfg = _cfg(max_iter=1500)
    s1, s2 = _two_solvers(cfg)                          # одинаковые пластины
    gap = 0.4 * s1.solve_uniform().w_max
    two = NonlinearTwoPlateMOR(s1, s2, cfg, gap=gap, f2=0.0).solve()
    assert two.r_max > 0.0 and two.n_contact > 0        # контакт состоялся
    assert two.w2_max > 1e-3 * two.w1_max               # 2-я заметно откликнулась на r
    # прогиб 1-й под реакцией меньше её свободного значения (реакция размягчает)
    assert two.w1_max < s1.solve_uniform().w_max
