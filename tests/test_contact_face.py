r"""Ворота ЭКСПЕРИМЕНТА (ветка ``exp/face-primary-contact``): контакт с ЛИЦЕВОЙ
поверхностью как основным полем (``contact_face.py``, ТЗ прототипа §6, §9).

Независимого эталона у реформулировки нет → проверяем (i) ВЫРОЖДЕНИЕМ в текущий
решатель (эталон корректности МОР-петли), (ii) внутренней согласованностью
(совпадение в зоне контакта, точность плато), (iii) СОХРАНЕНИЕМ физики
(кромочная сингулярность реакции — кольцо, Михайловский–Тарасов), (iv) целевым
эффектом (для линейной КТН слабая форма убирает «провал» реакции у Δw-звона).

Мат. обоснование ``weak``: ``v`` — Галёркинова L2-проекция соотношения (9) на
``V_h=span{ωT_j}`` с интегрированием кривизны по частям (граничный член ноль при
защемлении). Проектор — ортогональный (норма 1); первая производная под
интегралом НЕ усиливает высокочастотный хвост ``Δw`` (в отличие от поузельного
второго дифференцирования). См. docstring модуля.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver.config import Config
from plate_solver.contact_face import FacePrimaryContact, FacePrimaryContactKTN
from plate_solver.contact_nl import NonlinearContactMOR
from plate_solver.geometry import make_circle
from plate_solver.ktn_solver import KTNSolver

DOM = make_circle(1.0)
Z = 0.1


def _cfg(Q=64, max_iter=5000, tol=1e-7):
    return Config(E=1.0, nu=0.3, h=0.2, a=1.0, q0=0.02, p=12, Q=Q,
                  beta=1.5, max_iter=max_iter, tol=tol,
                  karman_tol=1e-9, karman_max_iter=200, n_load_steps=1,
                  karman_relax=1.0, contact_scheme="merged")


def _lin_solver(cfg):
    return KTNSolver.from_theory_name(DOM, cfg, "classic", bc_type="clamped",
                                      inplane_bc="immovable")


# --------------------------------------------------------------------------- #
#  (i) вырождение: pointwise (классика) = текущий решатель NonlinearContactMOR
# --------------------------------------------------------------------------- #
def test_classic_pointwise_reduces_to_official():
    """Эталон МОР-петли: ``pointwise`` (классика, v=w) = NonlinearContactMOR."""
    cfg = _cfg(Q=64, tol=1e-6)
    s = _lin_solver(cfg)
    mine = FacePrimaryContact(s, cfg, gap=Z, face_mode="pointwise", refined=False,
                              include_load_terms=False, stop="dr").solve()
    off = NonlinearContactMOR(s, cfg, gap=Z, scheme="merged").solve()
    assert abs(mine.w_max - off.w_max) / off.w_max < 1e-4
    assert abs(mine.r_max - off.r_max) / off.r_max < 1e-3
    assert mine.n_contact == off.n_contact


# --------------------------------------------------------------------------- #
#  (ii) внутренняя согласованность: классика weak≈pointwise, v≈w, плато v=z
# --------------------------------------------------------------------------- #
def test_classic_weak_agrees_with_pointwise():
    """Классика (c_curv=0): нечего чинить ⇒ weak совпадает с pointwise в зоне."""
    cfg = _cfg()
    s = _lin_solver(cfg)
    pt = FacePrimaryContact(s, cfg, gap=Z, face_mode="pointwise", refined=False).solve()
    wk = FacePrimaryContact(s, cfg, gap=Z, face_mode="weak", refined=False).solve()
    assert abs(pt.rho_c - wk.rho_c) < 0.02
    assert abs(pt.w_max - wk.w_max) / pt.w_max < 1e-2
    assert abs(pt.r_max - wk.r_max) / pt.r_max < 0.1


def test_weak_face_matches_w_for_classic():
    """Классика: слабое ``v`` есть L2-проекция ``w`` ⇒ v ≈ w (нет Δw)."""
    cfg = _cfg()
    s = _lin_solver(cfg)
    wk = FacePrimaryContact(s, cfg, gap=Z, face_mode="weak", refined=False).solve()
    w = wk.cw @ s._psi
    rel = np.max(np.abs(wk.v_nodes - w)) / np.max(np.abs(w))
    assert rel < 5e-3                                   # проекция гладкого w почти точна


@pytest.mark.parametrize("mode", ["pointwise", "weak"])
@pytest.mark.parametrize("refined", [False, True])
def test_plateau_v_equals_z(mode, refined):
    """Плато: на узлах контакта лицевой прогиб держит зазор ``v ≈ z`` (§6.2)."""
    cfg = _cfg()
    s = _lin_solver(cfg)
    res = FacePrimaryContact(s, cfg, gap=Z, face_mode=mode, refined=refined).solve()
    assert res.n_contact > 0
    assert res.plateau_dev / Z < 5e-3                   # |v−z| < 0.5% зазора


# --------------------------------------------------------------------------- #
#  (iii) физика: кромочная сингулярность (кольцо) СОХРАНЕНА, не сглажена (§6.4)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("mode", ["pointwise", "weak"])
def test_edge_ring_preserved(mode):
    """Классика: реакция — КОЛЬЦО (пик СМЕЩЁН от центра), центр разгружен.

    Жёсткий контакт защемлённой пластины с плоским основанием: реакция
    концентрируется у кромки пятна (Михайловский–Тарасов). Реформулировка НЕ
    ДОЛЖНА сглаживать это в центрированный купол.
    """
    cfg = _cfg()
    s = _lin_solver(cfg)
    res = FacePrimaryContact(s, cfg, gap=Z, face_mode=mode, refined=False).solve()
    q = s.quad
    rho = np.hypot(q.x, q.y)
    peak_rho = np.hypot(*res.peak_xy)
    assert peak_rho > 0.1                               # пик реакции СМЕЩЁН (кольцо)
    # центр разгружен: средняя реакция при ρ<0.1 много меньше пиковой
    core = res.r_nodes[rho < 0.1]
    assert core.mean() < 0.25 * res.r_max


# --------------------------------------------------------------------------- #
#  (iv) целевой эффект: линейная КТН — слабая форма убирает «провал» реакции
# --------------------------------------------------------------------------- #
def test_linear_ktn_weak_fills_reaction_dropout():
    """Линейная КТН: у эталона Δw-звон рвёт реакцию (провал → 0) между центром и
    кольцом; слабая форма заполняет провал (гладкий переход), зона совпадает."""
    cfg = _cfg(Q=96)
    s = _lin_solver(cfg)
    pt = FacePrimaryContact(s, cfg, gap=Z, face_mode="pointwise", refined=True).solve()
    wk = FacePrimaryContact(s, cfg, gap=Z, face_mode="weak", refined=True).solve()

    # совпадение в зоне контакта (§6.1)
    assert abs(pt.rho_c - wk.rho_c) < 0.03
    assert abs(pt.w_max - wk.w_max) / pt.w_max < 2e-2

    # «провал» реакции во внутренней части пятна: у слабой формы он мельче
    def interior_min(res):
        rho = np.hypot(s.quad.x, s.quad.y)
        nb, out = 8, []
        for i in range(nb):
            lo, hi = i * 0.07 * res.rho_c, (i + 1) * 0.07 * res.rho_c
            m = (rho >= lo) & (rho < hi)
            if m.sum() >= 3:
                out.append(res.r_nodes[m].mean())
        return min(out) if out else 0.0
    # слабая форма НЕ создаёт нулевого провала там, где у эталона реакция рвётся
    assert interior_min(wk) >= interior_min(pt) - 1e-9


# --------------------------------------------------------------------------- #
#  (v) Phase 2: полная КТН запускается; L-член мал при этих параметрах (§6.6)
# --------------------------------------------------------------------------- #
@pytest.mark.big
def test_phase2_ktn_runs_and_l_term_small():
    """Полная КТН: контакт с лицевым полем работает; эффект члена L(Φ,w) в
    эффективной нагрузке (9) мал при q0=0.02 (< 5% на r_max)."""
    cfg = _cfg(Q=64, max_iter=1500)
    s = KTNSolver.from_theory_name(DOM, cfg, "ktn_full", bc_type="clamped",
                                   inplane_bc="immovable")
    with_L = FacePrimaryContactKTN(s, cfg, gap=Z, face_mode="weak",
                                   effective_load=True).solve()
    no_L = FacePrimaryContactKTN(s, cfg, gap=Z, face_mode="weak",
                                 effective_load=False).solve()
    assert with_L.n_contact > 0 and with_L.r_max > 0
    assert abs(with_L.r_max - no_L.r_max) / no_L.r_max < 0.05
