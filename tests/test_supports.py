r"""Ворота точечных опор — секция [supports] (v0.7.0).

Эталоны:

* ГЛАВНЫЙ: защемлённый круг радиуса a под равномерной q с жёсткой центральной
  опорой — R_∞ = π·q·a²/4 ТОЧНО (суперпозиция табличных формул Тимошенко:
  w_free(0) = qa⁴/64D и G(0,0) = a²/16πD ⇒ R = w_free/G = πqa²/4);
* машинное тождество Шермана–Моррисона: R = k·w_free(P)/(1 + k·G_N(P, P)) —
  дискретно точное (не зависит от квадратуры);
* взаимный сертификат КР↔RFM на SS-квадрате + предел Тимошенко
  R_∞/(q·a²) = 0.00406/0.01160 = 0.3500;
* монотонность собственных частот (Курант–Фишер: K + k·ψψᵀ — PSD-возмущение).

Сходимость решения с опорой АЛГЕБРАИЧЕСКАЯ ~p⁻² (функция Грина ~r²ln r у
опоры), НЕ спектральная — допуски обоснованы измеренной p-лестницей
(rel(R): 8.9e-3 при p=12 → 2.0e-3 при p=28).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.config import Config
from plate_solver.fd_contact import FDPlateSS
from plate_solver.problem import CaseError, Problem

_D = Config().D


def _circle_case(k, *, points=((0.0, 0.0),), theory="classic", bc="clamped",
                 p=12, Q=64):
    return {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": bc},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": theory},
        "supports": {"points": [list(pt) for pt in points], "stiffness": k},
        "discretization": {"p": p, "Q": Q, "grid_n": 32},
        "verify": {"reference": "none"},
    }


def test_support_rigid_circle_center_reaction():
    """Жёсткая центральная опора защемлённого круга: R = π·q·a²/4 ТОЧНО.

    Измерено rel 8.8e-3 при p=12/Q=64 (алгебраическая ~p⁻² сходимость,
    фон w_free(0) на тех же сетках 9e-3) — допуск 2e-2 с запасом ×2.
    """
    res = dispatch.solve(Problem.from_dict(_circle_case(1e6 * _D)))
    R = res.support_reactions[0]
    R_exact = np.pi * 4.0 / 4.0                        # π·q·a²/4, q=4, a=1
    assert abs(R - R_exact) / R_exact < 2e-2


@pytest.mark.big
def test_support_rigid_circle_center_reaction_converges():
    """Та же реакция на p=20/Q=192: rel < 8e-3 (измерено 4.1e-3)."""
    res = dispatch.solve(Problem.from_dict(_circle_case(1e6 * _D, p=20, Q=192)))
    R = res.support_reactions[0]
    assert abs(R - np.pi) / np.pi < 8e-3


def test_support_sherman_morrison_identity():
    """Машинное тождество: R = k·w_free(P)/(1 + k·G_N(P, P)).

    Дискретная функция Грина G_N — из ТОГО ЖЕ оператора (обратный ход по
    факторизации), поэтому тождество не зависит от квадратуры; измерено 6e-7.
    """
    from plate_solver.clamped import ClampedPlate
    from plate_solver.geometry import make_circle

    P = (0.3, 0.2)
    k = 1e4 * _D
    cfg = Config(p=12, Q=64)
    dom = make_circle(1.0)
    free = ClampedPlate.from_config(dom, cfg)
    c_free = free.solve_uniform(4.0)
    w_free_P = float(free.deflection(c_free, np.array([P[0]]),
                                     np.array([P[1]]))[0])
    psi_P = free.structure_at(np.array([P[0]]), np.array([P[1]]))[:, 0]
    G_N = float(psi_P @ free.solve_from_b(psi_P))      # ψᵀ(D·S)⁻¹ψ
    R_pred = k * w_free_P / (1.0 + k * G_N)

    cfg_s = Config(p=12, Q=64, supports_points=(P,), supports_stiffness=k)
    sup = ClampedPlate.from_config(dom, cfg_s)
    c_s = sup.solve_uniform(4.0)
    R = k * float(sup.deflection(c_s, np.array([P[0]]), np.array([P[1]]))[0])
    assert abs(R - R_pred) / R_pred < 1e-4


def test_support_stiffness_sweep_monotone():
    """R(k) монотонно растёт; закон штрафа 1/(1+k·G_N) — стабилизация."""
    Rs = []
    for kf in (1e2, 1e3, 1e4, 1e5, 1e6):
        res = dispatch.solve(Problem.from_dict(_circle_case(kf * _D)))
        Rs.append(res.support_reactions[0])
    assert all(r2 > r1 for r1, r2 in zip(Rs, Rs[1:], strict=False))
    assert abs(Rs[-1] - Rs[-2]) / Rs[-1] < 2e-3        # закон: ~4.6e-4


def test_support_fd_certificate_ss_square():
    """Взаимный сертификат КР↔RFM (SS-квадрат, центральная опора; NOTES §25).

    КР: пружина в узле — интенсивность k/(hx·hy) на диагонали A (дискретная
    δ); предел Тимошенко R_∞/(q·a²) = 0.00406/0.01160 = 0.3500. Измерено:
    КР n=321 rel 4e-4 к Тимошенко; КР↔Ритц rel(R) 7.6e-3 при p=12.
    """
    q0, k = 1.0, 1e6 * _D
    fd = FDPlateSS(0.0, 1.0, 0.0, 1.0, 321, 321, _D,
                   springs=[(0.5, 0.5, k)])
    w = fd.solve(q0)
    R_fd = k * float(w.ravel()[fd.spring_idx[0]])
    assert abs(R_fd / (q0 * 1.0) - 0.3500) / 0.3500 < 2e-3

    d = {
        "geometry": {"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                     "y1": 0.0, "y2": 1.0},
        "bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
            {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]},
        "load": {"type": "uniform", "q0": q0},
        "model": {"theory": "classic"},
        "supports": {"points": [[0.5, 0.5]], "stiffness": k},
        "discretization": {"p": 12, "Q": 96, "grid_n": 32},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(d))
    R_ritz = res.support_reactions[0]
    assert abs(R_fd - R_ritz) / R_ritz < 1.5e-2
    assert abs(float(np.max(np.abs(w))) - res.w_max) / res.w_max < 5e-2


def test_support_karman_linear_limit():
    """Карман с опорой в линейном режиме = классика (кросс-тракт; фон ~7e-5)."""
    rk = dispatch.solve(Problem.from_dict(_circle_case(1e6 * _D,
                                                       theory="karman")))
    rc = dispatch.solve(Problem.from_dict(_circle_case(1e6 * _D)))
    assert abs(rk.w_max - rc.w_max) / rc.w_max < 2e-4
    assert abs(rk.support_reactions[0] - rc.support_reactions[0]) \
        / rc.support_reactions[0] < 2e-4


def test_support_eigen_frequencies_monotone():
    """Опора поднимает ВСЕ собственные частоты (Курант–Фишер, PSD ранг-1)."""
    base = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "model": {"theory": "classic"},
        "eigen": {"kind": "vibration", "n_modes": 6},
        "discretization": {"p": 10, "Q": 64, "grid_n": 24},
        "verify": {"reference": "none"},
    }
    import copy
    with_sup = copy.deepcopy(base)
    with_sup["supports"] = {"points": [[0.3, 0.2]], "stiffness": 1e6 * _D}
    e0 = dispatch.solve(Problem.from_dict(base)).eigen.values
    e1 = dispatch.solve(Problem.from_dict(with_sup)).eigen.values
    assert all(v1 >= v0 * (1.0 - 1e-12) for v0, v1 in zip(e0, e1, strict=False))
    assert e1[0] > e0[0] * 1.05                        # первая мода растёт заметно


def test_support_symmetry():
    """Зеркальные опоры — зеркальные поля (структурная корректность ранг-1)."""
    r1 = dispatch.solve(Problem.from_dict(_circle_case(
        1e5 * _D, points=((0.3, 0.2),))))
    r2 = dispatch.solve(Problem.from_dict(_circle_case(
        1e5 * _D, points=((-0.3, 0.2),))))
    w1, w2 = r1.w_grid, r2.w_grid[:, ::-1]             # отражение по x
    m = np.isfinite(w1) & np.isfinite(w2)
    # измерено 4.5e-8: округление Холецкого × cond(S) с жёсткой пружиной;
    # структурная симметрия ранг-1 добавки — с запасом
    assert np.max(np.abs(w1[m] - w2[m])) / np.max(np.abs(w1[m])) < 1e-6


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(bc={"type": "soft_hinge"}), "bc.type"),
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d["model"].update(theory="ktn_full"), "model.theory"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5},
                        model={"theory": "karman"}),
     "contact.enabled"),
    (lambda d: d["supports"].update(points=[[2.0, 0.0]]), "вне области"),
    (lambda d: d["supports"].update(points=[[1.0, 0.0]]), "вне области"),
    (lambda d: d["supports"].update(stiffness=-1.0), "supports.stiffness"),
    (lambda d: d["supports"].update(points=[[0.1, 0.1], [0.1, 0.1]]),
     "уникальная"),
])
def test_support_guards(mutate, match):
    d = _circle_case(1e6 * _D)
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_support_too_many_points_rejected():
    pts = [[0.01 * i, 0.0] for i in range(33)]
    with pytest.raises(CaseError, match="supports.points"):
        Problem.from_dict(_circle_case(1e6 * _D, points=pts))


def test_support_prestress_rejected():
    d = _circle_case(1e6 * _D, theory="karman")
    d["eigen"] = {"kind": "vibration", "n_modes": 3, "prestress": True}
    with pytest.raises(CaseError, match="prestress"):
        Problem.from_dict(d)


def test_support_extreme_stiffness_rejected():
    """k ≥ 1e12·D/a³ — жёсткий отказ (МНК-fallback молча искажает реакции)."""
    with pytest.raises(CaseError, match="supports.stiffness"):
        dispatch.solve(Problem.from_dict(_circle_case(1e13 * _D)))


def test_support_nonfinite_stiffness_rejected():
    """inf/nan-жёсткость — отказ на разборе (TOML умеет inf/nan)."""
    for bad in (float("nan"), float("inf")):
        with pytest.raises(CaseError):
            Problem.from_dict(_circle_case(bad))


def test_support_verify_reference_rejected():
    """[supports] + verify.reference ≠ none — отказ (заражение ворот)."""
    d = _circle_case(1e6 * _D)
    d["verify"] = {"reference": "analytic"}
    with pytest.raises(CaseError, match="verify.reference"):
        Problem.from_dict(d)


def test_support_with_contact_classic():
    """Контакт + опора (v0.7.0, classic clamped): редукции и KKT.

    Опора входит в S ДО факторизации ⇒ контактный оператор SPD, теорема 4
    как есть (gain на опёртом операторе). Ворота БЕЗ требования converged:
    компактная зона полусходится (документировано; числа стабильны ~1e-3).
    """
    import copy

    base = _circle_case(1e6 * _D, points=((0.5, 0.0),))
    r_free = dispatch.solve(Problem.from_dict(base))
    d_gap = copy.deepcopy(base)
    d_gap["contact"] = {"enabled": True, "gap": 1.0}
    r_gap = dispatch.solve(Problem.from_dict(d_gap))
    # R1: большой зазор — бит-точно чистые опоры, r ≡ 0
    assert r_gap.w_max == r_free.w_max
    assert float(np.max(r_gap.contact.r_nodes)) == 0.0

    d_c = copy.deepcopy(base)
    d_c["contact"] = {"enabled": True, "gap_factor": 0.5, "max_iter": 12000}
    r_c = dispatch.solve(Problem.from_dict(d_c))
    d_nc = copy.deepcopy(d_c)
    del d_nc["supports"]
    r_nc = dispatch.solve(Problem.from_dict(d_nc))
    assert np.all(r_c.contact.r_nodes >= 0.0)
    tot_s = float(np.sum(r_c.contact.r_nodes * r_c._plate.quad.w))
    tot_n = float(np.sum(r_nc.contact.r_nodes * r_nc._plate.quad.w))
    assert tot_s < tot_n                       # опора разгружает основание
    q = r_c._plate.quad
    near = (q.x - 0.5) ** 2 + q.y**2 < 0.05**2
    # жёсткая опора держит w(P) ≈ 0 < Δ ⇒ основание рядом не работает
    assert float(np.max(r_c.contact.r_nodes[near])) == 0.0
    # стабильность полусходящихся чисел (документированное свойство)
    d_half = copy.deepcopy(d_c)
    d_half["contact"]["max_iter"] = 6000
    r_h = dispatch.solve(Problem.from_dict(d_half))
    assert abs(r_h.w_max - r_c.w_max) / r_c.w_max < 5e-3


def test_support_contact_other_tracts_rejected():
    """Опоры с контактом вне classic+clamped+основание — отказ."""
    import copy

    base = _circle_case(1e6 * _D)
    for mut in (
        lambda d: d.update(contact={"enabled": True, "gap": 0.5},
                           model={"theory": "karman"}),
        lambda d: d.update(contact={"enabled": True, "force": 1.0}),
        lambda d: d.update(bc={"type": "soft_hinge"},
                           contact={"enabled": True, "gap": 0.5}),
    ):
        d = copy.deepcopy(base)
        mut(d)
        with pytest.raises(CaseError):
            Problem.from_dict(d)
