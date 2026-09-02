r"""Инвариантность модели к системе единиц (постоянные ворота, v0.8.0).

Задача изгиба и одностороннего контакта ОДНОРОДНА: при одновременном
масштабировании модуля упругости и нагрузки

    (E, q₀) → (s·E, s·q₀)

прогиб ``w ∝ q₀/E`` НЕ меняется, реакция ``r ∝ q₀`` масштабируется как ``s``,
а все безразмерные характеристики решения — ``r/q₀``, число узлов зоны,
``w/Δ``, доля переданной силы, невязка комплементарности — ИНВАРИАНТНЫ.
Шаг МОР сохраняет это свойство по построению: ``β_eff = β/gain``, ``gain ∝ 1/E``.

Ворота ловят класс дефектов «размерно несогласованный член»: в v0.7.0 и ранее
лицевое условие КТН содержало ``κ_r·D·r`` (Па·м⁴ вместо длины), и КТН-контакт
качественно зависел от выбора единиц — при s = 10³ контакт исчезал вовсе.
Исправление (опубликованная формула (9)): ``κ_r·r``. Историю см. CHANGELOG
[0.8.0], NOTES §21; символьная сторона — ``tests/test_face_deflection.py`` (т6, т6б).

Регламент: ФИКСИРОВАННЫЙ бюджет итераций и ``tol = 0`` — сравниваются сами
итераты (критерий ``stop="dr"`` абсолютен и сам по себе не инвариантен, а
``"comp"`` безразмерен и проверяется отдельно).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending

SCALES = (1.0, 1.0e3)
#: относительный допуск: пути исполнения одинаковы, различие — только округление
RTOL = 1e-9


def _dimensionless(cfg, ktn, budget=400, stop="dr"):
    """Безразмерные характеристики контактного решения при данном масштабе единиц."""
    dom = geometry.make_circle(1.0)
    pb = PlateBending.from_config(dom, cfg)
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    w_free = float(np.max(np.abs(pb.deflection(cw, q.x, q.y))))
    gap = 0.5 * w_free                       # зазор — доля свободного прогиба
    kp = None if ktn is None else KTNParams.from_config(cfg)
    mor = ContactMOR(pb, cfg, gap=gap, ktn=kp)
    res = mor.solve()
    r = res.r_nodes
    force = float(np.sum(q.w * r) / (cfg.q0 * np.sum(q.w)))
    return {
        "r_max/q0": float(r.max()) / cfg.q0,
        "n_contact": int((r > 0).sum()),
        "w_max/gap": float(res.w_nodes.max()) / gap,
        "force/load": force,
        "comp": float(res.comp_residual),
        "w_free/gap": w_free / gap,
        "iters": res.iters,
    }


def _run_pair(theory: str, budget: int = 400):
    out = []
    for s in SCALES:
        cfg = Config(E=2.1e6 * s, nu=0.3, q0=4.0 * s, h=0.15, p=8, Q=40,
                     beta=1.0, max_iter=budget, tol=0.0, grid_n=20)
        out.append(_dimensionless(cfg, theory, budget=budget))
    return out


@pytest.mark.parametrize("theory", [None, "ktn"], ids=["classic", "ktn_linear"])
def test_contact_invariant_under_unit_scaling(theory):
    """ГЛАВНЫЕ ВОРОТА: безразмерные характеристики контакта не зависят от единиц."""
    a, b = _run_pair(theory)
    assert a["n_contact"] == b["n_contact"] > 0
    assert a["iters"] == b["iters"]
    for key in ("r_max/q0", "w_max/gap", "force/load", "w_free/gap"):
        assert b[key] == pytest.approx(a[key], rel=RTOL), f"{theory}: {key} зависит от единиц"
    assert b["comp"] == pytest.approx(a["comp"], rel=1e-7)


def test_ktn_face_terms_are_lengths():
    """Размерность: κ_q·q и κ_r·r — ДЛИНЫ, поэтому κ ∝ 1/E при фиксированной h.

    Прямая проверка однородности коэффициентов (без решения задачи): при
    масштабировании E обе поправки лицевого условия масштабируются как 1/E,
    как и сам прогиб ⇒ их отношение к прогибу инвариантно.
    """
    k1 = KTNParams(E=2.1e6, nu=0.3, h=0.06)
    k2 = KTNParams(E=2.1e9, nu=0.3, h=0.06)
    assert k2.kappa_q == pytest.approx(k1.kappa_q / 1000.0, rel=1e-14)
    assert k2.kappa_r == pytest.approx(k1.kappa_r / 1000.0, rel=1e-14)
    # поправка лицевого при (E, q, r) → (sE, sq, sr) не меняется
    for kp, s in ((k1, 1.0), (k2, 1000.0)):
        assert kp.kappa_q * (4.0 * s) == pytest.approx(k1.kappa_q * 4.0, rel=1e-14)
        assert kp.kappa_r * (12.0 * s) == pytest.approx(k1.kappa_r * 12.0, rel=1e-14)
    # кривизный коэффициент от E не зависит вовсе (чистая длина²)
    assert k2.c_curv == pytest.approx(k1.c_curv, rel=1e-14)


def test_nonlinear_contact_invariant_under_unit_scaling():
    """Тот же инвариант для НЕЛИНЕЙНОГО тракта (ktn_full + МОР, схема merged)."""
    from plate_solver.contact_nl import NonlinearContactMOR
    from plate_solver.ktn_solver import KTNSolver

    vals = []
    for s in SCALES:
        cfg = Config(E=2.1e6 * s, nu=0.3, q0=4.0 * s, h=0.1, p=6, Q=40,
                     beta=1.0, max_iter=60, tol=0.0, grid_n=16,
                     karman_relax=0.5, karman_max_iter=40, karman_tol=1e-10)
        dom = geometry.make_circle(1.0)
        solver = KTNSolver.from_theory_name(dom, cfg, "ktn_full")
        free = solver.solve(np.full(solver.quad.x.size, cfg.q0))
        gap = 0.6 * float(np.max(np.abs(free.w_nodes)))
        res = NonlinearContactMOR(solver, cfg, gap=gap, scheme="merged").solve()
        vals.append({
            "r_max/q0": float(res.r_nodes.max()) / cfg.q0,
            "n_contact": int(res.n_contact),
            "w_max/gap": float(res.w_max) / gap,
            "gap/h": gap / cfg.h,
        })
    a, b = vals
    assert a["n_contact"] == b["n_contact"] > 0
    for key in ("r_max/q0", "w_max/gap", "gap/h"):
        assert b[key] == pytest.approx(a[key], rel=1e-8), f"ktn_full: {key} зависит от единиц"
