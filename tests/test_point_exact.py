r"""Ворота ТОЧНОЙ δ-силы — [load] point + exact = true (v0.7.0).

``b_i = P·ψ_i(x0, y0)`` — точечное вычисление ограничено на H² в 2D
(дискретизация точная, квадратуры нет); решение сходится по базису
АЛГЕБРАИЧЕСКИ (~p⁻¹·⁵ в точке приложения, функция Грина ~r²ln r).

Классические эталоны (Тимошенко–Войновский-Кригер):

* защемлённый круг, центр: ``w(0) = P·a²/(16πD)``; профиль
  ``w(r) = P/(16πD)·(a²−r²+2r²·ln(r/a))`` — дальнее поле;
* шарнирный круг, центр: ``w(0) = (3+ν)/(1+ν)·P·a²/(16πD)`` — истинный
  шарнир через полную форму (karman в линейном пределе).

Расщепление классики (soft_hinge) и КТН-теории под δ — честный отказ
(NOTES §18: (P1) с δ — функционал вне H¹; КТН-прогиб под δ расходится).
Предел eps→0 регуляризованного пятна в ворота НЕ берётся: малое пятно
упирается в документированный квадратурный пол маски (известное свойство
регуляризованной точечной нагрузки).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.config import Config
from plate_solver.problem import CaseError, Problem

_D = Config().D
_P = 10.0


def _case(bc="clamped", theory="classic", p=18, Q=96, x0=0.0, y0=0.0):
    return {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": bc},
        "load": {"type": "point", "P": _P, "x0": x0, "y0": y0, "exact": True},
        "model": {"theory": theory},
        "discretization": {"p": p, "Q": Q, "grid_n": 24},
        "verify": {"reference": "none"},
    }


def _w_at(res, x, y):
    return float(res._plate.deflection(res._c, np.array([x]), np.array([y]))[0])


def test_point_exact_clamped_center():
    """Центр защемлённого круга: w = P·a²/(16πD), алгебраическая p-лестница.

    Измерено rel: 2.2e-2 (p=10) → 6.2e-3 (p=24) — допуск 1.5e-2 при p=24
    + строгое убывание по p (ловит потерю сходимости).
    """
    w_ex = _P / (16.0 * np.pi * _D)
    err = {}
    for p in (10, 24):
        res = dispatch.solve(Problem.from_dict(_case(p=p, Q=max(64, 4 * p))))
        err[p] = abs(_w_at(res, 0.0, 0.0) - w_ex) / w_ex
    assert err[24] < 1.5e-2
    assert err[24] < err[10]


def test_point_exact_clamped_far_field():
    """Дальнее поле (r = a/2): полный профиль Грина защемлённого круга.

    w(r) = P/(16πD)·(a²−r²+2r²·ln(r/a)); измерено rel 6.0e-3 при p=18.
    """
    r = 0.5
    w_ex = _P / (16.0 * np.pi * _D) * (1.0 - r**2 + 2.0 * r**2 * np.log(r))
    res = dispatch.solve(Problem.from_dict(_case()))
    assert abs(_w_at(res, r, 0.0) - w_ex) / abs(w_ex) < 2e-2


def test_point_exact_true_hinge_center():
    """Истинный шарнир (karman, полная форма): w = (3+ν)/(1+ν)·P·a²/(16πD).

    Измерено rel 4.8e-3 при p=18 — допуск 1.5e-2 + убывание по p.
    """
    nu = 0.3
    w_ex = (3.0 + nu) / (1.0 + nu) * _P / (16.0 * np.pi * _D)
    err = {}
    for p in (10, 18):
        res = dispatch.solve(Problem.from_dict(
            _case(bc="soft_hinge", theory="karman", p=p, Q=max(64, 4 * p))))
        err[p] = abs(_w_at(res, 0.0, 0.0) - w_ex) / w_ex
    assert err[18] < 1.5e-2
    assert err[18] < err[10]


def test_point_exact_symmetry():
    """Зеркальные точки приложения — зеркальные поля.

    Абсолютная ошибка ~1e-14 (машинная симметрия); относительный порог 1e-6
    учитывает округление факторизации на разных BLAS (как у ворот опор).
    """
    r1 = dispatch.solve(Problem.from_dict(_case(p=12, Q=64, x0=0.3, y0=0.2)))
    r2 = dispatch.solve(Problem.from_dict(_case(p=12, Q=64, x0=-0.3, y0=0.2)))
    w1, w2 = r1.w_grid, r2.w_grid[:, ::-1]
    m = np.isfinite(w1) & np.isfinite(w2)
    assert np.max(np.abs(w1[m] - w2[m])) / np.max(np.abs(w1[m])) < 1e-6


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(bc={"type": "soft_hinge"}), "bc.type"),
    (lambda d: d["model"].update(theory="ktn_linear"), "model.theory"),
    (lambda d: d["model"].update(theory="ktn_full"), "model.theory"),
    (lambda d: d.update(contact={"enabled": True, "gap": 0.5}),
     "contact.enabled"),
    (lambda d: d.update(verify={"reference": "analytic"}), "verify.reference"),
    (lambda d: d["load"].update(eps=0.05), "load.eps"),
    (lambda d: d["load"].update(x0=1.0), "вне области"),
])
def test_point_exact_guards(mutate, match):
    d = _case(p=8, Q=32)
    mutate(d)
    with pytest.raises(CaseError, match=match):
        dispatch.solve(Problem.from_dict(d))


def test_point_exact_mixed_rejected():
    d = _case(p=8, Q=32)
    d["geometry"] = {"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                     "y1": 0.0, "y2": 1.0}
    d["load"].update(x0=0.5, y0=0.5)
    d["bc"] = {"type": "mixed", "sides": [
        {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
        {"side": "y1", "type": "hinge"}, {"side": "y2", "type": "hinge"}]}
    with pytest.raises(CaseError, match="bc.type"):
        Problem.from_dict(d)


def test_point_regularized_unchanged():
    """Без exact — прежний регуляризованный путь (дефолт бит-неизменен)."""
    d = _case(p=10, Q=64)
    del d["load"]["exact"]
    d["load"]["eps"] = 0.2
    res = dispatch.solve(Problem.from_dict(d))
    assert res.eps_eff is not None and res.w_max > 0.0


def test_point_exact_save_fields(tmp_path):
    """Сохранение полей при точной δ: q_top ≡ 0 (мера, не плотность)."""
    res = dispatch.solve(Problem.from_dict(_case(p=10, Q=48)))
    res.save(tmp_path)
    z = np.load(tmp_path / "fields.npz")
    assert "w" in z
    q_top, _ = res._q_faces_on_grid()
    assert float(np.max(np.abs(q_top))) == 0.0
