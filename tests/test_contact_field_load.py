r"""Ворота ПОЛЕВОЙ нагрузки (gaussian/expr) в контакте (v0.7.0).

МОР — свойство ОПЕРАТОРА (теорема 4: β_eff·‖G‖ < 2), нагрузка меняет лишь
правую часть. Отсюда и нормировка усиления: отклик на РАВНОМЕРНУЮ опорную
нагрузку амплитуды |q0| (v0.8.0), а не на фактическое поле — для локализованной
нагрузки отклик мал, и нормировка «по полю» занижала ‖G‖ в разы (при σ = 0.15
в 6.6 раза), выводя шаг за границу сходимости: итерация уходила к r ≡ 0
и контакт «терялся». Ворота:

* классический контакт + gaussian: ВЗАИМНЫЙ сертификат КР↔RFM (кирпич
  fd_contact принимает поле q(x, y); измерено rel(w)=2.0e-4, rel(∫r)=1.1e-3);
* нелинейный МОР+КТН (позиционное основание): R1 big-gap → свободное
  нелинейное решение (2.2e-15), nested == merged (2.1e-7 karman /
  7.4e-7 ktn_full — две схемы, одна неподвижная точка), линейный предел
  karman-контакта → классический контакт с той же нагрузкой (5.0e-3 —
  классический тракт сам сертифицирован КР);
* силовой/парный контакт под полем — по-прежнему отказ.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.fd_contact import FDPlateSS, fd_contact_foundation
from plate_solver.ladder import navier_uniform_center
from plate_solver.problem import CaseError, Problem


@pytest.mark.big          # ~60 c МОР Q=200: тяжёлый дублирующий
def test_classic_gaussian_contact_fd_certificate():
    """Классический контакт + gaussian: взаимный сертификат КР↔RFM."""
    x1, x2, y1, y2 = 0.0, 2.0, 0.0, 1.2
    w_ex, _ = navier_uniform_center(2.0, 1.2, 1.0, 1.0)
    gap, sig = 0.35 * w_ex, 0.35
    case = {
        "geometry": {"kind": "rectangle", "x1": x1, "x2": x2,
                     "y1": y1, "y2": y2},
        "bc": {"type": "soft_hinge"},
        "load": {"type": "gaussian", "q0": 3.0, "x0": 1.0, "y0": 0.6,
                 "sigma": sig},
        "model": {"theory": "classic", "E": 12 * (1 - 0.09), "nu": 0.3,
                  "h": 1.0},
        "contact": {"enabled": True, "gap": gap, "max_iter": 6000,
                    "tol": 1.0e-8},
        "discretization": {"p": 12, "Q": 200, "grid_n": 24},
        "verify": {"reference": "none"},
    }
    res = dispatch.solve(Problem.from_dict(case))
    tot_rfm = float(np.sum(res.contact.r_nodes * res._plate.quad.w))
    fd = FDPlateSS(x1, x2, y1, y2, 135, 81, 1.0)
    f = 3.0 * np.exp(-(((fd.X - 1.0) ** 2 + (fd.Y - 0.6) ** 2)
                       / (2 * sig**2)))
    fdc = fd_contact_foundation(x1, x2, y1, y2, D=1.0, q0=f, gap=gap,
                                nx=135, ny=81, tol=1e-7)
    assert fdc.n_contact > 0 and np.all(fdc.r >= 0.0)
    assert abs(fdc.w_max - res.w_max) / res.w_max < 1.5e-2
    assert abs(fdc.r_total - tot_rfm) / tot_rfm < 1.5e-2


def _nl_case(theory="karman", **contact):
    return {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "gaussian", "q0": 4.0, "x0": 0.0, "y0": 0.0,
                 "sigma": 0.5},
        "model": {"theory": theory},
        "contact": {"enabled": True, "max_iter": 2000, "tol": 1.0e-7,
                    **contact},
        "discretization": {"p": 8, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }


def test_gain_is_operator_property_not_load_shape():
    """ГЛАВНЫЕ ВОРОТА (v0.8.0): усиление ‖G‖ не зависит от ФОРМЫ нагрузки.

    Для узкой гауссианы отклик на само поле мал, и нормировка «по полю»
    занижала оценку нормы оператора в разы ⇒ β_eff·‖G‖ ≫ 2 ⇒ расходимость
    к r ≡ 0 при формально допустимом β. Проверяем: (i) gain одинаков для трёх
    ширин пятна; (ii) контакт находится во всех трёх случаях.
    """
    from plate_solver.config import Config
    from plate_solver.contact_nl import NonlinearContactMOR
    from plate_solver.geometry import make_circle
    from plate_solver.ktn_solver import KTNSolver

    cfg = Config(E=1e5, nu=0.3, h=0.2, q0=4.0, p=8, Q=48, beta=1.2, max_iter=400,
                 tol=1e-7, karman_relax=0.7, karman_max_iter=100, karman_tol=1e-10,
                 grid_n=20)
    solver = KTNSolver.from_theory_name(make_circle(1.0), cfg, "karman")
    x, y = solver.quad.x, solver.quad.y
    gains, contacts = [], []
    for sigma in (0.5, 0.25, 0.15):
        f = cfg.q0 * np.exp(-(x**2 + y**2) / (2 * sigma**2))
        free = solver.solve(f)
        gap = 0.55 * float(np.max(np.abs(free.w_nodes)))
        mor = NonlinearContactMOR(solver, cfg, gap=gap, scheme="merged", f_values=f)
        gains.append(mor.gain)
        res = mor.solve()
        contacts.append(res.n_contact)
        # непроникание выполняется с точностью шага МОР
        assert res.w_max <= gap * 1.02
    assert gains[1] == pytest.approx(gains[0], rel=1e-12)   # свойство оператора
    assert gains[2] == pytest.approx(gains[0], rel=1e-12)
    assert min(contacts) > 0, "контакт потерян при узком пятне (нормировка ‖G‖)"


def test_nl_gaussian_big_gap_reduction():
    r"""R1: большой зазор ⇒ ``r ≡ 0``, а прогиб — свободный нелинейный.

    Реакция обязана быть ТОЧНЫМ нулём (проекция ``[·]₊`` структурна), а вот
    прогиб сравнивается между ДВУМЯ РАЗНЫМИ путями итерации: контактный тракт
    идёт совмещённой схемой (шаг Пикара на шаг МОР), свободный — обычным
    Пикаром до ``karman_tol``. Сойтись бит-в-бит они не обязаны: расхождение
    имеет порядок сходимости нелинейной итерации, а не машинного эпсилон.
    Измерено 6.0e-13 (macOS/Accelerate) и 1.06e-12 (Linux/OpenBLAS, Python
    3.11) — прежний порог 1e-12 имел запас 1.66× и в CI отказал. Порог 1e-9
    держит запас ~10³ к худшему измеренному и при этом на четыре порядка
    строже собственного допуска решателя (``karman_tol = 1e-8``); смысловую
    поломку (контактный тракт меняет решение ТАМ, ГДЕ КОНТАКТА НЕТ) он ловит
    с огромным запасом — она даёт расхождение процентами.
    """
    d = _nl_case(gap=1.0)
    d["contact"]["max_iter"] = 300
    d["contact"]["tol"] = 1.0e-6
    r_big = dispatch.solve(Problem.from_dict(d))
    free = copy.deepcopy(d)
    free["contact"] = {"enabled": False}
    r_free = dispatch.solve(Problem.from_dict(free))
    assert float(np.max(r_big.contact.r_nodes)) == 0.0     # реакция — ТОЧНЫЙ ноль
    assert abs(r_big.w_max - r_free.w_max) / r_free.w_max < 1e-9


@pytest.mark.parametrize("theory", ["karman", "ktn_full"])
def test_nl_gaussian_nested_equals_merged(theory):
    """Две схемы композиции — одна неподвижная точка (изм. ≤7.4e-7)."""
    dn = _nl_case(theory=theory, gap_factor=0.55, scheme="nested")
    dm = copy.deepcopy(dn)
    dm["contact"]["scheme"] = "merged"
    rn = dispatch.solve(Problem.from_dict(dn))
    rm = dispatch.solve(Problem.from_dict(dm))
    tn = float(np.sum(rn.contact.r_nodes * rn._plate.quad.w))
    tm = float(np.sum(rm.contact.r_nodes * rm._plate.quad.w))
    assert rn.scalars()["n_contact"] > 0
    assert abs(rn.w_max - rm.w_max) / rn.w_max < 1e-5
    assert abs(tn - tm) / tn < 1e-5


def test_nl_gaussian_linear_limit_vs_classic_contact():
    """Малая нагрузка: karman-контакт → классический контакт (КР-сертифицирован).

    Оба тракта доводятся ДО СХОДИМОСТИ (нелинейный: 4615 итераций при tol 1e-9),
    и тогда совпадение машинно-качественное: rel(w) = 1.3e-5, rel(∫r) = 5.2e-6,
    зона совпадает узел-в-узел. Прежние допуски (2e-2 и 1e-3) сравнивали
    НЕсошедшийся нелинейный результат (2000 итераций, KKT-невязка 5e-2) с
    сошедшимся классическим — v0.8.0 ворота ужесточены на два порядка.
    """
    lk = _nl_case(gap_factor=0.55)
    lk["load"]["q0"] = 0.04
    lk["contact"]["max_iter"] = 10000
    lk["contact"]["tol"] = 1.0e-9
    lc = copy.deepcopy(lk)
    lc["model"]["theory"] = "classic"
    lc["contact"] = {"enabled": True, "gap_factor": 0.55, "max_iter": 20000,
                     "tol": 1.0e-9}
    rk = dispatch.solve(Problem.from_dict(lk))
    rc = dispatch.solve(Problem.from_dict(lc))
    tk = float(np.sum(rk.contact.r_nodes * rk._plate.quad.w))
    tc = float(np.sum(rc.contact.r_nodes * rc._plate.quad.w))
    assert rk.scalars()["converged"] and rc.scalars()["converged"]
    assert rk.scalars()["n_contact"] == rc.scalars()["n_contact"] > 0
    assert abs(rk.w_max - rc.w_max) / rc.w_max < 1e-4
    assert abs(tk - tc) / tc < 1e-4


def test_nl_expr_load_contact_runs():
    """expr-нагрузка в нелинейном контакте: маршрут + эквивалент гауссиане."""
    d1 = _nl_case(gap_factor=0.55)
    d2 = copy.deepcopy(d1)
    d2["load"] = {"type": "expr", "q0": 4.0,
                  "expr": "exp(-(x**2 + y**2)/(2*0.5**2))"}
    r1 = dispatch.solve(Problem.from_dict(d1))
    r2 = dispatch.solve(Problem.from_dict(d2))
    assert abs(r1.w_max - r2.w_max) / r1.w_max < 1e-10


def test_nl_field_load_force_pair_rejected():
    """Силовой и парный контакт под полем — по-прежнему отказ."""
    d = _nl_case()
    d["contact"] = {"enabled": True, "force": 1.0}
    with pytest.raises(CaseError, match="load.type"):
        Problem.from_dict(d)
    d2 = _nl_case()
    d2["contact"] = {"enabled": True, "target": "plate2", "gap": 0.1}
    d2["plate2"] = {"bc": {"type": "clamped"},
                    "load": {"type": "uniform", "q0": 0.0},
                    "model": {"theory": "karman"}}
    with pytest.raises(CaseError, match="load.type"):
        Problem.from_dict(d2)
