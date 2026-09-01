r"""Ворота ПОЛЕВОЙ нагрузки (gaussian/expr) в контакте (v0.7.0).

МОР — свойство ОПЕРАТОРА (теорема 4: β_eff·‖G‖ < 2), нагрузка меняет лишь
правую часть; нормировка усиления остаётся по амплитуде cfg.q0 (семантика
классического МОР). Ворота:

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


def test_nl_gaussian_big_gap_reduction():
    """R1: большой зазор ⇒ r ≡ 0, w = свободному нелинейному (машинно)."""
    d = _nl_case(gap=1.0)
    d["contact"]["max_iter"] = 300
    d["contact"]["tol"] = 1.0e-6
    r_big = dispatch.solve(Problem.from_dict(d))
    free = copy.deepcopy(d)
    free["contact"] = {"enabled": False}
    r_free = dispatch.solve(Problem.from_dict(free))
    assert float(np.max(r_big.contact.r_nodes)) == 0.0
    assert abs(r_big.w_max - r_free.w_max) / r_free.w_max < 1e-12


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
    """Малая нагрузка: karman-контакт → классический контакт (КР-сертиф.).

    Измерено rel(w)=5.0e-3 (зона груба на p=8), rel(∫r)=7.9e-5.
    """
    lk = _nl_case(gap_factor=0.55)
    lk["load"]["q0"] = 0.04
    lc = copy.deepcopy(lk)
    lc["model"]["theory"] = "classic"
    lc["contact"] = {"enabled": True, "gap_factor": 0.55, "max_iter": 20000,
                     "tol": 1.0e-9}
    rk = dispatch.solve(Problem.from_dict(lk))
    rc = dispatch.solve(Problem.from_dict(lc))
    tk = float(np.sum(rk.contact.r_nodes * rk._plate.quad.w))
    tc = float(np.sum(rc.contact.r_nodes * rc._plate.quad.w))
    assert abs(rk.w_max - rc.w_max) / rc.w_max < 2e-2
    assert abs(tk - tc) / tc < 1e-3


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
