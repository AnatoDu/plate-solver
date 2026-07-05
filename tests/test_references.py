"""Резолвер эталонов (фаза 2, P3.1): модельная согласованность, cross_1d, model_gap."""

from __future__ import annotations

import copy

import pytest

from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem
from plate_solver.references import resolve_reference, verify_result

BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "soft_hinge"},
    "load": {"type": "uniform", "q0": 4.0},
    "discretization": {"p": 6, "Q": 256, "grid_n": 24},
    "verify": {"reference": "analytic", "cross_1d": True, "tol": 1.0e-2,
               "model_gap": True},
}


def _problem(**over) -> Problem:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = v
    return Problem.from_dict(d)


def test_circle_soft_full_report():
    """Мягкий шарнир на круге: analytic+1D в допуске, model_gap — инфо-строка."""
    report = verify_result(solve(_problem()))
    assert len(report.rows) == 3 and report.ok
    gated = [r for r in report.rows if r.gated]
    assert all(r.rel < 1.0e-2 for r in gated), [(r.name, r.rel) for r in gated]
    info = [r for r in report.rows if not r.gated]
    assert len(info) == 1 and info[0].passed is None
    assert info[0].rel > 0.2                     # модельный разрыв ~26.4 % — вне допуска
    assert "Кирхгоф" in info[0].name and "инфо" in report.table()


def test_circle_clamped_no_model_gap_row():
    report = verify_result(solve(_problem(bc={"type": "clamped"},
                                          discretization={"p": 8, "Q": 256,
                                                          "grid_n": 24})))
    assert all(r.gated for r in report.rows)     # у защемления разрыва нет
    assert len(report.rows) == 2 and report.ok


def test_annulus_soft_references_small():
    """Кольцо (малый Q): резолвер собирает analytic+1D+model_gap; сами ворота — P3.4."""
    p = _problem(geometry={"kind": "annulus", "a": 1.0, "b": 0.4},
                 discretization={"p": 10, "Q": 256, "grid_n": 24},
                 verify={"reference": "analytic", "cross_1d": True, "tol": 5.0e-2,
                         "model_gap": True})
    report = verify_result(solve(p))
    assert len(report.rows) == 3
    assert report.ok, report.table()             # грубый допуск 5 % при Q=256


def test_resolver_errors():
    # с трека C прямоугольник/soft_hinge гейтится Навье — эталон существует
    refs = resolve_reference(_problem(geometry={"kind": "rectangle", "x1": 0.0,
                                                "x2": 1.0, "y1": 0.0, "y2": 1.0},
                                      verify={"reference": "analytic",
                                              "cross_1d": False}))
    assert refs[0].point is not None                  # сравнение в центре
    with pytest.raises(CaseError, match="mms | fem | none"):
        resolve_reference(_problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                                   verify={"reference": "analytic",
                                           "cross_1d": False}))
    tree = {"op": "union", "children": [{"kind": "circle", "a": 1.0},
                                        {"kind": "circle", "a": 0.5, "cx": 1.0,
                                         "cy": 0.0}]}
    with pytest.raises(CaseError, match="compose"):
        resolve_reference(_problem(geometry={"kind": "compose", "tree": tree},
                                   bc={"type": "clamped"},
                                   verify={"reference": "fem", "cross_1d": False}))
    with pytest.raises(CaseError, match="инварианты"):
        resolve_reference(_problem(contact={"enabled": True, "gap_factor": 0.5},
                                   verify={"reference": "analytic"}))
    with pytest.raises(CaseError, match="ЦЕНТРЕ"):
        resolve_reference(_problem(load={"type": "point", "P": 1.0,
                                         "x0": 0.3, "y0": 0.0},
                                   verify={"reference": "analytic",
                                           "cross_1d": False}))
    # сила в центре круга — эталон существует (P3.5)
    refs = resolve_reference(_problem(load={"type": "point", "P": 1.0,
                                            "x0": 0.0, "y0": 0.0},
                                      verify={"reference": "analytic",
                                              "cross_1d": False}))
    assert len(refs) == 1 and refs[0].w_max > 0.0


def test_fem_incompatibility_reported_before_skfem_requirement(monkeypatch):
    """P0.2 фазы 3: несовместимость постановки объясняется ДО требования skfem.

    Имитация CI без extra fem (скрываем импорт skfem): несовместимый случай
    fem+compose обязан падать сообщением про compose, а совместимый
    fem+circle/clamped — просьбой установить scikit-fem.
    """
    import builtins

    real_import = builtins.__import__

    def hide_skfem(name, *args, **kwargs):
        if name == "skfem" or name.startswith("skfem."):
            raise ImportError("skfem скрыт (имитация CI)")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", hide_skfem)
    tree = {"op": "union", "children": [{"kind": "circle", "a": 1.0},
                                        {"kind": "circle", "a": 0.5, "cx": 1.0,
                                         "cy": 0.0}]}
    with pytest.raises(CaseError, match="compose"):
        resolve_reference(_problem(geometry={"kind": "compose", "tree": tree},
                                   bc={"type": "clamped"},
                                   verify={"reference": "fem", "cross_1d": False}))
    with pytest.raises(CaseError, match="scikit-fem"):
        resolve_reference(_problem(bc={"type": "clamped"},
                                   verify={"reference": "fem", "cross_1d": False}))
