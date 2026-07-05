"""MMS-эталон (фаза 2, P3.8): обёртка MMS-шагов ladder.py в резолвере.

Прямоугольник: полиномиальная ω, решение в структуре ⇒ машинная точность.
Круг: кривая граница ⇒ остаток ступенчатой маски ~1/Q (NOTES §14).
"""

from __future__ import annotations

import copy

import pytest

from plate_solver.problem import CaseError, Problem
from plate_solver.references import resolve_reference, verify_result

BASE = {
    "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0, "y1": -0.5, "y2": 0.5},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 1.0},
    "discretization": {"p": 8, "Q": 64, "grid_n": 16},
    "verify": {"reference": "mms", "tol": 1.0e-10},
}


def _problem(**over) -> Problem:
    d = copy.deepcopy(BASE)
    for k, v in over.items():
        d[k] = v
    return Problem.from_dict(d)


def test_gate_mms_rectangle_machine_precision():
    """ВОРОТА: MMS на прямоугольнике (полиномиальная ω) — машинная точность."""
    refs = resolve_reference(_problem())
    assert len(refs) == 1 and refs[0].kind == "mms" and refs[0].value is not None
    rel = abs(refs[0].value - refs[0].w_max) / refs[0].w_max
    assert rel < 1e-10, rel                       # решение в структуре (NOTES §14)


def test_gate_mms_rectangle_offcenter():
    """Смещённый прямоугольник: поле и ω строятся вокруг его центра."""
    p = _problem(geometry={"kind": "rectangle", "x1": 0.0, "x2": 2.0,
                           "y1": 1.0, "y2": 2.0})
    refs = resolve_reference(p)
    rel = abs(refs[0].value - refs[0].w_max) / refs[0].w_max
    assert rel < 1e-10, rel


def test_gate_mms_circle_mask_floor():
    """Круг: остаток задаёт маска ~1/Q — проверяем уровень и убывание с Q."""
    rels = []
    for Q in (128, 256):
        p = _problem(geometry={"kind": "circle", "a": 1.0},
                     discretization={"p": 8, "Q": Q, "grid_n": 16},
                     verify={"reference": "mms", "tol": 5.0e-2})
        refs = resolve_reference(p)
        rels.append(abs(refs[0].value - refs[0].w_max) / refs[0].w_max)
    assert rels[0] < 5e-2 and rels[1] < rels[0]   # ~1/Q, убывает


def test_mms_through_verify_result():
    """verify_result использует value mms-прогона, а не w_max исходной задачи."""
    from plate_solver.dispatch import solve

    p = _problem()
    report = verify_result(solve(p))
    assert report.ok
    assert report.rows[0].value != pytest.approx(report.rows[0].reference, abs=0.0) \
        or report.rows[0].rel == 0.0


def test_mms_rejections():
    with pytest.raises(CaseError, match="clamped"):
        resolve_reference(_problem(bc={"type": "soft_hinge"},
                                   verify={"reference": "mms"}))
    with pytest.raises(CaseError, match="rectangle | circle"):
        resolve_reference(_problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                                   verify={"reference": "mms"}))
