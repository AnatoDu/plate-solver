"""MMS-эталон: обёртка MMS-шагов ladder.py в резолвере.

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
    """ВОРОТА: MMS-строка отчёта ГЕЙТУЕМА, сравнивает прогон MMS с точным решением.

    Три содержательных утверждения (прежде здесь стояло тождественно истинное
    ``value != approx(reference) or rel == 0``):

    * строка помечена ``gated`` и её вердикт — PASS в ЗАЯВЛЕННОМ допуске
      ``verify.tol = 1e-10`` (машинная точность полиномиальной ω);
    * ``value`` — прогиб ПРОГОНА MMS (с изготовленной нагрузкой), а не ``w_max``
      исходной задачи: величины различаются на шесть порядков, подмена видна;
    * при ухудшении дискретизации (кривая граница + грубая квадратура при том
      же допуске) ворота ПАДАЮТ — значит, проверка не вакуумна.
    """
    from plate_solver.dispatch import solve

    p = _problem()
    res = solve(p)
    report = verify_result(res)
    row = report.rows[0]
    ref = resolve_reference(p)[0]

    assert row.gated and row.passed is True and report.ok
    assert report.tol == 1.0e-10
    assert row.value == pytest.approx(ref.value, rel=1e-14)      # прогон MMS
    assert row.reference == pytest.approx(ref.w_max, rel=1e-14)  # точное решение
    assert row.rel == pytest.approx(abs(row.value - row.reference) / abs(row.reference))
    assert row.rel <= report.tol
    # исходная задача (равномерная нагрузка) даёт ДРУГОЙ прогиб — подмена заметна
    assert abs(res.w_max - row.value) > 0.5 * abs(row.value)

    # ухудшение дискретизации: круг (маска ~1/Q) при том же допуске 1e-10
    bad = _problem(geometry={"kind": "circle", "a": 1.0},
                   discretization={"p": 8, "Q": 64, "grid_n": 16})
    bad_report = verify_result(solve(bad))
    assert bad_report.rows[0].gated and bad_report.rows[0].passed is False
    assert not bad_report.ok and bad_report.rows[0].rel > 1.0e-10


def test_mms_rejections():
    with pytest.raises(CaseError, match="clamped"):
        resolve_reference(_problem(bc={"type": "soft_hinge"},
                                   verify={"reference": "mms"}))
    with pytest.raises(CaseError, match="rectangle | circle"):
        resolve_reference(_problem(geometry={"kind": "L", "side": 1.0, "cut": 0.5},
                                   verify={"reference": "mms"}))


# --------------------------------------------------------------------------- #
#  MMS полной КТН при замороженных усилиях (fixed-N, v0.6.4)
# --------------------------------------------------------------------------- #
_KTN = {"model": {"theory": "ktn_full", "h": 0.1}}


def test_gate_mms_ktn_rectangle_machine_precision():
    """ВОРОТА: MMS-КТН (fixed-N) на прямоугольнике — машинная точность (все члены сборки)."""
    refs = resolve_reference(_problem(**_KTN,
                                      discretization={"p": 10, "Q": 64, "grid_n": 16}))
    assert len(refs) == 1 and refs[0].kind == "mms"
    assert "КТН" in refs[0].name
    rel = abs(refs[0].value - refs[0].w_max) / refs[0].w_max
    assert rel < 1e-9, rel


def test_gate_mms_ktn_circle_mask_floor():
    """Круг+КТН: остаток задаёт маска ~1/Q — проверяем уровень и убывание с Q."""
    rels = []
    for Q in (128, 256):
        refs = resolve_reference(_problem(
            **_KTN, geometry={"kind": "circle", "a": 1.0},
            discretization={"p": 8, "Q": Q, "grid_n": 16},
            verify={"reference": "mms", "tol": 5.0e-2}))
        rels.append(abs(refs[0].value - refs[0].w_max) / refs[0].w_max)
    assert rels[0] < 5e-2 and rels[1] < rels[0]   # ~1/Q, убывает
