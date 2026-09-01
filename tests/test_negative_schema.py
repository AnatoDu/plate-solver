r"""Систематические негативные тесты входного языка case-схемы (v0.7.0).

Принцип: «некорректный вход отклоняется с диагностикой» — как
эталонное знание имеет явную границу применимости, так и
решатель обязан отклонять некорректную постановку, а не считать молча.
Этот модуль закрывает охват СИСТЕМАТИЧЕСКИ (а не точечно по фичам):

* неизвестный ключ в КАЖДОЙ секции схемы → CaseError, называющий секцию;
* неизвестная секция верхнего уровня → CaseError;
* неизвестное значение КАЖДОГО реестрового ключа (kind/type/theory/…) →
  CaseError с перечнем допустимого;
* неверный тип значения (строка вместо числа, число вместо секции…);
* нарушение числовых границ (p ≥ 1, Q, положительность E/h/σ…).

Точечные негативные тесты фич живут в своих файлах (test_expr_load,
test_supports, …) — здесь каркас языка целиком.
"""

from __future__ import annotations

import copy

import pytest

from plate_solver.problem import CaseError, Problem

_BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"theory": "classic"},
    "contact": {"enabled": False},
    "discretization": {"p": 8, "Q": 32, "grid_n": 16},
    "verify": {"reference": "none"},
    "output": {"dir": "results/x", "figures": False},
}

_SECTIONS = ("geometry", "bc", "load", "model", "contact",
             "discretization", "verify", "output")


def _case():
    return copy.deepcopy(_BASE)


def test_unknown_top_level_section_rejected():
    d = _case()
    d["surprise"] = {"a": 1}
    with pytest.raises(CaseError, match="case"):
        Problem.from_dict(d)


@pytest.mark.parametrize("section", _SECTIONS)
def test_unknown_key_in_every_section_rejected(section):
    """Неизвестный ключ любой секции — отказ, называющий секцию."""
    d = _case()
    d[section] = dict(d[section])
    d[section]["surprise_key"] = 1.0
    with pytest.raises(CaseError, match=section):
        Problem.from_dict(d)


def test_unknown_key_supports_eigen_plate2_rejected():
    """Секции, отсутствующие в базовом случае, — тот же контракт."""
    d = _case()
    d["supports"] = {"points": [[0.0, 0.0]], "stiffness": 1.0,
                     "surprise": 2}
    with pytest.raises(CaseError, match="supports"):
        Problem.from_dict(d)
    d = _case()
    del d["load"], d["contact"]
    d["eigen"] = {"kind": "vibration", "surprise": 1}
    with pytest.raises(CaseError, match="eigen"):
        Problem.from_dict(d)
    d = _case()
    d["contact"] = {"enabled": True, "target": "plate2", "gap": 0.01}
    d["plate2"] = {"bc": {"type": "clamped"},
                   "load": {"type": "uniform", "q0": 1.0}, "surprise": 1}
    with pytest.raises(CaseError, match="plate2"):
        Problem.from_dict(d)


@pytest.mark.parametrize("section,key,registry_hint", [
    ("geometry", "kind", "circle"),
    ("bc", "type", "clamped"),
    ("load", "type", "uniform"),
    ("model", "theory", "classic"),
    ("verify", "reference", "none"),
    ("contact", "stop", "dr"),
])
def test_unknown_registry_value_rejected(section, key, registry_hint):
    """Неизвестное значение реестрового ключа — отказ с перечнем допустимого."""
    d = _case()
    if section == "contact":
        d["contact"] = {"enabled": True, "gap": 0.5, "stop": "surprise"}
        d["bc"] = {"type": "soft_hinge"}
    else:
        d[section] = dict(d[section])
        d[section][key] = "surprise"
    with pytest.raises(CaseError) as e:
        Problem.from_dict(d)
    assert registry_hint in str(e.value)            # перечень допустимого назван


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(geometry="circle"), "geometry"),      # секция не таблица
    (lambda d: d["geometry"].update(a="big"), "geometry.a"),  # строка вместо числа
    (lambda d: d["model"].update(E="steel"), "model.E"),
    (lambda d: d["model"].update(E=True), "model.E"),         # bool не число
    (lambda d: d["load"].update(q0=None), "load.q0"),
    (lambda d: d["discretization"].update(p="ten"), "discretization.p"),
    (lambda d: d["verify"].update(tol="tight"), "verify"),
])
def test_wrong_value_types_rejected(mutate, match):
    d = _case()
    mutate(d)
    with pytest.raises(CaseError, match=match):
        Problem.from_dict(d)


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["geometry"].update(a=-1.0), "geometry.a"),
    (lambda d: d["geometry"].update(a=0.0), "geometry.a"),
    (lambda d: d["discretization"].update(p=0), "discretization.p"),
    (lambda d: d["model"].update(h=-0.1), "model.h"),
    (lambda d: d["model"].update(nu=0.7), "model.nu"),
    (lambda d: d["model"].update(winkler=-5.0), "model.winkler"),
])
def test_numeric_bounds_rejected(mutate, match):
    d = _case()
    mutate(d)
    with pytest.raises(CaseError, match=match):
        Problem.from_dict(d)


def test_missing_required_sections_rejected():
    for missing in ("geometry", "bc", "load"):
        d = _case()
        del d[missing]
        with pytest.raises(CaseError, match=missing):
            Problem.from_dict(d)


def test_diagnostics_carry_schema_pointer():
    """Каждая диагностика ведёт к документации схемы (docs/CASE_SCHEMA.md)."""
    d = _case()
    d["load"] = {"type": "surprise"}
    with pytest.raises(CaseError) as e:
        Problem.from_dict(d)
    assert "CASE_SCHEMA" in str(e.value)
