"""Валидатор case-файлов: валидные/невалидные случаи, тексты ошибок.

Каждая ошибка обязана быть человекочитаемой: «ключ: получено X, ожидалось Y,
см. docs/CASE_SCHEMA.md#секция». Round-trip: TOML → Problem → to_config.
"""

from __future__ import annotations

import copy

import pytest

from plate_solver.config import Config
from plate_solver.problem import CaseError, Problem

MINIMAL = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "soft_hinge"},
    "load": {"type": "uniform", "q0": 4.0},
}


def _case(**sections) -> dict:
    """Минимальный валидный случай с переопределением секций (глубоко по секциям)."""
    d = copy.deepcopy(MINIMAL)
    for name, content in sections.items():
        d[name] = content
    return d


def _expect_error(data: dict, *fragments: str) -> str:
    with pytest.raises(CaseError) as e:
        Problem.from_dict(data)
    msg = str(e.value)
    assert "ожидалось" in msg and "CASE_SCHEMA.md#" in msg, msg
    for frag in fragments:
        assert frag in msg, (frag, msg)
    return msg


# --------------------------------------------------------------------------- #
#  Валидные случаи и дефолты
# --------------------------------------------------------------------------- #
def test_minimal_case_and_defaults():
    p = Problem.from_dict(MINIMAL)
    assert p.geometry.kind == "circle" and p.geometry.a == 1.0
    assert p.bc.type == "soft_hinge" and p.load.q0 == 4.0
    assert not p.contact.enabled
    assert p.model.theory == "classic" and p.model.E is None  # дефолты НЕ дублируются
    assert p.verify.reference == "none" and p.verify.tol == 1e-2
    assert p.output.dir == "results" and p.output.figures is False


def test_to_config_inherits_config_defaults():
    cfg = Problem.from_dict(MINIMAL).to_config()
    ref = Config(q0=4.0, a=1.0)
    assert cfg == ref                       # всё прочее — дефолты Config


def test_to_config_maps_all_keys():
    data = _case(
        model={"theory": "ktn", "E": 1.0e6, "nu": 0.25, "h": 0.06},
        contact={"enabled": True, "gap": 5.0e-5, "beta": 1.0,
                 "max_iter": 500, "tol": 1e-6, "stop": "comp"},
        discretization={"p": 8, "Q": 40, "grid_n": 36},
    )
    cfg = Problem.from_dict(data).to_config()
    assert (cfg.E, cfg.nu, cfg.h, cfg.q0, cfg.a) == (1.0e6, 0.25, 0.06, 4.0, 1.0)
    assert (cfg.Delta, cfg.beta, cfg.max_iter, cfg.tol, cfg.stop) == \
        (5.0e-5, 1.0, 500, 1e-6, "comp")
    assert (cfg.p, cfg.Q, cfg.grid_n) == (8, 40, 36)


def test_round_trip_from_toml(tmp_path):
    case = tmp_path / "annulus.toml"
    case.write_text(
        """
        [geometry]
        kind = "annulus"
        a = 1.0
        b = 0.4

        [bc]
        type = "clamped"

        [load]
        type = "uniform"
        q0 = 4.0

        [discretization]
        p = 10
        Q = 128

        [verify]
        reference = "analytic"
        cross_1d = true
        tol = 1.0e-2
        """,
        encoding="utf-8",
    )
    p = Problem.from_toml(case)
    assert p.source.endswith("annulus.toml")
    assert p.geometry.kind == "annulus" and (p.geometry.a, p.geometry.b) == (1.0, 0.4)
    assert p.verify.cross_1d is True
    cfg = p.to_config()
    assert (cfg.p, cfg.Q, cfg.a) == (10, 128, 1.0)


def test_valid_patch_point_and_zone():
    p = Problem.from_dict(_case(load={"type": "patch", "q0": 4.0,
                                      "zone": {"kind": "circle", "a": 0.3}}))
    assert p.load.zone.kind == "circle" and p.load.zone.a == 0.3
    p = Problem.from_dict(_case(load={"type": "point", "P": 1.0, "x0": 0.0, "y0": 0.0}))
    assert p.load.P == 1.0 and p.load.eps is None   # eps — дефолт от bbox (диспетчер)
    p = Problem.from_dict(_case(
        contact={"enabled": True, "gap_factor": 0.5,
                 "zone": {"kind": "rectangle", "x1": 0.15, "x2": 0.45,
                          "y1": 0.15, "y2": 0.45}}))
    assert p.contact.zone.kind == "rectangle"


def test_valid_compose_depth3():
    tree = {"op": "difference", "children": [
        {"op": "union", "children": [
            {"kind": "circle", "a": 1.0},
            {"kind": "rectangle", "x1": 0.0, "x2": 2.0, "y1": -0.5, "y2": 0.5},
        ]},
        {"kind": "circle", "a": 0.3, "cx": 0.5, "cy": 0.0},
    ]}
    p = Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree}))
    assert p.geometry.tree["op"] == "difference"


# --------------------------------------------------------------------------- #
#  Невалидные случаи: тексты ошибок
# --------------------------------------------------------------------------- #
def test_bad_kind_and_missing_section():
    _expect_error(_case(geometry={"kind": "triangle"}), "geometry.kind", "circle")
    d = copy.deepcopy(MINIMAL)
    del d["bc"]
    _expect_error(d, "bc", "обязательная секция")


def test_unknown_key_rejected():
    _expect_error(_case(geometry={"kind": "circle", "a": 1.0, "radius": 2.0}),
                  "geometry.radius")


def test_geometry_param_checks():
    _expect_error(_case(geometry={"kind": "rectangle", "x1": 1.0, "x2": 0.0,
                                  "y1": 0.0, "y2": 1.0}), "x1 < x2")
    _expect_error(_case(geometry={"kind": "L", "side": 1.0, "cut": 1.5}), "cut < side")
    _expect_error(_case(geometry={"kind": "annulus", "a": 0.4, "b": 1.0}), "внутренний радиус")
    _expect_error(_case(geometry={"kind": "circle", "a": -1.0}), "положительное")


def test_load_checks():
    _expect_error(_case(load={"type": "patch", "q0": 4.0}), "load.zone", "обязательна")
    _expect_error(_case(load={"type": "point", "x0": 0.0, "y0": 0.0}), "load.P")
    _expect_error(_case(load={"type": "hydro", "q0": 1.0}), "load.type", "uniform")


def test_contact_checks():
    _expect_error(_case(contact={"enabled": True}), "contact.gap", "ровно одно")
    _expect_error(_case(contact={"enabled": True, "gap": 1e-4, "gap_factor": 0.5}),
                  "ровно одно")
    _expect_error(_case(contact={"enabled": True, "gap": 1e-4, "stop": "energy"}),
                  "contact.stop", "dr | comp")
    # контакт + защемление разрешён (A3.3)
    p = Problem.from_dict(_case(bc={"type": "clamped"},
                                contact={"enabled": True, "gap_factor": 0.5}))
    assert p.contact.enabled and p.bc.type == "clamped"


def test_verify_checks():
    tree = {"op": "union", "children": [{"kind": "circle", "a": 1.0},
                                        {"kind": "circle", "a": 0.5, "cx": 1.0, "cy": 0.0}]}
    _expect_error(_case(geometry={"kind": "compose", "tree": tree},
                        verify={"reference": "analytic"}),
                  "verify.reference", "mms | fem | none")
    _expect_error(_case(geometry={"kind": "rectangle", "x1": 0.0, "x2": 1.0,
                                  "y1": 0.0, "y2": 1.0},
                        verify={"cross_1d": True}),
                  "verify.cross_1d", "осесимметричных")
    # point-нагрузка неосесимметрична для cross_1d даже на круге
    _expect_error(_case(load={"type": "point", "P": 1.0, "x0": 0.3, "y0": 0.0},
                        verify={"cross_1d": True}),
                  "verify.cross_1d")


def test_model_checks():
    _expect_error(_case(model={"theory": "mindlin"}), "model.theory",
                  "classic | karman | ktn")
    _expect_error(_case(model={"nu": 0.7}), "model.nu")


def test_compose_fence():
    # глубина 4 отклоняется
    deep = {"op": "union", "children": [
        {"op": "union", "children": [
            {"op": "union", "children": [
                {"kind": "circle", "a": 1.0},
                {"kind": "circle", "a": 0.5}]},
            {"kind": "circle", "a": 0.4}]},
        {"kind": "circle", "a": 0.3}]}
    _expect_error(_case(geometry={"kind": "compose", "tree": deep}), "глубина")
    # > 7 узлов отклоняется (1 op + 7 примитивов = 8)
    wide = {"op": "union", "children": [{"kind": "circle", "a": float(i + 1)}
                                        for i in range(7)]}
    _expect_error(_case(geometry={"kind": "compose", "tree": wide}), "узлов")
    # difference строго бинарна
    tri = {"op": "difference", "children": [{"kind": "circle", "a": 1.0},
                                            {"kind": "circle", "a": 0.5},
                                            {"kind": "circle", "a": 0.2}]}
    _expect_error(_case(geometry={"kind": "compose", "tree": tri}), "ровно 2")
    # неизвестная операция
    bad = {"op": "xor", "children": [{"kind": "circle", "a": 1.0},
                                     {"kind": "circle", "a": 0.5}]}
    _expect_error(_case(geometry={"kind": "compose", "tree": bad}), "union | intersect")


def test_file_errors(tmp_path):
    with pytest.raises(CaseError, match="не найден"):
        Problem.from_toml(tmp_path / "нет_такого.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[geometry\nkind = ", encoding="utf-8")
    with pytest.raises(CaseError, match="TOML"):
        Problem.from_toml(broken)
