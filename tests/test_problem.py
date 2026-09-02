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


def _config_fields_filled_by_to_config() -> set[str]:
    """Поля ``Config``, которые ``Problem.to_config`` умеет заполнять.

    Реестр строится ИЗ ИСХОДНИКА метода: ключи ``kw[...]`` — строковые литералы
    (в том числе в кортежах циклов), поэтому берутся все строковые литералы
    метода, совпадающие с именем поля ``Config``. Множество заведомо не УЖЕ
    фактического: лишний литерал лишь потребовал бы лишнего покрытия и уронил
    тест — сторона безопасная, тихой дыры не возникает.
    """
    import ast
    import dataclasses
    import inspect
    import textwrap

    src = textwrap.dedent(inspect.getsource(Problem.to_config))
    literals = {n.value for n in ast.walk(ast.parse(src))
                if isinstance(n, ast.Constant) and isinstance(n.value, str)}
    return literals & {f.name for f in dataclasses.fields(Config)}


_GEOM = {"kind": "circle", "a": 1.5}          # a ≠ дефолта Config (1.0)
_CLAMPED = {"type": "clamped"}
_LOAD = {"type": "uniform", "q0": 3.0}        # q0 ≠ дефолта Config (4.0)
_DISC = {"p": 8, "Q": 40, "grid_n": 36}
_H_ORT = 0.1                                  # толщина ортотропного набора
_K_EL = 1.0 - 0.2 * (0.2 * 1.0 / 2.0)         # 1 − ν_xy·ν_yx при Ex=2, Ey=1, ν_xy=0.2


def _full_case(**sections) -> dict:
    """Случай на явной геометрии/нагрузке/дискретизации (все значения ≠ дефолтов)."""
    base = dict(geometry=_GEOM, bc=_CLAMPED, load=_LOAD, discretization=_DISC,
                verify={"reference": "none"})
    base.update(sections)
    return _case(**base)


#: (метка, case, ПОЛНЫЙ перечень полей Config, отличных от дефолта, с их значениями).
#: Одним случаем всё покрыть нельзя: схема запрещает совмещать ортотропию с
#: контактом, опоры с термомоментом, ускорение Андерсона с методом newton и т. д.
_TO_CONFIG_CASES = [
    ("классика + позиционный контакт", _full_case(
        model={"theory": "classic", "E": 1.0e6, "nu": 0.25, "h": 0.06},
        contact={"enabled": True, "gap": 5.0e-5, "beta": 1.0,
                 "max_iter": 500, "tol": 1e-6, "stop": "comp"}),
     {"E": 1.0e6, "nu": 0.25, "h": 0.06, "q0": 3.0, "a": 1.5, "Delta": 5.0e-5,
      "beta": 1.0, "max_iter": 500, "tol": 1e-6, "stop": "comp",
      "p": 8, "Q": 40, "grid_n": 36}),
    ("Карман + нелинейный контакт (Пикар с Андерсоном)", _full_case(
        model={"theory": "karman", "n_load_steps": 3, "karman_relax": 0.7,
               "karman_max_iter": 150, "karman_tol": 1e-7, "karman_anderson": 4},
        contact={"enabled": True, "gap": 5.0e-5, "scheme": "nested",
                 "gain": "linear", "mor_anderson": 5}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36, "Delta": 5.0e-5,
      "n_load_steps": 3, "karman_relax": 0.7, "karman_max_iter": 150,
      "karman_tol": 1e-7, "karman_anderson": 4, "contact_scheme": "nested",
      "contact_gain": "linear", "mor_anderson": 5}),
    ("Карман методом Ньютона", _full_case(
        model={"theory": "karman", "karman_method": "newton"}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36,
      "karman_method": "newton"}),
    ("полная КТН: метод и слагаемые лицевого условия", _full_case(
        model={"theory": "ktn_full", "ktn_method": "newton",
               "face_terms": {"curvature": True, "load": False, "reaction": True}}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36,
      "ktn_method": "newton", "face_terms": (True, False, True)}),
    ("основание Винклера + точечные опоры", _full_case(
        model={"theory": "classic", "winkler": 1.0e3},
        supports={"points": [[0.0, 0.0], [0.5, 0.25]], "stiffness": 2.0e4}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36, "winkler": 1.0e3,
      "supports_points": ((0.0, 0.0), (0.5, 0.25)), "supports_stiffness": 2.0e4}),
    ("термомомент", _full_case(load={"type": "uniform", "q0": 3.0,
                                     "thermal_moment": 0.5}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36, "thermal_moment": 0.5}),
    ("переменная толщина h(x, y)", _full_case(
        model={"theory": "classic", "h_expr": "0.1 + 0.01*x"}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36,
      "h_expr": "0.1 + 0.01*x"}),
    ("ортотропия прямым набором жёсткостей", _full_case(
        model={"theory": "classic",
               "orthotropy": {"D11": 2.0, "D12": 0.5, "D22": 1.0, "D66": 0.4}}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36,
      "ortho_D": (2.0, 0.5, 1.0, 0.4)}),
    ("ортотропия инженерным набором (Карман ⇒ ещё и мембранная A)", _full_case(
        model={"theory": "karman", "h": _H_ORT,
               "orthotropy": {"Ex": 2.0, "Ey": 1.0, "nu_xy": 0.2, "Gxy": 0.5}}),
     {"q0": 3.0, "a": 1.5, "p": 8, "Q": 40, "grid_n": 36, "h": _H_ORT,
      # D_ij = E_i h³/[12(1−ν_xy ν_yx)], A_ij = E_i h/(1−ν_xy ν_yx)
      "ortho_D": (2.0 * _H_ORT**3 / 12.0 / _K_EL, 0.2 * 1.0 * _H_ORT**3 / 12.0 / _K_EL,
                  1.0 * _H_ORT**3 / 12.0 / _K_EL, 0.5 * _H_ORT**3 / 12.0),
      "ortho_A": (2.0 * _H_ORT / _K_EL, 0.2 * 1.0 * _H_ORT / _K_EL,
                  1.0 * _H_ORT / _K_EL, 0.5 * _H_ORT)}),
]


def _matches(got, want) -> bool:
    """Кортеж вычисляемых жёсткостей — с допуском fp; всё прочее — точно."""
    if isinstance(want, tuple) and want and all(isinstance(v, float) for v in want):
        return got == pytest.approx(want, rel=1e-13)
    return got == want


def test_to_config_maps_all_keys():
    """Ворота ПОЛНОТЫ отображения Problem → Config (не выборка ключей).

    Прежняя редакция проверяла 13 полей из 31 и не замечала бы потерю нового
    ключа. Теперь: (1) для каждого случая набор полей, ОТЛИЧНЫХ от дефолта
    ``Config``, сверяется как множество — лишнее отображение так же красно, как
    потерянное; (2) объединение по случаям обязано покрыть ВСЕ поля, которые
    ``to_config`` умеет заполнять (реестр строится из исходника метода), поэтому
    новый ключ схемы без покрытия роняет тест.
    """
    import dataclasses

    dflt = Config()
    covered: set[str] = set()
    for label, data, expected in _TO_CONFIG_CASES:
        cfg = Problem.from_dict(data).to_config()
        changed = {f.name for f in dataclasses.fields(Config)
                   if getattr(cfg, f.name) != getattr(dflt, f.name)}
        assert changed == set(expected), (label, sorted(changed ^ set(expected)))
        for field, value in expected.items():
            assert _matches(getattr(cfg, field), value), (label, field,
                                                          getattr(cfg, field), value)
        covered |= set(expected)

    filled = _config_fields_filled_by_to_config()
    assert filled, "реестр отображаемых полей пуст — сломан разбор исходника"
    assert covered == filled, sorted(covered ^ filled)


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
                  "classic | karman | ktn_linear | ktn_full")
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


def test_compose_degenerate_region_rejected():
    """ВЫРОЖДЕННАЯ compose-область (пустая внутренность) — понятная CaseError, не падение.

    Структурный валидатор пропускает геометрически пустые деревья (difference с
    поглощением, intersect непересекающихся); их ловит `dispatch.build_domain`
    (иначе — деление на ноль в сборке и падение LAPACK на NaN, аудит v0.6.4).
    """
    import math

    from plate_solver import dispatch

    # difference: вычитаемое поглощает уменьшаемое ⇒ пусто
    engulf = {"op": "difference", "children": [
        {"kind": "circle", "a": 1.0, "cx": 0.0, "cy": 0.0},
        {"kind": "circle", "a": 2.0, "cx": 0.0, "cy": 0.0}]}
    with pytest.raises(CaseError, match="ПУСТАЯ область"):
        dispatch.solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": engulf})))

    # intersect непересекающихся примитивов ⇒ пустое пересечение bbox
    disjoint = {"op": "intersect", "children": [
        {"kind": "circle", "a": 0.5, "cx": -3.0, "cy": 0.0},
        {"kind": "circle", "a": 0.5, "cx": 3.0, "cy": 0.0}]}
    with pytest.raises(CaseError, match="вырожденная область"):
        dispatch.solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": disjoint})))

    # контроль: валидная compose-область с непустой внутренностью решается
    ok = {"op": "difference", "children": [
        {"kind": "rectangle", "x1": -1.0, "x2": 1.0, "y1": -1.0, "y2": 1.0},
        {"kind": "circle", "a": 0.4, "cx": 0.0, "cy": 0.0}]}
    r = dispatch.solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": ok})))
    assert math.isfinite(r.w_max)


def test_file_errors(tmp_path):
    with pytest.raises(CaseError, match="не найден"):
        Problem.from_toml(tmp_path / "нет_такого.toml")
    broken = tmp_path / "broken.toml"
    broken.write_text("[geometry\nkind = ", encoding="utf-8")
    with pytest.raises(CaseError, match="TOML"):
        Problem.from_toml(broken)


def test_mms_winkler_rejected():
    """mms + winkler > 0 — ложный FAIL ворот; честный отказ (v0.7.0)."""
    import pytest as _pytest

    from plate_solver.problem import CaseError, Problem

    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "classic", "winkler": 1.0e5},
        "verify": {"reference": "mms"},
    }
    with _pytest.raises(CaseError, match="verify.reference"):
        Problem.from_dict(d)
