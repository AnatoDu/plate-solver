r"""Ограды ДИСПЕТЧЕРА: то, что валидатор схемы поймать не может (v0.8.0).

Статический валидатор (``problem.py``) видит только текст постановки; часть
ошибок проявляется лишь на ГЕОМЕТРИИ и ДИСКРЕТИЗАЦИИ:

* разрешение квадратуры (узлов внутри Ω меньше базисных функций) зависит от
  доли bbox, занятой областью;
* смежные (лишь соприкасающиеся) операнды ``union`` дают внутреннюю линию
  ``ω = 0`` — невидимую опору внутри пластины;
* точка приложения силы вне Ω или пятно, вылезающее за кромку, прикладывают
  лишь ЧАСТЬ силы;
* физика первой пластины (Винклер, ортотропия, ``h(x, y)``, термомомент,
  опоры) молча наследовалась второй в тракте пары.

Каждый пункт раньше давал ТИХО неверный результат (или сырое исключение
LAPACK при формально валидной постановке) — ворота ниже фиксируют отказ или
явное предупреждение.
"""

from __future__ import annotations

import copy
import tomllib
import warnings as W
from pathlib import Path

import numpy as np
import pytest

from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]

_BASE = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"h": 0.1},
    "discretization": {"p": 6, "Q": 32, "grid_n": 16},
}


def _case(**over):
    """Базовая постановка с ЗАМЕНОЙ секций целиком (не слиянием ключей)."""
    d = copy.deepcopy(_BASE)
    d.update(copy.deepcopy(over))
    return d


# --------------------------------------------------------------------------- #
#  Разрешение квадратуры: M ≥ N (аудит S02)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p,Q", [(12, 8), (12, 16), (6, 2), (6, 8)])
def test_underresolved_ritz_rejected(p, Q):
    """M < N ⇒ отказ: система Ритца недоопределена (раньше — тихий мусор).

    При ``M < N`` ранг матрицы ≤ M: защемление молча уходило в МНК-фолбэк
    (прогиб НЕВЕРНОГО ЗНАКА, cond ~1e20, пустые warnings), мягкий шарнир падал
    сырым LinAlgError — при том что ``--check`` отвечал «постановка валидна».
    """
    with pytest.raises(CaseError, match="узлов квадратуры"):
        solve(Problem.from_dict(_case(discretization={"p": p, "Q": Q, "grid_n": 16})))


def test_marginal_resolution_warns():
    """N ≤ M < 2N ⇒ расчёт идёт, но с явным предупреждением о грубых интегралах."""
    res = solve(Problem.from_dict(_case(discretization={"p": 12, "Q": 24, "grid_n": 16})))
    assert any("дискретизация на грани" in w for w in res.warnings)
    assert res.w_max > 0.0


def test_shipped_cases_pass_resolution_guard():
    """Ни один поставляемый case не задевает ограду (ci и ladder)."""
    from plate_solver.dispatch import _check_resolution, build_domain

    for path in sorted((_ROOT / "cases").glob("*/*.toml")):
        prob = Problem.from_toml(path)
        warn: list[str] = []
        _check_resolution(prob.to_config(), build_domain(prob.geometry), warn)
        assert not warn, f"{path.name}: {warn}"


# --------------------------------------------------------------------------- #
#  Составная область: внутренние линии ω = 0 и связность (аудит S01)
# --------------------------------------------------------------------------- #
def _strip(y1, y2):
    return {"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": y1, "y2": y2}


def test_touching_union_operands_rejected():
    """ГЛАВНЫЕ ВОРОТА: смежные операнды union ⇒ отказ (внутренняя линия ω = 0).

    R-дизъюнкция обращается в нуль на общей границе: структура ``w = ω·Φ``
    зануляет там прогиб, и пластина считается с невидимой внутренней опорой —
    ``w_max`` занижается в 6.4 раза БЕЗ единого предупреждения.
    """
    tree = {"op": "union", "children": [_strip(0.0, 0.5), _strip(0.5, 1.0)]}
    with pytest.raises(CaseError, match="соприкасаются"):
        solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree},
                                      bc={"type": "soft_hinge"})))


def test_disjoint_union_operands_rejected():
    """Непересекающиеся операнды — две независимые пластины ⇒ отказ."""
    tree = {"op": "union", "children": [
        {"kind": "circle", "a": 0.3, "cx": -1.0, "cy": 0.0},
        {"kind": "circle", "a": 0.3, "cx": 1.0, "cy": 0.0}]}
    with pytest.raises(CaseError, match="не образуют ОДНУ область"):
        solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree},
                                      bc={"type": "soft_hinge"})))


def test_overlapping_union_accepted_and_close_to_whole():
    """Перекрывающиеся операнды ⇒ расчёт идёт и близок к цельной области."""
    tree = {"op": "union", "children": [_strip(0.0, 0.6), _strip(0.4, 1.0)]}
    res_u = solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree},
                                          bc={"type": "soft_hinge"},
                                          discretization={"p": 8, "Q": 48, "grid_n": 16})))
    res_r = solve(Problem.from_dict(_case(
        geometry={"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
        bc={"type": "soft_hinge"},
        discretization={"p": 8, "Q": 48, "grid_n": 16})))
    assert res_u.w_max == pytest.approx(res_r.w_max, rel=0.05)


def test_narrow_union_overlap_warns_across_the_whole_degradation_zone():
    r"""Предупреждение об УЗКОМ перекрытии покрывает всю зону занижения прогиба.

    Прежний критерий («значение ω в перекрытии < 10 % масштаба ω») срабатывал
    лишь при ширине перекрытия ≲ 0.025 размера пластины, а занижение прогиба
    при ширине 0.03…0.20 (от −87 % до −8 % при защемлении) проходило МОЛЧА:
    отношение ``min_ov/scale`` остаётся ≈ 0.12 даже при ширине 0.40 и потому
    негодно как мера. Мерой служит ШИРИНА перекрытия относительно размера
    пластины (порог :data:`dispatch.NARROW_OVERLAP_FRACTION`).

    Ворота проверяют обе стороны: узкое перекрытие ПРЕДУПРЕЖДАЕТ и при этом
    действительно занижает прогиб, широкое — молчит.
    """
    from plate_solver.dispatch import NARROW_OVERLAP_FRACTION

    def run(ov):
        tree = {"op": "union", "children": [_strip(0.0, 0.5 + ov / 2),
                                            _strip(0.5 - ov / 2, 1.0)]}
        with W.catch_warnings(record=True) as rec:
            W.simplefilter("always")
            res = solve(Problem.from_dict(_case(
                geometry={"kind": "compose", "tree": tree},
                bc={"type": "clamped"},
                discretization={"p": 8, "Q": 64, "grid_n": 16})))
        narrow = any("перекрываются УЗКО" in str(x.message) for x in rec)
        return res.w_max, narrow

    whole = solve(Problem.from_dict(_case(
        geometry={"kind": "rectangle", "x1": 0.0, "x2": 1.0, "y1": 0.0, "y2": 1.0},
        bc={"type": "clamped"},
        discretization={"p": 8, "Q": 64, "grid_n": 16}))).w_max

    for ov in (0.05, 0.10, 0.20):                  # зона деградации: предупреждать ОБЯЗАНО
        w, narrow = run(ov)
        assert narrow, f"ширина перекрытия {ov}: предупреждения нет"
        assert w < 0.95 * whole, f"ширина {ov}: занижения нет — ворота пересмотреть"
    w_wide, narrow_wide = run(0.5)                 # широкое перекрытие — молчание
    assert not narrow_wide
    assert 0.5 > NARROW_OVERLAP_FRACTION           # порог между этими режимами


def test_chain_of_three_overlapping_strips_accepted():
    """Цепочка A–B–C (каждая пара соседей перекрывается) — связная область."""
    tree = {"op": "union", "children": [_strip(0.0, 0.4), _strip(0.3, 0.7),
                                        _strip(0.6, 1.0)]}
    res = solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree},
                                        bc={"type": "soft_hinge"},
                                        discretization={"p": 8, "Q": 48, "grid_n": 16})))
    assert res.w_max > 0.0


def test_narrow_overlap_warns():
    """Узкое перекрытие ⇒ предупреждение о «долине» почти нулевой ω."""
    tree = {"op": "union", "children": [_strip(0.0, 0.505), _strip(0.495, 1.0)]}
    with W.catch_warnings(record=True) as rec:
        W.simplefilter("always")
        solve(Problem.from_dict(_case(geometry={"kind": "compose", "tree": tree},
                                      bc={"type": "soft_hinge"},
                                      discretization={"p": 8, "Q": 48, "grid_n": 16})))
    assert any("перекрываются УЗКО" in str(w.message) for w in rec)


# --------------------------------------------------------------------------- #
#  Точечная сила: точка вне Ω и пятно у кромки (аудит D08)
# --------------------------------------------------------------------------- #
def test_point_load_outside_domain_rejected():
    """Точка приложения вне области ⇒ отказ (симметрично exact = true)."""
    case = _case(load={"type": "point", "P": 1.0, "x0": 1.05, "y0": 0.0, "eps": 0.1},
                 discretization={"p": 8, "Q": 96, "grid_n": 16})
    with pytest.raises(CaseError, match="вне области"):
        solve(Problem.from_dict(case))


def test_point_load_near_edge_warns_with_fraction():
    """Пятно у кромки: приложена ЧАСТЬ силы — предупреждение с долей."""
    case = _case(load={"type": "point", "P": 1.0, "x0": 0.99, "y0": 0.0, "eps": 0.1},
                 discretization={"p": 8, "Q": 96, "grid_n": 16})
    res = solve(Problem.from_dict(case))
    msg = [w for w in res.warnings if "пятно радиуса" in w]
    assert msg, res.warnings
    assert "%" in msg[0]
    # внутреннее пятно предупреждения не даёт
    ok = _case(load={"type": "point", "P": 1.0, "x0": 0.0, "y0": 0.0, "eps": 0.1},
               discretization={"p": 8, "Q": 96, "grid_n": 16})
    assert not [w for w in solve(Problem.from_dict(ok)).warnings if "пятно радиуса" in w]


# --------------------------------------------------------------------------- #
#  Пара пластин: молчаливая потеря и молчаливое наследование (аудит D05)
# --------------------------------------------------------------------------- #
def _pair_case():
    return tomllib.loads((_ROOT / "cases" / "ci" / "two_plates_equal.toml")
                         .read_text(encoding="utf-8"))


@pytest.mark.parametrize("key,value", [
    ("winkler", 1.0), ("h_expr", "1.0"), ("karman_relax", 0.5),
    ("orthotropy", {"D11": 1.0, "D12": 0.3, "D22": 1.0, "D66": 0.35}),
])
def test_plate2_model_unsupported_keys_rejected(key, value):
    """Ключи [plate2.model], которые тракт пары не переносит, отклоняются явно."""
    d = _pair_case()
    d["plate2"].setdefault("model", {})[key] = value
    with pytest.raises(CaseError, match="plate2.model"):
        Problem.from_dict(d)


def test_plate2_theory_must_match_first_plate():
    """theory второй пластины допустима, но ОБЯЗАНА совпадать с первой.

    Тракт пары строит оба решателя по теории ПЕРВОЙ пластины; отличная
    ``[plate2.model] theory`` была бы молча потеряна.
    """
    d = _pair_case()
    d["plate2"].setdefault("model", {})["theory"] = "karman"
    with pytest.raises(CaseError, match="plate2.model.theory"):
        Problem.from_dict(d)
    # совпадающая теория принимается (так записаны поставляемые случаи пары)
    d2 = _pair_case()
    same = d2.get("model", {}).get("theory", "classic")
    d2["plate2"].setdefault("model", {})["theory"] = same
    assert Problem.from_dict(d2).plate2 is not None


def test_plate2_thermal_moment_rejected():
    """[plate2.load] thermal_moment не реализован ⇒ отказ вместо потери ключа."""
    d = _pair_case()
    d["plate2"]["load"]["thermal_moment"] = 1.0
    with pytest.raises(CaseError, match="thermal_moment"):
        Problem.from_dict(d)


def test_first_plate_physics_not_silently_inherited():
    """Физика ПЕРВОЙ пластины, не переносимая на вторую, отклоняется при [plate2].

    Раньше ``cfg2 = replace(cfg, …)`` наследовал её молча: вторая пластина
    «получала» чужое основание Винклера. Остальные ключи этого класса
    (ортотропия, ``h(x, y)``, термомомент, опоры) на кейсе пары отсекаются ещё
    более ранними оградами — здесь проверяется именно ограда наследования.
    """
    d = _pair_case()
    d.setdefault("model", {})["winkler"] = 10.0
    with pytest.raises(CaseError, match="plate2|отсутствие"):
        Problem.from_dict(d)


def test_plate2_thickness_still_allowed():
    """Разрешённые ключи ([plate2.model] E/nu/h) продолжают работать."""
    d = _pair_case()
    d["plate2"].setdefault("model", {})["h"] = 0.02
    prob = Problem.from_dict(d)
    assert prob.plate2.model.h == 0.02


# --------------------------------------------------------------------------- #
#  Пустая история невязок (аудит D01)
# --------------------------------------------------------------------------- #
def test_empty_residual_history_does_not_crash(tmp_path):
    """stop = "comp" без касания: KKT-невязка нулевая на первой проверке.

    История невязок пуста, и ``scalars()``/``save()`` падали IndexError на
    законной постановке.
    """
    case = _case(contact={"enabled": True, "gap_factor": 5.0, "stop": "comp",
                          "tol": 1e-6, "max_iter": 50})
    res = solve(Problem.from_dict(case))
    sc = res.scalars()
    assert sc["n_contact"] == 0
    assert sc["residual_first"] is None and sc["residual_last"] is None
    assert sc["residual_history"] == []
    path = res.save(tmp_path, fig_formats=())
    assert path.exists()
    import json

    assert json.loads(path.read_text(encoding="utf-8"))["scalars"]["n_contact"] == 0


# --------------------------------------------------------------------------- #
#  Ворота инвариантов контакта против «вакуумного PASS» (аудит S04, T01)
# --------------------------------------------------------------------------- #
def test_contact_invariants_catch_lost_contact():
    """ГЛАВНЫЕ ВОРОТА: развал МОР (r ≡ 0) при w_free > Δ ⇒ FAIL верификации.

    Раньше при ``reference = "none"`` список эталонов был пуст, и
    ``plate-verify`` выносил PASS, не проверив ничего: расходящаяся итерация
    возвращала свободное решение и была неотличима от контактного.
    """
    from plate_solver.references import verify_result

    case = _case(contact={"enabled": True, "gap_factor": 0.5, "max_iter": 400,
                          "tol": 1e-10},
                 verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    rep = verify_result(res)
    assert rep.ok and [r for r in rep.rows if r.gated and "инвариант" in r.name]
    assert res.free_overlap > 0.0                      # касание ОЖИДАЕТСЯ

    res.contact.r_nodes[:] = 0.0                       # имитация развала итерации
    object.__setattr__(res.contact, "gap_overshoot", float("nan"))
    bad = verify_result(res)
    assert not bad.ok
    assert any("контакт существует" in r.name and not r.passed
               for r in bad.rows if r.gated)


def test_contact_expected_by_zone_not_by_global_maximum():
    r"""Ожидаемость касания меряется ПОД ОСНОВАНИЕМ, а не по всей пластине.

    Штамп у кромки: свободный прогиб в зоне штампа МЕНЬШЕ зазора, значит
    верный ответ — ``r ≡ 0``. До v0.8.0-аудита ворота сравнивали с зазором
    ГЛОБАЛЬНЫЙ ``max|w_free|`` (он берётся в центре, вне штампа) и требовали
    контакта: ``plate-verify`` возвращал 1 на физически верном результате.
    Теперь та же постановка даёт PASS, а гейтом становится ОБРАТНЫЙ
    инвариант — «контакта нет».
    """
    from plate_solver.references import verify_result

    case = _case(discretization={"p": 8, "Q": 96, "grid_n": 24},
                 contact={"enabled": True, "gap": 2.2521e-04, "max_iter": 500,
                          "tol": 1e-9,
                          "zone": {"kind": "rectangle", "x1": 0.55, "x2": 0.85,
                                   "y1": -0.15, "y2": 0.15}},
                 verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    assert int((res.contact.r_nodes > 0).sum()) == 0          # касания нет — верно
    assert res.w_free_max > res.delta                          # прежний признак «ожидается»
    assert res.free_overlap < 0.0                              # но ПОД ШТАМПОМ перекрытия нет
    rep = verify_result(res)
    assert rep.ok, "\n" + rep.table()
    assert any("контакта нет" in r.name and r.gated and r.passed for r in rep.rows)

    # ворота невакуумны: паразитная реакция в зоне без перекрытия ⇒ FAIL
    res.contact.r_nodes[np.argmax(res.contact.r_nodes >= 0.0)] = 1.0
    assert not verify_result(res).ok


def test_pair_contact_expected_by_approach_not_by_first_plate():
    """У пары ожидаемость касания меряется СБЛИЖЕНИЕМ лицевых.

    Две одинаковые пластины под одинаковой нагрузкой прогибаются одинаково и
    не соприкасаются НИКОГДА, каким бы ни был прогиб. Прежний признак
    (``max|w₁_free| > Δ``) требовал контакта и красил верный результат.
    """
    from plate_solver.references import verify_result

    case = _case(discretization={"p": 6, "Q": 48, "grid_n": 16},
                 contact={"enabled": True, "target": "plate2", "gap": 1.0e-4,
                          "max_iter": 500, "tol": 1e-9},
                 plate2={"bc": {"type": "clamped"},
                         "load": {"type": "uniform", "q0": 4.0}},
                 verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    assert int((res.contact.r_nodes > 0).sum()) == 0
    assert res.w_free_max > res.delta                          # прогиб первой велик
    assert res.free_overlap < 0.0                              # а СБЛИЖЕНИЕ нулевое
    assert verify_result(res).ok


def test_pair_contact_gated_when_gap_exceeds_first_plate_deflection():
    """Контакт есть, а зазор больше прогиба первой пластины — ворота ОБЯЗАНЫ быть.

    Вторая пластина нагружена вверх: сближение перекрывает зазор, хотя
    ``Δ > max|w₁_free|``. Прежний признак снимал единственные нетривиальные
    ворота, и подсунутый ``r ≡ 0`` проходил верификацию (вакуумный PASS).
    """
    from plate_solver.references import verify_result

    case = _case(discretization={"p": 6, "Q": 48, "grid_n": 16},
                 contact={"enabled": True, "target": "plate2", "gap": 8.0e-4,
                          "max_iter": 800, "tol": 1e-9},
                 plate2={"bc": {"type": "clamped"},
                         "load": {"type": "uniform", "q0": -40.0}},
                 verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    assert int((res.contact.r_nodes > 0).sum()) > 0
    assert res.delta > res.w_free_max                          # прежний признак молчал бы
    assert res.free_overlap > 0.0
    rep = verify_result(res)
    assert rep.ok and any("контакт существует" in r.name and r.gated for r in rep.rows)

    res.contact.r_nodes[:] = 0.0                               # фабрикация развала
    object.__setattr__(res.contact, "gap_overshoot", float("nan"))
    assert not verify_result(res).ok


def test_contact_invariants_catch_negative_reaction():
    """Односторонняя связь: r < 0 ⇒ FAIL (структурный инвариант)."""
    from plate_solver.references import verify_result

    case = _case(contact={"enabled": True, "gap_factor": 0.5, "max_iter": 400,
                          "tol": 1e-10}, verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    res.contact.r_nodes[0] = -1.0
    rep = verify_result(res)
    assert not rep.ok
    assert any("r ≥ 0" in r.name and not r.passed for r in rep.rows if r.gated)


def test_contact_invariants_catch_penetration():
    """Проникание > 5 % зазора ⇒ FAIL (условие непроникания нарушено)."""
    from plate_solver.references import verify_result

    case = _case(contact={"enabled": True, "gap_factor": 0.5, "max_iter": 400,
                          "tol": 1e-10}, verify={"reference": "none"})
    res = solve(Problem.from_dict(case))
    object.__setattr__(res.contact, "gap_overshoot", 0.2)
    rep = verify_result(res)
    assert not rep.ok
    assert any("проникание" in r.name and not r.passed for r in rep.rows if r.gated)


# --------------------------------------------------------------------------- #
#  CLI: свип в обход валидатора и падение лестницы (аудит S05, S21)
# --------------------------------------------------------------------------- #
def test_sweep_values_obey_schema_bounds():
    """--sweep подчиняется тем же границам, что и ключи [discretization].

    Раньше свип шёл В ОБХОД валидатора: p = 0 считался молча, p < 0 давал
    сырую трассировку.
    """
    from plate_solver.cli import _parse_sweep

    for spec in ("p=0:4:2", "p=-2:4:2", "Q=1:8:2"):
        with pytest.raises(CaseError, match="ожидалось"):
            _parse_sweep(spec)
    key, vals = _parse_sweep("p=6:10:2")
    assert key == "p" and vals == [6, 8, 10]


def test_ladder_survives_numeric_failure(tmp_path, monkeypatch):
    """plate-ladder не обрушивается на ЧИСЛЕННОМ сбое одного случая (S21).

    Раньше перехватывался только CaseError: любое иное исключение (LinAlgError,
    переполнение ряда эталона) уносило весь прогон без сводки.
    """
    from plate_solver import cli
    from plate_solver.dispatch import solve as real_solve

    src = _ROOT / "cases" / "ci" / "circle_clamped.toml"
    for name in ("aaa_boom.toml", "bbb_ok.toml"):
        (tmp_path / name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    calls = {"n": 0}

    def flaky(problem, *a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:                       # первый случай — численный сбой
            raise np.linalg.LinAlgError("матрица вырождена (искусственно)")
        return real_solve(problem, *a, **kw)

    monkeypatch.setattr(cli, "Problem", cli.Problem)
    monkeypatch.setattr("plate_solver.dispatch.solve", flaky)
    out = tmp_path / "summary.md"
    code = cli.main_ladder([str(tmp_path), "--out", str(out)])
    assert code == 1                              # общий вердикт — FAIL
    text = out.read_text(encoding="utf-8")
    assert "сбой расчёта" in text and "LinAlgError" in text
    assert "bbb_ok.toml" in text                  # второй случай ВСЁ РАВНО посчитан


# --------------------------------------------------------------------------- #
#  Числовые ключи: nan/inf и знак нагрузки в контакте (аудит S17, D15)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("section,key,value", [
    ("model", "h", float("inf")), ("model", "h", float("nan")),
    ("load", "q0", float("inf")), ("model", "E", float("nan")),
    ("contact", "tol", float("nan")),
])
def test_nonfinite_numbers_rejected(section, key, value):
    """nan/inf в числовых ключах ⇒ отказ (раньше — тихий мусор или сбой LAPACK)."""
    d = _case()
    d.setdefault(section, {})[key] = value
    if section == "contact":
        d["contact"]["enabled"] = True
        d["contact"]["gap_factor"] = 0.5
    with pytest.raises(CaseError, match="КОНЕЧНОЕ"):
        Problem.from_dict(d)


@pytest.mark.parametrize("q0", [0.0, -4.0])
def test_nonpositive_load_with_contact_rejected(q0):
    """q0 ≤ 0 при контакте ⇒ отказ: основание недостижимо, метрики не определены."""
    d = _case(load={"type": "uniform", "q0": q0},
              contact={"enabled": True, "gap_factor": 0.5})
    with pytest.raises(CaseError, match="положительную нагрузку"):
        Problem.from_dict(d)


def test_negative_load_without_contact_allowed():
    """Без контакта отрицательная нагрузка законна (подъём)."""
    d = _case(load={"type": "uniform", "q0": -4.0})
    res = solve(Problem.from_dict(d))
    assert res.w_max > 0.0                        # модуль прогиба


# --------------------------------------------------------------------------- #
#  Эталон ↔ теория: подмена сертифицируемого оператора (аудит S06, S09, S10)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("model,verify", [
    ({"theory": "ktn_linear", "h": 0.1}, {"reference": "analytic", "tol": 0.05}),
    ({"theory": "karman", "h": 0.1}, {"reference": "analytic", "tol": 0.05}),
    ({"theory": "ktn_full", "h": 0.1}, {"reference": "fem", "tol": 0.05}),
    ({"theory": "ktn_linear", "h": 0.1}, {"reference": "none", "cross_1d": True}),
])
def test_kirchhoff_reference_rejected_for_refined_theories(model, verify):
    """Эталоны Кирхгофа не гейтят уточнённые/нелинейные теории.

    `analytic` / `fem` / `cross_1d` — решения КИРХГОФА; `w_max` уточнённой
    (ktn_linear) и нелинейной (karman, ktn_full) теорий отличается от них
    МОДЕЛЬНО. Прежде такая постановка принималась, и гейт срабатывал по
    модельному разрыву: ложный FAIL (уже при h = 0.06) или ложный PASS при
    малой нагрузке. Информационная сверка — `[verify] model_gap = true`.
    """
    d = _case(model=model, verify=verify)
    with pytest.raises(CaseError, match="theory ≠ classic"):
        Problem.from_dict(d)


@pytest.mark.parametrize("theory", ["karman", "ktn_linear"])
def test_mms_rejected_for_theories_without_manufactured_fields(theory):
    """MMS есть только для классики и полной КТН (fixed-N).

    Для karman/ktn_linear изготовленные поля не выведены, а прежний путь
    сертифицировал КЛАССИЧЕСКИЙ решатель, вовсе не касаясь результата
    постановки — тавтологический PASS.
    """
    d = _case(model={"theory": theory, "h": 0.1},
              verify={"reference": "mms", "tol": 1e-6})
    with pytest.raises(CaseError, match="mms"):
        Problem.from_dict(d)


def test_mms_still_allowed_for_classic_and_ktn_full():
    """Разрешённые сочетания MMS сохранены (classic и ktn_full)."""
    for theory in ("classic", "ktn_full"):
        d = _case(geometry={"kind": "rectangle", "x1": -1.0, "x2": 1.0,
                            "y1": -0.6, "y2": 0.6},
                  model={"theory": theory, "h": 0.1},
                  verify={"reference": "mms", "tol": 1e-6})
        assert Problem.from_dict(d).verify.reference == "mms"


def test_winkler_not_gated_by_references_without_foundation():
    """Эталоны реестра не знают основания Винклера ⇒ гейт по ним запрещён."""
    d = _case(model={"h": 0.1, "winkler": 10.0},
              verify={"reference": "analytic", "tol": 0.05})
    with pytest.raises(CaseError, match="winkler"):
        Problem.from_dict(d)


# --------------------------------------------------------------------------- #
#  Силовой штамп: брекет по лицевой поверхности (аудит D03)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("h", [0.06, 0.2])
def test_force_stamp_runs_for_refined_theory(h):
    """Силовой штамп при theory = ktn_linear доводится до уровня, а не падает.

    Брекет уровня штампа строился по СРЕДИННОМУ прогибу, тогда как условие
    непроникания ставится на ЛИЦЕВОМ: при уточнённой теории верхняя граница
    могла не быть уровнем непроникания, и `brentq` падал сырым ValueError
    «f(a) and f(b) must have different signs» (аудит D03).
    """
    import tomllib

    d = tomllib.loads((_ROOT / "cases" / "ci" / "lshape_stamp_force.toml")
                      .read_text(encoding="utf-8"))
    d.pop("output", None)
    d["model"] = {"theory": "ktn_linear", "h": h}
    res = solve(Problem.from_dict(d))
    P = float(d["contact"]["force"])
    assert res.level is not None and res.force_total == pytest.approx(P, rel=2e-2)
    assert int((res.contact.r_nodes > 0).sum()) > 0
    assert np.all(res.contact.r_nodes >= 0.0)
