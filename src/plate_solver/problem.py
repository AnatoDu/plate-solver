r"""problem.py — слой постановки задачи: case-файл TOML → неизменяемый Problem.

Комплекс из «библиотеки, которой пользуются программированием» становится
комплексом, которым пользуются ПОСТАНОВКОЙ ЗАДАЧИ: пользователь описывает
геометрию, закрепление, нагрузку и требования к верификации в case-файле
(секции ``[geometry]``, ``[bc]``, ``[load]``, ``[model]``, ``[contact]``,
``[discretization]``, ``[verify]``, ``[output]``), а решатель выбирается
диспетчером (``dispatch.py``). Полная схема — ``docs/CASE_SCHEMA.md``.

Принципы (TODO_PHASE2):

* обязательны только ``geometry``, ``bc``, ``load`` — остальное с дефолтами;
* physics-дефолты живут в ОДНОМ месте — :class:`~plate_solver.config.Config`;
  ``Problem`` их не дублирует: поля со значением ``None`` означают «взять
  дефолт Config», подстановка происходит в :meth:`Problem.to_config`;
* каждая ошибка валидации — :class:`CaseError` вида
  «ключ: получено X, ожидалось Y, см. docs/CASE_SCHEMA.md#секция»;
* ограда compose-языка v0.2: операции ``union | intersect | difference``,
  примитивы ``circle | rectangle``, глубина дерева ≤ 3, узлов ≤ 7.
"""

from __future__ import annotations

import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"

# Ограда compose-языка (зафиксирована в v0.2; расширение — только вместе
# с пересмотром реестра ворот).
COMPOSE_OPS = ("union", "intersect", "difference")
COMPOSE_PRIMITIVES = ("circle", "rectangle")
COMPOSE_MAX_DEPTH = 3
COMPOSE_MAX_NODES = 7

GEOMETRY_KINDS = ("circle", "rectangle", "L", "annulus", "compose")
BC_TYPES = ("soft_hinge", "clamped")
LOAD_TYPES = ("uniform", "patch", "point")
# Лестница моделей одним ключом [model] theory (v0.5.0, ЯВНЫЕ имена — §4):
#   classic    — линейный Кирхгоф;
#   karman     — геометрически-НЕЛИНЕЙНОЕ решение Фёппля–Кармана (L(Φ, w));
#   ktn_linear — ЛИНЕЙНЫЕ поправки сдвига/обжатия постобработкой на решении
#                Кирхгофа (не нелинейная теория; прежнее поведение "ktn");
#   ktn_full   — ПОЛНАЯ нелинейная КТН: Карман + оператор (I − h_ψ²Δ)L(Φ, w)
#                + нагрузочный член −h_*²Δq_n.
THEORIES = ("classic", "karman", "ktn_linear", "ktn_full")
# Депрекация-алиас: "ktn" неоднозначно (линейные поправки vs полная нелинейная
# теория). Сохраняем поведение старых case-файлов ⇒ алиас на ktn_linear (НЕ на
# ktn_full — это тихо сменило бы результат). Удаление алиаса — v1.0.0.
THEORY_ALIASES = {"ktn": "ktn_linear"}
# Нелинейные теории (мембранная итерация Пикара/Ньютона + шаги по нагрузке):
# для них осмысленны inplane_bc и параметры итерации.
NONLINEAR_THEORIES = ("karman", "ktn_full")
# Закрепление кромки в плане (осмысленно только для нелинейных теорий, §3.3):
#   immovable — u = v = 0 на ∂Ω (кромка не втягивается, натяжение максимально);
#   movable   — N·n = 0 (кромка свободна в плане; эффект слабее, но НЕнулевой).
INPLANE_BCS = ("immovable", "movable")
KARMAN_METHODS = ("picard", "newton")
KTN_METHODS = ("picard", "newton")
REFERENCES = ("analytic", "mms", "fem", "none")
STOP_CRITERIA = ("dr", "comp")

# Минимум узлов квадратуры в зоне (нагрузки или контакта) — защита от
# «зоны без узлов»: интеграл по маске теряет смысл.
MIN_ZONE_NODES = 20


class CaseError(ValueError):
    """Человекочитаемая ошибка case-файла (что получено, что ожидалось, где читать)."""


def _fail(key: str, got, expected: str, anchor: str) -> None:
    raise CaseError(f"{key}: получено {got!r}, ожидалось {expected}, см. {_SCHEMA_DOC}#{anchor}")


def _require_keys(section: str, data: dict, allowed: set[str], anchor: str) -> None:
    """Опечатки в ключах — самая частая ошибка; ловим их явно."""
    unknown = set(data) - allowed
    if unknown:
        _fail(f"{section}.{sorted(unknown)[0]}", data[sorted(unknown)[0]],
              f"один из ключей {sorted(allowed)}", anchor)


def _number(section: str, data: dict, key: str, anchor: str, *,
            required: bool = False, positive: bool = False):
    if key not in data:
        if required:
            _fail(f"{section}.{key}", None, "число (ключ обязателен)", anchor)
        return None
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        _fail(f"{section}.{key}", v, "число", anchor)
    v = float(v)
    if positive and v <= 0:
        _fail(f"{section}.{key}", v, "положительное число", anchor)
    return v


def _integer(section: str, data: dict, key: str, anchor: str, *, minimum: int = 1):
    if key not in data:
        return None
    v = data[key]
    if isinstance(v, bool) or not isinstance(v, int):
        _fail(f"{section}.{key}", v, "целое число", anchor)
    if v < minimum:
        _fail(f"{section}.{key}", v, f"целое ≥ {minimum}", anchor)
    return int(v)


def _boolean(section: str, data: dict, key: str, anchor: str, default: bool) -> bool:
    if key not in data:
        return default
    v = data[key]
    if not isinstance(v, bool):
        _fail(f"{section}.{key}", v, "true | false", anchor)
    return v


# --------------------------------------------------------------------------- #
#  Секции case-файла (frozen-датаклассы)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GeometrySpec:
    """Геометрия области Ω (или зоны нагрузки/контакта — язык тот же)."""

    kind: str
    a: float | None = None          # circle/annulus: радиус (annulus: внешний)
    b: float | None = None          # annulus: внутренний радиус
    x1: float | None = None         # rectangle: [x1, x2] × [y1, y2]
    x2: float | None = None
    y1: float | None = None
    y2: float | None = None
    side: float | None = None       # L: сторона
    cut: float | None = None        # L: вырез (0 < cut < side)
    tree: dict | None = None        # compose: дерево операций


@dataclass(frozen=True)
class BCSpec:
    """Закрепление края: один тип на всю границу или mixed (v0.3).

    ``mixed`` (только kind=rectangle): ``sides`` — кортеж пар
    (сторона, тип), все четыре стороны x1|x2|y1|y2 со значениями
    clamped|hinge. Структура (∏ω_c²)(∏ω_h)·Φ; статика шарнира —
    из полной билинейной формы (NOTES §20).
    """

    type: str
    sides: tuple = ()


@dataclass(frozen=True)
class LoadSpec:
    """Нагрузка: равномерная, зонная (patch) или точечная (point).

    Точечная сила — регуляризованный patch: круговое пятно радиуса ``eps``,
    ``q = P / (π·eps²)``. Истинная δ-нагрузка в схему сознательно не вводится
    (обоснование — docs/NOTES.md, раздел «Точечная сила и уточнённая теория»).
    """

    type: str
    q0: float | None = None         # uniform | patch
    P: float | None = None          # point: результирующая сила
    x0: float | None = None         # point: точка приложения
    y0: float | None = None
    eps: float | None = None        # point: радиус пятна (None ⇒ 0.05·min(ширина, высота bbox))
    zone: GeometrySpec | None = None  # patch: зона нагрузки


@dataclass(frozen=True)
class ModelSpec:
    """Модель: лестница теорий ``classic | karman | ktn_linear | ktn_full`` (§4).

    ``E``, ``nu``, ``h`` со значением None берутся из дефолтов Config
    (physics-дефолты не дублируются). ``inplane_bc`` и параметры нелинейной
    итерации (``n_load_steps``, ``karman_*``) осмысленны только для НЕЛИНЕЙНЫХ
    теорий (``karman``, ``ktn_full``); при ``classic``/``ktn_linear`` их задание
    отвергается валидатором. ``karman_method`` — только ``karman``,
    ``ktn_method`` — только ``ktn_full``. None-параметры наследуют дефолты Config.
    """

    theory: str = "classic"
    E: float | None = None
    nu: float | None = None
    h: float | None = None
    inplane_bc: str = "immovable"           # нелин.: immovable | movable (§3.3)
    n_load_steps: int | None = None         # нелин.: шагов по нагрузке (§5.2)
    karman_relax: float | None = None       # нелин.: недорелаксация θ ∈ (0, 1]
    karman_max_iter: int | None = None      # нелин.: предел итераций Пикара
    karman_tol: float | None = None         # нелин.: относит. порог останова
    karman_method: str | None = None        # karman: picard | newton
    ktn_method: str | None = None           # ktn_full: picard | newton


GAP_KINDS = ("const", "plane", "paraboloid", "steps")


@dataclass(frozen=True)
class GapSpec:
    r"""Поле зазора Δ(x, y): секция ``[contact.gap]``.

    * ``const``: Δ = value (алиас прежнего скалярного ``gap``);
    * ``plane``: Δ = a·x + b·y + c (наклонное основание);
    * ``paraboloid``: Δ = apex + ((x−cx)² + (y−cy)²) / (2·r_curv)
      (неплоский штамп; r_curv — радиус кривизны в вершине);
    * ``steps``: Δ = base, в зонах ``[[contact.gap.zones]]`` — своё value
      (несколько штампов разной высоты; зоны применяются по порядку).

    Положительность Δ на основании проверяется диспетчером (зависит от Ω).
    Произвольное поле — только через API (``ContactMOR(gap=массив)``).
    """

    kind: str
    value: float | None = None          # const
    a: float | None = None              # plane
    b: float | None = None
    c: float | None = None
    r_curv: float | None = None         # paraboloid
    cx: float | None = None
    cy: float | None = None
    apex: float | None = None
    base: float | None = None           # steps
    zones: tuple = ()                   # steps: пары (GeometrySpec, value)


@dataclass(frozen=True)
class ContactSpec:
    """Односторонний контакт (МОР): жёсткое препятствие с зазором.

    Ровно одно из: ``gap`` (скаляр Δ), ``gap_factor`` (Δ = gap_factor·w_free,
    вычисляет диспетчер), таблица ``[contact.gap]`` (поле Δ(x, y),
    :class:`GapSpec`). ``zone`` — геометрия зоны препятствия (дефолт: вся Ω).

    Силовой штамп (A2): ``force = P > 0`` — уровень штампа ищется из
    ``∫r dΩ = P``; скалярные ``gap``/``gap_factor`` при этом игнорируются
    (warning в result.json), а ``[contact.gap]`` осмыслен как ФОРМА штампа
    (профиль относительно неизвестного уровня): Δ(x, y) = level + shape(x, y).
    Параметры итерации со значением None берутся из дефолтов Config.
    """

    enabled: bool = False
    target: str = "foundation"          # foundation | plate2 (A4: две пластины)
    gap: float | None = None
    gap_factor: float | None = None
    gap_field: GapSpec | None = None
    force: float | None = None          # силовой штамп (A2): ∫r dΩ = force
    beta: float | None = None
    max_iter: int | None = None
    tol: float | None = None
    stop: str | None = None
    zone: GeometrySpec | None = None


@dataclass(frozen=True)
class Plate2Spec:
    """Вторая пластина (A4, ``[plate2]``; обязательна при contact.target=plate2).

    ``bc`` и ``load`` обязательны; ``geometry``/``model``/``discretization``
    со значением None наследуются от первой пластины (дефолт — та же
    планформа и та же дискретизация).
    """

    bc: BCSpec
    load: LoadSpec
    geometry: GeometrySpec | None = None
    model: ModelSpec | None = None
    discretization: DiscretizationSpec | None = None


@dataclass(frozen=True)
class DiscretizationSpec:
    """Дискретизация: степень Чебышёва p, квадратура Q, сетка вывода grid_n."""

    p: int | None = None
    Q: int | None = None
    grid_n: int | None = None


@dataclass(frozen=True)
class VerifySpec:
    """Верификация как свойство постановки (исполняется references.py, P3)."""

    reference: str = "none"
    cross_1d: bool = False
    tol: float = 1e-2
    model_gap: bool = False


@dataclass(frozen=True)
class OutputSpec:
    """Куда складывать result.json и фигуры."""

    dir: str = "results"
    figures: bool = False


@dataclass(frozen=True)
class Problem:
    """Неизменяемая постановка задачи (провалидированный case-файл)."""

    geometry: GeometrySpec
    bc: BCSpec
    load: LoadSpec
    model: ModelSpec = field(default_factory=ModelSpec)
    contact: ContactSpec = field(default_factory=ContactSpec)
    plate2: Plate2Spec | None = None
    discretization: DiscretizationSpec = field(default_factory=DiscretizationSpec)
    verify: VerifySpec = field(default_factory=VerifySpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    source: str = "<dict>"          # путь case-файла (для сообщений и result.json)

    # -- фабрики ---------------------------------------------------------- #
    @classmethod
    def from_toml(cls, path: str | Path) -> Problem:
        """Прочитать и провалидировать case-файл TOML."""
        path = Path(path)
        try:
            with open(path, "rb") as f:
                data = tomllib.load(f)
        except FileNotFoundError:
            raise CaseError(f"case-файл не найден: {path}") from None
        except tomllib.TOMLDecodeError as e:
            raise CaseError(f"{path}: некорректный TOML: {e}") from None
        return cls.from_dict(data, source=str(path))

    @classmethod
    def from_dict(cls, data: dict, source: str = "<dict>") -> Problem:
        """Построить Problem из словаря секций (с полной валидацией)."""
        if not isinstance(data, dict):
            _fail("case", type(data).__name__, "таблица секций TOML", "схема")
        _require_keys("case", data, {"geometry", "bc", "load", "model", "contact",
                                     "plate2", "discretization", "verify", "output"},
                      "схема")
        for sec in ("geometry", "bc", "load"):
            if sec not in data:
                _fail(sec, None, f"обязательная секция [{sec}]", sec)

        geometry = _parse_geometry("geometry", data["geometry"])
        bc = _parse_bc(data["bc"])
        load = _parse_load(data["load"])
        model = _parse_model(data.get("model", {}))
        contact = _parse_contact(data.get("contact", {}))
        plate2 = _parse_plate2(data["plate2"]) if "plate2" in data else None
        disc = _parse_discretization(data.get("discretization", {}))
        verify = _parse_verify(data.get("verify", {}))
        output = _parse_output(data.get("output", {}))

        problem = cls(geometry=geometry, bc=bc, load=load, model=model, contact=contact,
                      plate2=plate2, discretization=disc, verify=verify, output=output,
                      source=source)
        _validate_cross(problem)
        return problem

    # -- мост к лабораторному Config -------------------------------------- #
    def with_discretization(self, p: int | None = None, Q: int | None = None,
                            grid_n: int | None = None) -> Problem:
        """Копия постановки с заменой параметров дискретизации.

        Удобство циклов подбора (ноутбуки) и override сетки вывода:
        незаданные аргументы наследуются от текущей постановки; значения
        проходят тот же валидатор, что и секция [discretization]
        case-файла (p ≥ 1, Q ≥ 8, grid_n ≥ 2). Сетка вывода grid_n на
        числа решения не влияет.
        """
        from dataclasses import replace as _replace

        d = self.discretization
        raw: dict = {}
        for key, cur, new in (("p", d.p, p), ("Q", d.Q, Q),
                              ("grid_n", d.grid_n, grid_n)):
            val = cur if new is None else new
            if val is not None:
                raw[key] = val
        return _replace(self, discretization=_parse_discretization(raw))

    def to_config(self) -> Config:
        """Построить Config; None-поля Problem наследуют дефолты Config.

        Ключи с иной семантикой (gap_factor, зоны, point-нагрузка)
        разрешаются диспетчером, а не здесь: например
        ``Δ = gap_factor·w_free`` требует решения задачи без контакта.
        """
        kw: dict = {}
        for attr, key in (("E", "E"), ("nu", "nu"), ("h", "h")):
            v = getattr(self.model, attr)
            if v is not None:
                kw[key] = v
        # параметры нелинейной итерации Кармана/КТН (§5.4/§5.5); None ⇒ дефолт Config
        for attr in ("n_load_steps", "karman_relax", "karman_max_iter",
                     "karman_tol", "karman_method", "ktn_method"):
            v = getattr(self.model, attr)
            if v is not None:
                kw[attr] = v
        if self.load.q0 is not None:
            kw["q0"] = self.load.q0
        if self.geometry.kind in ("circle", "annulus") and self.geometry.a is not None:
            kw["a"] = self.geometry.a
        if self.contact.gap is not None:
            kw["Delta"] = self.contact.gap
        for attr in ("beta", "max_iter", "tol", "stop"):
            v = getattr(self.contact, attr)
            if v is not None:
                kw[attr] = v
        for attr in ("p", "Q", "grid_n"):
            v = getattr(self.discretization, attr)
            if v is not None:
                kw[attr] = v
        return Config(**kw)


# --------------------------------------------------------------------------- #
#  Парсеры секций
# --------------------------------------------------------------------------- #
def _parse_geometry(section: str, data, *, allow_compose: bool = True) -> GeometrySpec:
    anchor = "geometry" if section == "geometry" else section.split(".")[0]
    if not isinstance(data, dict):
        _fail(section, data, "таблица (секция TOML)", anchor)
    kind = data.get("kind")
    if kind not in GEOMETRY_KINDS:
        _fail(f"{section}.kind", kind, " | ".join(GEOMETRY_KINDS), anchor)

    if kind == "circle":
        _require_keys(section, data, {"kind", "a"}, anchor)
        a = _number(section, data, "a", anchor, required=True, positive=True)
        return GeometrySpec(kind=kind, a=a)

    if kind == "rectangle":
        _require_keys(section, data, {"kind", "x1", "x2", "y1", "y2"}, anchor)
        vals = {k: _number(section, data, k, anchor, required=True)
                for k in ("x1", "x2", "y1", "y2")}
        if not (vals["x1"] < vals["x2"] and vals["y1"] < vals["y2"]):
            _fail(f"{section}.x1..y2", (vals["x1"], vals["x2"], vals["y1"], vals["y2"]),
                  "x1 < x2 и y1 < y2", anchor)
        return GeometrySpec(kind=kind, **vals)

    if kind == "L":
        _require_keys(section, data, {"kind", "side", "cut"}, anchor)
        side = _number(section, data, "side", anchor, required=True, positive=True)
        cut = _number(section, data, "cut", anchor, required=True, positive=True)
        if not cut < side:
            _fail(f"{section}.cut", cut, f"0 < cut < side (= {side})", anchor)
        return GeometrySpec(kind=kind, side=side, cut=cut)

    if kind == "annulus":
        _require_keys(section, data, {"kind", "a", "b"}, anchor)
        a = _number(section, data, "a", anchor, required=True, positive=True)
        b = _number(section, data, "b", anchor, required=True, positive=True)
        if not b < a:
            _fail(f"{section}.b", b, f"0 < b < a (= {a}) — внутренний радиус меньше внешнего",
                  anchor)
        return GeometrySpec(kind=kind, a=a, b=b)

    # compose
    if not allow_compose:
        _fail(f"{section}.kind", kind, " | ".join(k for k in GEOMETRY_KINDS if k != "compose"),
              anchor)
    _require_keys(section, data, {"kind", "tree"}, "compose")
    tree = data.get("tree")
    validate_compose_tree(tree, f"{section}.tree")
    return GeometrySpec(kind=kind, tree=tree)


def validate_compose_tree(tree: dict, path: str = "geometry.tree") -> int:
    """Проверить compose-дерево против ограды v0.2; вернуть число узлов.

    Публичная точка входа для geometry.make_compose — единый источник правды
    об ограде (операции, примитивы, глубина ≤ 3, ≤ 7 узлов).
    """
    if not isinstance(tree, dict):
        _fail(path, tree, "таблица-дерево операций", "compose")
    n = _validate_compose_node(path, tree, depth=1)
    if n > COMPOSE_MAX_NODES:
        _fail(path, f"{n} узлов", f"≤ {COMPOSE_MAX_NODES} узлов", "compose")
    return n


def _validate_compose_node(path: str, node: dict, depth: int) -> int:
    """Структурная проверка узла compose-дерева; возвращает число узлов поддерева.

    Глубина считается в УЗЛАХ (примитив = 1); ограда v0.2: глубина ≤ 3, ≤ 7 узлов.
    """
    if depth > COMPOSE_MAX_DEPTH:
        _fail(path, f"глубина {depth}", f"глубина дерева ≤ {COMPOSE_MAX_DEPTH}", "compose")
    if not isinstance(node, dict):
        _fail(path, node, "таблица (узел дерева)", "compose")
    if "op" in node:
        op = node["op"]
        if op not in COMPOSE_OPS:
            _fail(f"{path}.op", op, " | ".join(COMPOSE_OPS), "compose")
        children = node.get("children")
        _require_keys(path, node, {"op", "children"}, "compose")
        if not isinstance(children, list) or len(children) < 2:
            _fail(f"{path}.children", children, "массив из ≥ 2 узлов", "compose")
        if op == "difference" and len(children) != 2:
            _fail(f"{path}.children", f"{len(children)} узлов",
                  "ровно 2 узла для difference", "compose")
        return 1 + sum(_validate_compose_node(f"{path}.children[{i}]", ch, depth + 1)
                       for i, ch in enumerate(children))
    # примитив
    kind = node.get("kind")
    if kind not in COMPOSE_PRIMITIVES:
        _fail(f"{path}.kind", kind, " | ".join(COMPOSE_PRIMITIVES) + " (примитив) или op",
              "compose")
    if kind == "circle":
        _require_keys(path, node, {"kind", "a", "cx", "cy"}, "compose")
        _number(path, node, "a", "compose", required=True, positive=True)
        _number(path, node, "cx", "compose")
        _number(path, node, "cy", "compose")
    else:
        _require_keys(path, node, {"kind", "x1", "x2", "y1", "y2"}, "compose")
        vals = {k: _number(path, node, k, "compose", required=True)
                for k in ("x1", "x2", "y1", "y2")}
        if not (vals["x1"] < vals["x2"] and vals["y1"] < vals["y2"]):
            _fail(f"{path}.x1..y2", (vals["x1"], vals["x2"], vals["y1"], vals["y2"]),
                  "x1 < x2 и y1 < y2", "compose")
    return 1


def _parse_bc(data) -> BCSpec:
    if not isinstance(data, dict):
        _fail("bc", data, "таблица (секция TOML)", "bc")
    _require_keys("bc", data, {"type", "sides"}, "bc")
    t = data.get("type")
    if t not in (*BC_TYPES, "mixed"):
        _fail("bc.type", t, " | ".join((*BC_TYPES, "mixed")), "bc")
    if t != "mixed":
        if "sides" in data:
            _fail("bc.sides", data["sides"], "только при type = mixed", "bc")
        return BCSpec(type=t)
    raw = data.get("sides")
    if not isinstance(raw, list) or not raw:
        _fail("bc.sides", raw, "массив [[bc.sides]] из четырёх сторон", "bc")
    sides = {}
    for i, s in enumerate(raw):
        if not isinstance(s, dict):
            _fail(f"bc.sides[{i}]", s, "таблица side/type", "bc")
        _require_keys(f"bc.sides[{i}]", s, {"side", "type"}, "bc")
        side = s.get("side")
        st = s.get("type")
        if side not in ("x1", "x2", "y1", "y2"):
            _fail(f"bc.sides[{i}].side", side, "x1 | x2 | y1 | y2", "bc")
        if st not in ("clamped", "hinge", "free"):
            _fail(f"bc.sides[{i}].type", st, "clamped | hinge | free", "bc")
        if side in sides:
            _fail(f"bc.sides[{i}].side", side, "каждая сторона один раз", "bc")
        sides[side] = st
    if set(sides) != {"x1", "x2", "y1", "y2"}:
        _fail("bc.sides", sorted(sides), "все четыре стороны x1, x2, y1, y2", "bc")
    # Правило жёстких смещений: кинематические условия обязаны
    # уничтожать ядро {1, x, y}. Достаточно ≥ 1 clamped (линейная функция,
    # зануляющаяся на прямой ВМЕСТЕ с нормальной производной, ≡ 0) ЛИБО
    # ≥ 2 hinge (линейная функция, зануляющаяся на двух РАЗЛИЧНЫХ прямых —
    # параллельных или пересекающихся, — ≡ 0). Иначе задача изгиба
    # вырождена (свободные стороны пластину не закрепляют).
    n_clamped = sum(1 for v in sides.values() if v == "clamped")
    n_hinge = sum(1 for v in sides.values() if v == "hinge")
    if n_clamped == 0 and n_hinge < 2:
        _fail("bc.sides", [f"{k}={v}" for k, v in sorted(sides.items())],
              "набор сторон, исключающий жёсткие смещения: не менее одной "
              "clamped либо не менее двух hinge (ядро {1, x, y})", "bc")
    return BCSpec(type=t, sides=tuple(sorted(sides.items())))


def _parse_load(data) -> LoadSpec:
    if not isinstance(data, dict):
        _fail("load", data, "таблица (секция TOML)", "load")
    t = data.get("type")
    if t not in LOAD_TYPES:
        _fail("load.type", t, " | ".join(LOAD_TYPES), "load")

    if t == "uniform":
        _require_keys("load", data, {"type", "q0"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        return LoadSpec(type=t, q0=q0)

    if t == "patch":
        _require_keys("load", data, {"type", "q0", "zone"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        if "zone" not in data:
            _fail("load.zone", None, "геометрия зоны нагрузки (обязательна для patch)", "load")
        zone = _parse_geometry("load.zone", data["zone"])
        return LoadSpec(type=t, q0=q0, zone=zone)

    # point: регуляризованный patch, q = P/(π·eps²)
    _require_keys("load", data, {"type", "P", "x0", "y0", "eps"}, "load")
    P = _number("load", data, "P", "load", required=True)
    x0 = _number("load", data, "x0", "load", required=True)
    y0 = _number("load", data, "y0", "load", required=True)
    eps = _number("load", data, "eps", "load", positive=True)
    return LoadSpec(type=t, P=P, x0=x0, y0=y0, eps=eps)


def _parse_model(data) -> ModelSpec:
    if not isinstance(data, dict):
        _fail("model", data, "таблица (секция TOML)", "model")
    _require_keys("model", data,
                  {"theory", "E", "nu", "h", "inplane_bc", "n_load_steps",
                   "karman_relax", "karman_max_iter", "karman_tol",
                   "karman_method", "ktn_method"}, "model")
    raw_theory = data.get("theory", "classic")
    if raw_theory in THEORY_ALIASES:
        # Депрекация-алиас (§4): "ktn" → "ktn_linear", поведение сохранено.
        canonical = THEORY_ALIASES[raw_theory]
        warnings.warn(
            f"model.theory = '{raw_theory}' неоднозначно и переименовано; "
            f"используйте '{canonical}' (линейные поправки сдвига/обжатия) или "
            "'ktn_full' (полная нелинейная КТН). Алиас "
            f"'{raw_theory}' будет удалён в v1.0.0.",
            DeprecationWarning, stacklevel=3)
        theory = canonical
    else:
        theory = raw_theory
    if theory not in THEORIES:
        _fail("model.theory", raw_theory,
              f"{' | '.join(THEORIES)} (или устаревший алиас 'ktn')", "model")
    E = _number("model", data, "E", "model", positive=True)
    nu = _number("model", data, "nu", "model")
    if nu is not None and not (-1.0 < nu < 0.5):
        _fail("model.nu", nu, "−1 < nu < 0.5", "model")
    h = _number("model", data, "h", "model", positive=True)
    # Закрепление кромки и параметры итерации осмысленны ТОЛЬКО для нелинейных
    # теорий (karman, ktn_full, §4); при classic/ktn_linear — ошибка постановки.
    nonlinear_only = {"inplane_bc", "n_load_steps", "karman_relax",
                      "karman_max_iter", "karman_tol"}
    provided = nonlinear_only & set(data)
    if provided and theory not in NONLINEAR_THEORIES:
        key = sorted(provided)[0]
        _fail(f"model.{key}", data[key],
              "ключ осмыслен только для нелинейных теорий "
              f"({' | '.join(NONLINEAR_THEORIES)}); classic/ktn_linear — "
              "линейный изгиб без мембранной связи", "model")
    if "karman_method" in data and theory != "karman":
        _fail("model.karman_method", data["karman_method"],
              "ключ осмыслен только при theory = 'karman' "
              "(для ktn_full — ktn_method)", "model")
    if "ktn_method" in data and theory != "ktn_full":
        _fail("model.ktn_method", data["ktn_method"],
              "ключ осмыслен только при theory = 'ktn_full'", "model")
    inplane_bc = data.get("inplane_bc", "immovable")
    if inplane_bc not in INPLANE_BCS:
        _fail("model.inplane_bc", inplane_bc, " | ".join(INPLANE_BCS), "model")
    n_load_steps = _integer("model", data, "n_load_steps", "model", minimum=1)
    karman_max_iter = _integer("model", data, "karman_max_iter", "model", minimum=1)
    karman_relax = _number("model", data, "karman_relax", "model", positive=True)
    if karman_relax is not None and karman_relax > 1.0:
        _fail("model.karman_relax", karman_relax,
              "0 < θ ≤ 1 (недорелаксация)", "model")
    karman_tol = _number("model", data, "karman_tol", "model", positive=True)
    karman_method = data.get("karman_method")
    if karman_method is not None and karman_method not in KARMAN_METHODS:
        _fail("model.karman_method", karman_method, " | ".join(KARMAN_METHODS), "model")
    ktn_method = data.get("ktn_method")
    if ktn_method is not None and ktn_method not in KTN_METHODS:
        _fail("model.ktn_method", ktn_method, " | ".join(KTN_METHODS), "model")
    return ModelSpec(theory=theory, E=E, nu=nu, h=h, inplane_bc=inplane_bc,
                     n_load_steps=n_load_steps, karman_relax=karman_relax,
                     karman_max_iter=karman_max_iter, karman_tol=karman_tol,
                     karman_method=karman_method, ktn_method=ktn_method)


def _parse_gap_field(data: dict) -> GapSpec:
    """Секция ``[contact.gap]`` — поле зазора Δ(x, y)."""
    sec = "contact.gap"
    kind = data.get("kind")
    if kind not in GAP_KINDS:
        _fail(f"{sec}.kind", kind, " | ".join(GAP_KINDS), "contact")
    if kind == "const":
        _require_keys(sec, data, {"kind", "value"}, "contact")
        return GapSpec(kind=kind,
                       value=_number(sec, data, "value", "contact",
                                     required=True, positive=True))
    if kind == "plane":
        _require_keys(sec, data, {"kind", "a", "b", "c"}, "contact")
        return GapSpec(kind=kind,
                       a=_number(sec, data, "a", "contact", required=True),
                       b=_number(sec, data, "b", "contact", required=True),
                       c=_number(sec, data, "c", "contact", required=True))
    if kind == "paraboloid":
        _require_keys(sec, data, {"kind", "r_curv", "cx", "cy", "apex"}, "contact")
        apex = _number(sec, data, "apex", "contact", required=True)
        if apex < 0:
            _fail(f"{sec}.apex", apex, "число ≥ 0 (зазор в вершине штампа)", "contact")
        cx = _number(sec, data, "cx", "contact")
        cy = _number(sec, data, "cy", "contact")
        return GapSpec(kind=kind,
                       r_curv=_number(sec, data, "r_curv", "contact",
                                      required=True, positive=True),
                       cx=0.0 if cx is None else cx, cy=0.0 if cy is None else cy,
                       apex=apex)
    # steps
    _require_keys(sec, data, {"kind", "base", "zones"}, "contact")
    base = _number(sec, data, "base", "contact", required=True, positive=True)
    zones_raw = data.get("zones")
    if not isinstance(zones_raw, list) or not zones_raw:
        _fail(f"{sec}.zones", zones_raw,
              "непустой массив таблиц [[contact.gap.zones]] (геометрия + value)",
              "contact")
    zones = []
    for i, z in enumerate(zones_raw):
        path = f"{sec}.zones[{i}]"
        if not isinstance(z, dict) or "value" not in z:
            _fail(f"{path}.value", None, "число > 0 (зазор в зоне; ключ обязателен)",
                  "contact")
        z = dict(z)
        value = z.pop("value")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            _fail(f"{path}.value", value, "число > 0", "contact")
        zones.append((_parse_geometry(path, z), float(value)))
    return GapSpec(kind=kind, base=base, zones=tuple(zones))


def _parse_contact(data) -> ContactSpec:
    if not isinstance(data, dict):
        _fail("contact", data, "таблица (секция TOML)", "contact")
    _require_keys("contact", data,
                  {"enabled", "target", "gap", "gap_factor", "force", "beta",
                   "max_iter", "tol", "stop", "zone"},
                  "contact")
    enabled = _boolean("contact", data, "enabled", "contact", default=False)
    target = data.get("target", "foundation")
    if target not in ("foundation", "plate2"):
        _fail("contact.target", target, "foundation | plate2", "contact")
    gap_raw = data.get("gap")
    gap = None
    gap_field = None
    if isinstance(gap_raw, dict):                     # [contact.gap] — поле Δ(x, y)
        gap_field = _parse_gap_field(gap_raw)
    else:
        gap = _number("contact", data, "gap", "contact")
    gap_factor = _number("contact", data, "gap_factor", "contact", positive=True)
    beta = _number("contact", data, "beta", "contact", positive=True)
    max_iter = _integer("contact", data, "max_iter", "contact", minimum=1)
    tol = _number("contact", data, "tol", "contact", positive=True)
    stop = data.get("stop")
    if stop is not None and stop not in STOP_CRITERIA:
        _fail("contact.stop", stop, " | ".join(STOP_CRITERIA), "contact")
    zone = _parse_geometry("contact.zone", data["zone"]) if "zone" in data else None
    force = _number("contact", data, "force", "contact", positive=True)
    if enabled and force is None:
        provided = sum(v is not None for v in (gap, gap_factor, gap_field))
        if provided != 1:
            _fail("contact.gap",
                  {"gap": gap, "gap_factor": gap_factor,
                   "[contact.gap]": None if gap_field is None else gap_field.kind},
                  "ровно одно из gap | gap_factor | таблица [contact.gap] "
                  "при enabled = true (либо силовой режим force)", "contact")
    return ContactSpec(enabled=enabled, target=target, gap=gap,
                       gap_factor=gap_factor,
                       gap_field=gap_field, force=force, beta=beta,
                       max_iter=max_iter, tol=tol, stop=stop, zone=zone)


def _parse_plate2(data) -> Plate2Spec:
    """Секция ``[plate2]`` (A4): bc и load обязательны, прочее — от первой."""
    if not isinstance(data, dict):
        _fail("plate2", data, "таблица (секция TOML)", "plate2")
    _require_keys("plate2", data, {"bc", "load", "geometry", "model",
                                   "discretization"}, "plate2")
    for key in ("bc", "load"):
        if key not in data:
            _fail(f"plate2.{key}", None, f"обязательная подсекция [plate2.{key}]",
                  "plate2")
    return Plate2Spec(
        bc=_parse_bc(data["bc"]),
        load=_parse_load(data["load"]),
        geometry=(_parse_geometry("plate2.geometry", data["geometry"])
                  if "geometry" in data else None),
        model=_parse_model(data["model"]) if "model" in data else None,
        discretization=(_parse_discretization(data["discretization"])
                        if "discretization" in data else None),
    )


def _parse_discretization(data) -> DiscretizationSpec:
    if not isinstance(data, dict):
        _fail("discretization", data, "таблица (секция TOML)", "discretization")
    _require_keys("discretization", data, {"p", "Q", "grid_n"}, "discretization")
    return DiscretizationSpec(
        p=_integer("discretization", data, "p", "discretization", minimum=1),
        Q=_integer("discretization", data, "Q", "discretization", minimum=2),
        grid_n=_integer("discretization", data, "grid_n", "discretization", minimum=2),
    )


def _parse_verify(data) -> VerifySpec:
    if not isinstance(data, dict):
        _fail("verify", data, "таблица (секция TOML)", "verify")
    _require_keys("verify", data, {"reference", "cross_1d", "tol", "model_gap"}, "verify")
    reference = data.get("reference", "none")
    if reference not in REFERENCES:
        _fail("verify.reference", reference, " | ".join(REFERENCES), "verify")
    tol = _number("verify", data, "tol", "verify", positive=True)
    return VerifySpec(
        reference=reference,
        cross_1d=_boolean("verify", data, "cross_1d", "verify", default=False),
        tol=1e-2 if tol is None else tol,
        model_gap=_boolean("verify", data, "model_gap", "verify", default=False),
    )


def _parse_output(data) -> OutputSpec:
    if not isinstance(data, dict):
        _fail("output", data, "таблица (секция TOML)", "output")
    _require_keys("output", data, {"dir", "figures"}, "output")
    d = data.get("dir", "results")
    if not isinstance(d, str) or not d:
        _fail("output.dir", d, "непустая строка (каталог)", "output")
    return OutputSpec(dir=d, figures=_boolean("output", data, "figures", "output", default=False))


# --------------------------------------------------------------------------- #
#  Перекрёстная валидация (несовместимости v0.2)
# --------------------------------------------------------------------------- #
def _validate_cross(p: Problem) -> None:
    if p.bc.type == "mixed":
        if p.geometry.kind != "rectangle":
            _fail("bc.type", "mixed",
                  "kind = rectangle (смешанные КУ на произвольных R-областях — "
                  "направление развития)", "bc")
        if p.contact.enabled:
            _fail("contact.enabled", True,
                  "false при bc.type = mixed (контакт при смешанных КУ — "
                  "направление развития)", "bc")
        if p.model.theory == "ktn_linear":
            _fail("model.theory", "ktn_linear", "classic при bc.type = mixed (v0.3)", "bc")
    if p.model.theory in NONLINEAR_THEORIES:
        # Рамки нелинейных теорий (§1): канонические области (круг,
        # прямоугольник/квадрат), изгибные КУ clamped|soft_hinge, БЕЗ контакта.
        # Нелинейный контакт (МОР поверх Кармана/КТН) — задел v0.6.0.
        th = p.model.theory
        if p.geometry.kind not in ("circle", "rectangle"):
            _fail("model.theory", th,
                  "geometry.kind = circle | rectangle (нелинейные теории — на "
                  "канонических областях; L-форма/кольцо/compose — направление "
                  "развития)", "model")
        if p.bc.type not in ("clamped", "soft_hinge"):
            _fail("bc.type", p.bc.type,
                  f"clamped | soft_hinge при theory = {th} (мембранная связь на "
                  "смешанных КУ — направление развития)", "bc")
        if p.contact.enabled:
            _fail("contact.enabled", True,
                  f"false при theory = {th} (нелинейный контакт — МОР поверх "
                  "Кармана/КТН — направление развития v0.6.0)", "model")
    c = p.contact
    if c.target == "plate2" or p.plate2 is not None:
        if not (c.enabled and c.target == "plate2" and p.plate2 is not None):
            _fail("contact.target", c.target,
                  "plate2 вместе с секцией [plate2] (и contact.enabled=true)",
                  "plate2")
        if c.force is not None:
            _fail("contact.force", c.force,
                  "отсутствие force — силовое управление парой пластин "
                  "отложено (направление развития)", "plate2")
        theories = {p.model.theory,
                    p.plate2.model.theory if p.plate2.model is not None else "classic"}
        if theories != {"classic"}:
            _fail("model.theory", sorted(theories),
                  "classic — контактное условие пары в v0.3 классическое "
                  "(срединные плоскости; геометрически контактируют нижняя "
                  "лицевая верхней и верхняя лицевая нижней пластины), КТН "
                  "для пары — направление развития", "plate2")
        if c.gap is not None and c.gap < 0:
            _fail("contact.gap", c.gap, "число ≥ 0 (Δ=0 — касание пластин)",
                  "plate2")
    elif c.enabled and c.gap is not None and c.gap <= 0:
        _fail("contact.gap", c.gap, "число > 0 (жёсткое основание)", "contact")
    if p.verify.reference == "analytic" and p.geometry.kind == "compose":
        _fail("verify.reference", "analytic",
              "mms | fem | none — для compose-геометрии аналитического эталона нет", "verify")
    axisymmetric = p.geometry.kind in ("circle", "annulus") and p.load.type == "uniform"
    if p.verify.cross_1d and not axisymmetric:
        _fail("verify.cross_1d", True,
              "false — сверка с 1D-Ритцем по радиусу доступна только для "
              "осесимметричных постановок (circle | annulus, равномерная нагрузка)", "verify")


__all__ = [
    "CaseError",
    "Problem",
    "GeometrySpec",
    "BCSpec",
    "LoadSpec",
    "ModelSpec",
    "ContactSpec",
    "DiscretizationSpec",
    "VerifySpec",
    "OutputSpec",
    "Plate2Spec",
    "GEOMETRY_KINDS",
    "GAP_KINDS",
    "GapSpec",
    "MIN_ZONE_NODES",
    "validate_compose_tree",
]
