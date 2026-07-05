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
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"

# Ограда compose-языка v0.2 (не расширять в этой фазе — TODO_PHASE2, п. 2).
COMPOSE_OPS = ("union", "intersect", "difference")
COMPOSE_PRIMITIVES = ("circle", "rectangle")
COMPOSE_MAX_DEPTH = 3
COMPOSE_MAX_NODES = 7

GEOMETRY_KINDS = ("circle", "rectangle", "L", "annulus", "compose")
BC_TYPES = ("soft_hinge", "clamped")
LOAD_TYPES = ("uniform", "patch", "point")
THEORIES = ("classic", "ktn")
REFERENCES = ("analytic", "mms", "fem", "none")
STOP_CRITERIA = ("dr", "comp")

# Минимум узлов квадратуры в зоне (нагрузки или контакта) — защита от
# «зоны без узлов»: интеграл по маске теряет смысл (P0.1/P0.2).
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
    tree: dict | None = None        # compose: дерево операций (P1.2)


@dataclass(frozen=True)
class BCSpec:
    """Закрепление края (в v0.2 — один тип на всю границу)."""

    type: str


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
    """Модель: классика (Кирхгоф, расщепление) или поправки КТН.

    ``E``, ``nu``, ``h`` со значением None берутся из дефолтов Config
    (physics-дефолты не дублируются).
    """

    theory: str = "classic"
    E: float | None = None
    nu: float | None = None
    h: float | None = None


GAP_KINDS = ("const", "plane", "paraboloid", "steps")


@dataclass(frozen=True)
class GapSpec:
    r"""Поле зазора Δ(x, y) (фаза 3, A1): секция ``[contact.gap]``.

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
    Параметры итерации со значением None берутся из дефолтов Config.
    """

    enabled: bool = False
    gap: float | None = None
    gap_factor: float | None = None
    gap_field: GapSpec | None = None
    beta: float | None = None
    max_iter: int | None = None
    tol: float | None = None
    stop: str | None = None
    zone: GeometrySpec | None = None


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
                                     "discretization", "verify", "output"}, "схема")
        for sec in ("geometry", "bc", "load"):
            if sec not in data:
                _fail(sec, None, f"обязательная секция [{sec}]", sec)

        geometry = _parse_geometry("geometry", data["geometry"])
        bc = _parse_bc(data["bc"])
        load = _parse_load(data["load"])
        model = _parse_model(data.get("model", {}))
        contact = _parse_contact(data.get("contact", {}))
        disc = _parse_discretization(data.get("discretization", {}))
        verify = _parse_verify(data.get("verify", {}))
        output = _parse_output(data.get("output", {}))

        problem = cls(geometry=geometry, bc=bc, load=load, model=model, contact=contact,
                      discretization=disc, verify=verify, output=output, source=source)
        _validate_cross(problem)
        return problem

    # -- мост к лабораторному Config -------------------------------------- #
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
    _require_keys("bc", data, {"type"}, "bc")
    t = data.get("type")
    if t not in BC_TYPES:
        _fail("bc.type", t, " | ".join(BC_TYPES), "bc")
    return BCSpec(type=t)


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
    _require_keys("model", data, {"theory", "E", "nu", "h"}, "model")
    theory = data.get("theory", "classic")
    if theory not in THEORIES:
        _fail("model.theory", theory, " | ".join(THEORIES), "model")
    E = _number("model", data, "E", "model", positive=True)
    nu = _number("model", data, "nu", "model")
    if nu is not None and not (-1.0 < nu < 0.5):
        _fail("model.nu", nu, "−1 < nu < 0.5", "model")
    h = _number("model", data, "h", "model", positive=True)
    return ModelSpec(theory=theory, E=E, nu=nu, h=h)


def _parse_gap_field(data: dict) -> GapSpec:
    """Секция ``[contact.gap]`` — поле зазора Δ(x, y) (фаза 3, A1.2)."""
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
                  {"enabled", "gap", "gap_factor", "beta", "max_iter", "tol", "stop", "zone"},
                  "contact")
    enabled = _boolean("contact", data, "enabled", "contact", default=False)
    gap_raw = data.get("gap")
    gap = None
    gap_field = None
    if isinstance(gap_raw, dict):                     # [contact.gap] — поле Δ(x, y)
        gap_field = _parse_gap_field(gap_raw)
    else:
        gap = _number("contact", data, "gap", "contact", positive=True)
    gap_factor = _number("contact", data, "gap_factor", "contact", positive=True)
    beta = _number("contact", data, "beta", "contact", positive=True)
    max_iter = _integer("contact", data, "max_iter", "contact", minimum=1)
    tol = _number("contact", data, "tol", "contact", positive=True)
    stop = data.get("stop")
    if stop is not None and stop not in STOP_CRITERIA:
        _fail("contact.stop", stop, " | ".join(STOP_CRITERIA), "contact")
    zone = _parse_geometry("contact.zone", data["zone"]) if "zone" in data else None
    if enabled:
        provided = sum(v is not None for v in (gap, gap_factor, gap_field))
        if provided != 1:
            _fail("contact.gap",
                  {"gap": gap, "gap_factor": gap_factor,
                   "[contact.gap]": None if gap_field is None else gap_field.kind},
                  "ровно одно из gap | gap_factor | таблица [contact.gap] "
                  "при enabled = true", "contact")
    return ContactSpec(enabled=enabled, gap=gap, gap_factor=gap_factor,
                       gap_field=gap_field, beta=beta,
                       max_iter=max_iter, tol=tol, stop=stop, zone=zone)


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
    if p.contact.enabled and p.bc.type == "clamped":
        _fail("bc.type", "clamped", "soft_hinge — в v0.2 контакт реализован "
              "для мягкого шарнира (снятие ограничения — фаза 3)", "contact")
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
    "GEOMETRY_KINDS",
    "GAP_KINDS",
    "GapSpec",
    "MIN_ZONE_NODES",
    "validate_compose_tree",
]
