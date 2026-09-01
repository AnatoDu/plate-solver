r"""problem.py — слой постановки задачи: case-файл TOML → неизменяемый Problem.

Комплекс из «библиотеки, которой пользуются программированием» становится
комплексом, которым пользуются ПОСТАНОВКОЙ ЗАДАЧИ: пользователь описывает
геометрию, закрепление, нагрузку и требования к верификации в case-файле
(секции ``[geometry]``, ``[bc]``, ``[load]``, ``[model]``, ``[contact]``,
``[discretization]``, ``[verify]``, ``[output]``), а решатель выбирается
диспетчером (``dispatch.py``). Полная схема — ``docs/CASE_SCHEMA.md``.

Принципы:

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

import math
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

from . import exprfield
from .config import Config

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"

# Ограда compose-языка (зафиксирована в v0.2; расширение — только вместе
# с пересмотром реестра ворот).
COMPOSE_OPS = ("union", "intersect", "difference")
COMPOSE_PRIMITIVES = ("circle", "rectangle")
COMPOSE_MAX_DEPTH = 3
COMPOSE_MAX_NODES = 7

GEOMETRY_KINDS = ("circle", "rectangle", "L", "annulus", "ellipse", "compose")
BC_TYPES = ("soft_hinge", "clamped")
LOAD_TYPES = ("uniform", "patch", "point", "gaussian", "expr", "line")
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
# Нелинейный контакт МОР+КТН (contact_nl.py, v0.6.3): схема композиции двух
# итераций (§4.2) и нормировка усиления оператора (теорема 4, §4.1).
CONTACT_SCHEMES = ("nested", "merged")
CONTACT_GAINS = ("secant", "linear")
# Собственные задачи (eigenmodes.py, v0.6.4): устойчивость | колебания.
EIGEN_KINDS = ("buckling", "vibration")

# Минимум узлов квадратуры в зоне (нагрузки или контакта) — защита от
# «зоны без узлов»: интеграл по маске теряет смысл.
MIN_ZONE_NODES = 20


class CaseError(ValueError):
    """Человекочитаемая ошибка case-файла (что получено, что ожидалось, где читать)."""


def _fail(key: str, got, expected: str, anchor: str) -> None:
    raise CaseError(f"{key}: получено {got!r}, ожидалось {expected}, см. {_SCHEMA_DOC}#{anchor}")


def _finite(key: str, value: float, anchor: str) -> float:
    """Конечность числа нового ключа (TOML допускает литералы inf/nan)."""
    if not math.isfinite(value):
        _fail(key, value, "конечное число (inf/nan не допускаются)", anchor)
    return value


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
    """Нагрузка: равномерная, зонная (patch), точечная (point), гауссова, выражением.

    Точечная сила — регуляризованный patch: круговое пятно радиуса ``eps``,
    ``q = P / (π·eps²)``. Истинная δ-нагрузка в схему сознательно не вводится
    (обоснование — docs/NOTES.md, раздел «Точечная сила и уточнённая теория»).
    ``gaussian`` — гладкая локализованная нагрузка
    ``q = q0·exp(−r²/(2σ²))`` (центр ``x0, y0``, ширина ``sigma``): у неё Δq
    аналитична, поэтому проявляется член КТН ``−h_*²Δq`` (§7) — под НЕравномерной
    нагрузкой уже и СРЕДИННЫЙ прогиб КТН отличается от классики.
    ``expr`` (v0.7.0) — произвольная гладкая нагрузка ``q = q0·g(x, y)``
    выражением (``exprfield``: sympy за токен-оградой, белый список имён);
    для гладкого ``g`` член КТН ``Δq`` берётся символьным дифференцированием.
    """

    type: str
    q0: float | None = None         # uniform | patch | gaussian | expr (амплитуда)
    P: float | None = None          # point: результирующая сила
    x0: float | None = None         # point/gaussian: центр
    y0: float | None = None
    eps: float | None = None        # point: радиус пятна (None ⇒ 0.05·min(ширина, высота bbox))
    sigma: float | None = None      # gaussian: ширина (СКО)
    zone: GeometrySpec | None = None  # patch: зона нагрузки
    expr: str | None = None         # expr: безразмерная форма g(x, y)
    p0: tuple | None = None         # line: начало отрезка (x, y)
    p1: tuple | None = None         # line: конец отрезка (x, y)
    intensity: float | None = None  # line: погонная интенсивность P (сила/длину)
    thermal_moment: float | None = None  # uniform: термомомент M_T (v0.7.0)
    exact: bool = False             # point: точная δ вместо регуляризации (v0.7.0)


@dataclass(frozen=True)
class OrthotropySpec:
    r"""Ортотропия классической теории (v0.7.0): подсекция ``[model.orthotropy]``.

    РОВНО один из двух наборов: инженерный (``Ex, Ey, nu_xy, Gxy``; конвенция
    ``ν_yx = ν_xy·Ey/Ex``, эллиптичность ``ν_xy²·Ey/Ex < 1``) либо прямой
    (``D11, D12, D22, D66``; положительная определённость
    ``D11 > 0, D66 > 0, D11·D22 > D12²``). Конверсия в жёсткости — в
    ``Problem.to_config`` (нужна толщина ``h``):
    ``k = 1 − ν_xy·ν_yx``; ``D11 = Ex·h³/(12k)``, ``D22 = Ey·h³/(12k)``,
    ``D12 = ν_xy·Ey·h³/(12k)``, ``D66 = Gxy·h³/12``.
    """

    Ex: float | None = None
    Ey: float | None = None
    nu_xy: float | None = None
    Gxy: float | None = None
    D11: float | None = None
    D12: float | None = None
    D22: float | None = None
    D66: float | None = None


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
    winkler: float | None = None            # упругое основание Винклера k_w ≥ 0 (v0.6.6)
    orthotropy: OrthotropySpec | None = None  # ортотропия классики (v0.7.0)
    h_expr: str | None = None               # переменная толщина h(x, y) (v0.7.0)


GAP_KINDS = ("const", "plane", "paraboloid", "steps")


@dataclass(frozen=True)
class GapSpec:
    r"""Поле зазора Δ(x, y): секция ``[contact.gap]``.

    * ``const``: Δ = value (алиас прежнего скалярного ``gap``);
    * ``plane``: Δ = a·x + b·y + c (наклонное основание);
    * ``paraboloid``: Δ = apex + ((x−cx)² + (y−cy)²) / (2·r_curv)
      (неплоский штамп; r_curv — радиус кривизны в вершине);
    * ``steps``: Δ = base, в зонах ``[[contact.gap.zones]]`` — своё value
      (несколько штампов разной высоты; зоны применяются по порядку);
    * ``expr`` (v0.7.0): Δ = f(x, y) выражением — конструируется ТОЛЬКО из
      строкового ключа ``[contact] gap_expr`` (не из таблицы ``[contact.gap]``);
      константное выражение редуцируется в скаляр — путь скалярного ``gap``
      бит-точно.

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
    expr: str | None = None             # expr: Δ = f(x, y) (ключ gap_expr)


@dataclass(frozen=True)
class SupportsSpec:
    r"""Точечные упругие опоры (v0.7.0): секция ``[supports]``.

    Энергия пружин ``Π_s = (k/2)·Σ_j w(P_j)²`` — ранг-1 добавки
    ``k·ψ(P_j)ψ(P_j)ᵀ`` к изгибной жёсткости ДО факторизации (паттерн
    Винклера). Жёсткая опора — штраф ``k ≈ 1e6·D/a³`` (ошибка штрафа ~5e-5,
    точный закон ``R(k) = w_free(P)/(G_N(P,P) + 1/k)``). Сходимость решения
    по базису с опорой АЛГЕБРАИЧЕСКАЯ ~p⁻² (функция Грина ~r²ln r), не
    спектральная. Реакции ``R_j = k·w(P_j)`` — в result.json.
    """

    points: tuple = ()        # пары (x, y)
    stiffness: float = 0.0    # жёсткость k каждой пружины (> 0)


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
    # нелинейный контакт МОР+КТН (theory = karman | ktn_full, v0.6.3):
    scheme: str | None = None           # nested | merged (§4.2); None ⇒ дефолт Config
    gain: str | None = None             # secant | linear (§4.1); None ⇒ дефолт Config
    mor_anderson: int | None = None     # окно проекц. Андерсона внешнего цикла (v0.6.5)


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
    """Куда складывать result.json, фигуры и VTK-экспорт."""

    dir: str = "results"
    figures: bool = False
    vtk: bool = False                       # + result.vtk для ParaView (v0.6.6)


@dataclass(frozen=True)
class EigenSpec:
    """Собственная задача (``[eigen]``, v0.6.4): устойчивость или колебания.

    ``kind = buckling`` — потеря устойчивости под равномерным мембранным
    усилием-эталоном ``(Nx, Ny, Nxy)`` (сжатие — отрицательный знак); критические
    множители ``λ_cr``. ``kind = vibration`` — свободные колебания; частоты ``ω``,
    ``rho_h`` — погонная масса ``ρh``. Анализ ЛИНЕЙНЫЙ (изгибная жёсткость
    Кирхгофа), поэтому ``[load]``/``[contact]`` не нужны и запрещены.
    """

    kind: str = "vibration"
    n_modes: int = 6
    Nx: float = -1.0                # buckling: эталонное усилие (сжатие < 0)
    Ny: float = 0.0
    Nxy: float = 0.0
    rho_h: float = 1.0              # vibration: погонная масса ρh
    # ПРЕДНАПРЯЖЁННАЯ собственная задача (v0.6.6): поле N(w) из кармановского
    # решения под [load] (theory = karman, равномерная нагрузка); анализ
    # остаётся линейным вокруг напряжённого состояния: K_eff = K + K_geo(N(w))
    prestress: bool = False


@dataclass(frozen=True)
class Problem:
    """Неизменяемая постановка задачи (провалидированный case-файл)."""

    geometry: GeometrySpec
    bc: BCSpec
    load: LoadSpec
    model: ModelSpec = field(default_factory=ModelSpec)
    contact: ContactSpec = field(default_factory=ContactSpec)
    supports: SupportsSpec = field(default_factory=SupportsSpec)  # v0.7.0
    plate2: Plate2Spec | None = None
    discretization: DiscretizationSpec = field(default_factory=DiscretizationSpec)
    verify: VerifySpec = field(default_factory=VerifySpec)
    output: OutputSpec = field(default_factory=OutputSpec)
    eigen: EigenSpec | None = None   # собственная задача (§eigen, v0.6.4)
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
                                     "supports", "plate2", "discretization",
                                     "verify", "output", "eigen"},
                      "схема")
        eigen_prestress = (isinstance(data.get("eigen"), dict)
                           and bool(data["eigen"].get("prestress", False)))
        if "eigen" in data and any(s in data for s in ("contact", "plate2")):
            _fail("eigen", "present",
                  "без секций [contact]/[plate2] — собственная задача без "
                  "контакта", "eigen")
        if "eigen" in data and "load" in data and not eigen_prestress:
            _fail("eigen", "present",
                  "без секции [load] — собственная задача линейная; нагрузка "
                  "осмысленна только при prestress = true (преднапряжение "
                  "кармановским полем N(w), v0.6.6)", "eigen")
        # [load] обязателен: для обычных постановок И для преднапряжённой
        # собственной задачи (prestress = true — источник поля N(w)).
        req = (("geometry", "bc", "load") if ("eigen" not in data or eigen_prestress)
               else ("geometry", "bc"))
        for sec in req:
            if sec not in data:
                _fail(sec, None, f"обязательная секция [{sec}]", sec)

        geometry = _parse_geometry("geometry", data["geometry"])
        bc = _parse_bc(data["bc"])
        load = (_parse_load(data["load"]) if "load" in data
                else LoadSpec(type="uniform", q0=0.0))
        model = _parse_model(data.get("model", {}))
        contact = _parse_contact(data.get("contact", {}))
        supports = (_parse_supports(data["supports"]) if "supports" in data
                    else SupportsSpec())
        plate2 = _parse_plate2(data["plate2"]) if "plate2" in data else None
        disc = _parse_discretization(data.get("discretization", {}))
        verify = _parse_verify(data.get("verify", {}))
        output = _parse_output(data.get("output", {}))
        eigen = _parse_eigen(data["eigen"]) if "eigen" in data else None

        problem = cls(geometry=geometry, bc=bc, load=load, model=model, contact=contact,
                      supports=supports, plate2=plate2, discretization=disc,
                      verify=verify, output=output, eigen=eigen, source=source)
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
                     "karman_tol", "karman_method", "ktn_method", "winkler"):
            v = getattr(self.model, attr)
            if v is not None:
                kw[attr] = v
        if self.load.q0 is not None:
            kw["q0"] = self.load.q0
        if self.load.thermal_moment is not None:       # термоизгиб (v0.7.0)
            kw["thermal_moment"] = self.load.thermal_moment
        if self.supports.points:                       # точечные опоры (v0.7.0)
            kw["supports_points"] = self.supports.points
            kw["supports_stiffness"] = self.supports.stiffness
        if self.model.h_expr is not None:              # переменная толщина (v0.7.0)
            kw["h_expr"] = self.model.h_expr
        o = self.model.orthotropy                      # ортотропия (v0.7.0)
        if o is not None:
            if o.D11 is not None:                      # прямой набор жёсткостей
                kw["ortho_D"] = (o.D11, o.D12, o.D22, o.D66)
            else:                                      # инженерные константы
                h_val = self.model.h if self.model.h is not None else Config().h
                k_el = 1.0 - o.nu_xy**2 * o.Ey / o.Ex  # 1 − ν_xy·ν_yx
                h3 = h_val**3 / 12.0
                kw["ortho_D"] = (o.Ex * h3 / k_el, o.nu_xy * o.Ey * h3 / k_el,
                                 o.Ey * h3 / k_el, o.Gxy * h3)
                if self.model.theory == "karman":
                    # мембранный закон N = A·ε (нужен нелинейной связке)
                    kw["ortho_A"] = (o.Ex * h_val / k_el,
                                     o.nu_xy * o.Ey * h_val / k_el,
                                     o.Ey * h_val / k_el, o.Gxy * h_val)
        if self.geometry.kind in ("circle", "annulus") and self.geometry.a is not None:
            kw["a"] = self.geometry.a
        if self.contact.gap is not None:
            kw["Delta"] = self.contact.gap
        for attr in ("beta", "max_iter", "tol", "stop"):
            v = getattr(self.contact, attr)
            if v is not None:
                kw[attr] = v
        # нелинейный контакт (§4): схема композиции и нормировка усиления
        if self.contact.scheme is not None:
            kw["contact_scheme"] = self.contact.scheme
        if self.contact.gain is not None:
            kw["contact_gain"] = self.contact.gain
        if self.contact.mor_anderson is not None:
            kw["mor_anderson"] = self.contact.mor_anderson
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

    if kind == "ellipse":
        # Эллипс с полуосями a (по x), b (по y); центр (0,0). Полуоси
        # независимы (a = b ⇒ круг); нет аналитического эталона (reference:
        # none | fem). Верхнеуровневый вид — НЕ примитив compose (ограда).
        _require_keys(section, data, {"kind", "a", "b"}, anchor)
        a = _number(section, data, "a", anchor, required=True, positive=True)
        b = _number(section, data, "b", anchor, required=True, positive=True)
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
        _require_keys("load", data, {"type", "q0", "thermal_moment"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        # термомомент M_T (v0.7.0): аддитивен к равномерной q (q0 = 0 — чистый
        # термоизгиб); у других типов нагрузки ключ не принимается
        mt = _number("load", data, "thermal_moment", "load")
        if mt is not None:
            _finite("load.thermal_moment", mt, "load")
        return LoadSpec(type=t, q0=q0, thermal_moment=mt)

    if t == "patch":
        _require_keys("load", data, {"type", "q0", "zone"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        if "zone" not in data:
            _fail("load.zone", None, "геометрия зоны нагрузки (обязательна для patch)", "load")
        zone = _parse_geometry("load.zone", data["zone"])
        return LoadSpec(type=t, q0=q0, zone=zone)

    if t == "gaussian":
        # гладкая локализованная: q = q0·exp(−r²/(2σ²)); Δq аналитична (§7)
        _require_keys("load", data, {"type", "q0", "x0", "y0", "sigma"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        x0 = _number("load", data, "x0", "load", required=True)
        y0 = _number("load", data, "y0", "load", required=True)
        sigma = _number("load", data, "sigma", "load", required=True, positive=True)
        return LoadSpec(type=t, q0=q0, x0=x0, y0=y0, sigma=sigma)

    if t == "expr":
        # нагрузка выражением: q = q0·g(x, y) (v0.7.0); синтаксис и белый
        # список проверяются ЗДЕСЬ (fail-fast при чтении case), значения на
        # узлах — диспетчером (нужна квадратура области)
        _require_keys("load", data, {"type", "q0", "expr"}, "load")
        q0 = _number("load", data, "q0", "load", required=True)
        s = data.get("expr")
        try:
            e = exprfield.parse_field("load.expr", s)
            if not e.free_symbols:                  # константа: значение сразу
                exprfield.field_values(e, 0.0, 0.0, key="load.expr")
        except ValueError as err:
            raise CaseError(f"{err}, см. {_SCHEMA_DOC}#load") from None
        return LoadSpec(type=t, q0=q0, expr=s)

    if t == "line":
        # погонная нагрузка вдоль отрезка (v0.7.0): b_i = P·∫_seg ψ_i ds
        _require_keys("load", data, {"type", "p0", "p1", "intensity"}, "load")
        pts = {}
        for key in ("p0", "p1"):
            raw = data.get(key)
            ok = (isinstance(raw, (list, tuple)) and len(raw) == 2
                  and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                          for v in raw))
            if not ok:
                _fail(f"load.{key}", raw, "пара чисел [x, y] (конец отрезка)",
                      "load")
            pts[key] = (_finite(f"load.{key}[0]", float(raw[0]), "load"),
                        _finite(f"load.{key}[1]", float(raw[1]), "load"))
        if pts["p0"] == pts["p1"]:
            _fail("load.p1", data.get("p1"),
                  "отрезок ненулевой длины (для точечной силы — load.type = "
                  "point)", "load")
        intensity = _number("load", data, "intensity", "load", required=True)
        _finite("load.intensity", intensity, "load")
        if intensity == 0.0:
            _fail("load.intensity", intensity, "число ≠ 0 (погонная сила/длину)",
                  "load")
        return LoadSpec(type=t, p0=pts["p0"], p1=pts["p1"], intensity=intensity)

    # point: регуляризованный patch q = P/(π·eps²) ЛИБО точная δ (exact=true,
    # v0.7.0: b_i = P·ψ_i(x0, y0) — функционал ограничен на H² в 2D; только
    # тракты с прямой сборкой — см. _validate_cross и NOTES §18)
    _require_keys("load", data, {"type", "P", "x0", "y0", "eps", "exact"}, "load")
    P = _number("load", data, "P", "load", required=True)
    x0 = _number("load", data, "x0", "load", required=True)
    y0 = _number("load", data, "y0", "load", required=True)
    eps = _number("load", data, "eps", "load", positive=True)
    exact = _boolean("load", data, "exact", "load", default=False)
    if exact and eps is not None:
        _fail("load.eps", eps,
              "отсутствие eps при exact = true (точная δ не регуляризуется)",
              "load")
    return LoadSpec(type=t, P=P, x0=x0, y0=y0, eps=eps, exact=exact)


def _parse_model(data) -> ModelSpec:
    if not isinstance(data, dict):
        _fail("model", data, "таблица (секция TOML)", "model")
    _require_keys("model", data,
                  {"theory", "E", "nu", "h", "inplane_bc", "n_load_steps",
                   "karman_relax", "karman_max_iter", "karman_tol",
                   "karman_method", "ktn_method", "winkler", "orthotropy",
                   "h_expr"},
                  "model")
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
    winkler = _number("model", data, "winkler", "model")
    if winkler is not None and winkler < 0.0:
        _fail("model.winkler", winkler, "число ≥ 0 (жёсткость основания Винклера)",
              "model")
    orthotropy = (_parse_orthotropy(data["orthotropy"])
                  if "orthotropy" in data else None)
    if orthotropy is not None and (E is not None or nu is not None):
        _fail("model.E", E if E is not None else nu,
              "отсутствие E и nu рядом с [model.orthotropy] — два источника "
              "жёсткости неоднозначны (h задавать можно и нужно)", "model")
    h_expr = data.get("h_expr")                     # переменная толщина (v0.7.0)
    if h_expr is not None:
        if not isinstance(h_expr, str):
            _fail("model.h_expr", h_expr, "строка-выражение h(x, y)", "model")
        try:
            exprfield.parse_field("model.h_expr", h_expr)
        except ValueError as err:
            raise CaseError(f"{err}, см. {_SCHEMA_DOC}#model") from None
        if h is not None:
            _fail("model.h", h,
                  "отсутствие h рядом с h_expr (толщина либо скаляром, либо "
                  "полем — двусмысленность не допускается)", "model")
    return ModelSpec(theory=theory, E=E, nu=nu, h=h, inplane_bc=inplane_bc,
                     n_load_steps=n_load_steps, karman_relax=karman_relax,
                     karman_max_iter=karman_max_iter, karman_tol=karman_tol,
                     karman_method=karman_method, ktn_method=ktn_method,
                     winkler=winkler, orthotropy=orthotropy, h_expr=h_expr)


def _parse_orthotropy(data) -> OrthotropySpec:
    """Подсекция ``[model.orthotropy]`` (v0.7.0): инженерный ЛИБО прямой набор."""
    sec = "model.orthotropy"
    if not isinstance(data, dict):
        _fail(sec, data, "таблица (подсекция TOML)", "model")
    _require_keys(sec, data, {"Ex", "Ey", "nu_xy", "Gxy",
                              "D11", "D12", "D22", "D66"}, "model")
    eng = {k: data.get(k) for k in ("Ex", "Ey", "nu_xy", "Gxy")}
    direct = {k: data.get(k) for k in ("D11", "D12", "D22", "D66")}
    has_eng = any(v is not None for v in eng.values())
    has_direct = any(v is not None for v in direct.values())
    if has_eng == has_direct:                       # оба или ни одного
        _fail(sec, sorted(k for k, v in {**eng, **direct}.items()
                          if v is not None) or None,
              "РОВНО один набор: инженерный (Ex, Ey, nu_xy, Gxy) либо прямой "
              "(D11, D12, D22, D66)", "model")
    if has_eng:
        Ex = _finite(f"{sec}.Ex", _number(sec, data, "Ex", "model",
                                          required=True, positive=True), "model")
        Ey = _finite(f"{sec}.Ey", _number(sec, data, "Ey", "model",
                                          required=True, positive=True), "model")
        Gxy = _finite(f"{sec}.Gxy", _number(sec, data, "Gxy", "model",
                                            required=True, positive=True), "model")
        nu_xy = _finite(f"{sec}.nu_xy", _number(sec, data, "nu_xy", "model",
                                                required=True), "model")
        if nu_xy < 0.0 or nu_xy**2 * Ey / Ex >= 1.0:
            _fail(f"{sec}.nu_xy", nu_xy,
                  "0 ≤ nu_xy и nu_xy²·Ey/Ex < 1 (эллиптичность энергии)",
                  "model")
        return OrthotropySpec(Ex=Ex, Ey=Ey, nu_xy=nu_xy, Gxy=Gxy)
    D11 = _finite(f"{sec}.D11", _number(sec, data, "D11", "model",
                                        required=True, positive=True), "model")
    D22 = _finite(f"{sec}.D22", _number(sec, data, "D22", "model",
                                        required=True, positive=True), "model")
    D66 = _finite(f"{sec}.D66", _number(sec, data, "D66", "model",
                                        required=True, positive=True), "model")
    D12 = _finite(f"{sec}.D12", _number(sec, data, "D12", "model",
                                        required=True), "model")
    if D11 * D22 <= D12**2:
        _fail(f"{sec}.D12", D12,
              "D11·D22 > D12² (положительная определённость энергии — иначе "
              "Ритц теряет смысл)", "model")
    return OrthotropySpec(D11=D11, D12=D12, D22=D22, D66=D66)


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
                  {"enabled", "target", "gap", "gap_factor", "gap_expr", "force",
                   "beta", "max_iter", "tol", "stop", "zone", "scheme", "gain",
                   "mor_anderson"},
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
    gap_expr = data.get("gap_expr")                   # Δ = f(x, y) выражением (v0.7.0)
    if gap_expr is not None:
        if not isinstance(gap_expr, str):
            _fail("contact.gap_expr", gap_expr, "строка-выражение Δ = f(x, y)",
                  "contact")
        try:
            e = exprfield.parse_field("contact.gap_expr", gap_expr)
            if not e.free_symbols:                    # константа: значение сразу
                exprfield.field_values(e, 0.0, 0.0, key="contact.gap_expr")
        except ValueError as err:
            raise CaseError(f"{err}, см. {_SCHEMA_DOC}#contact") from None
        if gap_field is not None:
            _fail("contact.gap_expr", gap_expr,
                  "ровно одно из gap_expr | таблица [contact.gap]", "contact")
        gap_field = GapSpec(kind="expr", expr=gap_expr)
    gap_factor = _number("contact", data, "gap_factor", "contact", positive=True)
    beta = _number("contact", data, "beta", "contact", positive=True)
    max_iter = _integer("contact", data, "max_iter", "contact", minimum=1)
    tol = _number("contact", data, "tol", "contact", positive=True)
    stop = data.get("stop")
    if stop is not None and stop not in STOP_CRITERIA:
        _fail("contact.stop", stop, " | ".join(STOP_CRITERIA), "contact")
    scheme = data.get("scheme")
    if scheme is not None and scheme not in CONTACT_SCHEMES:
        _fail("contact.scheme", scheme, " | ".join(CONTACT_SCHEMES), "contact")
    gain = data.get("gain")
    if gain is not None and gain not in CONTACT_GAINS:
        _fail("contact.gain", gain, " | ".join(CONTACT_GAINS), "contact")
    mor_anderson = _integer("contact", data, "mor_anderson", "contact", minimum=0)
    if mor_anderson is not None and mor_anderson > 20:
        # большие окна разваливают проекционный Андерсон (вырожденный МНК по
        # почти коллинеарной истории ⇒ дикая экстраполяция; аудит v0.6.5:
        # окно 50 РАСХОДИТСЯ там, где 0–20 сходятся). Рабочий диапазон 3–8.
        _fail("contact.mor_anderson", mor_anderson,
              "целое 0..20 (окно памяти проекционного Андерсона; оптимум 3–8, "
              "большие окна вырождают МНК и разваливают итерацию)", "contact")
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
                       max_iter=max_iter, tol=tol, stop=stop, zone=zone,
                       scheme=scheme, gain=gain, mor_anderson=mor_anderson)


_MAX_SUPPORTS = 32


def _parse_supports(data) -> SupportsSpec:
    """Секция ``[supports]`` (v0.7.0): points (пары [x, y]) + stiffness."""
    if not isinstance(data, dict):
        _fail("supports", data, "таблица (секция TOML)", "supports")
    _require_keys("supports", data, {"points", "stiffness"}, "supports")
    raw = data.get("points")
    if not isinstance(raw, list) or not raw:
        _fail("supports.points", raw, "непустой массив пар [[x, y], ...]",
              "supports")
    if len(raw) > _MAX_SUPPORTS:
        _fail("supports.points", f"{len(raw)} точек",
              f"≤ {_MAX_SUPPORTS} (близкие опоры вырождают обусловленность)",
              "supports")
    pts = []
    for i, pair in enumerate(raw):
        ok = (isinstance(pair, (list, tuple)) and len(pair) == 2
              and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                      for v in pair))
        if not ok:
            _fail(f"supports.points[{i}]", pair, "пара чисел [x, y]", "supports")
        pt = (_finite(f"supports.points[{i}].x", float(pair[0]), "supports"),
              _finite(f"supports.points[{i}].y", float(pair[1]), "supports"))
        if pt in pts:
            _fail(f"supports.points[{i}]", pair,
                  "уникальная точка (дубликат тихо удваивает жёсткость)",
                  "supports")
        pts.append(pt)
    k = _number("supports", data, "stiffness", "supports", required=True,
                positive=True)
    _finite("supports.stiffness", float(k), "supports")
    return SupportsSpec(points=tuple(pts), stiffness=float(k))


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
    _require_keys("output", data, {"dir", "figures", "vtk"}, "output")
    d = data.get("dir", "results")
    if not isinstance(d, str) or not d:
        _fail("output.dir", d, "непустая строка (каталог)", "output")
    return OutputSpec(dir=d,
                      figures=_boolean("output", data, "figures", "output", default=False),
                      vtk=_boolean("output", data, "vtk", "output", default=False))


def _parse_eigen(data) -> EigenSpec:
    if not isinstance(data, dict):
        _fail("eigen", data, "таблица (секция TOML)", "eigen")
    _require_keys("eigen", data, {"kind", "n_modes", "Nx", "Ny", "Nxy", "rho_h",
                                  "prestress"}, "eigen")
    kind = data.get("kind", "vibration")
    if kind not in EIGEN_KINDS:
        _fail("eigen.kind", kind, " | ".join(EIGEN_KINDS), "eigen")
    n_modes = _integer("eigen", data, "n_modes", "eigen", minimum=1)
    rho_h = _number("eigen", data, "rho_h", "eigen", positive=True)
    nx = _number("eigen", data, "Nx", "eigen")
    ny = _number("eigen", data, "Ny", "eigen")
    nxy = _number("eigen", data, "Nxy", "eigen")
    prestress = _boolean("eigen", data, "prestress", "eigen", default=False)
    if prestress and (nx is not None or ny is not None or nxy is not None):
        _fail("eigen.Nx", data.get("Nx", data.get("Ny", data.get("Nxy"))),
              "отсутствие Nx/Ny/Nxy при prestress = true (поле усилий берётся "
              "из кармановского решения под [load], а не задаётся вручную)",
              "eigen")
    return EigenSpec(
        kind=kind,
        n_modes=6 if n_modes is None else n_modes,
        Nx=-1.0 if nx is None else nx,
        Ny=0.0 if ny is None else ny,
        Nxy=0.0 if nxy is None else nxy,
        rho_h=1.0 if rho_h is None else rho_h,
        prestress=prestress)


# --------------------------------------------------------------------------- #
#  Перекрёстная валидация (несовместимости v0.2)
# --------------------------------------------------------------------------- #
def _validate_cross(p: Problem) -> None:
    if p.eigen is not None:
        # Собственная задача (устойчивость/колебания) — ЛИНЕЙНАЯ, изгиб Кирхгофа;
        # структура ω^m ⇒ КУ clamped | soft_hinge; поперечная нагрузка/контакт нет.
        if p.bc.type not in ("clamped", "soft_hinge"):
            _fail("bc.type", p.bc.type,
                  "clamped | soft_hinge для собственной задачи [eigen] "
                  "(структура ω^m; смешанные КУ — направление развития)", "eigen")
        if p.verify.reference != "none":
            _fail("verify.reference", p.verify.reference,
                  "none для [eigen] (верификация классическими эталонами — "
                  "tests/test_eigenmodes.py)", "eigen")
        if p.load.thermal_moment is not None and p.load.thermal_moment != 0.0:
            _fail("load.thermal_moment", p.load.thermal_moment,
                  "отсутствие при [eigen] (термо-преднапряжение требует "
                  "мембранной силы N_T, которая не моделируется — термо-"
                  "выпучивания нет)", "load")
        if p.model.h_expr is not None:
            _fail("model.h_expr", p.model.h_expr,
                  "отсутствие при [eigen] (собственные задачи с D(x, y) не "
                  "верифицированы — направление развития)", "model")
        if p.model.orthotropy is not None:
            # ортотропная собственная задача (v0.7.0): K = S_ortho, эталоны
            # ω_mn/N_cr Лехницкого — theory=classic гарантирует ветка ниже;
            # преднапряжение (кармановское N(w)) и опоры с ортотропией
            # не верифицированы
            if p.eigen.prestress:
                _fail("eigen.prestress", True,
                      "false при [model.orthotropy] (преднапряжение требует "
                      "ортотропной мембранной жёсткости — направление "
                      "развития)", "model")
            # опоры и Винклер с ортотропией в собственных задачах РАЗРЕШЕНЫ
            # (v0.7.0): вклады конститутив-независимы; ворота — сдвиг частот
            # ω² = (π⁴D_mn + k_w)/ρh и монотонность Куранта–Фишера
        if p.eigen.prestress and p.supports.points:
            # опоры в собственных задачах поддержаны (K + k·ψψᵀ, ворота
            # монотонности), но их сочетание с преднапряжением N(w) не
            # верифицировано — честный отказ
            _fail("eigen.prestress", True,
                  "false при [supports] (преднапряжение с опорами не "
                  "верифицировано; обычные vibration | buckling с опорами — "
                  "поддержаны)", "supports")
        if p.eigen.prestress:
            # преднапряжённая собственная задача (v0.6.6): источник поля N(w) —
            # кармановское решение под [load]; сам анализ линеен вокруг него
            if p.model.theory != "karman":
                _fail("model.theory", p.model.theory,
                      "karman при [eigen] prestress = true (поле N(w) — из "
                      "нелинейного кармановского решения под [load])", "eigen")
            if p.load.type != "uniform":
                _fail("load.type", p.load.type,
                      "uniform при [eigen] prestress = true (преднапряжение — "
                      "под равномерной нагрузкой)", "eigen")
        elif p.model.theory != "classic":
            # раньше нелинейная теория при [eigen] принималась и МОЛЧА
            # игнорировалась (анализ всегда линейный Кирхгоф) — честный отказ
            # (аудит v0.6.6); преднапряжение N(w) — prestress = true
            _fail("model.theory", p.model.theory,
                  "classic при [eigen] (собственная задача линейна — Кирхгоф; "
                  "преднапряжение полем N(w) — [eigen] prestress = true + "
                  "[load] + theory = karman, v0.6.6)", "eigen")
        return
    if p.model.h_expr is not None:
        # Переменная толщина (v0.7.0): D(x,y) весом ПОЛНОЙ билинейной формы;
        # честный объём — classic clamped и karman clamped immovable.
        if p.model.theory not in ("classic", "karman"):
            _fail("model.theory", p.model.theory,
                  "classic | karman при model.h_expr (лицевые параметры КТН "
                  "h_*², h_ψ² становятся полями — направление развития)",
                  "model")
        if p.bc.type != "clamped":
            _fail("bc.type", p.bc.type,
                  "clamped при model.h_expr (расщепление шарнира классики "
                  "неверно при D ≠ const: Δ(DΔw) ≠ DΔ²w; шарнир/смешанные КУ "
                  "с D(x, y) — направление развития)", "model")
        if p.contact.enabled or p.plate2 is not None:
            _fail("contact.enabled", True,
                  "false при model.h_expr (МОР поверх переменного D не "
                  "сертифицирован — нормировка усиления, теорема 4)", "model")
        if p.model.theory == "karman" and p.model.inplane_bc == "movable":
            _fail("model.inplane_bc", "movable",
                  "immovable при model.h_expr (ворота v0.7.0 покрывают только "
                  "immovable)", "model")
        if p.model.orthotropy is not None:
            _fail("model.orthotropy", "present",
                  "отсутствие при model.h_expr (комбинация двух конститутивных "
                  "полей не верифицирована)", "model")
        if p.load.thermal_moment is not None and p.load.thermal_moment != 0.0:
            _fail("load.thermal_moment", p.load.thermal_moment,
                  "отсутствие при model.h_expr (термомомент физически ~h² — "
                  "комбинация не верифицирована)", "load")
        if p.supports.points:
            _fail("supports.points", p.supports.points,
                  "отсутствие [supports] при model.h_expr (комбинация не "
                  "верифицирована)", "supports")
    mt = p.load.thermal_moment
    if mt is not None and mt != 0.0:
        # Термоизгиб (v0.7.0): D·Δ²w = q − ΔM_T; честный объём — classic
        # (clamped: w не меняется, сдвиг моментов; soft_hinge: расщепление)
        # и karman (истинный шарнир из полной формы; сфера на круге машинно).
        if p.model.theory in ("ktn_linear", "ktn_full"):
            _fail("model.theory", p.model.theory,
                  "classic | karman при load.thermal_moment (взаимодействие "
                  "термомомента с КТН-членами не выведено)", "load")
        if p.bc.type == "mixed":
            _fail("bc.type", "mixed",
                  "clamped | soft_hinge при load.thermal_moment (термо-член "
                  "на сторонах смешанной структуры — направление развития)",
                  "load")
        if p.contact.enabled or p.plate2 is not None:
            _fail("contact.enabled", True,
                  "false при load.thermal_moment (термо в цикле МОР не "
                  "верифицировано)", "load")
        if (p.model.winkler is not None and p.model.winkler > 0.0
                and p.bc.type == "soft_hinge"):
            _fail("model.winkler", p.model.winkler,
                  "0 при load.thermal_moment и bc = soft_hinge (комбинация "
                  "основание+термо+шарнир не верифицирована)", "load")
        if p.model.orthotropy is not None:
            _fail("model.orthotropy", "present",
                  "отсутствие при load.thermal_moment (ортотропный термо-член "
                  "M_T·(β_x, β_y) анизотропен — направление развития)", "load")
        if p.supports.points:
            _fail("supports.points", p.supports.points,
                  "отсутствие [supports] при load.thermal_moment (комбинация "
                  "не верифицирована)", "load")
    if p.model.orthotropy is not None:
        # Ортотропия (v0.7.0): полная квадратичная форма D_ij; объём —
        # classic (clamped | mixed clamped/hinge; eigen) и karman
        # (инженерный набор: мембранный закон N = A·ε; редукционная лестница).
        if p.model.theory not in ("classic", "karman"):
            _fail("model.theory", p.model.theory,
                  "classic | karman при [model.orthotropy] (поправкам КТН "
                  "нужен изотропный 3D-закон — направление развития)", "model")
        if p.model.theory == "karman":
            if p.model.orthotropy.Ex is None:
                _fail("model.orthotropy", "набор D11..D66",
                      "инженерный набор Ex, Ey, nu_xy, Gxy при theory = karman "
                      "(мембранная жёсткость A = f(Ex, Ey, ν, G, h) из прямых "
                      "D-жёсткостей неопределима)", "model")
            if p.model.inplane_bc == "movable":
                _fail("model.inplane_bc", "movable",
                      "immovable при [model.orthotropy] + karman (ворота "
                      "v0.7.0 покрывают только неподвижную кромку)", "model")
            if p.bc.type == "mixed":
                _fail("bc.type", "mixed",
                      "clamped | soft_hinge при [model.orthotropy] + karman "
                      "(смешанные КУ ортотропного Кармана — направление "
                      "развития)", "bc")
        if p.model.theory == "classic" and p.bc.type == "soft_hinge":
            _fail("bc.type", p.bc.type,
                  "clamped | mixed (hinge на сторонах) при [model.orthotropy]: "
                  "расщепление шарнира классики предполагает изотропный Δ² — "
                  "ортотропный оператор не квадрат Лапласиана (истинный "
                  "шарнир — theory = karman)", "model")
        if p.bc.type == "mixed":
            free_sides = [s for s, t in p.bc.sides if t == "free"]
            if free_sides:
                _fail(f"bc.sides[{free_sides[0]}]", "free",
                      "clamped | hinge при [model.orthotropy] (естественные "
                      "условия свободного края с D-матрицей не верифицированы)",
                      "model")
        if p.contact.enabled or p.plate2 is not None:
            _fail("contact.enabled", True,
                  "false при [model.orthotropy] (контактные эталоны изотропны; "
                  "МОР поверх ортотропного оператора — направление развития)",
                  "model")
        # Винклер и точечные опоры С ортотропией РАЗРЕШЕНЫ (v0.7.0): оба
        # вклада (+k_w∫ψψ и k·ψψᵀ) конститутив-независимы и складываются с
        # S_ortho по построению; сертификаты — MMS c k_w и машинное тождество
        # Шермана–Моррисона на ортотропном операторе (tests/test_orthotropy.py).
        if p.verify.reference != "none":
            _fail("verify.reference", p.verify.reference,
                  "none при [model.orthotropy] (реестр эталонов изотропен; "
                  "ворота — tests/test_orthotropy.py)", "verify")
    if p.verify.reference != "none":
        # Реестр эталонов ([verify] analytic|mms|fem) НЕ знает новых физических
        # ключей v0.7.0 — честный решатель получил бы ЛОЖНЫЙ FAIL ворот.
        infected = []
        if p.load.type == "expr":
            infected.append("load.type = expr")
        if p.load.thermal_moment is not None and p.load.thermal_moment != 0.0:
            infected.append("load.thermal_moment")
        if p.supports.points:
            infected.append("[supports]")
        if p.model.h_expr is not None:
            infected.append("model.h_expr")
        if p.contact.gap_field is not None and p.contact.gap_field.kind == "expr":
            infected.append("contact.gap_expr")
        if infected:
            _fail("verify.reference", p.verify.reference,
                  f"none при {', '.join(infected)} (реестр эталонов не "
                  "учитывает эти ключи — сравнение дало бы ложный вердикт; "
                  "ворота новых фич — tests/test_*.py v0.7.0)", "verify")
    if p.plate2 is not None and p.plate2.load.type == "line":
        # тракт пары собирает нагрузку второй пластины по площадной
        # квадратуре — line для неё не собирается
        _fail("plate2.load.type", "line",
              "uniform | patch | point | gaussian | expr для второй пластины "
              "(линейная нагрузка пары — направление развития)", "plate2")
    if p.load.type == "point" and p.load.exact:
        # Точная δ-сила (v0.7.0): b_i = P·ψ_i(P) — ограниченный функционал на
        # H² (2D), но ТОЛЬКО в трактах с прямой сборкой: расщепление классики
        # даёт (P1) с δ вне H⁻¹ (M ~ ln r), а КТН-прогиб под δ логарифмически
        # расходится — NOTES §18.
        if p.model.theory in ("ktn_linear", "ktn_full"):
            _fail("model.theory", p.model.theory,
                  "classic | karman при load.exact = true (КТН-прогиб под δ "
                  "логарифмически расходится — NOTES «Точечная сила и "
                  "уточнённая теория»; используйте регуляризованный point)",
                  "load")
        if p.model.theory == "classic" and p.bc.type == "soft_hinge":
            _fail("bc.type", p.bc.type,
                  "clamped при theory = classic с load.exact = true "
                  "(расщепление шарнира: (P1) с δ — функционал вне H¹; "
                  "истинный шарнир под δ — theory = karman)", "load")
        if p.bc.type == "mixed":
            _fail("bc.type", "mixed",
                  "clamped | soft_hinge при load.exact = true (вектор нагрузки "
                  "смешанного тракта — направление развития)", "load")
        if p.contact.enabled or p.plate2 is not None:
            _fail("contact.enabled", True,
                  "false при load.exact = true (контактные тракты собирают "
                  "нагрузку по площадной квадратуре)", "load")
        if p.verify.reference != "none":
            _fail("verify.reference", p.verify.reference,
                  "none при load.exact = true (эталоны реестра — для "
                  "регуляризованного пятна; ворота точной δ — "
                  "tests/test_point_exact.py)", "verify")
    if p.plate2 is not None and p.plate2.load.type == "point" and p.plate2.load.exact:
        _fail("plate2.load.exact", True,
              "false для второй пластины (точная δ пары — направление "
              "развития)", "plate2")
    if p.load.type == "line":
        # Линейная (погонная) нагрузка (v0.7.0): линейный функционал
        # b_i = P·∫_seg ψ_i ds — только тракты с готовым вектором нагрузки.
        if p.model.theory in ("ktn_linear", "ktn_full"):
            _fail("model.theory", p.model.theory,
                  "classic | karman при load.type = line (Δq линии сингулярен, "
                  "поправка ktn_linear требует скалярной амплитуды q0)", "load")
        if p.contact.enabled or p.plate2 is not None:
            _fail("contact.enabled", True,
                  "false при load.type = line (контактные тракты собирают "
                  "нагрузку по площадной квадратуре — направление развития)",
                  "load")
        if p.bc.type == "mixed":
            _fail("bc.type", "mixed",
                  "clamped | soft_hinge при load.type = line (вектор нагрузки "
                  "смешанного тракта — направление развития)", "bc")
        if p.verify.reference != "none":
            _fail("verify.reference", p.verify.reference,
                  "none при load.type = line (эталонов линии в реестре нет; "
                  "ворота — tests/test_line_load.py)", "verify")
    if p.supports.points:
        # Точечные опоры (v0.7.0): ранг-1 в ЕДИНУЮ матрицу изгибной жёсткости.
        if p.model.theory in ("ktn_linear", "ktn_full"):
            # реакция опоры — δ-сила; КТН-поправки требуют гладкой поверхностной
            # нагрузки (δ в схему сознательно не вводится — NOTES §18)
            _fail("model.theory", p.model.theory,
                  "classic | karman при [supports] (реакция опоры — δ-сила; "
                  "КТН-поправки требуют гладкой нагрузки, NOTES «Точечная сила "
                  "и уточнённая теория»)", "supports")
        if p.model.theory == "classic" and p.bc.type == "soft_hinge":
            # классический шарнир — РАСЩЕПЛЕНИЕ на две Пуассоны: единой матрицы
            # бигармоники нет, ранг-1 вставить некуда (прецедент Винклера)
            _fail("bc.type", p.bc.type,
                  "clamped | mixed при theory = classic с [supports] (мягкий "
                  "шарнир классики — расщепление; шарнир с опорами — "
                  "bc = mixed на прямоугольнике либо theory = karman)",
                  "supports")
        if p.contact.enabled or p.plate2 is not None:
            # v0.7.0: опоры входят в S ДО факторизации ⇒ контактный оператор
            # остаётся SPD и теорема 4 МОР применима как есть (gain считается
            # на ОПЁРТОМ операторе). Верифицирован классический позиционный
            # контакт об основание на защемлении (редукции + KKT,
            # tests/test_supports.py); остальные тракты — отказ.
            ok = (p.model.theory == "classic" and p.bc.type == "clamped"
                  and p.contact.target == "foundation"
                  and p.contact.force is None and p.plate2 is None)
            if not ok:
                _fail("contact.enabled", True,
                      "[supports] с контактом — только theory = classic, "
                      "bc = clamped, позиционное основание (нелинейный/"
                      "силовой/парный контакт с опорами — направление "
                      "развития)", "supports")
        # prestress×supports проверяется в ветке [eigen] выше (ранний return)
    if (p.model.winkler is not None and p.model.winkler > 0.0
            and p.verify.reference == "mms"):
        # MMS-эталон (ladder.mms_load_and_exact) НЕ учитывает основание k_w —
        # честный решатель получил бы ЛОЖНЫЙ FAIL ворот
        _fail("verify.reference", "mms",
              "none при model.winkler > 0 (MMS-эталон не учитывает основание "
              "Винклера — k_w вошёл бы в невязку; либо k_w = 0)", "verify")
    if (p.model.winkler is not None and p.model.winkler > 0.0
            and p.model.theory in ("classic", "ktn_linear")
            and p.bc.type == "soft_hinge"):
        # Винклер свёрнут в ПОЛНУЮ билинейную форму; классический мягкий шарнир
        # решается РАСЩЕПЛЕНИЕМ (две Пуассоны) — основание расщепление ломает.
        # clamped и mixed (v0.7.0: свёртка добавлена в MixedRectPlate,
        # сертификаты — MMS и ряд Навье с k_w) идут полной формой; полная
        # форма шарнира доступна нелинейным теориям (karman в линейном пределе).
        _fail("model.winkler", p.model.winkler,
              "0 либо bc = clamped | mixed при theory = classic | ktn_linear "
              "(мягкий шарнир классики — расщепление, несовместимое с "
              "основанием; используйте theory = karman: его линейный предел — "
              "полная форма шарнира)", "model")
    if p.bc.type == "mixed":
        if p.geometry.kind != "rectangle":
            _fail("bc.type", "mixed",
                  "kind = rectangle (смешанные КУ на произвольных R-областях — "
                  "направление развития)", "bc")
        if p.contact.enabled:
            _fail("contact.enabled", True,
                  "false при bc.type = mixed (контакт при смешанных КУ — "
                  "направление развития)", "bc")
        if p.model.theory in ("ktn_linear", "ktn_full"):
            _fail("model.theory", p.model.theory,
                  "classic или karman при bc.type = mixed (смешанные КУ: линейный "
                  "Кирхгоф v0.3 либо геом.-нелинейный Карман v0.6.5; ktn_full со "
                  "смешанными КУ — граничный член §3.5 на шарнирных сторонах, v0.7+)",
                  "bc")
    if p.model.theory in NONLINEAR_THEORIES:
        # Рамки нелинейных теорий: любые области, включая неканонические и
        # МНОГОСВЯЗНЫЕ (L, кольцо, compose — R-операции над ω, §5). Изгибные КУ
        # clamped | soft_hinge. Полная КТН на мягком шарнире (граничный член §3.5)
        # требует звёздной квадратуры ∂Ω ⇒ только circle | ellipse.
        th = p.model.theory
        # Гладкие/выпуклые границы (без ВХОДЯЩИХ углов) — для них квадратура ∂Ω
        # граничного члена §3.5 (мягкий шарнир полной КТН) точна: circle/ellipse/
        # rectangle (звёздные) и annulus (многосвязная, контурная квадратура).
        # L/compose с вырезом — реентрантный угол рвёт точность (v0.7).
        star = p.geometry.kind in ("circle", "ellipse", "rectangle", "annulus")
        # Карман допускает и СМЕШАННЫЕ КУ (только rectangle — проверено выше);
        # полная КТН — пока clamped | soft_hinge (§3.5 на шарнирных сторонах — v0.7+).
        allowed_bc = (("clamped", "soft_hinge", "mixed") if th == "karman"
                      else ("clamped", "soft_hinge"))
        if p.bc.type not in allowed_bc:
            _fail("bc.type", p.bc.type,
                  f"{' | '.join(allowed_bc)} при theory = {th} (мембранная связь на "
                  "смешанных КУ — Карман поддержан, ktn_full — v0.7+)", "bc")
        if th == "karman" and p.bc.type == "mixed":
            # СВОБОДНЫЕ стороны для нелинейного Кармана не верифицированы
            # (глубоко-нелинейный режим free+movable не сходится Пикаром —
            # аудит v0.6.5); поддержаны clamped | hinge (сертификат Леви).
            free_sides = [s for s, t in p.bc.sides if t == "free"]
            if free_sides:
                _fail(f"bc.sides[{free_sides[0]}]", "free",
                      "clamped | hinge при theory = karman (free-стороны "
                      "нелинейной теории — направление развития; для free "
                      "используйте theory = classic)", "bc")
        if th == "ktn_full" and p.bc.type == "soft_hinge" and not star:
            _fail("geometry.kind", p.geometry.kind,
                  "circle | ellipse | rectangle | annulus при theory = ktn_full и "
                  "bc = soft_hinge (граничный член §3.5 — квадратура ∂Ω; область с "
                  "входящим углом (L/compose) — направление развития v0.7)", "geometry")
        if p.contact.enabled:
            # Нелинейный контакт МОР+КТН (contact_nl.py): позиционный штамп/
            # основание. Защемление — любая R-область; мягкий шарнир — только
            # circle | ellipse (звёздная квадратура ∂Ω для граничного члена §3.5).
            if p.bc.type == "soft_hinge" and not star:
                _fail("bc.type", p.bc.type,
                      f"clamped при theory = {th} с контактом на НЕзвёздной области "
                      "(мягкий шарнир — circle | ellipse | rectangle | annulus, "
                      "одиночная ИЛИ пара; область с входящим углом L/compose — "
                      "направление развития v0.7)", "bc")
            # v0.7.0: гладкое поле нагрузки (gaussian/expr) допущено в
            # ПОЗИЦИОННОМ контакте об основание (МОР — свойство оператора,
            # ворота — редукции и nested==merged); силовой режим и пара —
            # по-прежнему равномерная нагрузка.
            positional = (p.contact.target == "foundation"
                          and p.contact.force is None and p.plate2 is None)
            allowed_loads = (("uniform", "gaussian", "expr") if positional
                             else ("uniform",))
            if p.load.type not in allowed_loads:
                _fail("load.type", p.load.type,
                      f"{' | '.join(allowed_loads)} при theory = {th} с контактом "
                      "(гладкое поле нагрузки — только позиционное основание, "
                      "v0.7.0; силовой/парный контакт — uniform, §4)", "load")
    # Ключи схемы/усиления осмысленны ТОЛЬКО для нелинейного контакта (§4).
    _nl_keys = {"scheme": p.contact.scheme, "gain": p.contact.gain,
                "mor_anderson": p.contact.mor_anderson}
    _nl_given = {k: v for k, v in _nl_keys.items() if v is not None}
    if _nl_given and not (p.contact.enabled and p.model.theory in NONLINEAR_THEORIES):
        k0 = sorted(_nl_given)[0]                    # анкер — реально заданный ключ
        _fail(f"contact.{k0}", _nl_given[k0],
              "задан только при contact.enabled = true и theory = karman | "
              "ktn_full (схема композиции / нормировка усиления / ускорение "
              "Андерсона нелинейного контакта МОР+КТН, §4.1–4.2)", "contact")
    c = p.contact
    if c.target == "plate2" or p.plate2 is not None:
        if not (c.enabled and c.target == "plate2" and p.plate2 is not None):
            _fail("contact.target", c.target,
                  "plate2 вместе с секцией [plate2] (и contact.enabled=true)",
                  "plate2")
        theories = {p.model.theory,
                    p.plate2.model.theory if p.plate2.model is not None else "classic"}
        # Силовое управление парой (∫r = P через поиск начального зазора z,
        # v0.6.5) — только НЕЛИНЕЙНАЯ пара; классическая пара — позиционная.
        if c.force is not None and not (theories <= {"karman"} or theories <= {"ktn_full"}):
            _fail("contact.force", c.force,
                  "отсутствие force для КЛАССИЧЕСКОЙ пары (силовое управление "
                  "парой — нелинейная пара karman | ktn_full, v0.6.5)", "plate2")
        if p.plate2.bc.type not in ("clamped", "soft_hinge"):
            # mixed для ВТОРОЙ пластины не реализован ни в одном тракте пары:
            # классический тихо игнорировал бы [[plate2.bc.sides]] (аудит v0.6.5)
            _fail("plate2.bc", p.plate2.bc.type,
                  "clamped | soft_hinge (смешанные КУ второй пластины пары — "
                  "направление развития)", "plate2")
        if c.scheme == "nested":
            # пара реализована ТОЛЬКО совмещённым циклом (по шагу Пикара на
            # пластину + шаг МОР); "nested" тихо игнорировался бы (аудит v0.6.5)
            _fail("contact.scheme", "nested",
                  "merged для пары пластин (вложенной схемы пары нет — "
                  "совместная реакция определяется совмещённым циклом, §9.2)",
                  "plate2")
        if theories == {"classic"}:
            pass                                    # классическая пара (v0.3)
        elif theories <= {"karman"} or theories <= {"ktn_full"}:
            # Нелинейная пара МОР+КТН (v0.6.3): ОБЕ пластины — одна нелинейная
            # теория, общая квадратура (одна планформа, один Q — проверит
            # решатель, NonlinearTwoPlateMOR); защемление и равномерная нагрузка
            # (для первой — общий нелинейный блок выше). Условие непроникания —
            # на лицевых прогибах u_c1 − u_c2 ≤ z (§9.2).
            star2 = p.geometry.kind in ("circle", "ellipse", "rectangle", "annulus")
            if p.plate2.bc.type not in ("clamped", "soft_hinge"):
                _fail("plate2.bc", p.plate2.bc.type,
                      f"clamped | soft_hinge при theory = {p.model.theory} "
                      "(нелинейная пара МОР+КТН, §9.2)", "plate2")
            if p.plate2.bc.type == "soft_hinge" and not star2:
                _fail("plate2.bc", p.plate2.bc.type,
                      "clamped на НЕзвёздной области (мягкий шарнир пары — только "
                      "circle | ellipse | rectangle | annulus; L/compose — v0.7)",
                      "plate2")
            if p.plate2.load.type != "uniform":
                _fail("plate2.load", p.plate2.load.type,
                      f"uniform при theory = {p.model.theory} (нелинейная пара — "
                      "равномерная нагрузка, §9.2)", "plate2")
        else:
            _fail("model.theory", sorted(theories),
                  "либо classic (классическая пара по срединным плоскостям, v0.3), "
                  "либо ОБЕ пластины одной нелинейной теории karman | ktn_full "
                  "(нелинейная пара МОР+КТН, v0.6.3, §9.2); смешанные пары и "
                  "ktn_linear для пары — направление развития", "plate2")
        if c.gap is not None and c.gap < 0:
            _fail("contact.gap", c.gap, "число ≥ 0 (Δ=0 — касание пластин)",
                  "plate2")
    elif c.enabled and c.gap is not None and c.gap <= 0:
        _fail("contact.gap", c.gap, "число > 0 (жёсткое основание)", "contact")
    if p.verify.reference == "analytic" and p.geometry.kind in ("compose", "ellipse"):
        _fail("verify.reference", "analytic",
              f"mms | fem | none — для геометрии {p.geometry.kind} аналитического "
              "эталона нет (замкнутого решения свободной задачи нет)", "verify")
    axisymmetric = p.geometry.kind in ("circle", "annulus") and p.load.type == "uniform"
    if p.verify.cross_1d and not axisymmetric:
        _fail("verify.cross_1d", True,
              "false — сверка с 1D-Ритцем по радиусу доступна только для "
              "осесимметричных постановок (circle | annulus, равномерная нагрузка)", "verify")
    # Эталон верификации — СТРУКТУРНАЯ применимость (fail-fast на этапе разбора,
    # а не лениво в resolve_reference; аудит устойчивости v0.6.4).
    ref = p.verify.reference
    if p.contact.enabled and ref == "analytic":
        from . import references as _refs  # ленивый импорт (references → Problem)
        if not _refs._is_axisym_contact_case(p):
            _fail("verify.reference", "analytic",
                  "none — сертифицированный контактный эталон существует только для "
                  "circle + (soft_hinge | clamped) + classic + основание со скалярным "
                  "зазором; прочий контакт (ktn_linear/ktn_full, не круг) — ворота "
                  "инвариантов (reference = none)", "verify")
    if ref == "mms" and (p.bc.type != "clamped"
                         or p.geometry.kind not in ("rectangle", "circle")):
        _fail("verify.reference", "mms",
              "clamped-постановка на rectangle | circle (изготовленное решение —"
              " MMS-ступени лестницы; для ktn_full — MMS полной КТН при фикс. N)", "verify")


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
    "EigenSpec",
    "GEOMETRY_KINDS",
    "GAP_KINDS",
    "GapSpec",
    "MIN_ZONE_NODES",
    "validate_compose_tree",
]
