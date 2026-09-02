"""cli.py — командная строка комплекса.

Пять команд (pyproject ``[project.scripts]``):

* ``plate-solve case.toml`` — решить постановку (result.json, fields.npz,
  фигуры); ``--new circle|rectangle|L|annulus|ellipse|compose
  [--out путь.toml]`` —
  закомментированный шаблон case-файла; ``--check`` — только валидация;
  ``--report`` — одностраничный md-отчёт;
* ``plate-verify case.toml`` — сверка с эталонами ``[verify]`` (exit 0/1);
  ``--sweep`` — сходимость по параметру;
* ``plate-ladder каталог/`` — сводный md по каталогу case-файлов;
* ``plate-replot dir/`` — перерисовка фигур из fields.npz без пересчёта;
* ``plate-profile dir/ --key w --from x0,y0 --to x1,y1`` — профиль поля
  вдоль сечения (+CSV, наложения нескольких результатов).

Новый случай делается копией шаблона и правкой нескольких строк
(docs/CASE_SCHEMA.md). Шаблон — исполняемая документация: его расширения
даны блоками «``#:`` », и снятие сигила с любого ОДНОГО блока (вместе с
правкой, названной строкой «``# требует:``» над ним) оставляет постановку
валидной; это проверяется тестами, а не декларируется.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

from .config import Config
from .problem import CaseError, Problem

_TEMPLATE_KINDS = ("circle", "rectangle", "L", "annulus", "ellipse", "compose")

#: сигил «готового блока» шаблона: снятие «#: » оставляет постановку валидной
_SIGIL = "#:"

#: аннотация над блоком: правки, вносимые ВМЕСТЕ с ним (точечные ключи TOML)
#: и — необязательно — секции, которые блок требует убрать
_REQUIRES_RE = re.compile(
    r"^#\s*требует:\s*(\{.*\})\s*(?:;\s*без секции:\s*(.+?))?\s*$")

#: повторяющаяся правка аннотаций: реестр эталонов покрывает только базовую
#: классическую задачу, поэтому расширения снимают эталон Кирхгофа
_REQ_NONE = 'verify.reference = "none", verify.cross_1d = false'

#: ключи, чьи значения в шаблоне берутся ИЗ ``Config`` (ворота — tests/test_cli.py):
#: обещание «это дефолты» проверяется, а не декларируется (аудит S13)
_TEMPLATE_DEFAULT_KEYS = ("E", "nu", "h", "winkler", "beta", "max_iter")

_GEOMETRY = {
    "circle": '''[geometry]
kind = "circle"
a = 1.0                  # радиус''',
    "rectangle": '''[geometry]
kind = "rectangle"
x1 = 0.0                 # [x1, x2] × [y1, y2]
x2 = 1.0
y1 = 0.0
y2 = 1.0''',
    "L": '''[geometry]
kind = "L"
side = 1.0               # сторона квадрата
cut = 0.5                # квадратный вырез (0 < cut < side, входящий угол)''',
    "annulus": '''[geometry]
kind = "annulus"
a = 1.0                  # внешний радиус
b = 0.4                  # внутренний радиус (0 < b < a)''',
    "ellipse": '''[geometry]
kind = "ellipse"
a = 1.0                  # полуось по x
b = 0.6                  # полуось по y''',
    "compose": '''[geometry]
kind = "compose"         # дерево R-операций (union|intersect|difference;
                         #  примитивы circle|rectangle; глубина ≤ 3, ≤ 7 узлов)

[geometry.tree]
op = "difference"        # квадрат с круглым вырезом (пример)

[[geometry.tree.children]]
kind = "rectangle"
x1 = 0.0
x2 = 1.0
y1 = 0.0
y2 = 1.0

[[geometry.tree.children]]
kind = "circle"
a = 0.2
cx = 0.5
cy = 0.5''',
}

# Эталон по умолчанию — что доступно данной геометрии (см. CASE_SCHEMA#verify).
_VERIFY = {
    "circle": '''[verify]
reference = "analytic"   # analytic | mms | fem | none
cross_1d = true          # сверка с 1D-Ритцем по радиусу
tol = 1.0e-2
#: model_gap = true      # информационная строка «истинный Кирхгоф»''',
    "annulus": '''[verify]
reference = "analytic"   # analytic | mms | fem | none
cross_1d = true          # сверка с 1D-Ритцем по радиусу [b, a]
tol = 1.0e-2
#: model_gap = true      # информационная строка «истинный Кирхгоф»''',
    "rectangle": '''[verify]
reference = "analytic"   # ряд Навье (soft_hinge + равномерная); mms — только clamped
tol = 1.0e-2''',
    "L": '''[verify]
reference = "fem"        # fem | mms | none (нужен pip install -e ".[fem]")
tol = 5.0e-2''',
    "ellipse": '''[verify]
reference = "none"       # аналитического эталона эллипса в реестре нет
                         # (верификация — тест-ворота, CASE_SCHEMA#geometry)''',
    "compose": '''[verify]
reference = "none"       # для compose доступны mms | fem | none (не analytic)''',
}


def _num(v) -> str:
    """Число в TOML-записи: целое — как есть, дробное — с точкой либо порядком."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    s = f"{v:g}".replace("e+0", "e").replace("e+", "e").replace("e-0", "e-")
    return s if ("." in s or "e" in s) else s + ".0"


def template(kind: str) -> str:
    """Текст закомментированного case-файла для геометрии ``kind``.

    Активная часть — полная постановка со СВОИМ эталоном ``[verify]``.
    Расширения даны блоками «``#:`` »: снятие сигила с ЛЮБОГО ОДНОГО блока
    (плюс правки, перечисленные строкой «``# требует:``» над ним) оставляет
    постановку валидной — ворота ``tests/test_cli.py``. Прежде шаблон
    предлагал блоки, которые валидатор отклонял (реестр эталонов покрывает
    только базовую классическую задачу), и путь «скопировать шаблон и
    раскомментировать» обрывался ошибкой (аудит S15).
    """
    if kind not in _TEMPLATE_KINDS:
        raise CaseError(
            f"--new: получено {kind!r}, ожидалось {' | '.join(_TEMPLATE_KINDS)}, "
            "см. docs/CASE_SCHEMA.md#geometry"
        )
    d = Config()
    return f'''# case-файл plate-solver — шаблон «{kind}».
# Схема и все ключи: docs/CASE_SCHEMA.md. Обязательны [geometry], [bc], [load].
#
# Строки «#: » — ГОТОВЫЕ блоки расширений: снимите «#: » с ОДНОГО блока, и
# постановка останется валидной. Строка «# требует: {{…}}» над блоком —
# правки, которые вносятся ВМЕСТЕ с ним (точечные ключи: «verify.reference»
# — это ключ reference секции [verify]). Реестр эталонов покрывает только
# базовую классическую задачу, поэтому почти всякое расширение снимает
# эталон Кирхгофа. Числа блоков — дефолты plate_solver.config.Config,
# кроме помеченных как «не дефолт».

{_GEOMETRY[kind]}

[bc]
type = "soft_hinge"      # soft_hinge (M=0) | clamped (w=∂w/∂n=0) |
                         #  mixed (прямоугольник, стороны [[bc.sides]])

[load]
type = "uniform"         # uniform | patch | point | gaussian | expr | line
q0 = {_num(d.q0)}                 # равномерная нагрузка (q0 > 0 «вниз»)
# прочие виды нагрузки (docs/CASE_SCHEMA.md#load):
#   точечная сила — type = "point", P, x0, y0 (+ exact = true: ТОЧНАЯ δ
#     вместо пятна; classic clamped | karman);
#   пятно — type = "patch", q0 и подсекция [load.zone];
#   гауссова — type = "gaussian", q0, x0, y0, sigma (Δq аналитичен);
#   выражением — type = "expr", q0, expr = "sin(pi*x/2.0)";
#   вдоль отрезка — type = "line", p0, p1, intensity.
# термомомент M_T (аддитивен к равномерной q; при q0 = 0 — чистый термоизгиб):
# требует: {{{_REQ_NONE}}}
#: thermal_moment = 3.0   # не дефолт: без ключа термомомента нет

[model]
theory = "classic"       # classic (Кирхгоф) | karman (геом. нелинейность) |
                         #  ktn_linear (линейные поправки сдвига/обжатия) |
                         #  ktn_full (полная нелинейная КТН). Флаги CLI
                         #  --theory и --inplane-bc переопределяют этот блок.
# упругие постоянные, толщина и упругое основание:
#: E = {_num(d.E)}
#: nu = {_num(d.nu)}
#: h = {_num(d.h)}                # толщина (существенна для КТН-теорий)
#: winkler = {_num(d.winkler)}          # основание Винклера k_w ≥ 0 (0 — без основания)
# геометрическая нелинейность (docs/THEORY.md):
# требует: {{model.theory = "karman", {_REQ_NONE}}}
#: inplane_bc = "immovable"   # immovable (u = v = 0, основной) | movable (N·n = 0)
#: n_load_steps = {_num(d.n_load_steps)}           # шагов по нагрузке (большой прогиб — увеличить)
#: karman_relax = {_num(d.karman_relax)}         # недорелаксация θ ∈ (0, 1] итерации Пикара
#: karman_method = "{d.karman_method}"   # picard | newton (ускоритель)
# переменная толщина h(x, y) вместо постоянной h:
# требует: {{bc.type = "clamped", {_REQ_NONE}}}
#: h_expr = "0.5*(1+0.3*x)"   # не дефолт: без ключа толщина постоянна

[discretization]
p = 10                   # степень Чебышёва по оси (N = (p+1)²)
Q = 256                  # узлов квадратуры по оси (точность маски ~1/Q)
grid_n = 80              # сетка вывода полей и графиков; на ТОЧНОСТЬ решения
                         # НЕ влияет; увеличивайте для гладких картин
                         # (кольцо: >= 96); на лету: --grid N или Result.regrid(N)

{_VERIFY[kind]}

[output]
dir = "results/{kind}_case"
figures = false          # true — сохранить фигуры viz.py
#: vtk = true             # не дефолт: result.vtk (legacy VTK, ParaView)

# ─── расширения: снять «#: » с ОДНОГО блока (см. шапку файла) ─────────────── #

# точечные упругие опоры (реакции R печатаются в сводке прогона):
# требует: {{bc.type = "clamped", {_REQ_NONE}}}
#: [supports]
#: points = [[0.0, 0.0]]
#: stiffness = 2.0e5      # не дефолт: жёсткая опора ≈ 1e6·D/a³

# односторонний контакт с жёстким основанием (метод обобщённой реакции);
# силовой штамп — force = P, профиль зазора выражением — gap_expr, а для
# нелинейных теорий — scheme = "merged" | "nested" и mor_anderson > 0:
# требует: {{{_REQ_NONE}}}
#: [contact]
#: enabled = true
#: gap_factor = 0.5       # не дефолт: Δ = gap_factor·w_free (абсолютный — gap)
#: beta = {_num(d.beta)}             # 0 < β < 2 (теорема 4)
#: max_iter = {_num(d.max_iter)}

# зона препятствия — плоский штамп (по умолчанию препятствие под всей Ω):
# требует: {{contact.enabled = true, contact.gap_factor = 0.5, {_REQ_NONE}}}
#: [contact.zone]
#: kind = "rectangle"
#: x1 = 0.15
#: x2 = 0.45
#: y1 = 0.15
#: y2 = 0.45

# собственная задача: частоты колебаний либо критические множители
# (преднапряжение N(w) — prestress = true при theory = "karman" и [load]):
# требует: {{{_REQ_NONE}}}; без секции: load
#: [eigen]
#: kind = "vibration"     # vibration | buckling
#: n_modes = 6

# ортотропия классической теории (прямой набор жёсткостей энергии; либо
# инженерный набор Ex, Ey, nu_xy, Gxy):
# требует: {{bc.type = "clamped", {_REQ_NONE}}}
#: [model.orthotropy]
#: D11 = 1.0e5            # не дефолты: без секции пластина изотропна
#: D12 = 0.3e5
#: D22 = 0.6e5
#: D66 = 0.35e5
'''


def _template_blocks(text: str) -> list[dict]:
    """Разобрать шаблон на блоки «``#:`` » с их аннотациями «``# требует:``».

    Блок — максимальная непрерывная цепочка строк с сигилом; аннотация —
    строка непосредственно НАД блоком. Возвращает список словарей
    ``{title, start, end, requires, drop}``: ``requires`` — вложенный
    словарь правок (точечные ключи TOML), ``drop`` — секции, которые блок
    требует убрать. Обещание шаблона проверяется этим разбором в тестах.
    """
    import tomllib

    lines = text.splitlines()
    blocks: list[dict] = []
    i = 0
    while i < len(lines):
        if not lines[i].startswith(_SIGIL):
            i += 1
            continue
        j = i
        while j < len(lines) and lines[j].startswith(_SIGIL):
            j += 1
        requires: dict = {}
        drop: tuple[str, ...] = ()
        m = _REQUIRES_RE.match(lines[i - 1]) if i else None
        if m is not None:
            requires = tomllib.loads(f"req = {m.group(1)}")["req"]
            drop = tuple(s.strip() for s in (m.group(2) or "").split(",") if s.strip())
        blocks.append({"title": lines[i][len(_SIGIL):].strip(),
                       "start": i, "end": j, "requires": requires, "drop": drop})
        i = j
    return blocks


def _uncomment_block(text: str, block: dict) -> str:
    """Текст шаблона со снятым сигилом у ОДНОГО блока (остальные — как есть)."""
    lines = text.splitlines()
    for k in range(block["start"], block["end"]):
        s = lines[k]
        lines[k] = s[len(_SIGIL) + 1:] if s.startswith(_SIGIL + " ") else s[len(_SIGIL):]
    return "\n".join(lines) + "\n"


def write_template(kind: str, out: str | Path | None = None) -> Path:
    """Записать шаблон в файл (по умолчанию ``<kind>.toml``); перезапись запрещена."""
    path = Path(out) if out is not None else Path(f"{kind}.toml")
    if path.exists():
        raise CaseError(f"{path}: файл уже существует — перезапись запрещена, "
                        "укажите другой --out")
    text = template(kind)
    Problem.from_dict(_parse_for_selfcheck(text), source=str(path))  # самопроверка шаблона
    path.write_text(text, encoding="utf-8")
    return path


def _parse_for_selfcheck(text: str) -> dict:
    import tomllib

    return tomllib.loads(text)


# --------------------------------------------------------------------------- #
#  Свип по дискретизации: --sweep p=2:12:2 [--sweep Q=64:256:64]
# --------------------------------------------------------------------------- #
def _parse_sweep(spec: str) -> tuple[str, list[int]]:
    """Разобрать ``p=A:B:S`` / ``Q=A:B:S`` в (ключ, список значений)."""
    key, _, rng = spec.partition("=")
    key = key.strip()
    if key not in ("p", "Q"):
        raise CaseError(f"--sweep: получено {key!r}, ожидалось p | Q, "
                        "см. docs/CASE_SCHEMA.md#discretization")
    parts = rng.split(":")
    try:
        a, b, s = (int(v) for v in parts)
    except (ValueError, TypeError):
        raise CaseError(f"--sweep {spec!r}: ожидался формат КЛЮЧ=нач:кон:шаг "
                        "(целые числа)") from None
    if len(parts) != 3 or s <= 0 or b < a:
        raise CaseError(f"--sweep {spec!r}: ожидался формат КЛЮЧ=нач:кон:шаг, "
                        "шаг > 0, кон ≥ нач")
    # Значения свипа обязаны удовлетворять тем же ограничениям схемы, что и
    # ключи [discretization] (p ≥ 1, Q ≥ 2): раньше свип шёл В ОБХОД валидатора
    # (p = 0 считался молча, p < 0 давал сырую трассировку; аудит S05).
    low = 1 if key == "p" else 2
    if a < low:
        raise CaseError(
            f"--sweep {spec!r}: получено {key} = {a}, ожидалось {key} ≥ {low} "
            "(та же ограда, что у ключей [discretization]), "
            "см. docs/CASE_SCHEMA.md#discretization")
    return key, list(range(a, b + 1, s))


def _sweep_rows(problem: Problem, sweeps: list[tuple[str, list[int]]],
                do_verify: bool) -> list[dict]:
    """Прогнать декартово произведение точек свипа; собрать строки таблицы."""
    import dataclasses
    import itertools

    from .dispatch import solve as _solve

    keys = [k for k, _ in sweeps]
    rows: list[dict] = []
    for combo in itertools.product(*[vals for _, vals in sweeps]):
        disc = dataclasses.replace(problem.discretization, **dict(zip(keys, combo, strict=True)))
        prob = dataclasses.replace(problem, discretization=disc)
        res = _solve(prob)
        row: dict = dict(zip(keys, combo, strict=True))
        row["w_max"] = res.w_max
        row["cond_A"] = res.cond
        if do_verify:
            from .references import verify_result

            rep = verify_result(res)
            gated = [r for r in rep.rows if r.gated]
            row["rel"] = max((r.rel for r in gated), default=float("nan"))
            row["ok"] = rep.ok
        rows.append(row)
    return rows


def _write_sweep_outputs(rows: list[dict], keys: list[str], out_dir: Path,
                         do_verify: bool) -> None:
    """md + csv + png (semilogy rel против параметра) — публикационный формат."""
    # headless-дружественный бэкенд ТЕМ ЖЕ приёмом, что и в _run_case: явный
    # matplotlib.use("Agg") молча ломал выбор пользователя (аудит S26).
    os.environ.setdefault("MPLBACKEND", "Agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys())
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    csv = [",".join(cols)]
    for r in rows:
        md.append("| " + " | ".join(_fmt(r[c]) for c in cols) + " |")
        csv.append(",".join(str(r[c]) for c in cols))
    (out_dir / "sweep.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (out_dir / "sweep.csv").write_text("\n".join(csv) + "\n", encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
    except ImportError:                            # png — опциональный артефакт
        return
    xs = [r[keys[0]] for r in rows]
    fig, ax = plt.subplots(figsize=(6, 4))
    if do_verify:
        ax.semilogy(xs, [r["rel"] for r in rows], "o-")
        ax.set_ylabel("относительная ошибка (max по эталонам)")
    else:
        ax.plot(xs, [r["w_max"] for r in rows], "o-")
        ax.set_ylabel("w_max")
    ax.set_xlabel(keys[0])
    ax.grid(True, which="both", alpha=0.3)
    fig.savefig(out_dir / "sweep.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _fmt(v) -> str:
    if isinstance(v, bool):
        return "PASS" if v else "FAIL"
    if isinstance(v, float):
        return f"{v:.6e}"
    return str(v)


#: запасной перечень форматов фигур (когда matplotlib недоступен для опроса)
_FIG_FORMATS_FALLBACK = ("eps", "pdf", "png", "ps", "raw", "rgba", "svg", "svgz")


def _supported_fig_formats() -> tuple[str, ...]:
    """Форматы, которые умеет писать matplotlib данной установки."""
    try:
        from matplotlib.backend_bases import FigureCanvasBase
    except ImportError:                        # фигуры — опциональный артефакт
        return _FIG_FORMATS_FALLBACK
    return tuple(sorted(FigureCanvasBase.get_supported_filetypes()))


def _parse_fig_formats(spec: str) -> tuple[str, ...]:
    """Разобрать ``--fig-format png,pdf`` в кортеж форматов.

    Пробелы вокруг значений отбрасываются (``"png, pdf"`` — законная запись
    оболочки), регистр приводится к нижнему, ведущая точка отбрасывается.
    Неизвестный формат отклоняется ЗДЕСЬ, до расчёта: прежде он доходил до
    ``savefig`` и давал сырую трассировку ``ValueError`` matplotlib уже
    после решения задачи (аудит S12).
    """
    items = [f.strip().lower().lstrip(".") for f in spec.split(",")]
    items = [f for f in items if f]
    if not items:
        raise CaseError("--fig-format: пустой список форматов, "
                        "ожидалось например png,pdf")
    allowed = _supported_fig_formats()
    bad = [f for f in items if f not in allowed]
    if bad:
        raise CaseError(f"--fig-format: получено {', '.join(bad)}, ожидалось "
                        f"из {' | '.join(allowed)} (через запятую)")
    return tuple(items)


def _apply_model_overrides(args_problem, args):
    """Переопределить ``[model] theory``/``inplane_bc`` флагами ``--theory``/``--inplane-bc``.

    Флаги CLI имеют приоритет над case-файлом (удобно гонять одну постановку
    тремя теориями). Переопределённая постановка проходит ту же перекрёстную
    валидацию рамок (геометрия/КУ/контакт для karman), что и case-файл.
    """
    import dataclasses

    theory = getattr(args, "theory", None)
    inplane = getattr(args, "inplane_bc", None)
    if theory is None and inplane is None:
        return args_problem
    kw: dict = {}
    if theory is not None:
        kw["theory"] = theory
    if inplane is not None:
        kw["inplane_bc"] = inplane
    model = dataclasses.replace(args_problem.model, **kw)
    from .problem import NONLINEAR_THEORIES, _validate_cross

    if inplane is not None and model.theory not in NONLINEAR_THEORIES:
        raise CaseError("--inplane-bc: осмыслен только для нелинейных теорий "
                        f"({' | '.join(NONLINEAR_THEORIES)}); добавьте "
                        "--theory karman|ktn_full, см. docs/CASE_SCHEMA.md#model")

    problem = dataclasses.replace(args_problem, model=model)
    # Смена теории обесценивает эталоны КИРХГОФА, записанные в case-файле
    # (analytic | fem | cross_1d, а также mms для karman/ktn_linear): гейт
    # сравнивал бы разные МОДЕЛИ. Верификация в этом прогоне отключается — с
    # явной записью в stderr, чтобы «зелёный» вывод не выглядел сертификатом
    # (v0.8.0; сама ограда — в _validate_cross, аудит S06/S09).
    v = problem.verify
    kirchhoff_ref = v.reference in ("analytic", "fem") or v.cross_1d
    mms_wrong = v.reference == "mms" and model.theory in ("karman", "ktn_linear")
    if theory is not None and model.theory != "classic" and (kirchhoff_ref or mms_wrong):
        dropped = v.reference if v.reference != "none" else "cross_1d"
        problem = dataclasses.replace(
            problem, verify=dataclasses.replace(v, reference="none", cross_1d=False))
        print(f"внимание: --theory {model.theory} — эталон '{dropped}' относится к "
              "другой модели (решение Кирхгофа), верификация в этом прогоне "
              "отключена; для информационной сверки — [verify] model_gap = true",
              file=sys.stderr)
    _validate_cross(problem)                          # рамки karman (§1) после override
    return problem


def _run_case(args, do_verify: bool) -> int:
    """Общий путь plate-solve/plate-verify: решить case (или свип) и отчитаться."""
    # headless-дружественный бэкенд для фигур (уважает явный выбор пользователя)
    os.environ.setdefault("MPLBACKEND", "Agg")

    from .dispatch import solve as _solve

    # значения флагов разбираются ДО расчёта: ошибка в записи форматов не
    # должна вскрываться после решения задачи (аудит S12)
    formats = _parse_fig_formats(getattr(args, "fig_format", "png,pdf"))
    problem = Problem.from_toml(args.case)
    problem = _apply_model_overrides(problem, args)
    if getattr(args, "grid", None) is not None:
        problem = problem.with_discretization(grid_n=args.grid)
    out_dir = Path(args.out) if args.out else Path(problem.output.dir)

    if args.sweep:
        sweeps = [_parse_sweep(s) for s in args.sweep]
        keys = [k for k, _ in sweeps]
        if len(set(keys)) != len(keys):
            raise CaseError("--sweep: ключи повторяются; допустимо по одному p и Q")
        if len(sweeps) > 1:
            print("предупреждение: два свипа — декартово произведение "
                  f"{'×'.join(str(len(v)) for _, v in sweeps)} прогонов", file=sys.stderr)
        rows = _sweep_rows(problem, sweeps, do_verify)
        _write_sweep_outputs(rows, keys, out_dir, do_verify)
        for line in (out_dir / "sweep.md").read_text(encoding="utf-8").splitlines():
            print(line)
        print(f"артефакты свипа: {out_dir}/sweep.{{md,csv,png}}")
        if do_verify:
            return 0 if rows[-1]["ok"] else 1      # вердикт — по самой точной точке
        return 0

    if getattr(args, "figures", False):
        import dataclasses

        problem = dataclasses.replace(
            problem, output=dataclasses.replace(problem.output, figures=True))
    res = _solve(problem)
    if do_verify:
        from .references import verify_result

        rep = verify_result(res)
        print(rep.table())
        if res.eigen is not None and not rep.rows:
            # НЕ вакуумный PASS молча: у собственных задач эталонов в [verify]
            # нет — честно сообщаем (числовые ворота — tests/test_eigenmodes.py)
            print("предупреждение: эталонов для собственной задачи в [verify] "
                  "нет — вердикт не выносится (ворота — tests/test_eigenmodes.py)")
        print(f"допуск tol = {rep.tol:g}; вердикт: {'PASS' if rep.ok else 'FAIL'}")
        return 0 if rep.ok else 1
    path = res.save(out_dir, fig_formats=formats,
                    surface=getattr(args, "surface", "mid"))
    s = res.scalars()
    if res.eigen is not None:
        label = ("критич. множители λ" if res.eigen.kind == "buckling"
                 else "частоты ω")
        print(f"{args.case}: собственная задача [{res.eigen.kind}], {label}:")
        for i, v in enumerate(res.eigen.values, 1):
            print(f"  мода {i}: {v:.6e}")
        print(f"cond(A) = {res.cond:.2e}")
        for w in res.warnings:
            print(f"предупреждение: {w}")
        print(f"результат: {path}")
        return 0
    print(f"{args.case}: w_max = {res.w_max:.6e}, cond(A) = {res.cond:.2e}")
    if problem.model.winkler is not None and problem.model.winkler > 0.0:
        print(f"основание Винклера: k_w = {problem.model.winkler:g}")
    if res.support_reactions is not None:
        rs = ", ".join(f"{v:.4e}" for v in res.support_reactions)
        print(f"опоры: реакции R = [{rs}]")
    if problem.model.theory in ("ktn_linear", "ktn_full"):
        # только УТОЧНЁННЫЕ теории: у Кармана поправок сдвига и обжатия нет
        # по построению (h_ψ² = h_*² = 0), и строка была шумом (аудит S22)
        tp = res.thickness_params()          # интроспекция §6.3
        print(f"толщина (КТН): h_ψ² = {tp['h_psi_sq']:.4e}, h_*² = {tp['h_star_sq']:.4e}, "
              f"h_c² = {tp['h_c_sq']:.4e}, h/L = {tp.get('h_over_L', float('nan')):.3f}")
    if res.contact is not None:
        print(f"контакт: итераций {s['iters']}, узлов {s['n_contact']}/{s['n_quad']}, "
              f"r_max = {s['r_max']:.4e}, комплементарность {s['comp_residual']:.2e}")
    for w in res.warnings:
        print(f"предупреждение: {w}")
    if getattr(args, "report", False):
        rp = _write_report(args.case, res, out_dir)
        print(f"отчёт: {rp}")
    print(f"результат: {path}")
    return 0


def _write_report(case_path: str, res, out_dir: Path) -> Path:
    """Одностраничный md-отчёт по кейсу: «нажал — получил документ».

    Состав: постановка листингом (исходный TOML), сводные числа
    (Result.scalars), verify-таблица при наличии эталона, фигуры
    относительными ссылками (если сохранены рядом).
    """
    lines = [f"# Отчёт: {Path(case_path).name}", ""]
    lines += ["## Постановка", "", "```toml",
              Path(case_path).read_text(encoding="utf-8").rstrip(), "```", ""]
    lines += ["## Сводные числа", "", "| величина | значение |", "|---|---|"]
    for k, v in res.scalars().items():
        if v is None:
            continue
        val = f"{v:.6e}" if isinstance(v, float) else str(v)
        lines.append(f"| {k} | {val} |")
    lines.append("")
    if res.problem.verify.reference != "none":
        from .references import verify_result

        rep = verify_result(res)
        verdict = "PASS" if rep.ok else "FAIL"
        lines += ["## Верификация", "", "```", rep.table(),
                  f"допуск tol = {rep.tol:g}; вердикт: {verdict}", "```", ""]
    figs = sorted(q.name for q in out_dir.glob("*.png"))
    if figs:
        lines += ["## Фигуры", ""]
        lines += [f"![{f}]({f})" for f in figs]
        lines.append("")
    for w in res.warnings:
        lines.append(f"> предупреждение: {w}")
    out = out_dir / "report.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


# --------------------------------------------------------------------------- #
#  Точки входа
# --------------------------------------------------------------------------- #
_EXAMPLES = {
    "plate-solve": """примеры:
  plate-solve --new annulus            шаблон case-файла annulus.toml
  plate-solve case.toml                решить: result.json + fields.npz
  plate-solve case.toml --figures --report --out results/run1
  plate-solve case.toml --check        только валидация (exit 0/1)
  plate-solve case.toml --surface bottom --figures   прогиб нижней лицевой""",
    "plate-verify": """примеры:
  plate-verify case.toml               таблица эталонов, exit 0/1 по tol
  plate-verify case.toml --sweep p=2:12:2      сходимость по p (md+csv+png)
  plate-verify case.toml --sweep p=4:12:4 --sweep Q=64:256:64""",
}


def _base_parser(prog: str, descr: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog, description=descr, epilog=_EXAMPLES.get(prog),
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", nargs="?", help="case-файл TOML (docs/CASE_SCHEMA.md)")
    parser.add_argument("--sweep", action="append", metavar="p=2:12:2",
                        help="свип по p или Q (можно оба — декартово произведение)")
    parser.add_argument("--out", metavar="DIR", default=None,
                        help="каталог результатов (по умолчанию output.dir "
                             "case-файла; у plate-verify — только артефакты --sweep)")
    parser.add_argument("--figures", action="store_true",
                        help="форсировать output.figures = true (png 300 dpi + pdf); "
                             "только plate-solve — верификация фигур не пишет")
    parser.add_argument("--fig-format", metavar="png,pdf", default="png,pdf",
                        help="форматы фигур через запятую (по умолчанию png,pdf); "
                             "только plate-solve")
    parser.add_argument("--grid", type=int, metavar="N", default=None,
                        help="сетка ВЫВОДА grid_n (полей и фигур); на числа "
                             "решения не влияет; целое ≥ 2")
    parser.add_argument("--surface", choices=("mid", "top", "bottom"),
                        default="mid",
                        help="поверхность на w-фигуре: срединная (mid) или "
                             "лицевые top/bottom (лицевые величины КТН, NOTES §21); "
                             "только plate-solve")
    parser.add_argument("--theory",
                        choices=("classic", "karman", "ktn_linear", "ktn_full"),
                        default=None,
                        help="переопределить [model] theory: classic (Кирхгоф) | "
                             "karman (геометрическая нелинейность) | ktn_linear "
                             "(линейные поправки сдвига/обжатия) | ktn_full "
                             "(полная нелинейная КТН). Устаревшее имя 'ktn' "
                             "принимается только в case-файле (депрекация-алиас "
                             "на ktn_linear, docs/MIGRATION.md), из CLI оно "
                             "исключено — задавайте имя явно")
    parser.add_argument("--inplane-bc", dest="inplane_bc",
                        choices=("immovable", "movable"), default=None,
                        help="переопределить [model] inplane_bc (нелинейные теории): "
                             "immovable (u=v=0) | movable (N·n=0)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    return parser


def main(argv: list[str] | None = None) -> int:
    """``plate-solve``: решить case-файл (``--new`` — сгенерировать шаблон)."""
    parser = _base_parser("plate-solve", "Решение постановки из case-файла "
                                         "(геометрия+КУ+нагрузка -> Result).")
    parser.add_argument("--new", dest="new_kind", metavar="KIND",
                        help=f"создать шаблон case-файла: {' | '.join(_TEMPLATE_KINDS)}")
    parser.add_argument("--check", action="store_true",
                        help="только валидация постановки (схема + статические "
                             "несовместимости), НИЧЕГО не считает; exit 0/1 — "
                             "для пользовательских CI. Проверяется постановка "
                             "С УЧЁТОМ --theory/--inplane-bc/--grid — та же, "
                             "что была бы посчитана")
    parser.add_argument("--report", action="store_true",
                        help="одностраничный md-отчёт по кейсу (постановка, "
                             "сводные числа, verify-таблица, фигуры) в каталог "
                             "результата")
    args = parser.parse_args(argv)
    try:
        if args.new_kind is not None:
            path = write_template(args.new_kind, args.out)
            print(f"шаблон записан: {path} (схема — docs/CASE_SCHEMA.md)")
            return 0
        if args.case is None:
            parser.print_help()
            return 0
        if args.check:
            from . import __version__

            # Проверяется ТА ЖЕ постановка, которую посчитал бы plate-solve с
            # теми же флагами: переопределения теории/КУ в плоскости и сетки
            # вывода применяются ДО валидации (прежде --check их игнорировал и
            # подтверждал другую постановку — аудит S14).
            problem = Problem.from_toml(args.case)   # вся статика — валидатор
            problem = _apply_model_overrides(problem, args)
            if args.grid is not None:
                problem = problem.with_discretization(grid_n=args.grid)
            print(f"{args.case}: постановка валидна — theory = "
                  f"{problem.model.theory}, grid_n = {problem.discretization.grid_n} "
                  f"(plate-solver {__version__}, схема — docs/CASE_SCHEMA.md)")
            return 0
        return _run_case(args, do_verify=False)
    except CaseError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1


def _check_verify_flags(args) -> None:
    """Флаги ``plate-verify``: отклонить недействующие, назвать молчащие (S16).

    Верификация не сохраняет ни результата, ни фигур — она печатает таблицу
    эталонов и возвращает код. Флаги ``--figures``/``--fig-format``/
    ``--surface`` парсер принимал, но они не делали НИЧЕГО: пользователь
    получал молчание вместо картинок. Действующие флаги команды —
    ``--sweep``, ``--grid``, ``--theory``, ``--inplane-bc`` и ``--out``
    (последний — только каталог артефактов свипа).
    """
    inert = [name for name, value, default in
             (("--figures", args.figures, False),
              ("--fig-format", args.fig_format, "png,pdf"),
              ("--surface", args.surface, "mid"))
             if value != default]
    if inert:
        raise CaseError(
            f"plate-verify: {', '.join(inert)} не применяются — верификация "
            "печатает таблицу эталонов и не сохраняет фигур; фигуры делает "
            "plate-solve (или plate-replot по готовому fields.npz)")
    if args.out and not args.sweep:
        print("предупреждение: --out без --sweep верификацией не используется "
              "(артефакты пишет свип; результат и фигуры — plate-solve)",
              file=sys.stderr)


def main_verify(argv: list[str] | None = None) -> int:
    """``plate-verify``: таблица «эталон | значение | rel | статус», exit 0/1."""
    parser = _base_parser("plate-verify", "Верификация постановки по эталонам "
                                          "case-файла (секция [verify]).")
    args = parser.parse_args(argv)
    if args.case is None:
        parser.print_help()
        return 0
    try:
        _check_verify_flags(args)
        return _run_case(args, do_verify=True)
    except CaseError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1


def main_ladder(argv: list[str] | None = None) -> int:
    """``plate-ladder``: каталог case-файлов → сводный md с провенансом."""
    parser = argparse.ArgumentParser(
        prog="plate-ladder",
        description="Прогнать каталог case-файлов (лестница верификации) и "
                    "собрать сводный markdown-отчёт.",
        epilog="пример:\n  plate-ladder cases/ci --out ladder.md",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", help="каталог с *.toml")
    parser.add_argument("--out", metavar="FILE", default=None,
                        help="файл отчёта (по умолчанию <каталог>/ladder_summary.md)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    folder = Path(args.directory)
    cases = sorted(folder.glob("*.toml"))
    if not cases:
        print(f"ошибка: в {folder} нет case-файлов *.toml", file=sys.stderr)
        return 1
    from .dispatch import _provenance
    from .dispatch import solve as _solve
    from .references import verify_result

    lines = ["# Лестница верификации — сводка", "",
             "| case | w_max | эталоны (rel) | статус |", "|---|---|---|---|"]
    all_ok = True
    for case in cases:
        try:
            problem = Problem.from_toml(case)
            res = _solve(problem)
            rep = verify_result(res)
            rels = "; ".join(f"{r.name}: {r.rel:.2e}" for r in rep.rows) or "—"
            ok = rep.ok
            lines.append(f"| {case.name} | {res.w_max:.6e} | {rels} | "
                         f"{'PASS' if ok else 'FAIL'} |")
        except CaseError as e:
            ok = False
            lines.append(f"| {case.name} | — | ошибка постановки: {e} | FAIL |")
        except Exception as e:                    # noqa: BLE001 — лестница не должна
            # обрушиваться на ОДНОМ случае: численный сбой (LinAlgError,
            # переполнение ряда эталона) раньше уносил весь прогон без сводки
            # (аудит S21). Фиксируем строкой FAIL и идём дальше.
            ok = False
            lines.append(f"| {case.name} | — | сбой расчёта: "
                         f"{type(e).__name__}: {str(e)[:120]} | FAIL |")
        all_ok &= ok
        print(f"{case.name}: {'PASS' if ok else 'FAIL'}")
    prov = _provenance()
    lines += ["", "## Провенанс",
              "", f"- plate-solver {prov['plate_solver']}, git {prov['git']}",
              f"- numpy {prov['numpy']}, scipy {prov['scipy']}, sympy {prov['sympy']}"]
    out = Path(args.out) if args.out else folder / "ladder_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"сводка: {out}")
    return 0 if all_ok else 1


def main_replot(argv: list[str] | None = None) -> int:
    """``plate-replot <dir>`` — перерисовать фигуры из fields.npz БЕЗ пересчёта.

    Каталог результата (``[output] dir``) обязан содержать ``fields.npz``;
    решатель не запускается — фигуры строятся из снимка полей (v0.6.6).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="plate-replot",
        description="Перерисовка фигур из fields.npz без пересчёта")
    parser.add_argument("dir", help="каталог результата с fields.npz")
    parser.add_argument("--fig-format", default="png,pdf",
                        help="форматы через запятую (по умолчанию png,pdf)")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--surface", default="mid",
                        choices=("mid", "top", "bottom"),
                        help="поверхность w-фигуры (лицевые — NOTES §21)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    from .viz import replot

    try:                    # запись форматов проверяется ДО чтения полей (S12)
        formats = _parse_fig_formats(args.fig_format)
    except CaseError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    target = Path(args.dir)
    if not (target / "fields.npz").exists():
        print(f"plate-replot: в {target} нет fields.npz "
              "(укажите каталог результата [output] dir)", file=sys.stderr)
        return 1
    paths = replot(target, formats=formats, dpi=args.dpi, surface=args.surface)
    for p in paths:
        print(p)
    return 0


def main_profile(argv: list[str] | None = None) -> int:
    """``plate-profile <dir> [<dir2> …]`` — профиль поля вдоль сечения (v0.6.6).

    Один каталог — профиль + CSV; несколько — НАЛОЖЕНИЕ (сравнение
    теорий/кейсов вдоль одного сечения). Всё из fields.npz, без пересчёта.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="plate-profile",
        description="Профиль поля вдоль сечения из fields.npz (без пересчёта); "
                    "несколько каталогов — наложение кривых")
    parser.add_argument("dirs", nargs="+", help="каталог(и) результатов с fields.npz")
    parser.add_argument("--key", default="w",
                        help="ключ поля (w, Mx, Qx, sx_bot, svm_top, r, …)")
    parser.add_argument("--from", dest="p0", required=True,
                        help="начало сечения: x0,y0 (отрицательные координаты "
                             "— формой --from=-1,0)")
    parser.add_argument("--to", dest="p1", required=True,
                        help="конец сечения: x1,y1")
    parser.add_argument("-n", type=int, default=200, help="число точек")
    parser.add_argument("--csv", default=None,
                        help="записать CSV (s, значения по каталогам)")
    parser.add_argument("--fig", default=None,
                        help="записать фигуру (расширение задаёт формат)")
    from . import __version__

    parser.add_argument("--version", action="version",
                        version=f"%(prog)s {__version__}")
    args = parser.parse_args(argv)
    from .viz import overlay_profiles, section_profile

    try:
        p0 = tuple(float(v) for v in args.p0.split(","))
        p1 = tuple(float(v) for v in args.p1.split(","))
    except ValueError:
        print("plate-profile: --from/--to ожидают пару координат x,y",
              file=sys.stderr)
        return 1
    if len(p0) != 2 or len(p1) != 2:
        print("plate-profile: --from/--to ожидают пару координат x,y",
              file=sys.stderr)
        return 1
    curves = []
    for d in args.dirs:
        if not (Path(d) / "fields.npz").exists():
            print(f"plate-profile: в {d} нет fields.npz "
                  "(укажите каталог результата [output] dir)", file=sys.stderr)
            return 1
        try:
            s, vals = section_profile(d, args.key, p0, p1, n=args.n)
        except (KeyError, ValueError) as e:
            print(f"plate-profile: {e}", file=sys.stderr)
            return 1
        curves.append((Path(d).name, s, vals))
    if args.csv:
        header = "s," + ",".join(name for name, _, _ in curves)
        cols = [curves[0][1]] + [v for _, _, v in curves]
        np.savetxt(args.csv, np.column_stack(cols), delimiter=",",
                   header=header, comments="")
        print(args.csv)
    if args.fig:
        overlay_profiles(args.dirs, args.key, p0, p1, n=args.n, save=args.fig)
        print(args.fig)
    if not args.csv and not args.fig:                    # по умолчанию — в консоль
        s = curves[0][1]
        for i in range(0, len(s), max(1, len(s) // 20)):
            row = "  ".join(f"{v[i]:.6e}" for _, _, v in curves)
            print(f"s={s[i]:.4f}  {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
