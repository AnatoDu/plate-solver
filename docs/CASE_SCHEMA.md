# CASE_SCHEMA — схема case-файла (v0.2)

Case-файл — TOML-описание постановки задачи: геометрия, закрепление,
нагрузка, модель, контакт, дискретизация, верификация, вывод. Загружается
`plate_solver.problem.Problem.from_toml`, решается `dispatch.solve(problem)`.
Новый случай = копия шаблона (`plate-solve --new <kind>`) и правка
нескольких строк — см. пример внизу.

Правило дефолтов: физика и дискретизация по умолчанию живут в ОДНОМ месте —
`plate_solver.config.Config`; отсутствующий ключ означает «взять дефолт
Config», а не «второй экземпляр дефолта в схеме».

## Схема

Секции: `[geometry]`, `[bc]`, `[load]` — **обязательные**;
`[model]`, `[contact]`, `[discretization]`, `[verify]`, `[output]` —
необязательные. Неизвестные секции и ключи отклоняются с ошибкой
(защита от опечаток). Ошибки валидатора имеют вид
«ключ: получено X, ожидалось Y, см. docs/CASE_SCHEMA.md#секция».

## Полный аннотированный пример

```toml
# Кольцо под равномерной нагрузкой, защемление, сверка с аналитикой и 1D.

[geometry]
kind = "annulus"        # circle | rectangle | L | annulus | compose
a = 1.0                 # внешний радиус
b = 0.4                 # внутренний радиус (0 < b < a)

[bc]
type = "clamped"        # soft_hinge | clamped (в v0.2 один тип на всю границу)

[load]
type = "uniform"        # uniform | patch | point
q0 = 4.0                # равномерная поперечная нагрузка (q0 > 0 «вниз»)

[model]
theory = "classic"      # classic | karman | ktn_linear | ktn_full (§model);
                        #   устаревший алиас "ktn" = ktn_linear (DeprecationWarning)
# E = 2.1e6             # дефолты Config: E=2.1e6, nu=0.3, h=1.0
# nu = 0.3
h = 0.06                # толщина (важно для ktn_linear/ktn_full)
# --- только для НЕЛИНЕЙНЫХ теорий (karman, ktn_full) --- #
# inplane_bc = "immovable"  # immovable (u=v=0) | movable (N·n=0)
# n_load_steps = 1          # шагов по нагрузке (большой прогиб — увеличить)
# karman_relax = 1.0        # недорелаксация θ ∈ (0, 1] итерации Пикара
# karman_max_iter = 200     # предел итераций Пикара на уровень нагрузки
# karman_tol = 1.0e-8       # относит. порог останова ‖Δw‖/‖w‖
# karman_method = "picard"  # picard | newton (только karman)
# ktn_method = "picard"     # picard | newton (только ktn_full)

[discretization]
p = 10                  # степень Чебышёва по оси (N = (p+1)²); дефолт 12
Q = 1024                # узлов Гаусса–Лежандра по оси; дефолт 64
# grid_n = 80           # фоновая сетка вывода

[verify]
reference = "analytic"  # analytic | mms | fem | none
cross_1d = true         # сверка с 1D-Ритцем по радиусу (только circle/annulus,
                        # равномерная нагрузка)
tol = 1.0e-2            # допуск ворот верификации
model_gap = false       # печатать строку «истинный Кирхгоф» (вне допуска)

[output]
dir = "results/annulus_clamped"
figures = true
```

## Таблица ключей

| Ключ | Тип | Дефолт | Допустимые значения | Исполняет |
|---|---|---|---|---|
| geometry.kind | str | — (обяз.) | circle, rectangle, L, annulus, compose | geometry.py |
| geometry.a | float | — | > 0 (circle; annulus: внешний) | geometry.py |
| geometry.b | float | — | 0 < b < a (annulus: внутренний) | geometry.py |
| geometry.x1..y2 | float | — | x1 < x2, y1 < y2 (rectangle) | geometry.py |
| geometry.side, cut | float | — | 0 < cut < side (L) | geometry.py |
| geometry.tree | таблица | — | compose-дерево (см. #compose) | geometry.py |
| bc.type | str | — (обяз.) | soft_hinge, clamped, mixed | plate.py / clamped.py |
| bc.sides | массив таблиц | — | side: x1..y2; type: clamped, hinge, free | clamped.py (MixedRectPlate) |
| load.type | str | — (обяз.) | uniform, patch, point | dispatch.py |
| load.q0 | float | — | число (uniform, patch) | dispatch.py |
| load.zone | таблица | — | геометрия зоны (patch; язык — как geometry) | dispatch.py |
| load.P | float | — | число (point: результирующая сила) | dispatch.py |
| load.x0, y0 | float | — | точка приложения (point) | dispatch.py |
| load.eps | float | 0.05·min(шир., выс. bbox) | > 0 (point: радиус пятна) | dispatch.py |
| model.theory | str | "classic" | classic, karman, ktn_linear, ktn_full (алиас ktn=ktn_linear) | dispatch.py |
| model.E, nu, h | float | дефолты Config | E>0, −1<nu<0.5, h>0 | config.py |
| model.inplane_bc | str | "immovable" | immovable, movable (нелинейные теории) | membrane.py |
| model.n_load_steps | int | дефолт Config (1) | ≥ 1 (нелинейные теории) | membrane.py |
| model.karman_relax | float | дефолт Config (1.0) | 0 < θ ≤ 1 (нелинейные теории) | membrane.py |
| model.karman_max_iter | int | дефолт Config (200) | ≥ 1 (нелинейные теории) | membrane.py |
| model.karman_tol | float | дефолт Config (1e-8) | > 0 (нелинейные теории) | membrane.py |
| model.karman_method | str | дефолт Config ("picard") | picard, newton (только karman) | membrane.py |
| model.ktn_method | str | дефолт Config ("picard") | picard, newton (только ktn_full) | ktn_full.py |
| contact.enabled | bool | false | true, false | contact.py |
| contact.gap | float | — | > 0 (абсолютный зазор Δ) | contact.py |
| contact.gap_factor | float | — | > 0 (Δ = gap_factor·w_free) | dispatch.py |
| contact.beta | float | дефолт Config (1.2) | 0 < β < 2 (теорема 4) | contact.py |
| contact.max_iter | int | дефолт Config | ≥ 1 | contact.py |
| contact.tol | float | дефолт Config | > 0 | contact.py |
| contact.stop | str | дефолт Config ("dr") | dr, comp | contact.py |
| contact.zone | таблица | вся Ω | геометрия зоны препятствия | dispatch.py |
| [contact.gap] kind | str | — | const, plane, paraboloid, steps (поле Δ(x,y), v0.3) | dispatch.py |
| contact.gap.value | float | — | > 0 (const: алиас скалярного gap) | dispatch.py |
| contact.gap.a, b, c | float | — | plane: Δ = a·x + b·y + c (>0 на основании) | dispatch.py |
| contact.gap.r_curv, cx, cy, apex | float | cx=cy=0 | paraboloid: Δ = apex + ((x−cx)²+(y−cy)²)/(2·r_curv) | dispatch.py |
| contact.gap.base, [[zones]] | float, массив | — | steps: base + зоны (геометрия + value > 0) | dispatch.py |
| contact.force | float | — | > 0: силовой штамп, ∫r dΩ = force (v0.3) | dispatch.py |
| contact.target | str | "foundation" | foundation, plate2 (пара пластин, v0.3) | dispatch.py |
| [plate2] bc, load | секции | — (обяз. при target=plate2) | как у первой пластины | dispatch.py |
| [plate2] geometry, model, discretization | секции | от первой | как у первой пластины | dispatch.py |
| discretization.p | int | дефолт Config (12) | ≥ 1 | basis.py |
| discretization.Q | int | дефолт Config (64) | ≥ 2 | quadrature.py |
| discretization.grid_n | int | дефолт Config (80) | ≥ 2 | contact.py/viz.py |
| verify.reference | str | "none" | analytic, mms, fem, none | references.py |
| verify.cross_1d | bool | false | true, false (осесимметричные) | radial.py |
| verify.tol | float | 1e-2 | > 0 | references.py |
| verify.model_gap | bool | false | true, false | references.py |
| output.dir | str | "results" | непустая строка | dispatch.py |
| output.figures | bool | false | true, false | viz.py |

Требование к зонам (`load.zone`, `contact.zone`, пятно `point`): пересечение
зоны с Ω должно накрывать **не менее 20 узлов квадратуры**, иначе интеграл по
маске теряет смысл. Для `point` пятно расширяется автоматически до
наименьшего радиуса с ≥ 20 узлами (факт и новое eps — в `result.json`,
поле `warnings`); для `patch` авторасширение зоны произвольной формы не
определено — это ошибка с советом «увеличьте Q или зону».

## Несовместимости v0.2

| Комбинация | Решение |
|---|---|
| contact.enabled + bc.type = clamped | ошибка: в v0.2 контакт реализован для мягкого шарнира |
| verify.reference = analytic + geometry.kind = compose | ошибка: используйте mms \| fem \| none |
| verify.cross_1d + неосесимметричная постановка | ошибка: cross_1d — только circle/annulus с равномерной нагрузкой |

## geometry

Пять видов области. `circle` — круг радиуса `a` с центром в начале координат;
`rectangle` — `[x1, x2] × [y1, y2]`; `L` — квадрат `side` с квадратным
вырезом `cut` (входящий угол); `annulus` — кольцо `b < r < a`;
`compose` — конструктор (см. #compose). Граница всюду задаётся R-функцией
ω(x, y): ω > 0 внутри, ω = 0 на границе.

## compose

Дерево операций над примитивами (ограда v0.2 — не расширяется):
операции `union | intersect | difference` (difference строго бинарна),
примитивы `circle (a, cx, cy)` и `rectangle (x1, x2, y1, y2)`,
глубина дерева ≤ 3 (узлы считаются по вертикали, примитив = 1),
всего ≤ 7 узлов. bbox: union — объединение, intersect — пересечение,
difference — bbox первого операнда.

```toml
[geometry]
kind = "compose"

[geometry.tree]             # квадрат с круглым вырезом
op = "difference"

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
cy = 0.5
```

## bc

Один тип закрепления на всю границу: `soft_hinge` — «мягкий шарнир»
(M = 0 на ∂Ω, расщепление бигармоники на две задачи Пуассона, plate.py);
`clamped` — жёсткое защемление (структура w = ω²Φ, прямой Ритц, clamped.py).
Смешанные закрепления по участкам — `mixed` + `[[bc.sides]]` (только
kind = rectangle): каждой стороне x1|x2|y1|y2 назначается тип
`clamped` (w = ∂w/∂n = 0, множитель ω² в структуре), `hinge`
(w = 0, истинный шарнир: M_n = 0 естественно из полной билинейной формы)
или `free` (СВОБОДНЫЙ край: кинематических условий нет; M_n = 0 и
обобщённая перерезывающая Кирхгофа V_n = 0 — естественные условия).
Правило жёстких смещений: набор сторон обязан исключать ядро {1, x, y} —
не менее ОДНОЙ clamped либо не менее ДВУХ hinge; FFFF и SFFF отклоняются
валидатором. Контакт и theory=ktn при mixed (в т.ч. free) — направление
развития (валидатор отклоняет с пояснением).

```toml
[bc]
type = "mixed"
[[bc.sides]]
side = "x1"
type = "clamped"   # консоль: заделка x1, остальные free
[[bc.sides]]
side = "x2"
type = "free"
[[bc.sides]]
side = "y1"
type = "free"
[[bc.sides]]
side = "y2"
type = "free"
```

## load

`uniform` — равномерная `q0` по всей Ω. `patch` — `q0` внутри зоны
`[load.zone]` (геометрия зоны — тем же языком, что `[geometry]`), ноль вне:
`q̃ = q0·[ω_zone > 0]`. `point` — сосредоточенная сила `P` в точке
`(x0, y0)` как регуляризованный patch: круговое пятно радиуса `eps`,
`q = P/(π·eps²)`; дефолт `eps = 0.05·min(ширина, высота bbox)`.

### Неравномерная нагрузка

Через схему доступны только `patch` и `point`. Истинная δ-нагрузка в схему
сознательно не вводится: в расщеплении (P1) — это функционал вне H¹
(M ~ ln r), а КТН-прогиб под δ логарифмически расходится — см.
docs/NOTES.md, раздел «Точечная сила и уточнённая теория». Произвольная
гладкая f(x, y) — только через API, значениями в узлах квадратуры:

```python
from plate_solver import Config
from plate_solver.geometry import make_circle
from plate_solver.plate import PlateBending

pb = PlateBending.from_config(make_circle(1.0), Config(p=10, Q=256))
q = pb.quad
f_values = 4.0 * (1.0 + 0.5 * q.x)        # свой закон нагрузки в узлах
cM, cw = pb.solve(f_values)
print(float(pb.deflection(cw, 0.0, 0.0)))
```

## model

Лестница теорий одним ключом `theory` — с ЧЕСТНЫМИ, однозначными терминами
(нельзя называть линейную поправку именем полной нелинейной теории):

* `classic` — линейная теория Кирхгофа (расщепление бигармоники);
* `karman` — геометрически-**НЕЛИНЕЙНОЕ** решение Фёппля–Кармана: прогиб
  входит квадратично в мембранные деформации, поле усилий `N` ужесточает
  пластину (мембранная связь `L(Φ, w)`). Итерация Пикара по замороженным
  усилиям `N_k` с наращиванием нагрузки (`membrane.py`, `THEORY.md`);
* `ktn_linear` — **ЛИНЕЙНЫЕ** поправки поперечного сдвига/обжатия (часть теории
  Тимошенко–Нагди) ПОСТОБРАБОТКОЙ на решении Кирхгофа (`corrected_deflection`
  при r = 0; кривизна как Δw = −M/D, без численного дифференцирования). Это НЕ
  нелинейная теория (прежнее поведение `ktn`, число-в-число);
* `ktn_full` — **ПОЛНАЯ нелинейная КТН**: Карман + регуляризация связи
  `(I − h_ψ²Δ)L(Φ, w)` + нагрузочный член `−h_*²Δq_n` (`ktn_full.py`, §THEORY).

**Депрекация `ktn`.** Значение `ktn` неоднозначно и переименовано: `ktn` —
устаревший алиас на `ktn_linear` (сохраняет поведение старых case-файлов,
число-в-число) с `DeprecationWarning`; будет удалён в v1.0.0. Миграция —
`docs/MIGRATION.md`.

**Только для НЕЛИНЕЙНЫХ теорий (`karman`, `ktn_full`).** `inplane_bc` —
закрепление кромки В ПЛАНЕ: `immovable` (u = v = 0 на ∂Ω — кромка не
втягивается, натяжение максимально; основной режим) либо `movable` (N·n = 0 —
кромка свободна в плане, эффект слабее). Параметры итерации: `n_load_steps`
(шагов по нагрузке; тёплый старт по уровням), `karman_relax` (недорелаксация
θ ∈ (0, 1]), `karman_max_iter`, `karman_tol` (относительный останов ‖Δw‖/‖w‖);
метод — `karman_method` (только `karman`) / `ktn_method` (только `ktn_full`),
оба `picard` по умолчанию | `newton`. При `classic`/`ktn_linear` задание любого
из этих ключей — ошибка постановки. Рамки: круг и прямоугольник/квадрат, КУ
`clamped`/`soft_hinge`, без контакта (нелинейный контакт — v0.6.0).

**CLI-переопределение.** Флаги `plate-solve`/`plate-verify` `--theory`
(`classic | karman | ktn_linear | ktn_full`) и `--inplane-bc`
(`immovable | movable`) переопределяют блок `[model]` — удобно гонять одну
постановку разными теориями
(`--inplane-bc` — только вместе с `--theory karman`).

## contact

Односторонний контакт с жёстким ПЛОСКИМ препятствием при постоянном зазоре
Δ (метод обобщённой реакции). Зона препятствия `[contact.zone]` — тем же
геометрическим языком; дефолт — вся Ω (основание), подобласть — плоский
штамп. Ровно одно из `gap` (абсолютный Δ) / `gap_factor`
(Δ = gap_factor·w_free; w_free считает диспетчер). Критерий останова
`stop`: `dr` — ‖r_k − r_{k−1}‖ < tol; `comp` — безразмерная KKT-невязка
Синьорини < tol (см. докстринг `ContactMOR.solve`).

### Поле зазора Δ(x, y) (v0.3)

Вместо скаляра зазор задаётся таблицей `[contact.gap]` (ровно одно из:
`gap`, `gap_factor`, `[contact.gap]`): `const` (алиас скаляра), `plane`
(наклонное основание), `paraboloid` (неплоский штамп: apex — зазор в
вершине, r_curv — радиус кривизны), `steps` (base + `[[contact.gap.zones]]`
— несколько штампов разной высоты; зоны применяются по порядку, зазор в
зоне = value). Положительность Δ на основании проверяет диспетчер.
Для отчёта (`Result.delta`) у поля берётся min Δ на основании; той же
величиной нормируются метрики комплементарности. Произвольное гладкое
поле — только через API, по образцу нагрузки:

```toml
[contact]
enabled = true

[contact.gap]
kind = "paraboloid"      # штамп-параболоид
r_curv = 1.0e4
apex = 1.0e-4            # зазор в вершине (cx = cy = 0 по умолчанию)
```

```python
mor = ContactMOR(pb, cfg, gap=gap_values)   # gap_values — массив в узлах квадратуры
```

### Силовой штамп (v0.3)

`force = P` (> 0): задана равнодействующая реакции, уровень штампа ищется
из скалярного уравнения `∫r dΩ = P` (монотонного по уровню; brentq на
[касание, отрыв] с тёплым стартом МОР). Скалярные `gap`/`gap_factor` при
`force` игнорируются (warning в result.json); таблица `[contact.gap]`
осмыслена как ФОРМА штампа относительно искомого уровня:
Δ(x, y) = level + shape(x, y). `Result.level` — найденный уровень,
`Result.force_total` — фактическая ∫r. P выше достижимого максимума
(касание) — ошибка с указанием диапазона. Точность замыкания
позиционной и силовой постановок требует сходимости МОР (tol достижим,
а не «по лимиту итераций») — см. tests/test_force_stamp.py.

## plate2

Контакт двух пластин (v0.3): `contact.target = "plate2"` + секция `[plate2]`
(bc и load обязательны; geometry/model/discretization по умолчанию — от
первой пластины). Зазор между пластинами — те же `gap`/`[contact.gap]`
(`gap = 0.0` — касание). Итерация МОР на разности прогибов
`r ← [r + β((w₁−w₂) − Δ)]₊`, нагрузки `q₁ − r` и `q₂ + r`;
сходимость — теорема 4 с G = G₁ + G₂. Узлы контакта — квадратура первой
пластины на пересечении планформ (ω₂ > 0); межсеточного переноса нет:
прогиб второй вычисляется в узлах первой прямо через её структуру, вклад
реакции в нагрузку второй интегрируется по родной квадратуре реакции.
Контактное условие пары в v0.3 — КЛАССИЧЕСКОЕ (по срединным плоскостям);
геометрически при этом контактируют нижняя лицевая ВЕРХНЕЙ пластины и
верхняя лицевая НИЖНЕЙ (валидатор отклоняет theory=ktn для пары с этим
пояснением). Канон напряжений (§19, NOTES): у верхней пластины реакция
приходит на нижнюю грань (q⁻₁ = r), у нижней — на верхнюю (q⁺₂ = r);
fields.npz содержит обе шестёрки σ (вторая — с суффиксом «2»).
Ограничения v0.3: theory=classic; силовое управление парой — направление развития.

```toml
[contact]
enabled = true
target = "plate2"
gap = 0.0                # касание

[plate2]
[plate2.bc]
type = "clamped"
[plate2.load]
type = "uniform"
q0 = 1.0
```

## discretization

`p` — степень Чебышёва по каждой оси (N = (p+1)² функций); `Q` — узлов
квадратуры Гаусса–Лежандра по каждой оси (маска ω > 0 отбирает внутренние);
`grid_n` — фоновая сетка для полей вывода и фигур.

Сетка вывода `grid_n` влияет ТОЛЬКО на вывод полей и фигур: числа решения
(w_max, реакция, комплементарность) считаются на узлах квадратуры и от
`grid_n` не зависят. Grid-зависима лишь ДИАГНОСТИКА топологии зоны
контакта (число связных компонент по сетке): при сравнении топологий
фиксируйте `grid_n`. Изменить сетку без правки файла: флаг
`plate-solve … --grid N`; программно — `solve(problem, grid_n=…)`,
`problem.with_discretization(grid_n=…)`; мгновенно после расчёта (без
повторного МОР) — `result.regrid(N)`.

## verify

`reference` — эталон: `analytic` (модельно-согласованная аналитика — circle,
annulus, point на круге), `mms` (изготовленное решение — rectangle, circle),
`fem` (независимый scikit-fem), `none`. `cross_1d` — дополнительная сверка
с 1D-Ритцем по радиусу (только осесимметричные постановки). `tol` — допуск
ворот. `model_gap` — печатать строку «истинный Кирхгоф» для документирования
модельной погрешности мягкого шарнира (NOTES §8); в допуск не входит.

Что покрывает `reference = "analytic"` (включая ФАБРИКУ автогенерации
с самосертификацией — plate_solver.analytic_auto; каждый эталон
проверяется подстановкой в уравнение и КУ, ряды — контролем остатка):

| Геометрия | КУ | Нагрузка | Эталон |
|---|---|---|---|
| circle | soft_hinge / clamped | uniform | замкнутые формулы |
| circle | soft_hinge / clamped | point в центре | замкнутые формулы (soft — модельный предел ν→1, NOTES §18) |
| annulus | soft_hinge / clamped (обе кромки) | uniform | система 4×4 |
| rectangle | soft_hinge или mixed все hinge | uniform | ряд Навье |
| rectangle | soft_hinge или mixed все hinge | patch (прямоугольная зона) | ряд Навье, замкнутые коэффициенты; сравнение в центре зоны |
| rectangle | soft_hinge или mixed все hinge | point (произвольная точка) | ряд Навье; сравнение в точке приложения |
| rectangle | mixed: hinge-пара + clamped-пара | uniform | ряд Леви (симметричный) |
| rectangle | mixed: hinge-пара + hinge/clamped кромки | uniform | ряд Леви, константы 4×4 на моду (фабрика) |
| rectangle | mixed: hinge-пара + free/clamped/hinge кромки (SFSF, SFCF, …) | uniform | ряд Леви-free: M_y = 0, V_y = 0 на модах (фабрика) |
| rectangle | mixed с free (консоль CFFF и др.) | uniform | fem: Кирхгоф (Морли), сравнение в точке свободной кромки |

Вне таблицы (L, compose, контактные задачи, patch на кривых зонах) —
`mms` | `fem` | `none`: свободная граница контакта на произвольной
области замкнутых решений не имеет.

## output

`dir` — каталог результатов (result.json: снимок Problem, git-hash, версии
зависимостей, warnings; фигуры при `figures = true`). Каталог `results/`
не коммитится (кроме замороженного `results/golden/`).

Управление фигурами и отчётом из CLI: `--figures` форсирует
`output.figures = true`; `--fig-format png,pdf` — форматы (по умолчанию
оба, png — 300 dpi); `--surface mid|top|bottom` — какую поверхность
рисовать на w-фигуре (лицевые — при `theory = "ktn"`, NOTES §21);
`--out DIR` переопределяет `dir`; `--report` пишет одностраничный
`report.md` (постановка, сводные числа, verify-таблица, фигуры);
`--check` только валидирует постановку (ничего не считает).

## Ошибки, FAIL и exit-коды

- **Ошибка схемы/несовместимость** — `CaseError` с шаблоном «ключ:
  получено X, ожидалось Y, см. docs/CASE_SCHEMA.md#секция»; CLI печатает
  её в stderr и возвращает exit 1. Это ошибки ПОСТАНОВКИ — исправляются
  правкой case-файла.
- **`plate-verify` FAIL** — постановка валидна и решена, но отличие от
  эталона превысило `verify.tol`: exit 1, таблица показывает rel по
  каждому эталону. Это сигнал о ТОЧНОСТИ (дискретизация мала, допуск
  нереалистичен или регресс метода).
- **exit 0** — решение получено; при `[verify]` — все гейтуемые эталоны
  в допуске.
- `plate-solve --check case.toml` — только валидация: exit 0/1 без
  расчёта (для пользовательских CI).
- Предупреждения (не ошибки) копятся в `Result.warnings` и печатаются
  CLI; например, скалярный `gap` при силовом режиме игнорируется.

Требование к зазору: Δ > 0 НА ОСНОВАНИИ (поле зазора обязано быть
строго положительным под зоной контакта; `gap = 0` допустим только
для пары пластин — касание срединных плоскостей).

## Оси, «верх/низ» и поверхности (сводная таблица)

Конвенция пакета: прогиб w > 0 и нагрузка q > 0 направлены ВНИЗ, ось z —
тоже вниз (NOTES §0/§19). Отсюда однозначно:

| Вопрос | Ответ |
|---|---|
| Куда действует q > 0 | вниз (на ВЕРХНЮЮ лицевую, z = −h/2) |
| Где основание/штамп | СНИЗУ (у нижней лицевой, z = +h/2); реакция r ≥ 0 действует вверх |
| Какая лицевая контактирует | нижняя (z = +h/2); при theory=ktn зазор проверяется её прогибом (NOTES §21) |
| Что значит `sx_top`/`sx_bot` в fields.npz | напряжения на верхней (z = −h/2) / нижней (z = +h/2) лицевой; обжатие ν/(1−ν)·q_n на стороне давления |
| Что значит `w_top`/`w_bot`/`dh` в fields.npz | при theory=ktn: w_bot — прогиб контактирующей (нижней) лицевой по канону КТН (NOTES §21.1), w_top ≡ w (срединная), dh = w_bot − w — смещение контактирующей лицевой (в зоне dh < 0); classic — все совпадают с w |
| Что выбирает `--surface` (CLI) | поверхность на w-фигуре: `mid` (срединная, по умолчанию) / `top` / `bottom` |
| Пара пластин | первая — ВЕРХНЯЯ (q⁻₁ = r), вторая — НИЖНЯЯ (q⁺₂ = r); поля второй — суффикс «2» |
| Физический контроль | в пролёте при q > 0: низ растянут (σ_bot > 0), верх сжат; в зоне контакта dh < 0 (обжатие) |

## Новый случай за ≤ 5 строк

Шаблон: `plate-solve --new annulus` → `annulus.toml` со всеми секциями
и комментариями. Для перехода, например, к кольцу другого размера с мягким
шарниром достаточно изменить 3–5 строк:

```diff
 [geometry]
 kind = "annulus"
-a = 1.0
-b = 0.4
+a = 2.0
+b = 0.8

 [bc]
-type = "clamped"
+type = "soft_hinge"
```
