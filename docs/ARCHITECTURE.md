# Архитектура

Обнародованный результат: [github.com/AnatoDu/plate-solver](https://github.com/AnatoDu/plate-solver),
архивная запись Zenodo (постоянный идентификатор проекта, concept DOI):
[10.5281/zenodo.21218627](https://doi.org/10.5281/zenodo.21218627).

## Слои (направление зависимостей — сверху вниз)

1. **Постановка и CLI** — `problem` (валидатор case-файлов, `CaseError`),
   `cli` (plate-solve / plate-verify / plate-ladder), `references`
   (верификация как свойство постановки).
2. **Диспетчер** — `dispatch`: `solve(problem) → Result`; маршрутизация
   по решателям и теориям (блок-схема — [dispatch_flow.md](dispatch_flow.md)).
3. **Контакт** — `contact` (линейный МОР: `ContactMOR`, `TwoPlateMOR` над
   classic/КТН-поправками), `contact_nl` (НЕЛИНЕЙНЫЙ контакт МОР поверх
   полной КТН: `NonlinearContactMOR`, `NonlinearTwoPlateMOR`; схемы
   nested|merged, выбор нормировки шага `gain_mode`), `diagnostics`
   (размер, топология и сила зоны контакта), `contact_face`
   (ЭКСПЕРИМЕНТАЛЬНЫЙ задел: лицевое условие Синьорини на самостоятельном
   поле лицевого прогиба; вне публичного фасада, API нестабилен).
4. **Единая модель теорий** — `theory` (`TheoryParams`; пресеты
   `classic`/`karman`/`ktn_linear`/`ktn_full`; морфинг уточнения),
   `ktn_solver` (`KTNSolver` — ОДИН решатель для всех пресетов, редукции
   точны по построению), `ktn` (линейные поправки сдвига/обжатия,
   напряжения лицевых поверхностей), `ktn_full` (полная нелинейная КТН),
   `membrane` (геометрическая нелинейность Кармана), `faces` (лицевые
   величины первым классом).
5. **Решатели изгиба (ядро)** — `plate` (расщепление, мягкий шарнир),
   `clamped` (прямой Ритц, защемление; `MixedRectPlate` — смешанные КУ и
   свободный край), `poisson` (кирпич расщепления), `radial` (1D по
   радиусу, осесимметрия).
6. **Геометрия и дискретизация** — `geometry` (R-функции, система R0),
   `rfunctions` (R-операции `r_and`/`r_or`/`r_diff` — многосвязность),
   `basis` (Чебышёв), `quadrature` (гауссова квадратура с маской ω > 0),
   `assembler`.
7. **Эталоны и верификация** — `analytic` (ручные замкнутые решения),
   `analytic_auto` (фабрика с самосертификацией), `benchmarks`
   (кармановские эталоны Hencky/Way/Levy), `ladder` (верификационная
   лестница, MMS), `verify_fem` (независимый МКЭ, scikit-fem), `config`.
8. **Вывод** — `viz` (фигуры, `replot` из fields.npz), `export`
   (результирующие усилия и запись сеточных полей в LEGACY-VTK).
9. **1D-задел** — `green1d`, `mor1d`, `stamp`, `stamp_ritz`,
   `strip_contact`, `penalty`, `problems` (историческое ядро 1D-контакта
   и сравнений; используется эталонными воротами).

## Граф импортов (фактический; генератор — scripts/import_graph.py)

```mermaid
flowchart TD
    analytic --> ladder
    clamped --> basis
    clamped --> geometry
    clamped --> ladder
    clamped --> quadrature
    clamped --> verify_fem
    cli --> dispatch
    cli --> problem
    cli --> references
    contact --> config
    contact --> ktn
    contact --> plate
    contact_face --> config
    contact_face --> faces
    contact_face --> ktn
    contact_face --> ktn_solver
    contact_face --> membrane
    contact_nl --> config
    contact_nl --> diagnostics
    contact_nl --> ktn_solver
    dispatch --> clamped
    dispatch --> config
    dispatch --> contact
    dispatch --> faces
    dispatch --> ktn
    dispatch --> ktn_full
    dispatch --> ladder
    dispatch --> membrane
    dispatch --> plate
    dispatch --> problem
    faces --> ktn
    geometry --> problem
    ktn_full --> basis
    ktn_full --> faces
    ktn_full --> membrane
    ktn_full --> quadrature
    ktn_solver --> basis
    ktn_solver --> ktn_full
    ktn_solver --> quadrature
    ktn_solver --> theory
    ladder --> clamped
    ladder --> geometry
    membrane --> basis
    membrane --> clamped
    membrane --> ladder
    membrane --> quadrature
    mor1d --> green1d
    penalty --> basis
    penalty --> config
    penalty --> plate
    penalty --> quadrature
    plate --> basis
    plate --> poisson
    plate --> quadrature
    poisson --> assembler
    problem --> config
    problems --> ktn
    quadrature --> geometry
    references --> analytic_auto
    references --> clamped
    references --> dispatch
    references --> geometry
    references --> ladder
    references --> problem
    references --> radial
    references --> verify_fem
    stamp --> green1d
    stamp --> mor1d
    stamp_ritz --> mor1d
    stamp_ritz --> stamp
    theory --> faces
    verify_fem --> plate
    viz --> config
    viz --> contact
    viz --> plate
```

Часть рёбер — ленивые импорты внутри функций (тяжёлые зависимости:
matplotlib в `viz`, scikit-fem в `verify_fem`, sympy-фабрика в
`references`); пакет загружается и без них.

## Принцип «плоский пакет + фасад»

Пакет НАМЕРЕННО плоский (38 модулей в `src/plate_solver/`): стабильность
листинга исходного текста к регистрации, короткие ссылки из документации
(NOTES/THEORY ссылаются на имена файлов), обозримость для стороннего
читателя. «Нелоскость» достигается фасадом: `__init__.py` экспортирует
публичные точки секциями (геометрия / решатели / контакт / верификация /
ввод-вывод / графика), а навигацией служит граф выше и docs/API.md.
Физическая перегруппировка по подпакетам сознательно отложена (вне
freeze): она бы поменяла все ссылки без выигрыша в ясности.

## Как устроен регресс (для стороннего читателя)

Три пояса защиты:

1. **Реестр случаев = ворота.** Каждый файл `cases/ci/*.toml`
   автоматически становится CI-тестом (`tests/test_ci_cases.py`,
   `plate-verify` exit 0); тяжёлые ступени — `cases/ladder/*.toml`
   (маркер `big`). Допуски заморожены протоколом «потолок задан → факт
   × 3» с диагнозом природы ошибки в комментарии case-файла.
2. **Эталонный отчёт под хеш-воротами.** Единый прогон
   (`scripts/run_reference.py`, см. README) порождает отчёт с ключевыми
   числами; файл заморожен SHA-256 — любое изменение чисел = красный
   тест, а не тихое обновление. Обновление отчёта — только осознанным
   коммитом с обоснованием в CHANGELOG.
3. **Регресс-снимки.** `cases/baselines.json` — информационные снимки
   (профили, топология зон, свипы) с кросс-платформенными допусками;
   каждая запись самодокументирована.

Плюс независимые пояса корректности: аналитика (в т.ч. фабрика с
самосертификацией — эталона без проверенного сертификата не существует),
MMS, независимый МКЭ, 1D↔2D-сверка на осесимметрии, sympy-тождества
формул (NOTES §19–21).

## Роли эталонов

| Эталон | Что проверяет | Где |
|---|---|---|
| analytic | модельно-согласованные замкнутые решения; фабрика покрывает канонику автоматически | `analytic`, `analytic_auto` |
| mms | метод на изготовленном решении той же дискретизацией | `ladder` |
| fem | независимая дискретизация (Морли/Аргирис/P2) | `verify_fem` |
| cross_1d | осесимметрия: 1D-Ритц по радиусу | `radial` |
| golden/reference | воспроизводимость всей серии одним прогоном | `scripts/` |
