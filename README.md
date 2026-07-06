# plate-solver

Комплекс программ для расчёта **изгиба и одностороннего контакта упругих
пластин произвольного очертания**. Защищаемый результат кандидатской
диссертации (специальность 1.2.2 «Математическое моделирование, численные
методы и комплексы программ», физико-математические науки).

Триада «Модель — Алгоритм — Программа» (по Самарскому):

- **Модель** — изгиб пластины (расщепление бигармоники `D·Δ²w = q̃` на две
  задачи Пуассона) и поправки уточнённой теории типа Кармана–Тимошенко–Нагди
  (поперечный сдвиг + поперечное обжатие, `ktn.py`);
- **Геометрия** — граница произвольной области R-функциями В. Л. Рвачёва
  (ω(x, y) = 0); структура решения `w = ω·Φ` удовлетворяет краевым условиям
  тождественно;
- **Численный метод** — Ритц на базисе `ω·Φ` (Φ — полиномы Чебышёва),
  матрица Ритца факторизуется один раз (Холецкий);
- **Контакт** — метод обобщённой реакции (МОР) для одностороннего контакта
  со свободной границей: `r ← max(0, r + β(w − Δ))`.

Статус: **работает**. Изгиб (мягкий шарнир и жёсткое защемление), контакт
МОР на неканонических областях (L-форма с входящим углом), поправки КТН,
1D-задел (балка-полоса, штамп) и верификационная лестница — всё покрыто
тест-воротами (см. таблицу ниже) и воспроизводится одним прогоном.

## Установка

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # + ".[dev,fem]" для независимой МКЭ-верификации
```

## Быстрый старт через case-файлы (v0.2)

Постановка задачи описывается TOML-файлом (схема — [docs/CASE_SCHEMA.md](docs/CASE_SCHEMA.md)):

```bash
plate-solve --new annulus        # шаблон annulus.toml (закомментирован)
plate-solve annulus.toml         # решить: result.json + фигуры
plate-verify annulus.toml        # таблица «эталон | значение | rel | статус», exit 0/1
plate-verify annulus.toml --sweep p=2:12:2   # сходимость: md + csv + png
plate-ladder cases/ci            # каталог случаев → сводный md с провенансом
```

| Команда | Что делает |
|---|---|
| `plate-solve case.toml` | решает постановку, пишет `result.json` (+фигуры) |
| `plate-solve --new KIND` | шаблон case-файла (circle/rectangle/L/annulus) |
| `plate-verify case.toml` | сверка с эталонами `[verify]`, exit 0/1 по tol |
| `--sweep p=2:12:2`, `Q=…` | свип дискретизации (md+csv+png), в solve и verify |
| `plate-ladder КАТАЛОГ` | прогон реестра случаев, сводка с провенансом |

Нагрузки в схеме: `uniform` (q0), `patch` (q0 + зона тем же геометрическим
языком), `point` (P, x0, y0 — регуляризованное пятно eps; истинная δ
сознательно не вводится — см. docs/NOTES.md §18). Произвольная гладкая
f(x, y) — через API (`f_values` в узлах квадратуры, пример в CASE_SCHEMA).

Реестры: [cases/ladder/](cases/ladder) — полные ступени верификации
(лестница), [cases/ci/](cases/ci) — лёгкие копии, каждая автоматически
является CI-тестом. Блок-схема диспетчера — [docs/dispatch_flow.md](docs/dispatch_flow.md).

## Быстрый старт через API: контакт L-формы с жёстким основанием

```python
from plate_solver import Config, viz
from plate_solver.contact import solve_contact
from plate_solver.geometry import make_L

cfg = Config(h=0.06, p=10, Q=120, Delta=5.0e-5, max_iter=8000)
res = solve_contact(cfg, make_L(side=1.0, cut=0.5))  # изгиб + односторонний контакт (МОР)
print(f"итераций МОР: {res.iters}, узлов контакта: {int((res.r_nodes > 0).sum())}")
print(f"комплементарность: {res.comp_residual:.1e}, перелёт зазора: {res.gap_overshoot:.1e}")
viz.plot_contact_summary(cfg, res).savefig("contact_L.png", dpi=150)
```

## Числа ↔ скрипты ↔ тесты

Все числа главы 4 доклада получаются ОДНИМ прогоном
`python scripts/run_golden.py` (из корня; результат — `golden_results.md`,
фигуры — `figures/`) и защищены тест-воротами:

| Результат | Ключевые числа | Скрипт | Тест-ворота |
|---|---|---|---|
| Табл. 4.1 — верификация на круге | ошибка < 0.1 %; модельный разрыв мягкого шарнира 26.42 % | `run_circle.py` | `test_plate_circle.py` |
| Верификация L-формы (МКЭ) | RFM↔FEM-Marcus 2.64 %; парадокс Сапонджяна 54.86 % | `run_lshape_verify.py` | `test_lshape.py` |
| Табл. 4.2 — контакт МОР | 8000 итер.; 67/10800 узлов; r_max = 112.58; комплементарность 8.59e-2 | `run_lshape_contact.py` | `test_contact.py` |
| Табл. 4.3 — поправки КТН | пик реакции ×0.095; узлы ×7.75; w_max +22.3 % | `run_ktn.py` | `test_ktn.py` |
| 1D-штамп (эталон Maple) | согласие 2.4 % (L²) | `run_stamp_1d.py` | `test_stamp.py` |
| Верификация 1D↔2D (круг) | 1D↔2D↔аналитика, 0.1 % | `run_circle_1d_2d.py` | `test_circle_1d_2d.py` |
| Вклад R-функций (vs штраф) | 1 % при N=9 против N=25; лучше cond | `run_rvachev_vs_penalty.py` | `test_rvachev_vs_penalty.py` |
| Лестница верификации изгиба | машинная точность → эталоны | `run_ladder_*.py` | `test_ladder.py` |

## Запуск

```bash
pytest -m "not big and not fem"     # быстрые ворота (~1 мин)
pytest                              # все ворота (big: Q≥1024; fem: scikit-fem)
python examples/circular_plate.py   # минимальный пример (аналитика)
python scripts/run_golden.py        # золотой прогон главы 4 (из корня)
ruff check .                        # стиль
```

`pytest` работает и без установки пакета: пути `src/` и `scripts/` прописаны
в `pyproject.toml`.

Архив для ревью (исходники + лёгкий снимок полей + свежий pytest-лог)
собирается одной командой: `scripts/make_review_zip.sh` (по умолчанию —
в неотслеживаемый `private/review/`).

## Структура

```
src/plate_solver/    модули плоско (имена соответствуют главе 3 доклада):
  geometry, basis, quadrature, assembler   область (R-функции) и дискретизация
  poisson, plate, clamped, radial          решатели изгиба (Ритц)
  contact, ktn                             контакт (МОР) и поправки КТН
  mor1d, green1d, stamp, stamp_ritz        1D-задел (балка-полоса, штамп)
  analytic, ladder, penalty, verify_fem    эталоны и верификация
  viz                                      графика
  data/                                    эталон Maple (см. data/README.md)
scripts/             run_*.py — расчётные серии; golden_config.py — единый конфиг
tests/               тест-ворота (маркеры big и fem — см. pyproject)
examples/            минимальные воспроизводимые примеры
docs/NOTES.md        тонкости и подводные камни (журнал заметок)
```

## Лицензия и цитирование

MIT (см. `LICENSE`). Для ссылок — `CITATION.cff`.
