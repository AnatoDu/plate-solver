# PROGRESS — журнал фазы 2 (слой постановки задачи)

Журнал для пост-фактум просмотра автором (TODO_PHASE2, автономный режим,
п. 4). Каждая задача дописывает раздел: что сделано, откалиброванные числа,
применённые fallback'и, отступления от TODO.

## Отступления от TODO, действующие на всю фазу

- **PR вместо merge.** Открыть pull request на GitHub из рабочей сессии
  невозможно: репозиторий приватный, доступа из сессии нет (gh-аккаунт на
  машине не имеет прав на AnatoDu/plate-solver). Применён fallback фазы 1:
  каждая задача — ветка от `main` + зелёные тесты + `merge --no-ff` локально.
  Автор может ретроспективно смотреть по merge-коммитам (история эквивалентна
  PR-ам).
- **Расположение снимка golden.** TODO ссылается на
  `results/golden/golden_results.md`; на момент старта фазы файл жил в корне.
  Первым коммитом P0 создан замороженный снимок `results/golden/` (копия
  корневого golden v0.1.0, SHA-256 = 7a646bd8…), .gitignore получил
  исключение для этого каталога. Живой корневой файл хеш-воротами обязан
  совпадать со снимком (run_golden.py и golden_config.py не тронуты).

## P0. Схема случая и слой Problem

- P0 (коммит 1): хеш-ворота golden — `tests/test_golden_hash.py`
  (SHA-256 снимка + байтовое равенство корневого golden снимку), снимок
  `results/golden/golden_results.md`, исключение в .gitignore. Ворота
  входят в общий pytest и работают в CI (workflow фазы 1 без изменений).
- P0.1+P0.2: `problem.py` — frozen-датаклассы Problem/секций, CaseError,
  tomllib (requires-python поднят до >=3.11), валидатор со схемными
  якорями и защитой от опечаток; несовместимости v0.2; ограда compose
  (3 операции / 2 примитива / глубина ≤ 3 / ≤ 7 узлов, difference бинарна).
  Дефолты физики НЕ дублируются: None → дефолт Config в to_config().
- P0.3: `docs/CASE_SCHEMA.md` — пример, таблица ключей (тип/дефолт/
  допустимые/исполнитель), несовместимости, «неравномерная нагрузка»
  (f(x,y) — только API), «новый случай = правка ≤ 5 строк» (diff).
- P0.4: `cli.py` + [project.scripts] plate-solve; `--new circle|rectangle|
  L|annulus` пишет закомментированный шаблон с самопроверкой (шаблон
  обязан парситься валидатором); перезапись запрещена. Решение case-файлов
  отложено на P4 (exit-код 2 с пояснением).
- Уточнения схемы, введённые в P0 (разрешено: «ключи, перечисленные в P0»,
  определяются здесь): [model] E/nu/h; [contact] beta/max_iter/tol/stop
  (нужны cases P3.7: mor_iter как в golden); [load.zone]/[contact.zone] —
  GeometrySpec тем же языком. Проверки «≥ 20 узлов в зоне» отнесены к
  диспетчеру (P2): требуют квадратуры, статически невычислимы.
- Приёмка P0: 22 юнит-теста (валидатор, шаблоны, round-trip
  TOML→Problem→to_config), хеш-ворота зелёные, entry point работает
  (`plate-solve --new annulus` из venv).

## P1. Реестр геометрий

- P1.1: `r_not(f) = −f`, `r_diff(f1, f2) = r_and(f1, −f2)`,
  `make_annulus(a, b)` (R-разность кругов, bbox (−a, a)²); `circle_expr`
  получил центр (cx, cy) — нужен примитивам compose. Формулы в докстрингах.
- P1.2: `make_compose(tree)` — дерево TOML → Domain; ops
  union|intersect|difference (difference бинарна), примитивы
  circle(a, cx, cy) / rectangle(x1..y2); bbox: union — объединение,
  intersect — пересечение (пустое ⇒ ValueError), difference — bbox первого.
  Ограда — ЕДИНЫЙ валидатор `problem.validate_compose_tree` (публичная
  обёртка, переиспользуется geometry ⇒ нет дублирования пределов).
- P1.3 ворота: (а) площадь кольца по маске против π(a²−b²): rel < 3/Q,
  убывает (Q=128, 256); (б) знаки ω кольца и compose (тело/дырка/снаружи/
  границы); (в) символьный ∇ω против центральной разности в 20 случайных
  внутренних точках (rel < 1e-6, annulus и compose); (г) smoke: compose
  глубины 3 (6 узлов) через PoissonSolver — cond конечен, v > 0;
  (д) глубина 4 → CaseError; бонус: символьная проверка тождеств
  r_not/r_diff (sympy.simplify == 0).

## P2. Диспетчер

- P2.1: `dispatch.py` — solve(problem)→Result; маршрутизация по докстрингу
  (soft_hinge→PlateBending, clamped→ClampedPlate, contact→ContactMOR c
  foundation_mask из [contact.zone], ktn→KTNParams; в чистом изгибе —
  corrected_deflection при r=0). Нагрузки: uniform/patch/point;
  point — авторасширение eps до ≥20 узлов (warning в Result), patch/зона
  контакта <20 узлов — CaseError. build_domain — реестр геометрий.
- ContactMOR расширен опциональным load_values (нагрузка в узлах для
  patch/point из диспетчера); дефолт None — путь и арифметика прежние
  (golden: контактные и хеш-ворота зелёные).
- P2.2: Result (frozen) — w_max, cond(A), поля на сетке, ContactResult,
  Δ/w_free/eps_eff, warnings, тайминги; save(dir) → result.json (снимок
  Problem+Config, git-хеш, версии numpy/scipy/sympy, warnings) + фигуры viz.
- Решения, принятые самостоятельно (в TODO не оговорено):
  (а) theory=ktn + bc=clamped отклоняется CaseError — КТН-кривизна берётся
  из (P1) расщепления, у прямого Ритца поля M нет (интерфейс Δw через
  гессиан структуры — фаза 3); (б) центр (cx, cy) есть только у
  compose-примитива circle — верхнеуровневый circle центрирован (ограда
  «новых ключей после P0 не вводим»), смещённые зоны — rectangle/compose.
- Приёмка P2: эквивалентность прямому API — circle/soft и L/contact
  (rel ≤ 1e-12, включая iters и n_contact); circle/clamped против
  аналитики; rect/clamped; штамп-зона (r ≡ 0 вне зоны); ∫q̃dΩ=q0·|Ω_patch|
  (факт ~5e-3 при Q=256, ворота 2e-2); результанта point=P (пятно из
  ~20 узлов — ворота 0.35, дефолтное eps — без предупреждений);
  КТН-поправка изгиба > 0; result.json с провенансом.

## P3. Верификация как свойство постановки

- P3.2: analytic.annulus_uniform — общее решение кольца, 4×4 для
  clamped / soft / true_ss; sympy-подстановка в D·ΔΔw=q — машинный нуль;
  КУ до 1e-12. soft = ТОЧНОЕ решение расщепления (проверено выводом).
- P3.3: radial.py — RadialClampedAnnulus (ω²Φ, ω=(a−r)(r−b)),
  RadialPoissonAnnulus (ω·Φ), solve_radial_soft_hinge_annulus; базис —
  обычный Чебышёв на [b,a] (чётность нужна только при b=0 — прежний путь,
  big-регрессия test_circle_1d_2d зелёная). Ворота: 1D↔analytic
  rel < 1e-8 при p=16 (clamped и soft).
- P3.1: references.py — resolve_reference / verify_result / VerifyReport
  (таблица «эталон | значение | rel | статус», ok по гейтуемым строкам).
  Модельная согласованность NOTES §8; cross_1d (p=16, nq=400);
  model_gap-строка вне допуска (у clamped не создаётся). Отказы: analytic
  для rectangle/L, fem→P3.6, mms→P3.8, point→P3.5, контакт→инварианты P3.7.
- P3.4 (самокалибровка, потолок rel ≤ 1e-2 ПРОЙДЕН): факты Q=1024, p=10:
  annulus_soft — 2D↔analytic 1.140e-3, 2D↔1D 1.140e-3 (ожидание ~3e-3
  подтверждено), model_gap 9.70e-2 (инфо); annulus_clamped — 1.610e-3 /
  1.610e-3. Заморожено факт×3: tol = 3.5e-3 (soft), 4.9e-3 (clamped) в
  cases/annulus_{soft,clamped}.toml. Ворота — tests/test_annulus_cases.py
  (big: полный Q=1024; light: валидация case-файлов). Интерпретация:
  «sweep p=2..10» из TODO — инструмент P4.2 (plate-solve --sweep),
  калибровка ворот выполнена на рабочей точке p=10.
