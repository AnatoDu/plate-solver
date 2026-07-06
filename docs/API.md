# API — публичные точки входа

Пакет плоский; всё публичное перечислено здесь по модулям (`__all__`).
Главный путь пользователя: `Problem` → `solve` → `Result` (плюс CLI —
см. README). Мини-примеры минимальны; больше — `examples/` и `notebooks/`.

## Постановка и диспетчер

### problem — валидатор case-файлов

- `Problem` — неизменяемая постановка. `Problem.from_toml(path)` /
  `Problem.from_dict(d)`; `p.to_config()` → `Config`.
- `CaseError` — диагностика схемы: «ключ: получено X, ожидалось Y,
  см. docs/CASE_SCHEMA.md#секция».
- Спецификации секций: `GeometrySpec`, `BCSpec`, `LoadSpec`, `ModelSpec`,
  `ContactSpec`, `GapSpec`, `Plate2Spec`, `DiscretizationSpec`,
  `VerifySpec`, `OutputSpec`.
- Реестры и константы: `GEOMETRY_KINDS`, `GAP_KINDS`, `MIN_ZONE_NODES`,
  `validate_compose_tree` (ограда мини-языка compose).

```python
from plate_solver.dispatch import solve
from plate_solver.problem import Problem

res = solve(Problem.from_toml("cases/ci/circle_soft.toml"))
print(res.w_max)
```

### dispatch — расчёт постановки

- `solve(problem)` → `Result` — маршрутизация по решателям
  (docs/dispatch_flow.md).
- `Result` — скаляры (`w_max`, `scalars()`), поля на сетке (`w_grid`,
  `moments_on_grid()`, `faces_on_grid()` — w_top/w_bot/dh по NOTES §21),
  контакт (`contact`), выгрузка `save(dir)` (result.json + fields.npz
  + фигуры), `save_fields(path)`; `regrid(N)` — мгновенное уплотнение
  сетки вывода без пересчёта (МОР не перезапускается).
- Сетка вывода: `solve(problem, grid_n=…)` и
  `Problem.with_discretization(p=…, Q=…, grid_n=…)` — программные
  override'ы; CLI — флаг `--grid N`. На числа решения не влияют.
- `build_domain(spec)` — GeometrySpec → `Domain` (реестр геометрий).

### config — численные параметры

- `Config` — материал (E, nu, h → D), дискретизация (p, Q, grid_n),
  контакт (Delta, beta, max_iter, tol, stop). Единственный источник
  дефолтов физики.

## Геометрия (R-функции)

- `x`, `y` — sympy-символы; `Domain(omega_expr, bbox)` — область
  ω(x, y) ≥ 0; `BBox`.
- Операции системы R0: `r_and`, `r_or`, `r_not`, `r_diff`.
- Примитивы-выражения: `circle_expr`, `rectangle_expr`.
- Фабрики областей: `make_circle`, `make_rectangle`, `make_L`,
  `make_annulus`, `make_compose` (дерево операций из case-файла).

## Решатели изгиба

- `plate.PlateBending` — мягкий шарнир: расщепление бигармоники на две
  задачи Пуассона; `from_config(domain, cfg)`, `solve_uniform(q0)`,
  `deflection(cw, X, Y)`; протокол контакта `solve(f)` / `w_at_quad` /
  `lap_w_at_quad` / `coeffs_w`.
- `poisson.PoissonSolver` — кирпич расщепления (структура ω·Φ, кэш
  матриц структуры).
- `clamped.ClampedPlate` — защемление: прямой Ритц по (D/2)∫(Δw)² на
  структуре ω²Φ; та же сигнатура протокола.
- `clamped.MixedRectPlate` — смешанные КУ на прямоугольнике
  (clamped|hinge|free по сторонам, полная билинейная форма);
  `solve_uniform()`, `deflection`, `moments_at`, `w_max_on_grid`.
- `clamped.assemble_biharmonic_full` — полная билинейная форма
  a(w, v) = D∬[ΔwΔv − (1−ν)(…)] (NOTES §20).
- Служебные fem-обёртки защемления (Аргирис, scikit-fem):
  `clamped.solve_clamped_fem`, `clamped.clamped_fem_circle`,
  `clamped.clamped_fem_lshape`, контейнер `clamped.ClampedFem`.
- `poisson.CACHE_NM_MAX` — порог включения кэша матриц структуры
  (N·M ≤ 5·10⁷; выше — экономия памяти важнее скорости).

## Контакт (метод обобщённой реакции)

- `contact.solve_contact(cfg, domain, foundation_mask=None, gap=None,
  ktn=None)` — фасад: изгиб + односторонний контакт с жёстким основанием.
- `contact.ContactMOR` — итерация r ← [r + β(u − Δ)]₊ (тёплый старт
  `solve(r0)`, поле зазора, критерии останова dr|comp).
- `contact.ContactResult` — r в узлах/на сетке, зона, история сходимости,
  комплементарность.
- `contact.TwoPlateMOR` / `contact.TwoPlateResult` — контакт двух пластин
  (узлы — квадратура первой; пара ±r; без межсеточного переноса).
- `contact.sample_fields_on_grid` / `contact.sample_pair_fields_on_grid` —
  единая точка истины сэмплинга вывода на фоновую сетку (используется
  и решателем, и `Result.regrid`).

```python
from plate_solver import Config, viz
from plate_solver.contact import solve_contact
from plate_solver.geometry import make_L

cfg = Config(h=0.06, p=10, Q=120, Delta=5.0e-5, max_iter=8000)
res = solve_contact(cfg, make_L(side=1.0, cut=0.5))
viz.plot_contact_summary(cfg, res, save="contact_L.png")
```

## Уточнённая теория (КТН)

- `ktn.KTNParams` — коэффициенты поправок (сдвиг h_Ψ², обжатие h_*²);
  `contact_displacement` (= `w_face_bottom` — прогиб нижней лицевой,
  которым проверяется зазор), `corrected_deflection` (прогиб для w_max).
- `ktn.stresses_faces(Mx, My, Mxy, h, nu, q_top, q_bottom)` — шестёрка
  напряжений лицевых поверхностей (канон NOTES §19).
- `ktn.PlateMaterial`, `ktn.flexural_rigidity` — классика.

## Эталоны и верификация

- `references.resolve_reference(problem)` → список `Reference`;
  `references.verify_result(result)` → `VerifyReport` (строки `RefRow`,
  `table()`, `ok`).
- `analytic` — ручные замкнутые решения: круг (`clamped_uniform`,
  `clamped_uniform_wmax`, `simply_supported_uniform`,
  `simply_supported_uniform_wmax`, `circular_plate_clamped`,
  `circular_plate_simply_supported`, `circular_plate_soft_hinge`,
  `circular_plate_soft_hinge_wmax`), кольцо (`annulus_uniform`,
  `annulus_uniform_wmax`, `ANNULUS_BCS`), точечная сила
  (`circle_point_clamped`, `circle_point_clamped_wmax`,
  `circle_point_soft`, `circle_point_soft_wmax`,
  `circle_point_soft_moment`), прямоугольник (`navier_rect_uniform`,
  `levy_rect_uniform`), мембранные кирпичи (`disk_poisson_uniform`,
  `disk_poisson_uniform_center`, `disk_poisson_unit`).
- `analytic_auto` — ФАБРИКА эталонов с самосертификацией:
  `axisym_solution` (круг/кольцо, полином q(r), сила в центре),
  `navier_solution` (uniform|patch|point), `levy_solution`
  (hinge|clamped|free кромки), `strip_solution` (1D-полоса),
  `axisym_contact_solution` (замкнутый контакт круг+основание),
  `CertifiedSolution`, `FactoryError`.
- `ladder` — верификационная лестница: 1D-полоса (`strip_hinge_exact`,
  `strip_hinge_wmax`, `strip_clamped_exact`, `strip_clamped_wmax`,
  `Strip1DResult`, `solve_strip_1d`), синус-нагрузка (`rect_sin_load`,
  `rect_sin_exact`, `rect_sin_wmax`), ряд Навье (`navier_uniform`,
  `navier_uniform_center`), MMS (`mms_load_and_exact`,
  `mms_clamped_rect_w`, `mms_clamped_disk_w`), моменты RFM-решения
  (`bending_moments`, `bending_moments_full`).
- `verify_fem` — независимый МКЭ (scikit-fem): сетки (`lshape_mesh`,
  `annulus_mesh`, `rect_mesh`), решатели (`solve_plate_fem` —
  Кирхгоф-Морли/Marcus-P2, `solve_rect_fem_mixed` — смешанные стороны,
  включая свободные), сравнение (`FemSolution`, `FemComparison`,
  `compare_l2`, `compare_rfm_vs_fem`).

## Графика

- Из результата: `viz.replot(dir, formats, surface=mid|top|bottom)` —
  фигуры из fields.npz БЕЗ пересчёта; `viz.surface3d(X, Y, W, elev,
  azim)`; `viz.stress_maps(X, Y, stresses, components)`.
- Из объектов: `viz.plot_deflection_surface`, `viz.plot_deflection_contour`,
  `viz.plot_reaction`, `viz.plot_contact_zone`, `viz.plot_convergence`,
  `viz.plot_contact_summary` (планшет 2×2), `viz.plot_pair_summary`
  (планшет пары с w₁/w₂).
