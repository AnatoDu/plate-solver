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

## Лицевые величины первым классом (`faces.py`, v0.5.0)

Самостоятельный, переиспользуемый слой лицевых величин для ЛЮБОЙ теории
(§6): прогиб лицевой поверхности `u_c`, напряжения на гранях, интроспекция.

- `faces.FaceParams` — параметры лицевых величин с КАНОНИЧЕСКИМИ именами (§3.2):
  `h_psi_sq` (h_ψ²), `h_star_sq` (h_*²), `h_c_sq` (h_c² = h_ψ²−h_*², assert),
  `c_curv`, `mu`, `D`; `from_config`, `introspection(length)` (§6.3: h/L и
  порядок (h/L)²), `face_deflection(w, Δw, q, r, surface)` (нижняя грань — канон
  §21.1, число-в-число `ktn_linear` через `ktn()`), `mid_corrected` (для w_max).
- `faces.face_stresses(Mx, My, Mxy, h, nu, q_top, q_bottom, Nx, Ny, Nxy)` —
  полные лицевые напряжения: изгиб + обжатие (`ktn.stresses_faces`) + мембрана
  `N/h` (нелинейные теории).
- `faces.membrane_face_stress(Nx, Ny, Nxy, h)` — мембранная составляющая `N/h`.
- `Result.thickness_params()` — интроспекция параметров толщины из результата.

## Геометрическая нелинейность (теория Кармана, v0.4.0)

- `membrane.KarmanPlate` — нелинейный решатель Фёппля–Кармана:
  `from_config(domain, cfg, bc_type=..., inplane_bc=...)`, `solve(f_values)` /
  `solve_uniform(q0)` → `KarmanResult`; поля `deflection`, `deflection_at_quad`,
  `structure_at`, `moments_at`, `membrane_forces_at`, `w_max_on_grid`, `cond`.
  Итерация Пикара (ускорение Андерсона) с шагами по нагрузке; `inplane_bc`
  `immovable` (u=v=0) | `movable` (N·n=0).
- `membrane.KarmanResult` — прогиб `cw`/`w_nodes`, `w_max`, `w_max_classic`
  (линейный Кирхгоф), перемещения `cu`/`cv`, усилия `Nx`/`Ny`/`Nxy`,
  `converged`, `n_iter`, `history` (сходимость по уровням нагрузки).
- `benchmarks` — независимые эталоны большого прогиба (чистые формулы и
  замороженные таблицы с источниками; единый источник чисел для тестов и
  ноутбука). Нормировки (прил. B): `pbar(p, a, E, h)`,
  `pbar_to_pa4_over_Dh`, `pbar_to_pa4_over_64Dh`. Линейные наклоны Gate L:
  `kirchhoff_clamped_circle`, `kirchhoff_hinge_circle`,
  `kirchhoff_clamped_square`, `kirchhoff_hinge_square`. Нелинейные эталоны:
  `hencky_center_deflection` (мембрана, `HENCKY_W_COEFF`, `HENCKY_SIGMA_COEFF`),
  `way_clamped_circle` (ряды), `timoshenko_clamped_circular` и
  `timoshenko_clamped_circular_inverse` (одночлен), `levy_square_clamped`
  (ряды Фурье), `levy_square_ss_immovable` (`LEVY_SQUARE_SS_IMMOVABLE`,
  `LEVY_SQUARE_SS_MOVABLE`).

```python
from plate_solver import Config
from plate_solver.geometry import make_circle
from plate_solver.membrane import KarmanPlate
from plate_solver import benchmarks as bm

# защемлённый круг, неподвижная кромка, P̄ = 6.321 (E=h=a=1 ⇒ P̄ = q0)
cfg = Config(E=1.0, h=1.0, nu=0.3, a=1.0, q0=6.321, p=12, Q=200, n_load_steps=4)
kp = KarmanPlate.from_config(make_circle(1.0), cfg, bc_type="clamped",
                             inplane_bc="immovable")
r = kp.solve_uniform()
print(r.w_max, bm.way_clamped_circle())     # ~0.80 (Way: P̄=6.321 → w/h=0.800)
```

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
  Копируемые примеры (исполняются тестом `tests/test_api_examples.py`,
  протухнуть молча не могут):

```python
# кольцо: разные КУ на кромках, полиномиальная нагрузка q(r) = q0 + q2·r²
import numpy as np
from plate_solver.analytic_auto import axisym_solution

sol = axisym_solution(a=1.0, b=0.4, bc_outer="clamped", bc_inner="soft",
                      D=100.0, nu=0.3, q_coeffs=(4.0, 0.0, 1.5))
print("w(r=0.7):", float(sol.w(0.7, 0.0)))
print("сертификат:", max(abs(v) for v in sol.certificate.values()) < 1e-10)

# круг с СИЛОЙ в центре (q_coeffs опускаем, задаём P)
sol_p = axisym_solution(a=1.0, bc_outer="clamped", D=100.0, nu=0.3, P=5.0)
print("w(0) при силе:", float(sol_p.w(0.0, 0.0)))
```

```python
# ряд Леви: x-края шарнир; y-кромки любые из hinge|clamped|free
# (nu обязателен при free — естественные условия зависят от ν)
from plate_solver.analytic_auto import levy_solution

sfsf = levy_solution(x1=0.0, x2=1.0, y1=0.0, y2=1.0, D=100.0, q0=4.0,
                     bc_y1="free", bc_y2="free", nu=0.3)
print("w в центре:", float(sfsf.w(0.5, 0.5)))
print("w на свободной кромке:", float(sfsf.w(0.5, 0.0)))
```

```python
# замкнутый контакт: круг (мягкий шарнир) над плоским основанием с зазором
from plate_solver.analytic_auto import axisym_contact_solution

D = 100.0
w_free0 = 3 * 4.0 * 1.0**4 / (64 * D)          # прогиб центра без контакта
ref = axisym_contact_solution(a=1.0, D=D, q0=4.0, gap=0.5 * w_free0)
print("радиус зоны c:", ref.meta["c"])
print("кольцевая реакция P_ring:", ref.meta["P_ring"])
print("сертификат пройден:", all(abs(v) <= 1e-8
                                 for v in ref.certificate.values()))
```
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
