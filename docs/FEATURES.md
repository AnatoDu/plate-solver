# Матрица возможностей (автогенерация: scripts/doc_matrix.py)

Каждая возможность обязана быть описана в документации и покрыта
случаем/примером/тестом. Пустая клетка — задача (гейт
tests/test_doc_matrix.py).

## Ключи схемы case-файлов

| секция.ключ | описано | покрыто (cases) |
|---|---|---|
| bc.sides | CASE_SCHEMA.md | rect_cfff.toml, rect_cfff.toml, rect_levy.toml |
| bc.type | API.md, CASE_SCHEMA.md, README.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.bc | API.md, CASE_SCHEMA.md, MIGRATION.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.contact | ALGORITHMS.md, API.md, ARCHITECTURE.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| case.discretization | API.md, CASE_SCHEMA.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.geometry | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.load | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.model | CASE_SCHEMA.md, MIGRATION.md, THEORY.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.output | CASE_SCHEMA.md, THEORY.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.plate2 | CASE_SCHEMA.md | two_plates_equal.toml, two_plates_equal.toml, two_plates_mixed_bc.toml |
| case.verify | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| contact.beta | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.enabled | CASE_SCHEMA.md, dispatch_flow.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.force | ALGORITHMS.md, API.md, CASE_SCHEMA.md | lshape_stamp_force.toml, lshape_stamp_force.toml |
| contact.gap | API.md, CASE_SCHEMA.md, README.md | annulus_soft.toml, annulus_soft.toml, annulus_soft_contact.toml |
| contact.gap_factor | CASE_SCHEMA.md, dispatch_flow.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.max_iter | API.md, CASE_SCHEMA.md, NOTES.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.stop | API.md, CASE_SCHEMA.md, THEORY.md | test_contact.py, test_doc_policy.py, test_golden_config.py |
| contact.target | CASE_SCHEMA.md | two_plates_equal.toml, two_plates_equal.toml, two_plates_mixed_bc.toml |
| contact.tol | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| contact.zone | API.md, CASE_SCHEMA.md, dispatch_flow.md | lshape_stamp.toml, lshape_stamp.toml, lshape_stamp_force.toml |
| discretization.Q | API.md, CASE_SCHEMA.md, NOTES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| discretization.grid_n | API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| discretization.p | ALGORITHMS.md, API.md, ARCHITECTURE.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.P | ALGORITHMS.md, API.md, ARCHITECTURE.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| load.eps | ALGORITHMS.md, API.md, CASE_SCHEMA.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.q0 | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.sigma | ALGORITHMS.md, CASE_SCHEMA.md | ktn_full_circle_clamped_gaussian.toml |
| load.type | API.md, CASE_SCHEMA.md, README.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.x0 | CASE_SCHEMA.md, README.md, dispatch_flow.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.y0 | CASE_SCHEMA.md, README.md, dispatch_flow.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.zone | API.md, CASE_SCHEMA.md, dispatch_flow.md | lshape_stamp.toml, lshape_stamp.toml, lshape_stamp_force.toml |
| model.E | ALGORITHMS.md, API.md, ARCHITECTURE.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| model.h | ALGORITHMS.md, API.md, ARCHITECTURE.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| model.inplane_bc | API.md, CASE_SCHEMA.md, MIGRATION.md | karman_circle_clamped_immovable.toml, karman_circle_hencky_limit.toml, karman_square_clamped_immovable.toml |
| model.karman_max_iter | CASE_SCHEMA.md | karman_circle_clamped_immovable.toml, karman_circle_hencky_limit.toml, karman_square_clamped_immovable.toml |
| model.karman_method | CASE_SCHEMA.md | test_karman.py |
| model.karman_relax | CASE_SCHEMA.md | karman_circle_hencky_limit.toml |
| model.karman_tol | CASE_SCHEMA.md | karman_circle_clamped_immovable.toml, karman_circle_hencky_limit.toml, karman_square_clamped_immovable.toml |
| model.ktn_method | CASE_SCHEMA.md | test_karman.py, test_ktn_full.py |
| model.n_load_steps | ALGORITHMS.md, API.md, CASE_SCHEMA.md | karman_circle_clamped_immovable.toml, karman_circle_hencky_limit.toml, karman_square_clamped_immovable.toml |
| model.nu | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| model.theory | API.md, CASE_SCHEMA.md, MIGRATION.md | karman_circle_clamped_immovable.toml, karman_circle_hencky_limit.toml, karman_square_clamped_immovable.toml |
| output.dir | API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| output.figures | CASE_SCHEMA.md, README.md | golden_config.py, run_circle_1d_2d.py, run_clamped_circle.py |
| plate2.bc | API.md, CASE_SCHEMA.md, MIGRATION.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.discretization | API.md, CASE_SCHEMA.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.geometry | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.load | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.model | CASE_SCHEMA.md, MIGRATION.md, THEORY.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.cross_1d | ARCHITECTURE.md, CASE_SCHEMA.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.model_gap | CASE_SCHEMA.md, dispatch_flow.md | annulus_soft.toml, annulus_soft.toml, circle_soft.toml |
| verify.reference | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.tol | ALGORITHMS.md, API.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |

## Флаги CLI

| команда | флаг | описано | покрыто (тесты/доки) |
|---|---|---|---|
| plate-ladder | `--out` | CASE_SCHEMA.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-ladder | `--version` | README.md | doc_matrix.py |
| plate-solve | `--check` | CASE_SCHEMA.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--fig-format` | CASE_SCHEMA.md, README.md | test_stresses.py |
| plate-solve | `--figures` | CASE_SCHEMA.md, README.md | test_stresses.py |
| plate-solve | `--grid` | API.md, CASE_SCHEMA.md, README.md | test_regrid.py |
| plate-solve | `--help` | README.md | README.md |
| plate-solve | `--inplane-bc` | CASE_SCHEMA.md, MIGRATION.md | test_cli.py |
| plate-solve | `--new` | CASE_SCHEMA.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--out` | CASE_SCHEMA.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-solve | `--report` | CASE_SCHEMA.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--surface` | CASE_SCHEMA.md, README.md | CASE_SCHEMA.md, README.md |
| plate-solve | `--sweep` | README.md | 02_annulus_case.ipynb, test_cli.py |
| plate-solve | `--theory` | CASE_SCHEMA.md, MIGRATION.md | test_cli.py |
| plate-solve | `--version` | README.md | doc_matrix.py |
| plate-verify | `--fig-format` | CASE_SCHEMA.md, README.md | test_stresses.py |
| plate-verify | `--figures` | CASE_SCHEMA.md, README.md | test_stresses.py |
| plate-verify | `--grid` | API.md, CASE_SCHEMA.md, README.md | test_regrid.py |
| plate-verify | `--help` | README.md | README.md |
| plate-verify | `--inplane-bc` | CASE_SCHEMA.md, MIGRATION.md | test_cli.py |
| plate-verify | `--out` | CASE_SCHEMA.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-verify | `--surface` | CASE_SCHEMA.md, README.md | CASE_SCHEMA.md, README.md |
| plate-verify | `--sweep` | README.md | 02_annulus_case.ipynb, test_cli.py |
| plate-verify | `--theory` | CASE_SCHEMA.md, MIGRATION.md | test_cli.py |
| plate-verify | `--version` | README.md | doc_matrix.py |

## Публичные функции и классы

| модуль | имя | описано | покрыто (examples/notebooks/tests) |
|---|---|---|---|
| analytic | `ANNULUS_BCS` | API.md | test_annulus_analytic.py |
| analytic | `annulus_uniform` | API.md | test_analytic_factory.py, test_annulus_analytic.py |
| analytic | `annulus_uniform_wmax` | API.md | test_annulus_analytic.py |
| analytic | `circle_point_clamped` | API.md | 04_point_load.ipynb, test_analytic_factory.py, test_point_analytic.py |
| analytic | `circle_point_clamped_wmax` | API.md | 04_point_load.ipynb, test_point_analytic.py |
| analytic | `circle_point_soft` | API.md, dispatch_flow.md | test_analytic_factory.py, test_doc_matrix.py, test_point_analytic.py |
| analytic | `circle_point_soft_moment` | API.md | test_doc_matrix.py |
| analytic | `circle_point_soft_wmax` | API.md | test_point_analytic.py |
| analytic | `circular_plate_clamped` | API.md | run_circle_1d_2d.py |
| analytic | `circular_plate_simply_supported` | API.md | run_circle.py, run_clamped_circle.py, run_ladder_circle.py |
| analytic | `circular_plate_soft_hinge` | API.md, NOTES.md | 01_circle_api.ipynb, run_circle.py, run_circle_1d_2d.py |
| analytic | `circular_plate_soft_hinge_wmax` | API.md | 01_circle_api.ipynb, run_circle.py, run_clamped_circle.py |
| analytic | `clamped_uniform` | API.md | 06_theory_comparison.ipynb, circular_plate.py, run_clamped_circle.py |
| analytic | `clamped_uniform_wmax` | API.md | circular_plate.py, run_clamped_circle.py, run_ladder_circle.py |
| analytic | `disk_poisson_uniform` | API.md | test_doc_matrix.py |
| analytic | `disk_poisson_uniform_center` | API.md | test_doc_matrix.py |
| analytic | `disk_poisson_unit` | API.md | test_poisson_disk.py |
| analytic | `levy_rect_uniform` | API.md | test_analytic_factory.py, test_mixed_bc.py |
| analytic | `navier_rect_uniform` | API.md | test_analytic_factory.py, test_mixed_bc.py |
| analytic | `simply_supported_uniform` | API.md | test_analytic.py |
| analytic | `simply_supported_uniform_wmax` | API.md | test_analytic.py |
| analytic_auto | `CertifiedSolution` | API.md | analytic_auto.py |
| analytic_auto | `FactoryError` | API.md | test_analytic_factory.py |
| analytic_auto | `axisym_contact_solution` | API.md | test_analytic_factory.py |
| analytic_auto | `axisym_solution` | API.md | test_analytic_factory.py |
| analytic_auto | `levy_solution` | API.md | test_analytic_factory.py, test_free_edge.py |
| analytic_auto | `navier_solution` | API.md | test_analytic_factory.py |
| analytic_auto | `strip_solution` | API.md | test_analytic_factory.py |
| benchmarks | `HENCKY_SIGMA_COEFF` | API.md | test_karman.py |
| benchmarks | `HENCKY_W_COEFF` | API.md | test_karman.py |
| benchmarks | `LEVY_SQUARE_SS_IMMOVABLE` | API.md | test_karman.py |
| benchmarks | `LEVY_SQUARE_SS_MOVABLE` | API.md | test_karman.py |
| benchmarks | `hencky_center_deflection` | API.md | 06_theory_comparison.ipynb, test_karman.py |
| benchmarks | `kirchhoff_clamped_circle` | API.md | test_karman.py, test_ktn_full.py, test_unified_theory.py |
| benchmarks | `kirchhoff_clamped_square` | API.md | test_karman.py |
| benchmarks | `kirchhoff_hinge_circle` | API.md | test_karman.py |
| benchmarks | `kirchhoff_hinge_square` | API.md | test_karman.py |
| benchmarks | `levy_square_clamped` | API.md | test_karman.py |
| benchmarks | `levy_square_ss_immovable` | API.md | test_karman.py |
| benchmarks | `pbar` | API.md | test_karman.py |
| benchmarks | `pbar_to_pa4_over_64Dh` | API.md | test_karman.py |
| benchmarks | `pbar_to_pa4_over_Dh` | API.md | benchmarks.py |
| benchmarks | `timoshenko_clamped_circular` | API.md | test_karman.py |
| benchmarks | `timoshenko_clamped_circular_inverse` | API.md | test_karman.py |
| benchmarks | `way_clamped_circle` | API.md | 06_theory_comparison.ipynb, test_karman.py |
| clamped | `ClampedFem` | API.md | clamped.py |
| clamped | `ClampedPlate` | API.md, dispatch_flow.md | run_circle_1d_2d.py, run_clamped_circle.py, run_clamped_lshape.py |
| clamped | `clamped_fem_circle` | API.md | run_clamped_circle.py |
| clamped | `clamped_fem_lshape` | API.md | run_clamped_lshape.py, test_clamped.py |
| clamped | `solve_clamped_fem` | API.md | test_clamped.py |
| config | `Config` | API.md, CASE_SCHEMA.md, README.md | 01_circle_api.ipynb, 06_theory_comparison.ipynb, golden_config.py |
| contact | `ContactMOR` | ALGORITHMS.md, API.md, ARCHITECTURE.md | run_ktn.py, run_lshape_contact.py, test_contact.py |
| contact | `ContactResult` | API.md | run_lshape_contact.py |
| contact | `TwoPlateMOR` | API.md, ARCHITECTURE.md | test_two_plates.py |
| contact | `TwoPlateResult` | API.md | __init__.py, contact.py, viz.py |
| contact | `sample_fields_on_grid` | API.md | contact.py, dispatch.py |
| contact | `sample_pair_fields_on_grid` | API.md | contact.py, dispatch.py |
| contact | `solve_contact` | API.md, README.md | test_analytic_factory.py, test_face_deflection.py |
| contact_nl | `NonlinearContactMOR` | API.md | test_contact_ktn.py |
| contact_nl | `NonlinearContactResult` | API.md | __init__.py, contact_nl.py |
| dispatch | `Result` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | 01_circle_api.ipynb, 06_theory_comparison.ipynb, run_lshape_contact.py |
| dispatch | `build_domain` | API.md, dispatch_flow.md | test_dispatch.py, test_gap_field.py, test_lshape_stamp.py |
| dispatch | `solve` | ALGORITHMS.md, API.md, ARCHITECTURE.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| faces | `FaceParams` | ALGORITHMS.md, API.md, MIGRATION.md | test_faces.py |
| faces | `face_stresses` | ALGORITHMS.md, API.md | test_faces.py |
| faces | `membrane_face_stress` | API.md | test_faces.py |
| geometry | `BBox` | API.md | basis.py, geometry.py, quadrature.py |
| geometry | `Domain` | API.md | run_ladder_mms.py, test_stresses.py |
| geometry | `circle_expr` | API.md | test_geometry_registry.py |
| geometry | `make_L` | API.md, README.md | run_clamped_lshape.py, run_ktn.py, run_lshape_contact.py |
| geometry | `make_annulus` | API.md | test_geometry_registry.py, test_multiply_connected.py |
| geometry | `make_circle` | API.md, CASE_SCHEMA.md | 01_circle_api.ipynb, 06_theory_comparison.ipynb, run_circle.py |
| geometry | `make_compose` | API.md | test_geometry_registry.py |
| geometry | `make_plate_with_hole` | API.md | test_multiply_connected.py |
| geometry | `make_rectangle` | API.md, NOTES.md | run_ladder_rect_clamped.py, run_ladder_rect_hinge.py, test_clamped.py |
| geometry | `r_and` | API.md, NOTES.md | test_contact.py, test_geometry.py, test_geometry_registry.py |
| geometry | `r_diff` | API.md | test_geometry_registry.py |
| geometry | `r_not` | API.md | test_geometry_registry.py |
| geometry | `r_or` | API.md, NOTES.md | test_geometry.py |
| geometry | `rectangle_expr` | API.md | geometry.py |
| geometry | `x` | ALGORITHMS.md, API.md, ARCHITECTURE.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| geometry | `y` | ALGORITHMS.md, API.md, ARCHITECTURE.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| ktn | `KTNParams` | API.md, MIGRATION.md, NOTES.md | run_ktn.py, test_face_deflection.py, test_faces.py |
| ktn | `PlateMaterial` | API.md | circular_plate.py, conftest.py, test_smoke.py |
| ktn | `flexural_rigidity` | API.md | test_analytic.py, test_smoke.py |
| ktn | `stresses_faces` | ALGORITHMS.md, API.md, MIGRATION.md | 01_circle_api.ipynb, 03_compose_cutout.ipynb, test_faces.py |
| ktn_full | `KTNPlate` | ALGORITHMS.md, API.md, MIGRATION.md | 06_theory_comparison.ipynb, test_ktn_full.py, test_unified_theory.py |
| ktn_solver | `KTNSolver` | API.md, NOTES.md | test_contact_ktn.py, test_multiply_connected.py, test_unified_theory.py |
| ladder | `Strip1DResult` | API.md | ladder.py |
| ladder | `bending_moments` | API.md, NOTES.md | run_ladder_rect_clamped.py, run_ladder_rect_hinge.py, test_stresses.py |
| ladder | `bending_moments_full` | API.md, NOTES.md | test_stresses.py |
| ladder | `mms_clamped_disk_w` | API.md | run_ladder_mms.py |
| ladder | `mms_clamped_rect_w` | API.md | run_ladder_mms.py |
| ladder | `mms_load_and_exact` | API.md | run_ladder_mms.py |
| ladder | `navier_uniform` | API.md | run_ladder_rect_hinge.py |
| ladder | `navier_uniform_center` | API.md | run_ladder_rect_hinge.py |
| ladder | `rect_sin_exact` | API.md | test_doc_matrix.py |
| ladder | `rect_sin_load` | API.md | run_ladder_rect_hinge.py |
| ladder | `rect_sin_wmax` | API.md | run_ladder_rect_hinge.py, test_doc_matrix.py |
| ladder | `solve_strip_1d` | API.md | run_ladder_1d.py |
| ladder | `strip_clamped_exact` | API.md | run_ladder_1d.py |
| ladder | `strip_clamped_wmax` | API.md | run_ladder_1d.py |
| ladder | `strip_hinge_exact` | API.md | run_ladder_1d.py |
| ladder | `strip_hinge_wmax` | API.md | run_ladder_1d.py |
| membrane | `KarmanPlate` | ALGORITHMS.md, API.md, THEORY.md | 06_theory_comparison.ipynb, test_karman.py, test_ktn_full.py |
| membrane | `KarmanResult` | API.md | __init__.py, dispatch.py, membrane.py |
| plate | `PlateBending` | API.md, CASE_SCHEMA.md, dispatch_flow.md | 01_circle_api.ipynb, run_circle.py, run_circle_1d_2d.py |
| poisson | `CACHE_NM_MAX` | API.md | test_poisson_disk.py |
| poisson | `PoissonSolver` | API.md | test_geometry_registry.py, test_poisson_disk.py |
| problem | `BCSpec` | API.md | problem.py |
| problem | `CaseError` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | test_analytic_factory.py, test_cli.py, test_dispatch.py |
| problem | `ContactSpec` | API.md | problem.py |
| problem | `DiscretizationSpec` | API.md | problem.py |
| problem | `GAP_KINDS` | API.md | problem.py |
| problem | `GEOMETRY_KINDS` | API.md | problem.py |
| problem | `GapSpec` | API.md | dispatch.py, problem.py |
| problem | `GeometrySpec` | API.md | test_gap_field.py |
| problem | `LoadSpec` | API.md | dispatch.py, problem.py |
| problem | `MIN_ZONE_NODES` | API.md | dispatch.py |
| problem | `ModelSpec` | API.md | problem.py |
| problem | `OutputSpec` | API.md | problem.py |
| problem | `Plate2Spec` | API.md | problem.py |
| problem | `Problem` | API.md, CASE_SCHEMA.md, dispatch_flow.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| problem | `VerifySpec` | API.md | problem.py |
| problem | `validate_compose_tree` | API.md | geometry.py, problem.py |
| references | `RefRow` | API.md | references.py |
| references | `Reference` | API.md | references.py |
| references | `VerifyReport` | API.md | references.py |
| references | `resolve_reference` | API.md | test_analytic_factory.py, test_mms_reference.py, test_references.py |
| references | `verify_result` | API.md, dispatch_flow.md | 02_annulus_case.ipynb, run_reference.py, test_analytic_factory.py |
| theory | `TheoryParams` | API.md, NOTES.md | test_unified_theory.py |
| theory | `classic` | ALGORITHMS.md, API.md, CASE_SCHEMA.md | 06_theory_comparison.ipynb, run_ktn.py, run_reference.py |
| theory | `from_preset` | API.md | test_unified_theory.py |
| theory | `karman` | ALGORITHMS.md, API.md, CASE_SCHEMA.md | 06_theory_comparison.ipynb, run_reference.py, test_cli.py |
| theory | `ktn_full` | ALGORITHMS.md, API.md, CASE_SCHEMA.md | 06_theory_comparison.ipynb, doc_matrix.py, run_reference.py |
| theory | `ktn_linear` | ALGORITHMS.md, API.md, CASE_SCHEMA.md | 06_theory_comparison.ipynb, test_dispatch.py, test_face_deflection.py |
| verify_fem | `FemComparison` | API.md | verify_fem.py |
| verify_fem | `FemSolution` | API.md | verify_fem.py |
| verify_fem | `annulus_mesh` | API.md | references.py |
| verify_fem | `compare_l2` | API.md | verify_fem.py |
| verify_fem | `compare_rfm_vs_fem` | API.md | run_lshape_verify.py, test_lshape.py |
| verify_fem | `lshape_mesh` | API.md | clamped.py, references.py, verify_fem.py |
| verify_fem | `solve_plate_fem` | API.md | references.py, verify_fem.py |
| viz | `plot_contact_summary` | API.md, README.md | run_lshape_contact.py, test_viz.py |
| viz | `plot_contact_zone` | API.md | test_viz.py |
| viz | `plot_convergence` | API.md | test_viz.py |
| viz | `plot_deflection_contour` | API.md | test_viz.py |
| viz | `plot_deflection_surface` | API.md | 01_circle_api.ipynb, run_circle.py, test_viz.py |
| viz | `plot_pair_summary` | API.md | 05_two_plates.ipynb, test_two_plates.py |
| viz | `plot_reaction` | API.md | test_viz.py |
| viz | `replot` | API.md, ARCHITECTURE.md | 01_circle_api.ipynb, test_stresses.py |
| viz | `stress_maps` | API.md | 01_circle_api.ipynb, 03_compose_cutout.ipynb |
| viz | `surface3d` | API.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |

Дыр (пустых клеток): **0**
