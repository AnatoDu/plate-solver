# Матрица возможностей (автогенерация: scripts/doc_matrix.py)

Каждая возможность обязана быть описана в документации и покрыта
случаем/примером/тестом. Пустая клетка — задача (гейт
tests/test_doc_matrix.py).

## Ключи схемы case-файлов

| секция.ключ | описано | покрыто (cases) |
|---|---|---|
| bc.sides | CASE_SCHEMA.md, FEATURES.md | rect_cfff.toml, rect_cfff.toml, rect_levy.toml |
| bc.type | CASE_SCHEMA.md, FEATURES.md, README.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.bc | CASE_SCHEMA.md, FEATURES.md, THEORY.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.contact | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| case.discretization | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.geometry | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.load | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.model | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.output | CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| case.plate2 | CASE_SCHEMA.md, FEATURES.md | two_plates_equal.toml, two_plates_equal.toml, two_plates_mixed_bc.toml |
| case.verify | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| contact.beta | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.enabled | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.force | CASE_SCHEMA.md, FEATURES.md | lshape_stamp_force.toml, lshape_stamp_force.toml |
| contact.gap | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_soft.toml, annulus_soft.toml, annulus_soft_contact.toml |
| contact.gap_factor | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.max_iter | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_soft_contact.toml, circle_clamped_contact.toml, circle_clamped_contact.toml |
| contact.stop | API.md, CASE_SCHEMA.md, FEATURES.md | test_contact.py, test_doc_policy.py, test_golden_config.py |
| contact.target | CASE_SCHEMA.md, FEATURES.md | two_plates_equal.toml, two_plates_equal.toml, two_plates_mixed_bc.toml |
| contact.tol | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| contact.zone | API.md, CASE_SCHEMA.md, FEATURES.md | lshape_stamp.toml, lshape_stamp.toml, lshape_stamp_force.toml |
| discretization.Q | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| discretization.grid_n | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| discretization.p | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.P | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| load.eps | CASE_SCHEMA.md, FEATURES.md, NOTES.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.q0 | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.type | CASE_SCHEMA.md, FEATURES.md, README.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| load.x0 | CASE_SCHEMA.md, FEATURES.md, README.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.y0 | CASE_SCHEMA.md, FEATURES.md, README.md | circle_point.toml, circle_point_clamped.toml, circle_point_soft.toml |
| load.zone | API.md, CASE_SCHEMA.md, FEATURES.md | lshape_stamp.toml, lshape_stamp.toml, lshape_stamp_force.toml |
| model.E | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_soft.toml, annulus_soft_contact.toml |
| model.h | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| model.nu | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| model.theory | CASE_SCHEMA.md, FEATURES.md, README.md | test_dispatch.py, test_face_deflection.py, test_problem.py |
| output.dir | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| output.figures | CASE_SCHEMA.md, FEATURES.md, README.md | golden_config.py, run_circle_1d_2d.py, run_clamped_circle.py |
| plate2.bc | CASE_SCHEMA.md, FEATURES.md, THEORY.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.discretization | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.geometry | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.load | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| plate2.model | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.cross_1d | ARCHITECTURE.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.model_gap | CASE_SCHEMA.md, FEATURES.md, dispatch_flow.md | annulus_soft.toml, annulus_soft.toml, circle_soft.toml |
| verify.reference | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |
| verify.tol | API.md, CASE_SCHEMA.md, FEATURES.md | annulus_clamped.toml, annulus_clamped.toml, annulus_soft.toml |

## Флаги CLI

| команда | флаг | описано | покрыто (тесты/доки) |
|---|---|---|---|
| plate-ladder | `--out` | CASE_SCHEMA.md, FEATURES.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-ladder | `--version` | FEATURES.md, README.md | doc_matrix.py |
| plate-solve | `--check` | CASE_SCHEMA.md, FEATURES.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--fig-format` | CASE_SCHEMA.md, FEATURES.md, README.md | test_stresses.py |
| plate-solve | `--figures` | CASE_SCHEMA.md, FEATURES.md, README.md | test_stresses.py |
| plate-solve | `--help` | FEATURES.md | FEATURES.md |
| plate-solve | `--new` | CASE_SCHEMA.md, FEATURES.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--out` | CASE_SCHEMA.md, FEATURES.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-solve | `--report` | CASE_SCHEMA.md, FEATURES.md, README.md | doc_matrix.py, test_cli.py |
| plate-solve | `--surface` | CASE_SCHEMA.md, FEATURES.md, README.md | CASE_SCHEMA.md, FEATURES.md, README.md |
| plate-solve | `--sweep` | FEATURES.md, README.md | 02_annulus_case.ipynb, test_cli.py |
| plate-solve | `--version` | FEATURES.md, README.md | doc_matrix.py |
| plate-verify | `--fig-format` | CASE_SCHEMA.md, FEATURES.md, README.md | test_stresses.py |
| plate-verify | `--figures` | CASE_SCHEMA.md, FEATURES.md, README.md | test_stresses.py |
| plate-verify | `--help` | FEATURES.md | FEATURES.md |
| plate-verify | `--out` | CASE_SCHEMA.md, FEATURES.md | doc_matrix.py, test_ci_cases.py, test_cli.py |
| plate-verify | `--surface` | CASE_SCHEMA.md, FEATURES.md, README.md | CASE_SCHEMA.md, FEATURES.md, README.md |
| plate-verify | `--sweep` | FEATURES.md, README.md | 02_annulus_case.ipynb, test_cli.py |
| plate-verify | `--version` | FEATURES.md, README.md | doc_matrix.py |

## Публичные функции и классы

| модуль | имя | описано | покрыто (examples/notebooks/tests) |
|---|---|---|---|
| analytic | `ANNULUS_BCS` | API.md, FEATURES.md | test_annulus_analytic.py |
| analytic | `annulus_uniform` | API.md, FEATURES.md | test_analytic_factory.py, test_annulus_analytic.py |
| analytic | `annulus_uniform_wmax` | API.md, FEATURES.md | test_annulus_analytic.py |
| analytic | `circle_point_clamped` | API.md, FEATURES.md | 04_point_load.ipynb, test_analytic_factory.py, test_point_analytic.py |
| analytic | `circle_point_clamped_wmax` | API.md, FEATURES.md | 04_point_load.ipynb, test_point_analytic.py |
| analytic | `circle_point_soft` | API.md, FEATURES.md, dispatch_flow.md | test_analytic_factory.py, test_doc_matrix.py, test_point_analytic.py |
| analytic | `circle_point_soft_moment` | API.md, FEATURES.md | test_doc_matrix.py |
| analytic | `circle_point_soft_wmax` | API.md, FEATURES.md | test_point_analytic.py |
| analytic | `circular_plate_clamped` | API.md, FEATURES.md | run_circle_1d_2d.py |
| analytic | `circular_plate_simply_supported` | API.md, FEATURES.md | run_circle.py, run_clamped_circle.py, run_ladder_circle.py |
| analytic | `circular_plate_soft_hinge` | API.md, FEATURES.md, NOTES.md | 01_circle_api.ipynb, run_circle.py, run_circle_1d_2d.py |
| analytic | `circular_plate_soft_hinge_wmax` | API.md, FEATURES.md | 01_circle_api.ipynb, run_circle.py, run_clamped_circle.py |
| analytic | `clamped_uniform` | API.md, FEATURES.md | circular_plate.py, run_clamped_circle.py, run_ladder_circle.py |
| analytic | `clamped_uniform_wmax` | API.md, FEATURES.md | circular_plate.py, run_clamped_circle.py, run_ladder_circle.py |
| analytic | `disk_poisson_uniform` | API.md, FEATURES.md | test_doc_matrix.py |
| analytic | `disk_poisson_uniform_center` | API.md, FEATURES.md | test_doc_matrix.py |
| analytic | `disk_poisson_unit` | API.md, FEATURES.md | test_poisson_disk.py |
| analytic | `levy_rect_uniform` | API.md, FEATURES.md | test_analytic_factory.py, test_mixed_bc.py |
| analytic | `navier_rect_uniform` | API.md, FEATURES.md | test_analytic_factory.py, test_mixed_bc.py |
| analytic | `simply_supported_uniform` | API.md, FEATURES.md | test_analytic.py |
| analytic | `simply_supported_uniform_wmax` | API.md, FEATURES.md | test_analytic.py |
| analytic_auto | `CertifiedSolution` | API.md, FEATURES.md | analytic_auto.py |
| analytic_auto | `FactoryError` | API.md, FEATURES.md | test_analytic_factory.py |
| analytic_auto | `axisym_contact_solution` | API.md, FEATURES.md | test_analytic_factory.py |
| analytic_auto | `axisym_solution` | API.md, FEATURES.md | test_analytic_factory.py |
| analytic_auto | `levy_solution` | API.md, FEATURES.md | test_analytic_factory.py, test_free_edge.py |
| analytic_auto | `navier_solution` | API.md, FEATURES.md | test_analytic_factory.py |
| analytic_auto | `strip_solution` | API.md, FEATURES.md | test_analytic_factory.py |
| clamped | `ClampedFem` | API.md, FEATURES.md | clamped.py |
| clamped | `ClampedPlate` | API.md, FEATURES.md, dispatch_flow.md | run_circle_1d_2d.py, run_clamped_circle.py, run_clamped_lshape.py |
| clamped | `clamped_fem_circle` | API.md, FEATURES.md | run_clamped_circle.py |
| clamped | `clamped_fem_lshape` | API.md, FEATURES.md | run_clamped_lshape.py, test_clamped.py |
| clamped | `solve_clamped_fem` | API.md, FEATURES.md | test_clamped.py |
| config | `Config` | API.md, CASE_SCHEMA.md, FEATURES.md | 01_circle_api.ipynb, golden_config.py, run_circle.py |
| contact | `ContactMOR` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | run_ktn.py, run_lshape_contact.py, test_contact.py |
| contact | `ContactResult` | API.md, FEATURES.md | run_lshape_contact.py |
| contact | `TwoPlateMOR` | API.md, ARCHITECTURE.md, FEATURES.md | test_two_plates.py |
| contact | `TwoPlateResult` | API.md, FEATURES.md | contact.py, viz.py |
| contact | `solve_contact` | API.md, FEATURES.md, README.md | test_analytic_factory.py, test_face_deflection.py |
| dispatch | `Result` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | 01_circle_api.ipynb, run_lshape_contact.py, run_stamp_1d.py |
| dispatch | `build_domain` | API.md, FEATURES.md, dispatch_flow.md | test_dispatch.py, test_gap_field.py, test_lshape_stamp.py |
| dispatch | `solve` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| geometry | `BBox` | API.md, FEATURES.md | basis.py, geometry.py, quadrature.py |
| geometry | `Domain` | API.md, FEATURES.md | run_ladder_mms.py, test_stresses.py |
| geometry | `circle_expr` | API.md, FEATURES.md | test_geometry_registry.py |
| geometry | `make_L` | API.md, FEATURES.md, README.md | run_clamped_lshape.py, run_ktn.py, run_lshape_contact.py |
| geometry | `make_annulus` | API.md, FEATURES.md | test_geometry_registry.py |
| geometry | `make_circle` | API.md, CASE_SCHEMA.md, FEATURES.md | 01_circle_api.ipynb, run_circle.py, run_circle_1d_2d.py |
| geometry | `make_compose` | API.md, FEATURES.md | test_geometry_registry.py |
| geometry | `make_rectangle` | API.md, FEATURES.md, NOTES.md | run_ladder_rect_clamped.py, run_ladder_rect_hinge.py, test_clamped.py |
| geometry | `r_and` | API.md, FEATURES.md, NOTES.md | test_contact.py, test_geometry.py, test_geometry_registry.py |
| geometry | `r_diff` | API.md, FEATURES.md | test_geometry_registry.py |
| geometry | `r_not` | API.md, FEATURES.md | test_geometry_registry.py |
| geometry | `r_or` | API.md, FEATURES.md, NOTES.md | test_geometry.py |
| geometry | `rectangle_expr` | API.md, FEATURES.md | geometry.py |
| geometry | `x` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| geometry | `y` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| ktn | `KTNParams` | API.md, FEATURES.md, NOTES.md | run_ktn.py, test_face_deflection.py, test_ktn.py |
| ktn | `PlateMaterial` | API.md, FEATURES.md | circular_plate.py, conftest.py, test_smoke.py |
| ktn | `flexural_rigidity` | API.md, FEATURES.md | test_analytic.py, test_smoke.py |
| ktn | `stresses_faces` | API.md, FEATURES.md | 01_circle_api.ipynb, 03_compose_cutout.ipynb, test_stresses.py |
| ladder | `Strip1DResult` | API.md, FEATURES.md | ladder.py |
| ladder | `bending_moments` | API.md, FEATURES.md, NOTES.md | run_ladder_rect_clamped.py, run_ladder_rect_hinge.py, test_stresses.py |
| ladder | `bending_moments_full` | API.md, FEATURES.md, NOTES.md | test_stresses.py |
| ladder | `mms_clamped_disk_w` | API.md, FEATURES.md | run_ladder_mms.py |
| ladder | `mms_clamped_rect_w` | API.md, FEATURES.md | run_ladder_mms.py |
| ladder | `mms_load_and_exact` | API.md, FEATURES.md | run_ladder_mms.py |
| ladder | `navier_uniform` | API.md, FEATURES.md | run_ladder_rect_hinge.py |
| ladder | `navier_uniform_center` | API.md, FEATURES.md | run_ladder_rect_hinge.py |
| ladder | `rect_sin_exact` | API.md, FEATURES.md | test_doc_matrix.py |
| ladder | `rect_sin_load` | API.md, FEATURES.md | run_ladder_rect_hinge.py |
| ladder | `rect_sin_wmax` | API.md, FEATURES.md | run_ladder_rect_hinge.py, test_doc_matrix.py |
| ladder | `solve_strip_1d` | API.md, FEATURES.md | run_ladder_1d.py |
| ladder | `strip_clamped_exact` | API.md, FEATURES.md | run_ladder_1d.py |
| ladder | `strip_clamped_wmax` | API.md, FEATURES.md | run_ladder_1d.py |
| ladder | `strip_hinge_exact` | API.md, FEATURES.md | run_ladder_1d.py |
| ladder | `strip_hinge_wmax` | API.md, FEATURES.md | run_ladder_1d.py |
| plate | `PlateBending` | API.md, CASE_SCHEMA.md, FEATURES.md | 01_circle_api.ipynb, run_circle.py, run_circle_1d_2d.py |
| poisson | `CACHE_NM_MAX` | API.md, FEATURES.md | test_poisson_disk.py |
| poisson | `PoissonSolver` | API.md, FEATURES.md | test_geometry_registry.py, test_poisson_disk.py |
| problem | `BCSpec` | API.md, FEATURES.md | problem.py |
| problem | `CaseError` | API.md, ARCHITECTURE.md, CASE_SCHEMA.md | test_analytic_factory.py, test_cli.py, test_dispatch.py |
| problem | `ContactSpec` | API.md, FEATURES.md | problem.py |
| problem | `DiscretizationSpec` | API.md, FEATURES.md | problem.py |
| problem | `GAP_KINDS` | API.md, FEATURES.md | problem.py |
| problem | `GEOMETRY_KINDS` | API.md, FEATURES.md | problem.py |
| problem | `GapSpec` | API.md, FEATURES.md | dispatch.py, problem.py |
| problem | `GeometrySpec` | API.md, FEATURES.md | test_gap_field.py |
| problem | `LoadSpec` | API.md, FEATURES.md | dispatch.py, problem.py |
| problem | `MIN_ZONE_NODES` | API.md, FEATURES.md | dispatch.py |
| problem | `ModelSpec` | API.md, FEATURES.md | problem.py |
| problem | `OutputSpec` | API.md, FEATURES.md | problem.py |
| problem | `Plate2Spec` | API.md, FEATURES.md | problem.py |
| problem | `Problem` | API.md, CASE_SCHEMA.md, FEATURES.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |
| problem | `VerifySpec` | API.md, FEATURES.md | problem.py |
| problem | `validate_compose_tree` | API.md, FEATURES.md | geometry.py, problem.py |
| references | `RefRow` | API.md, FEATURES.md | references.py |
| references | `Reference` | API.md, FEATURES.md | references.py |
| references | `VerifyReport` | API.md, FEATURES.md | references.py |
| references | `resolve_reference` | API.md, FEATURES.md | test_analytic_factory.py, test_mms_reference.py, test_references.py |
| references | `verify_result` | API.md, FEATURES.md, dispatch_flow.md | 02_annulus_case.ipynb, test_analytic_factory.py, test_annulus_cases.py |
| verify_fem | `FemComparison` | API.md, FEATURES.md | verify_fem.py |
| verify_fem | `FemSolution` | API.md, FEATURES.md | verify_fem.py |
| verify_fem | `annulus_mesh` | API.md, FEATURES.md | references.py |
| verify_fem | `compare_l2` | API.md, FEATURES.md | verify_fem.py |
| verify_fem | `compare_rfm_vs_fem` | API.md, FEATURES.md | run_lshape_verify.py, test_lshape.py |
| verify_fem | `lshape_mesh` | API.md, FEATURES.md | clamped.py, references.py, verify_fem.py |
| verify_fem | `solve_plate_fem` | API.md, FEATURES.md | references.py, verify_fem.py |
| viz | `plot_contact_summary` | API.md, FEATURES.md, README.md | run_lshape_contact.py, test_viz.py |
| viz | `plot_contact_zone` | API.md, FEATURES.md | test_viz.py |
| viz | `plot_convergence` | API.md, FEATURES.md | test_viz.py |
| viz | `plot_deflection_contour` | API.md, FEATURES.md | test_viz.py |
| viz | `plot_deflection_surface` | API.md, FEATURES.md | 01_circle_api.ipynb, run_circle.py, test_viz.py |
| viz | `plot_pair_summary` | API.md, FEATURES.md | 05_two_plates.ipynb, test_two_plates.py |
| viz | `plot_reaction` | API.md, FEATURES.md | test_viz.py |
| viz | `replot` | API.md, ARCHITECTURE.md, FEATURES.md | 01_circle_api.ipynb, test_stresses.py |
| viz | `stress_maps` | API.md, FEATURES.md | 01_circle_api.ipynb, 03_compose_cutout.ipynb |
| viz | `surface3d` | API.md, FEATURES.md | 01_circle_api.ipynb, 02_annulus_case.ipynb, 03_compose_cutout.ipynb |

Дыр (пустых клеток): **0**
