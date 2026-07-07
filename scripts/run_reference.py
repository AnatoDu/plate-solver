#!/usr/bin/env python3
"""run_reference.py — ЕДИНЫЙ эталонный прогон комплекса.

Один запуск порождает results/reference/reference_v0.5.md (+ csv чисел):

* секции «золотой серии» — ТЕМИ ЖЕ функциями, что исторический
  run_golden.py (числа совпадают с legacy побайтово в напечатанной
  точности; ворота преемственности);
* серии вне реестра случаев: 1D-штамп против Maple-эталона; вклад
  структуры Рвачёва против штрафного учёта ГУ;
* ЛЕСТНИЦА: прогон всех cases/ladder/*.toml с verify-таблицами
  (требует scikit-fem для fem-ступеней; включает нелинейные karman-ступени
  v0.4/v0.5 — Hencky, Levy и ktn_full_thickness_sweep, reference="none").

v0.5.0: отчёт вынесен в НОВЫЙ файл reference_v0.5.md; исторические
reference_v0.3.md / reference_v0.4.md заморожены и НЕ трогаются (протокол §0).

Отчёт ДЕТЕРМИНИРОВАН (без даты и git-хеша — они в спутнике
provenance.json, вне хеш-ворот): двойной прогон обязан дать идентичные
файлы (tests/test_reference_run.py). Файл reference_v0.5.md заморожен
SHA-256 (tests/test_reference_hash.py): любое изменение чисел — красный
тест; обновление — только осознанным коммитом с записью в CHANGELOG.

Запуск (из корня): .venv/bin/python scripts/run_reference.py
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from datetime import UTC
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "reference"
# v0.5.0: НОВЫЙ файл отчёта (старые reference_v0.3.md / v0.4.md заморожены, не
# трогаются — протокол слияния §0). Лестница теперь включает нелинейные ступени
# Кармана и полной КТН (ktn_full_thickness_sweep).
OUT_MD = OUT_DIR / "reference_v0.5.md"
OUT_CSV = OUT_DIR / "reference_v0.5.csv"
OUT_PROV = OUT_DIR / "provenance.json"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))


# --------------------------------------------------------------------------- #
#  Секция 1: золотая серия (те же функции — числа преемственны)
# --------------------------------------------------------------------------- #
def golden_series(csv_rows: list) -> list[str]:
    from golden_config import GoldenConfig
    from run_circle import run_circle_golden
    from run_ktn import run_ktn_golden
    from run_lshape_contact import compute_w_free_lshape, run_lshape_contact_golden
    from run_lshape_verify import run_lshape_verify_golden

    cfg = GoldenConfig()
    w_free = compute_w_free_lshape(cfg)
    Delta = cfg.gap_factor * w_free
    circle, fig_c = run_circle_golden(cfg)
    verify = run_lshape_verify_golden(cfg)
    contact, fig_k, classic_res = run_lshape_contact_golden(cfg, Delta)
    ktn, fig_t = run_ktn_golden(cfg, Delta, classic_res)
    import matplotlib.pyplot as plt

    plt.close("all")

    # инварианты серии (как в историческом прогоне)
    assert contact["r_min"] >= 0
    assert abs(contact["w_under_max"] - Delta) / Delta < 0.05
    assert contact["residual_drop"] > 100
    assert verify["rfm_vs_marcus_pct"] < 5

    L: list[str] = []
    A = L.append
    A("\n## Золотая серия (одна серия, один конфиг GoldenConfig)\n")
    A(f"> **Толщина.** Круг — при `h=1.0` (`D` = {cfg.D:.6g}); вся L-серия "
      f"(`w_free`, верификация, контакт, КТН) — при ЕДИНОЙ `h={cfg.h_ktn}` "
      f"(`D` = {cfg.D_lshape:.6g}), чтобы зазор `Δ` был один и тот же в "
      "контакте и КТН. Относительные погрешности и `cond(A)` от `h` не "
      "зависят.\n")
    A(f"**Единый зазор.** `w_free = {w_free:.6e}` (L-форма без контакта, "
      f"h={cfg.h_ktn}, p={cfg.p}, Q={cfg.Q_lshape}); "
      f"`Δ = {cfg.gap_factor}·w_free = {Delta:.6e}`.\n")

    A("\n### Верификация на круге (эталонная таблица)\n")
    A(f"`a={cfg.a}`, `q0={cfg.q0}`, `ν={cfg.nu}`, `Q={cfg.Q_circle}`, "
      f"`h={cfg.h_circle}`. Эталоны: мягкий шарнир "
      f"`w_soft = {circle['w_soft']:.6e}`, Кирхгоф `w_SS = {circle['w_SS']:.6e}`.\n")
    A("| p | w_max (числ.) | отн. ошибка | cond(A) |")
    A("|---|---|---|---|")
    for p, r in circle["rows"].items():
        A(f"| {p} | {r['w_max']:.6e} | {r['err']:.2e} | {r['cond']:.2e} |")
        csv_rows.append(("golden.circle", f"w_max(p={p})", f"{r['w_max']:.6e}"))
    A(f"\nМодельный разрыв мягкого шарнира: `w_soft/w_SS = {circle['ratio']:.4f} "
      f"= 3(1+ν)/(5+ν)`, расхождение **{circle['model_gap_pct']:.2f} %**.\n")
    csv_rows.append(("golden.circle", "model_gap_pct", f"{circle['model_gap_pct']:.2f}"))

    A("\n### Верификация L-формы — независимый МКЭ (две колонки)\n")
    A(f"`h={cfg.h_ktn}`, `p={cfg.p}`, `Q={cfg.Q_lshape}`, сетка МКЭ "
      f"m={cfg.fem_mesh_m}, refine={cfg.fem_refine}; точек сверки "
      f"{verify['n_points']}.\n")
    A("| Модель | w_max | отн. L² к RFM |")
    A("|---|---|---|")
    A(f"| RFM (расщепление) | {verify['rfm_w_max']:.6e} | — |")
    A(f"| FEM-Marcus (P2) | {verify['marcus_w_max']:.6e} | "
      f"**{verify['rfm_vs_marcus_pct']:.2f} %** |")
    A(f"| FEM-Kirchhoff (Морли) | {verify['kirch_w_max']:.6e} | "
      f"**{verify['rfm_vs_kirchhoff_pct']:.2f} %** (парадокс Сапонджяна) |")
    A(f"\nДвойная проверка: `w_free = {w_free:.6e}` ≈ `RFM w_max = "
      f"{verify['rfm_w_max']:.6e}` (отн. "
      f"{abs(w_free - verify['rfm_w_max']) / verify['rfm_w_max']:.1e}).\n")
    csv_rows.append(("golden.lshape_verify", "rfm_vs_marcus_pct",
                     f"{verify['rfm_vs_marcus_pct']:.2f}"))
    csv_rows.append(("golden.lshape_verify", "rfm_vs_kirchhoff_pct",
                     f"{verify['rfm_vs_kirchhoff_pct']:.2f}"))

    A("\n### Контакт L-формы (МОР)\n")
    A(f"`q0={cfg.q0}`, `β={cfg.beta}`, основание под всей Ω, зазор "
      f"`Δ = {Delta:.6e}`.\n")
    A("| Показатель | Значение |")
    A("|---|---|")
    A(f"| итераций МОР | {contact['iters']} |")
    A(f"| невязка ‖Δr‖ | {contact['residual_first']:.2e} → "
      f"{contact['residual_last']:.2e} "
      f"(падение ×{contact['residual_drop']:.0f}, монотонно) |")
    A(f"| узлов в контакте | {contact['n_contact']} из {contact['n_total']} "
      f"(узлы квадратуры Q={cfg.Q_lshape}) |")
    A(f"| реакция | r ≥ 0 (min={contact['r_min']:.2e}), "
      f"пик = {contact['r_max']:.3e} |")
    A(f"| пик реакции | точка ({contact['peak_xy'][0]:.3f}, "
      f"{contact['peak_xy'][1]:.3f}), "
      f"`d` до угла (0.5,0.5) = {contact['d_corner']:.3f} |")
    A(f"| max w под контактом | {contact['w_under_max']:.6e} ≈ Δ = {Delta:.6e} |")
    A(f"| комплементарность max\\|r·(w−Δ)\\|/(q0·Δ) | "
      f"{contact['comp_residual']:.2e} |")
    A(f"| перелёт зазора (max w_cont − Δ)/Δ | {contact['gap_overshoot']:.2e} |")
    csv_rows.append(("golden.contact", "r_max", f"{contact['r_max']:.3e}"))
    csv_rows.append(("golden.contact", "n_contact", str(contact["n_contact"])))
    csv_rows.append(("golden.contact", "comp_residual",
                     f"{contact['comp_residual']:.2e}"))

    A("\n### Поправки уточнённой теории (классика ↔ КТН)\n")
    A(f"`h={ktn['h']}` (h/L≈{ktn['h'] / cfg.L_cut:.2f}), тот же "
      f"`Δ = {Delta:.6e}`, основание под всей Ω. "
      f"Длины поправок: `h_Ψ²={ktn['h_psi2']:.3e}`, `h_*²={ktn['h_star2']:.3e}`, "
      f"коэф. кривизны `2h_*²−h_Ψ²={ktn['c_curv']:.3e}`.\n")
    A("| Величина | классика | КТН | отношение |")
    A("|---|---|---|---|")
    c, k = ktn["classic"], ktn["ktn"]
    A(f"| пик реакции r_max | {c['peak']:.2f} | {k['peak']:.2f} | "
      f"×{ktn['peak_ratio']:.3f} |")
    A(f"| узлов в контакте | {c['nodes']} | {k['nodes']} | "
      f"×{k['nodes'] / c['nodes']:.2f} |")
    A(f"| шероховатость r | {c['rough']:.2e} | {k['rough']:.2e} | "
      f"×{ktn['rough_ratio']:.3f} |")
    A(f"| w_max | {c['w_max']:.6e} | {k['w_max']:.6e} | "
      f"{ktn['wmax_corr_pct']:+.1f} % |")
    A("\n(колонка «классика» — тот же расчёт, что таблица контакта выше.)\n")
    csv_rows.append(("golden.ktn", "peak_ratio", f"{ktn['peak_ratio']:.3f}"))
    csv_rows.append(("golden.ktn", "wmax_corr_pct", f"{ktn['wmax_corr_pct']:+.1f}"))
    return L


# --------------------------------------------------------------------------- #
#  Секция 2: серии вне реестра случаев
# --------------------------------------------------------------------------- #
def extra_series(csv_rows: list) -> list[str]:
    from run_rvachev_vs_penalty import run_rvachev_vs_penalty
    from run_stamp_1d import run_stamp

    L: list[str] = []
    A = L.append
    A("\n## 1D-штамп против эталона Maple\n")
    res, _maple, (dmax, l2pct) = run_stamp()
    A("| max|Δw| | отн. L², % |")
    A("|---|---|")
    A(f"| {dmax:.3e} | {l2pct:.2f} |")
    csv_rows.append(("stamp_1d", "maple_l2_pct", f"{l2pct:.2f}"))

    A("\n## Вклад структуры Рвачёва (структура против штрафа)\n")
    d, d_small = run_rvachev_vs_penalty()
    A("| N | ошибка структуры, % | ошибка штрафа, % |")
    A("|---|---|---|")
    for r in d["rows"]:
        A(f"| {r['N']} | {100 * r['err_struct']:.3f} | {100 * r['err_pen']:.3f} |")
    A(f"\nЦель достигается структурой при N = {d['N_struct']}, штрафом — "
      f"при N = {d['N_pen']}; при малом γ штраф не выполняет ГУ "
      f"(пол ошибки {100 * min(r['err_pen'] for r in d_small['rows']):.2f} %).\n")
    csv_rows.append(("rvachev_vs_penalty", "N_struct", str(d["N_struct"])))
    csv_rows.append(("rvachev_vs_penalty", "N_pen", str(d["N_pen"])))
    return L


# --------------------------------------------------------------------------- #
#  Секция 3: лестница (cases/ladder)
# --------------------------------------------------------------------------- #
def ladder_section(csv_rows: list) -> list[str]:
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem
    from plate_solver.references import verify_result

    L: list[str] = []
    A = L.append
    A("\n## Лестница верификации (cases/ladder, боевые параметры)\n")
    A("| случай | w_max | эталон | rel | tol | статус |")
    A("|---|---|---|---|---|---|")
    for case in sorted((ROOT / "cases" / "ladder").glob("*.toml")):
        problem = Problem.from_toml(case)
        res = solve(problem)
        if problem.verify.reference == "none":
            A(f"| {case.stem} | {res.w_max:.6e} | — | — | — | run |")
            csv_rows.append((f"ladder.{case.stem}", "w_max", f"{res.w_max:.6e}"))
            continue
        rep = verify_result(res)
        for row in rep.rows:
            status = "PASS" if (not row.gated or row.rel <= rep.tol) else "FAIL"
            A(f"| {case.stem} | {res.w_max:.6e} | {row.name} | {row.rel:.3e} | "
              f"{rep.tol:g} | {status} |")
            csv_rows.append((f"ladder.{case.stem}", f"rel[{row.name}]",
                             f"{row.rel:.3e}"))
            assert not row.gated or row.rel <= rep.tol, \
                f"{case.stem}: {row.name} rel={row.rel:.3e} > tol={rep.tol:g}"
    return L


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_rows: list[tuple[str, str, str]] = []
    L = ["# Эталонный отчёт plate-solver v0.5",
         "",
         "Все числа получены ОДНИМ прогоном `scripts/run_reference.py` и",
         "заморожены хеш-воротами (tests/test_reference_hash.py). Обновление",
         "отчёта — только осознанным коммитом с обоснованием в CHANGELOG.",
         "Провенанс прогона (git-хеш, версии, дата) — provenance.json рядом",
         "(вне хеша: отчёт детерминирован)."]
    print("[1/3] золотая серия…")
    L += golden_series(csv_rows)
    print("[2/3] серии вне реестра…")
    L += extra_series(csv_rows)
    print("[3/3] лестница cases/ladder…")
    L += ladder_section(csv_rows)
    L.append("")
    OUT_MD.write_text("\n".join(L), encoding="utf-8")
    with OUT_CSV.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["серия", "величина", "значение"])
        w.writerows(csv_rows)

    import numpy
    import scipy
    import sympy

    import plate_solver

    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip()
    except OSError:
        git = "unknown"
    from datetime import datetime

    OUT_PROV.write_text(json.dumps({
        "git": git,
        "date": datetime.now(UTC).isoformat(),
        "plate_solver": plate_solver.__version__,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sympy": sympy.__version__, "python": sys.version.split()[0],
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"готово: {OUT_MD} ({len(csv_rows)} чисел в csv)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
