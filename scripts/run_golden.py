#!/usr/bin/env python3
"""run_golden.py — золотой прогон: вся глава 4 из ОДНОГО конфига.

Порядок: единый Δ из свободного прогиба L-формы → круг (4.1) → верификация МКЭ →
контакт (4.2) → КТН (4.3). Все числа в golden_results.md, три фигуры в figures/.
Падение инварианта = числа в доклад НЕ годятся.

Запуск (из корня репозитория; вывод пишется в CWD):
    plate-solver/.venv/bin/python scripts/run_golden.py
"""

from __future__ import annotations

import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from golden_config import GoldenConfig  # noqa: E402
from run_circle import run_circle_golden  # noqa: E402
from run_ktn import run_ktn_golden  # noqa: E402
from run_lshape_contact import compute_w_free_lshape, run_lshape_contact_golden  # noqa: E402
from run_lshape_verify import run_lshape_verify_golden  # noqa: E402


def _save(fig, cfg, name):
    os.makedirs(cfg.fig_dir, exist_ok=True)
    path = os.path.join(cfg.fig_dir, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _count_gates() -> int | None:
    """Число тест-ворот комплекса: `pytest --collect-only -q` из корня репозитория.

    Вывод -q — строки «tests/test_x.py: N»; суммируем N. При любом сбое
    (нет pytest, изменился формат) возвращаем None — подпись без числа,
    но с командой; хардкод числа не допускается (P1.5).
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            cwd=root, capture_output=True, text=True, timeout=300, check=True,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    counts = re.findall(r"^\S+\.py: (\d+)$", out, flags=re.MULTILINE)
    return sum(map(int, counts)) if counts else None


def write_markdown(cfg, w_free, Delta, circle, verify, contact, ktn, path):
    L = []
    A = L.append
    A("# Золотой прогон — глава 4 (одна серия, один конфиг)\n")
    A("Все числа и рисунки ниже получены ОДНИМ прогоном `run_golden.py` из "
      "`golden_config.GoldenConfig`. Воспроизведение — в конце.\n")
    A(f"> **Толщина.** Круг (4.1) — при `h=1.0` (`D` = {cfg.D:.6g}); вся L-серия "
      f"(`w_free`, верификация, контакт 4.2, КТН 4.3) — при ЕДИНОЙ `h={cfg.h_ktn}` "
      f"(`D` = {cfg.D_lshape:.6g}), чтобы зазор `Δ` был один и тот же в 4.2 и 4.3. "
      "Относительные погрешности и `cond(A)` от `h` не зависят.\n"
      )
    A(f"**Единый зазор.** `w_free = {w_free:.6e}` (L-форма без контакта, "
      f"h={cfg.h_ktn}, p={cfg.p}, Q={cfg.Q_lshape}); "
      f"`Δ = {cfg.gap_factor}·w_free = {Delta:.6e}` — используется и в 4.2, и в 4.3.\n")

    # --- 4.1 круг ---
    A("\n## Таблица 4.1 — верификация на круге\n")
    A(f"`a={cfg.a}`, `q0={cfg.q0}`, `ν={cfg.nu}`, `Q={cfg.Q_circle}`, `h={cfg.h_circle}`. "
      f"Эталоны: мягкий шарнир `w_soft = {circle['w_soft']:.6e}`, "
      f"Кирхгоф (4.2) `w_SS = {circle['w_SS']:.6e}`.\n")
    A("| p | w_max (числ.) | отн. ошибка | cond(A) |")
    A("|---|---|---|---|")
    for p, r in circle["rows"].items():
        A(f"| {p} | {r['w_max']:.6e} | {r['err']:.2e} | {r['cond']:.2e} |")
    A(f"\nСтрока 2 (модель): `w_soft/w_SS = {circle['ratio']:.4f} = 3(1+ν)/(5+ν)`, "
      f"расхождение **{circle['model_gap_pct']:.2f} %**.\n")

    # --- верификация L ---
    A("\n## Верификация L-формы — независимый МКЭ (две колонки)\n")
    A(f"`h={cfg.h_ktn}`, `p={cfg.p}`, `Q={cfg.Q_lshape}`, сетка МКЭ "
      f"m={cfg.fem_mesh_m}, refine={cfg.fem_refine}; точек сверки {verify['n_points']}.\n")
    A("| Модель | w_max | отн. L² к RFM |")
    A("|---|---|---|")
    A(f"| RFM (расщепление) | {verify['rfm_w_max']:.6e} | — |")
    A(f"| FEM-Marcus (P2) | {verify['marcus_w_max']:.6e} | **{verify['rfm_vs_marcus_pct']:.2f} %** |")
    A(f"| FEM-Kirchhoff (Морли) | {verify['kirch_w_max']:.6e} | "
      f"**{verify['rfm_vs_kirchhoff_pct']:.2f} %** (парадокс Сапонджяна) |")
    A(f"\nДвойная проверка: `w_free = {w_free:.6e}` ≈ `RFM w_max = {verify['rfm_w_max']:.6e}` "
      f"(отн. {abs(w_free - verify['rfm_w_max']) / verify['rfm_w_max']:.1e}).\n")

    # --- 4.2 контакт ---
    A("\n## Таблица 4.2 — контакт L-формы (МОР)\n")
    A(f"`q0={cfg.q0}`, `β={cfg.beta}`, основание под всей Ω, зазор `Δ = {Delta:.6e}`.\n")
    A("| Показатель | Значение |")
    A("|---|---|")
    A(f"| итераций МОР | {contact['iters']} |")
    A(f"| невязка ‖Δr‖ | {contact['residual_first']:.2e} → {contact['residual_last']:.2e} "
      f"(падение ×{contact['residual_drop']:.0f}, монотонно) |")
    A(f"| узлов в контакте | {contact['n_contact']} из {contact['n_total']} (узлы квадратуры Q={cfg.Q_lshape}) |")
    A(f"| реакция | r ≥ 0 (min={contact['r_min']:.2e}), пик = {contact['r_max']:.3e} |")
    A(f"| пик реакции | точка ({contact['peak_xy'][0]:.3f}, {contact['peak_xy'][1]:.3f}), "
      f"`d` до угла (0.5,0.5) = {contact['d_corner']:.3f} |")
    A(f"| max w под контактом | {contact['w_under_max']:.6e} ≈ Δ = {Delta:.6e} |")
    A(f"| комплементарность max\\|r·(w−Δ)\\|/(q0·Δ) | {contact['comp_residual']:.2e} |")
    A(f"| перелёт зазора (max w_cont − Δ)/Δ | {contact['gap_overshoot']:.2e} |")

    # --- 4.3 КТН ---
    A("\n## Таблица 4.3 — поправки КТН (классика ↔ КТН)\n")
    A(f"`h={ktn['h']}` (h/L≈{ktn['h']/cfg.L_cut:.2f}), тот же `Δ = {Delta:.6e}`, основание под всей Ω. "
      f"Длины поправок: `h_Ψ²={ktn['h_psi2']:.3e}`, `h_*²={ktn['h_star2']:.3e}`, "
      f"коэф. кривизны `2h_*²−h_Ψ²={ktn['c_curv']:.3e}`.\n")
    A("| Величина | классика | КТН | отношение |")
    A("|---|---|---|---|")
    c, k = ktn["classic"], ktn["ktn"]
    A(f"| пик реакции r_max | {c['peak']:.2f} | {k['peak']:.2f} | ×{ktn['peak_ratio']:.3f} |")
    A(f"| узлов в контакте | {c['nodes']} | {k['nodes']} | ×{k['nodes'] / c['nodes']:.2f} |")
    A(f"| шероховатость r | {c['rough']:.2e} | {k['rough']:.2e} | ×{ktn['rough_ratio']:.3f} |")
    A(f"| w_max | {c['w_max']:.6e} | {k['w_max']:.6e} | {ktn['wmax_corr_pct']:+.1f} % |")
    A("\n(колонка «классика» совпадает с Табл. 4.2: тот же расчёт.)\n")

    # --- фигуры + воспроизведение ---
    A("\n## Фигуры (из этой же серии)\n")
    A(f"- `figures/circle_w_surface.png` — поверхность прогиба круга (p={cfg.p}).")
    A("- `figures/lshape_contact_summary.png` — планшет контакта (прогиб, реакция, зона, сходимость).")
    A("- `figures/ktn_reaction_compare.png` — классика vs КТН при едином Δ.\n")
    n_gates = _count_gates()
    gates_txt = f"{n_gates} ворот-тестов" if n_gates is not None else "ворота-тесты"
    A("## Воспроизведение\n```\nplate-solver/.venv/bin/python scripts/run_golden.py\n```\n"
      f"Корректность математики — {gates_txt} (`pytest`).\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def main():
    cfg = GoldenConfig()
    print("== Золотой прогон ==")

    # 0) единый Δ
    w_free = compute_w_free_lshape(cfg)
    Delta = cfg.gap_factor * w_free
    print(f"[0] w_free(L,h={cfg.h_ktn}) = {w_free:.4e};  Δ = {Delta:.4e}")

    # 1) круг
    circle, fig_c = run_circle_golden(cfg)
    _save(fig_c, cfg, "circle_w_surface.png")
    print(f"[1] круг: модельный разрыв {circle['model_gap_pct']:.2f} %")

    # 2) верификация L
    verify = run_lshape_verify_golden(cfg)
    print(f"[2] верификация: RFM↔Marcus {verify['rfm_vs_marcus_pct']:.2f} %, "
          f"RFM↔Kirchhoff {verify['rfm_vs_kirchhoff_pct']:.2f} %")

    # 3) контакт (единый Δ)
    contact, fig_k, classic_res = run_lshape_contact_golden(cfg, Delta)
    _save(fig_k, cfg, "lshape_contact_summary.png")
    print(f"[3] контакт: {contact['iters']} итер., падение ×{contact['residual_drop']:.0f}, "
          f"пик {contact['peak_xy']}, d={contact['d_corner']:.3f}")

    # 4) КТН (единый Δ, классика из шага 3)
    ktn, fig_t = run_ktn_golden(cfg, Delta, classic_res)
    _save(fig_t, cfg, "ktn_reaction_compare.png")
    print(f"[4] КТН: пик ×{ktn['peak_ratio']:.3f}, узлы ×"
          f"{ktn['ktn']['nodes'] / ktn['classic']['nodes']:.2f}, w_max {ktn['wmax_corr_pct']:+.1f} %")

    # --- инварианты ---
    assert contact["r_min"] >= 0, "реакция должна быть >= 0"
    assert abs(contact["w_under_max"] - Delta) / Delta < 0.05, "max w под контактом != Δ"
    assert contact["residual_drop"] > 100, "МОР не сошёлся монотонно (падение < 100×)"
    assert verify["rfm_vs_marcus_pct"] < 5, "RFM vs FEM-Marcus должно совпадать (~2-3%)"
    print("[OK] инварианты пройдены")

    write_markdown(cfg, w_free, Delta, circle, verify, contact, ktn, cfg.out_md)
    print(f"Готово. Числа -> {cfg.out_md}, фигуры -> {cfg.fig_dir}/")


if __name__ == "__main__":
    main()
