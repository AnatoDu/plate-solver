r"""Тест-ворота Блока Р — вклад структуры Рвачёва против штрафного учёта ГУ.

Один решатель Ритца, один базис Чебышёва, круг (шарнир). Структура ``ω·Φ`` (ГУ
тождественно) против сырого базиса со штрафом ``γ∮w²ds``. Показываем: структура
достигает целевой точности при СУЩЕСТВЕННО меньшем числе функций N и с лучшей
обусловленностью.
"""

from __future__ import annotations

import os

from plate_solver.penalty import penalty_vs_structure_circle

NU, E, H = 0.3, 2.1e6, 1.0
D = E / (12 * (1 - NU**2))
A, Q0 = 1.0, 4.0


def _study(gamma, Q=256, p_list=(2, 4, 6, 8, 10), n_bnd=400, target=1.0):
    return penalty_vs_structure_circle(A, Q0, D, NU, E, H, p_list=p_list, Q=Q,
                                       gamma=gamma, n_bnd=n_bnd, target_pct=target)


def test_rvachev_vs_penalty():
    """ВОРОТА: структура достигает 1 % при меньшем N и лучшей обусловленности, чем штраф."""
    d = _study(gamma=1e4)
    assert d["N_struct"] is not None and d["N_pen"] is not None
    assert d["N_struct"] < d["N_pen"]                           # структура — меньше функций
    assert d["cond_struct_target"] < d["cond_pen_target"]       # и лучше обусловлена
    # при равном минимальном N=9 структура НА ПОРЯДОК+ точнее штрафа
    r0 = d["rows"][0]
    assert r0["err_struct"] < 0.1 * r0["err_pen"]               # ≥ 10× (факт ~36× при Q=256)


def test_penalty_floor_with_small_gamma():
    """Слишком малый γ ⇒ ГУ не выполнены: штраф упирается в «пол» и не берёт цель."""
    d = _study(gamma=1e2)
    assert d["N_pen"] is None                                   # 1 % не достигается
    floor = min(r["err_pen"] for r in d["rows"])
    assert floor > 0.02                                         # «пол» ~ единицы % (ГУ нарушены)
    # структура при том же базисе цель берёт
    assert d["N_struct"] is not None


def test_rvachev_png_created(tmp_path):
    """ВОРОТА: ``rvachev_vs_penalty.png`` создаётся и непустой."""
    import run_rvachev_vs_penalty as rp

    d = _study(gamma=1e4, p_list=(2, 4, 6))
    d_small = _study(gamma=1e2, p_list=(2, 4, 6))
    out = tmp_path / "rvachev_vs_penalty.png"
    path = rp.make_figure(d, d_small, save=str(out))
    assert os.path.exists(path) and os.path.getsize(path) > 5000
