r"""Площадной режим контакта: ПЛАТО реакции на неканонической форме (v0.8.0).

При малом зазоре (``Δ = 0.002·w_free``) пластина ложится на основание почти
целиком. Тогда в ГЛУБИНЕ зоны реакция обязана выйти на плато ``r ≈ q₀``:
нагрузка передаётся основанию напрямую, изгибные усилия остаются лишь в узкой
полосе у кромки зоны и у входящего угла. Это классическая картина для круга;
здесь она проверяется на L-форме — то есть на области, для которой замкнутого
решения нет.

Физические ворота (не зависят от бюджета итераций, поэтому пригодны как
регресс):

* зона ОДНОСВЯЗНА (топология по решётке квадратуры);
* среднее ``r/q₀`` в глубине зоны близко к единице, разброс мал;
* доля переданной основанию силы ``∫r/∫q`` близка к доле площади зоны;
* реакция неотрицательна, проникание в пределах шага МОР.

Уточнённая теория здесь необходима для СХОДИМОСТИ: член податливости
``−κ_r·r`` делает задачу дополнительности строго монотонной (NOTES §11), и МОР
сходится за ~8000 итераций; классическая итерация при таком зазоре за тот же
бюджет не сходится, что и проверяется последним тестом.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plate_solver.diagnostics import contact_interior_stats, contact_report
from plate_solver.dispatch import solve
from plate_solver.problem import Problem

_ROOT = Path(__file__).resolve().parents[1]
_CASE = _ROOT / "cases" / "ci" / "lshape_ktn_plateau.toml"


@pytest.fixture(scope="module")
def plateau():
    res = solve(Problem.from_toml(_CASE))
    quad = res._plate.quad
    return res, quad


def test_plateau_converges_and_zone_is_single_patch(plateau):
    """Итерация сходится, зона контакта — ОДНО связное пятно почти всей области."""
    res, quad = plateau
    sc = res.scalars()
    assert sc["converged"], "МОР не сошёлся: плато проверяется на сошедшемся решении"
    rep = contact_report(res.contact.r_nodes, quad)
    assert rep["n_components"] == 1
    assert rep["contact_fraction"] > 0.7          # площадной режим (измерено 0.84)
    assert np.all(res.contact.r_nodes >= 0.0)


def test_reaction_plateau_in_zone_interior(plateau):
    """ГЛАВНЫЕ ВОРОТА: в глубине зоны реакция выходит на плато ``r ≈ q₀``."""
    res, quad = plateau
    q0 = float(res.config.q0)
    st = contact_interior_stats(res.contact.r_nodes, quad, q_ref=q0, depth=0.15)
    assert st["n_interior"] > 100
    assert st["mean"] == pytest.approx(1.0, abs=0.15)     # измерено 1.058
    assert st["std"] < 0.25                               # измерено 0.101
    assert st["share_within_band"] > 0.5                  # измерено 0.657
    # пик остаётся кромочным и умеренным: это не сингулярность жёсткого штампа
    assert 1.0 < float(res.contact.r_nodes.max()) / q0 < 3.0


def test_force_balance_matches_zone_area(plateau):
    """Доля переданной основанию силы согласована с долей площади зоны.

    На плато ``r ≈ q₀`` и ``∫r ≈ q₀·|зона|``, поэтому ``F/Q ≈ доля площади``:
    проверка баланса сил, независимая от деталей итерации.
    """
    res, quad = plateau
    q0 = float(res.config.q0)
    rep = contact_report(res.contact.r_nodes, quad)
    total_load = q0 * float(np.sum(quad.w))
    share_force = rep["r_total"] / total_load
    assert share_force == pytest.approx(rep["contact_fraction"], rel=0.15)
    assert 0.7 < share_force < 0.95                       # измерено 0.803


def test_refined_theory_is_what_makes_it_converge(plateau):
    """Классика при том же зазоре и бюджете НЕ сходится — сходимость даёт κ_r.

    Ворота фиксируют физическую причину (член податливости), а не совпадение:
    та же постановка с ``theory = classic`` при том же числе итераций остаётся
    с большой KKT-невязкой.
    """
    import tomllib

    res_ktn, _ = plateau
    data = tomllib.loads(_CASE.read_text(encoding="utf-8"))
    data.pop("output", None)
    data["model"] = {"theory": "classic", "h": data["model"]["h"]}
    res_cls = solve(Problem.from_dict(data))
    assert not res_cls.scalars()["converged"]
    assert res_cls.scalars()["comp_residual"] > 10.0 * res_ktn.scalars()["comp_residual"]
