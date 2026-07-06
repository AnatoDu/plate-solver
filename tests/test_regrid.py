"""Сетка вывода: --grid, solve(grid_n=…), with_discretization, regrid.

Числа решения от grid_n НЕ зависят (считаются на узлах квадратуры);
grid_n — только вывод. regrid — мгновенное уплотнение из удержанных
коэффициентов/узловой реакции: МОР не перезапускается (контракт: iters
и residual_history идентичны исходным). т1 — тождество n₀ побайтово;
т2 — контакт на 4·n₀ без повторных итераций; т3 — согласованность
поверхностей на новой сетке.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plate_solver.dispatch import solve
from plate_solver.problem import CaseError, Problem

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> Problem:
    return Problem.from_toml(_ROOT / "cases" / "ci" / f"{name}.toml")


def test_with_discretization_and_solve_override():
    p = _load("circle_soft")
    p2 = p.with_discretization(grid_n=40)
    assert p2.discretization.grid_n == 40
    assert p2.discretization.p == p.discretization.p      # наследуется
    assert p2.source == p.source
    res = solve(p, grid_n=24)
    assert res.w_grid.shape == (24, 24)
    with pytest.raises(CaseError, match="grid_n"):
        p.with_discretization(grid_n=1)


def test_gate_t1_regrid_same_n_is_bitwise_identical():
    """т1: regrid(n₀) ≡ исходные поля ПОБАЙТОВО (изгиб и контакт)."""
    res = solve(_load("circle_soft"))
    n0 = res.w_grid.shape[0]
    r2 = res.regrid(n0)
    assert r2.w_grid.tobytes() == res.w_grid.tobytes()
    assert r2.Xg.tobytes() == res.Xg.tobytes()
    resc = solve(_load("lshape_stamp"))
    n0 = resc.w_grid.shape[0]
    rc = resc.regrid(n0)
    assert rc.w_grid.tobytes() == resc.w_grid.tobytes()
    assert rc.contact.r_grid.tobytes() == resc.contact.r_grid.tobytes()
    assert np.array_equal(rc.contact.contact_zone, resc.contact.contact_zone)


def test_gate_t2_contact_regrid_no_new_iterations():
    """т2: уплотнение 4·n₀ — формы новые, МОР не перезапускался."""
    res = solve(_load("lshape_stamp"))
    n0 = res.w_grid.shape[0]
    r4 = res.regrid(4 * n0)
    assert r4.w_grid.shape == (4 * n0, 4 * n0)
    assert r4.contact.r_grid.shape == (4 * n0, 4 * n0)
    # контракт: итерации/история/узловые величины — исходные объекты
    assert r4.contact.iters == res.contact.iters
    assert r4.contact.residual_history is res.contact.residual_history
    assert r4.contact.r_nodes is res.contact.r_nodes
    # числа решения не зависят от сетки вывода
    assert r4.w_max == res.w_max
    assert float(r4.contact.r_nodes.max()) == float(res.contact.r_nodes.max())
    # исходный результат не тронут (regrid — копия)
    assert res.w_grid.shape == (n0, n0)
    assert res.config.grid_n == n0


def test_gate_t3_surfaces_consistent_after_regrid():
    """т3: classic ⇒ w_top ≡ w_mid ≡ w_bot на НОВОЙ сетке; ktn — dh < 0 в зоне."""
    import tomllib

    res = solve(_load("lshape_stamp")).regrid(32)
    w_top, w_bot, dh = res.faces_on_grid()
    inside = np.isfinite(res.w_grid)
    assert w_top.shape == (32, 32)
    assert np.array_equal(w_top[inside], res.w_grid[inside])
    assert np.array_equal(w_bot[inside], res.w_grid[inside])
    assert float(np.max(np.abs(dh[inside]))) == 0.0
    d = tomllib.loads((_ROOT / "cases" / "ci" / "lshape_stamp.toml")
                      .read_text(encoding="utf-8"))
    d["model"] = {"theory": "ktn", "h": 0.1}
    d.pop("output", None)
    rk = solve(Problem.from_dict(d)).regrid(32)
    _, _, dhk = rk.faces_on_grid()
    zone = rk.contact.contact_zone
    assert zone.any()
    # кромка интерполированной зоны grid-зависима (см. CASE_SCHEMA):
    # гейтим ЯДРО зоны — пиксели с реакцией ≥ 0.5·max (там dh < 0 всюду)
    r_grid = np.nan_to_num(rk.contact.r_grid, nan=0.0)
    core = zone & (r_grid >= 0.5 * float(np.max(r_grid)))
    assert core.any() and float(np.nanmax(dhk[core])) < 0.0


def test_pair_regrid():
    """Пара пластин: regrid переоценивает w₁, w₂, r и зону без итераций."""
    res = solve(_load("two_plates_ring"))
    r2 = res.regrid(48)
    assert r2.contact.w2_grid.shape == (48, 48)
    assert r2.contact.iters == res.contact.iters
    assert r2.contact.residual_history is res.contact.residual_history
    in2 = np.isfinite(r2.contact.w2_grid)
    assert in2.sum() > 0
    # тождество при родном n₀
    n0 = res.w_grid.shape[0]
    r0 = res.regrid(n0)
    assert r0.contact.w2_grid.tobytes() == res.contact.w2_grid.tobytes()


def test_cli_grid_flag(tmp_path):
    """--grid N: поля формы (N, N); равный кейсовому — побайтово те же npz."""
    from plate_solver.cli import main

    case = str(_ROOT / "cases" / "ci" / "circle_soft.toml")
    out1 = tmp_path / "a"
    out2 = tmp_path / "b"
    out3 = tmp_path / "c"
    assert main([case, "--out", str(out1)]) == 0
    n0 = Problem.from_toml(case).discretization.grid_n
    assert main([case, "--out", str(out2), "--grid", str(n0)]) == 0
    a = np.load(out1 / "fields.npz")
    b = np.load(out2 / "fields.npz")
    for key in a.files:
        if key == "problem_json":                # снимок постановки — не поле
            continue
        assert a[key].tobytes() == b[key].tobytes(), key
    assert main([case, "--out", str(out3), "--grid", "24"]) == 0
    c = np.load(out3 / "fields.npz")
    assert c["w"].shape == (24, 24)
