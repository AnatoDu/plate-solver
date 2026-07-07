r"""Ворота экспорта усилий и VTK (export.py, N8 v0.6.0, §9.4).

Постобработка: результирующие усилия как поля + запись в legacy-VTK. Проверяем
структуру (моменты на сетке, контактная реакция), круговой рейс VTK (запись →
чтение заголовка и значений) и согласованность с полями результата.
"""

from __future__ import annotations

import numpy as np

from plate_solver.dispatch import solve
from plate_solver.export import forces_on_grid, to_vtk
from plate_solver.problem import Problem

_CLASSIC = Problem.from_dict({
    "geometry": {"kind": "circle", "a": 1.0}, "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 1.0e-2},
    "model": {"theory": "classic", "E": 1.0, "nu": 0.3, "h": 0.2},
    "discretization": {"p": 8, "Q": 48, "grid_n": 24},
})


def test_forces_on_grid_has_moments():
    """Усилия на сетке всегда содержат изгибные моменты (форма = сетка)."""
    res = solve(_CLASSIC)
    f = forces_on_grid(res)
    assert {"Mx", "My", "Mxy"} <= set(f)
    assert f["Mx"].shape == res.Xg.shape
    # моменты определены внутри Ω (не всюду NaN)
    assert np.isfinite(f["Mx"]).any()


def test_to_vtk_roundtrip(tmp_path):
    """VTK: запись → чтение заголовка; размеры и значение w совпадают с сеткой."""
    res = solve(_CLASSIC)
    path = to_vtk(res, tmp_path / "plate")
    assert path.suffix == ".vtk" and path.exists()
    text = path.read_text(encoding="utf-8")
    ny, nx = res.Xg.shape
    assert "DATASET STRUCTURED_POINTS" in text
    assert f"DIMENSIONS {nx} {ny} 1" in text
    assert f"POINT_DATA {nx * ny}" in text
    assert "SCALARS w double 1" in text
    # блок w: nx*ny числовых строк, порядок x-быстрее ⇒ совпадает с ravel('C')
    body = text.splitlines()
    i = body.index("SCALARS w double 1")
    vals = np.array([float(v) for v in body[i + 2:i + 2 + nx * ny]])
    ref = np.asarray(res.w_grid, float).ravel(order="C")
    fin = np.isfinite(ref)
    assert np.allclose(vals[fin], ref[fin], rtol=1e-6, atol=1e-12)


def test_forces_and_vtk_include_contact_reaction(tmp_path):
    """Контактная задача: реакция r попадает в усилия и в VTK."""
    prob = Problem.from_dict({
        "geometry": {"kind": "circle", "a": 1.0}, "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0}, "contact": {"enabled": True, "gap": 0.2},
        "model": {"theory": "classic", "E": 1.0, "nu": 0.3, "h": 1.0},
        "discretization": {"p": 8, "Q": 48, "grid_n": 24},
    })
    res = solve(prob)
    f = forces_on_grid(res)
    assert "r" in f and f["r"].shape == res.Xg.shape
    text = to_vtk(res, tmp_path / "contact").read_text(encoding="utf-8")
    assert "SCALARS r double 1" in text
