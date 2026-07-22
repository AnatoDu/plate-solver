"""Ворота примитива `ellipse` (v0.6.3): форма произвольного очертания в case-файле.

Эллипс с полуосями (a, b) как верхнеуровневый вид геометрии (не примитив
compose — ограда неизменна). Проверяется: сведение к кругу при a = b (нормировка
ω), решение изгиба и КОНТАКТА (в т.ч. по полной КТН на защемлении — лицевая
поверхность), понятные отказы для несовместимых постановок.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.geometry import make_circle, make_ellipse
from plate_solver.problem import CaseError, Problem

_GEO = '[geometry]\nkind = "ellipse"\na = 1.5\nb = 1.0\n'


def _solve(tmp_path, body):
    p = tmp_path / "case.toml"
    p.write_text(_GEO + body, encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


def _fail(tmp_path, body):
    p = tmp_path / "case.toml"
    p.write_text(_GEO + body, encoding="utf-8")
    with pytest.raises(CaseError) as e:
        Problem.from_toml(str(p))
    return str(e.value)


def test_ellipse_reduces_to_circle_at_equal_axes():
    """Нормировка ω: эллипс с a = b совпадает с кругом того же радиуса."""
    e = make_ellipse(1.0, 1.0)
    c = make_circle(1.0)
    xs = np.array([0.0, 0.3, -0.5, 0.7])
    ys = np.array([0.0, 0.4, 0.2, -0.1])
    assert np.allclose(e.omega(xs, ys), c.omega(xs, ys), atol=1e-14)
    assert e.bbox == (-1.0, 1.0, -1.0, 1.0)


def test_ellipse_bending_solves(tmp_path):
    """Эллипс + шарнир + классический изгиб — решается на произвольной R-области."""
    res = _solve(tmp_path,
                 '[bc]\ntype = "soft_hinge"\n[load]\ntype = "uniform"\nq0 = 4.0\n'
                 '[model]\nh = 0.06\n[discretization]\np = 8\nQ = 64\ngrid_n = 16\n')
    assert res.w_max > 0.0 and np.isfinite(res.w_max)


def test_ellipse_soft_hinge_contact_linear(tmp_path):
    """Эллипс + ШАРНИР + контакт (классика): шарнир+контакт доступны для линейных теорий."""
    res = _solve(tmp_path,
                 '[bc]\ntype = "soft_hinge"\n[load]\ntype = "uniform"\nq0 = 4.0\n'
                 '[model]\nh = 0.06\n[contact]\nenabled = true\ngap_factor = 0.6\n'
                 'max_iter = 1500\ntol = 1e-7\n[discretization]\np = 8\nQ = 64\ngrid_n = 16\n'
                 '[verify]\nreference = "none"\n')
    c = res.contact
    assert (c.r_nodes >= -1e-9).all() and c.r_nodes.max() > 0.0 and (c.r_nodes > 0).sum() > 0


def test_ellipse_ktn_full_contact_face(tmp_path):
    """Эллипс + защемление + контакт по полной КТН: лицевая поверхность (dh ≠ 0)."""
    res = _solve(tmp_path,
                 '[bc]\ntype = "clamped"\n[load]\ntype = "uniform"\nq0 = 0.01\n'
                 '[model]\ntheory = "ktn_full"\nh = 0.2\nn_load_steps = 3\n'
                 'karman_max_iter = 100\nkarman_tol = 1e-6\n[contact]\nenabled = true\n'
                 'gap_factor = 0.5\nscheme = "merged"\nbeta = 1.5\nmax_iter = 3000\ntol = 3e-4\n'
                 '[discretization]\np = 8\nQ = 64\ngrid_n = 16\n[verify]\nreference = "none"\n')
    c = res.contact
    assert (c.r_nodes >= -1e-9).all() and c.r_nodes.max() > 0.0
    _wt, _wb, dh = res.faces_on_grid()
    assert np.nanmax(np.abs(dh)) > 0.0                    # контакт по лицевой


def test_ellipse_ktn_full_soft_hinge_solves(tmp_path):
    """Эллипс + ШАРНИР + полная КТН — изгиб решается (граничный член §3.5, v0.6.3).

    Полная верификация граничного члена — tests/test_soft_hinge_ktn.py.
    """
    res = _solve(tmp_path,
                 '[bc]\ntype = "soft_hinge"\n[load]\ntype = "uniform"\nq0 = 0.01\n'
                 '[model]\ntheory = "ktn_full"\nh = 0.2\nn_load_steps = 3\n'
                 'karman_max_iter = 100\nkarman_tol = 1e-6\n'
                 '[discretization]\np = 8\nQ = 64\ngrid_n = 16\n[verify]\nreference = "none"\n')
    assert res.w_max > 0.0 and np.isfinite(res.w_max)
    _wt, _wb, dh = res.faces_on_grid()
    assert np.nanmax(np.abs(dh)) > 0.0                    # лицевые различаются (КТН)


def test_ellipse_analytic_reference_rejected(tmp_path):
    """Эллипс + analytic — понятный отказ (замкнутого решения нет)."""
    msg = _fail(tmp_path,
                '[bc]\ntype = "clamped"\n[load]\ntype = "uniform"\nq0 = 4.0\n'
                '[model]\nh = 0.06\n[discretization]\np = 8\nQ = 64\n'
                '[verify]\nreference = "analytic"\n')
    assert "эталон" in msg
