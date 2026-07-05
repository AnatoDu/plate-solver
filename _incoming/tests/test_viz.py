"""Smoke-тесты графики (viz.py): функции строят и сохраняют фигуры без ошибок.

Бэкенд Agg (headless). Проверяем, что каждый график создаётся и пишется в файл
ненулевого размера. Физику не проверяем — это чистая визуализация.
"""
# ruff: noqa: E402, I001  — matplotlib.use("Agg") должен идти ДО импорта pyplot.

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from plates import geometry, viz  # noqa: E402
from plates.config import Config  # noqa: E402
from plates.contact import ContactMOR  # noqa: E402
from plates.plate import PlateBending  # noqa: E402


@pytest.fixture(scope="module")
def setup():
    dom = geometry.make_L(1.0, 0.5)
    cfg = Config(nu=0.3, q0=4.0, h=0.06, p=6, Q=24, beta=1.0, max_iter=300, tol=1e-12, grid_n=24)
    pb = PlateBending.from_config(dom, cfg)
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    gap = 0.6 * float(pb.deflection(cw, q.x, q.y).max())
    res = ContactMOR(pb, cfg, gap=gap).solve()
    return cfg, pb, cw, res


def _assert_saved(fig, path):
    assert path.exists() and path.stat().st_size > 0
    plt.close(fig)


def test_deflection_plots(setup, tmp_path):
    cfg, pb, cw, _ = setup
    _assert_saved(viz.plot_deflection_surface(cfg, pb, cw, save=str(tmp_path / "surf.png")),
                  tmp_path / "surf.png")
    _assert_saved(viz.plot_deflection_contour(cfg, pb, cw, save=str(tmp_path / "cont.png")),
                  tmp_path / "cont.png")


def test_reaction_and_zone(setup, tmp_path):
    cfg, _, _, res = setup
    _assert_saved(viz.plot_reaction(cfg, res, save=str(tmp_path / "r.png")), tmp_path / "r.png")
    _assert_saved(viz.plot_contact_zone(cfg, res, save=str(tmp_path / "z.png")), tmp_path / "z.png")


def test_convergence_and_summary(setup, tmp_path):
    cfg, _, _, res = setup
    _assert_saved(viz.plot_convergence(res, save=str(tmp_path / "conv.png")), tmp_path / "conv.png")
    _assert_saved(viz.plot_contact_summary(cfg, res, save=str(tmp_path / "sum.png")),
                  tmp_path / "sum.png")
