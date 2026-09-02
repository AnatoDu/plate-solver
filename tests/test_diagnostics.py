r"""Ворота диагностики зоны контакта (diagnostics.py, N5 v0.6.0, §8).

Число связных пятен контакта — граф близости + union–find. Проверяем на
синтетических масках (топология известна точно) и на согласованность сводки
``contact_report`` с прямым счётом.

Мат. обоснование (v0.8.0). Штатный путь — разметка 4-связностью ПО РЕШЁТКЕ
квадратуры: узлы тензорного правила отображаются в индексы по осям, и топология
считается на индексной решётке. Это не зависит от НЕРАВНОМЕРНОСТИ шага: узлы
Гаусса сгущены у кромок bbox, поэтому прежний порог ``1.8·медиана(шаг)``
определялся кромочными узлами и рвал одно центральное пятно на десятки
«компонент» (на прямоугольнике — по одной на узел). Запасной путь (граф
близости) сохранён для нерегулярных наборов точек и при явном ``radius``;
он корректен при зазорах между пятнами ≥ 2 шагов.

Ниже проверяются ОБА пути: синтетические маски на равномерной сетке (топология
известна точно) и те же топологии на РЕАЛЬНЫХ узлах Гаусса–Лежандра.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from plate_solver.diagnostics import contact_components, contact_report


def _grid(n=20, lo=0.0, hi=1.0):
    """Регулярная сетка n×n на [lo,hi]² — узлы «квадратуры» для тестов."""
    t = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(t, t)
    return X.ravel(), Y.ravel()


def test_empty_mask_zero_components():
    """Контакта нет ⇒ 0 компонент."""
    x, y = _grid()
    assert contact_components(x, y, np.zeros(x.size, bool)) == 0


def test_single_node_one_component():
    """Один узел ⇒ 1 компонента."""
    x, y = _grid()
    m = np.zeros(x.size, bool)
    m[0] = True
    assert contact_components(x, y, m) == 1


def test_one_connected_patch():
    """Одно сплошное пятно (центральный диск) ⇒ 1 компонента."""
    x, y = _grid(n=30)
    m = (x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.2**2
    assert m.sum() > 3
    assert contact_components(x, y, m) == 1


def test_two_separated_patches():
    """Два разнесённых пятна (у противоположных углов) ⇒ 2 компоненты."""
    x, y = _grid(n=30)
    left = (x - 0.15) ** 2 + (y - 0.15) ** 2 < 0.1**2
    right = (x - 0.85) ** 2 + (y - 0.85) ** 2 < 0.1**2
    m = left | right
    assert left.sum() > 1 and right.sum() > 1
    assert contact_components(x, y, m) == 2


def test_ring_is_one_component():
    """Кольцевая зона (многосвязная геометрически, но связная как множество) ⇒ 1."""
    x, y = _grid(n=40)
    rr = (x - 0.5) ** 2 + (y - 0.5) ** 2
    m = (rr < 0.35**2) & (rr > 0.2**2)
    assert m.sum() > 10
    assert contact_components(x, y, m) == 1


# --------------------------------------------------------------------------- #
#  РЕАЛЬНЫЕ узлы квадратуры Гаусса–Лежандра (неравномерный шаг) — регресс v0.8.0
# --------------------------------------------------------------------------- #
def _gauss_nodes(kind="rectangle", Q=64):
    from plate_solver import geometry
    from plate_solver.quadrature import interior_nodes

    dom = {"rectangle": lambda: geometry.make_rectangle(-1.0, 1.0, -1.0, 1.0),
           "circle": lambda: geometry.make_circle(1.0),
           "L": lambda: geometry.make_L(1.0, 0.5),
           "ellipse": lambda: geometry.make_ellipse(1.0, 0.6)}[kind]()
    q = interior_nodes(dom, Q)
    return q.x, q.y


def test_gauss_nodes_single_patch_is_one_component():
    """ГЛАВНЫЕ ВОРОТА (v0.8.0): сплошное пятно на узлах Гаусса ⇒ РОВНО 1 компонента.

    Прежняя реализация (глобальный порог по медиане шага) давала на
    прямоугольнике по компоненте на каждый узел зоны: узлы Гаусса у кромок
    bbox сгущены, медиана меньше центрального шага, и соседи в центре не
    связывались.
    """
    for kind in ("rectangle", "circle", "L", "ellipse"):
        x, y = _gauss_nodes(kind)
        cx, cy = 0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max())
        mask = (x - cx) ** 2 + (y - cy) ** 2 < 0.3**2
        assert mask.sum() > 20, kind
        assert contact_components(x, y, mask) == 1, kind


def test_gauss_nodes_two_patches_are_two_components():
    """Два разнесённых пятна на узлах Гаусса ⇒ 2 компоненты."""
    x, y = _gauss_nodes("rectangle")
    left = (x + 0.6) ** 2 + (y + 0.6) ** 2 < 0.2**2
    right = (x - 0.6) ** 2 + (y - 0.6) ** 2 < 0.2**2
    assert left.sum() > 5 and right.sum() > 5
    assert contact_components(x, y, left | right) == 2


def test_scattered_points_use_proximity_fallback():
    """Нерегулярный набор точек: работает запасной путь (граф близости)."""
    rng = np.random.default_rng(12345)
    pts = rng.uniform(0.0, 1.0, size=(400, 2))
    x, y = pts[:, 0], pts[:, 1]
    far = np.array([[5.0, 5.0], [5.02, 5.0], [5.0, 5.02]])
    x = np.concatenate([x, far[:, 0]])
    y = np.concatenate([y, far[:, 1]])
    mask = np.zeros(x.size, bool)
    mask[-3:] = True                       # три близких узла вдали от облака
    assert contact_components(x, y, mask) == 1


def test_interior_stats_detect_plateau():
    """Статистика внутренности: плато ``r ≈ q`` отделено от кромочного пика."""
    from plate_solver.diagnostics import contact_interior_stats

    x, y = _grid(n=40)
    quad = SimpleNamespace(x=x, y=y, w=np.full(x.size, (1.0 / 39) ** 2))
    rr = np.hypot(x - 0.5, y - 0.5)
    r = np.zeros(x.size)
    zone = rr < 0.3
    r[zone] = 4.0                                   # плато r = q
    edge = zone & (rr > 0.27)
    r[edge] = 40.0                                  # кромочный пик
    st = contact_interior_stats(r, quad, q_ref=4.0)
    assert st["n_interior"] > 10
    assert st["mean"] == pytest.approx(1.0, rel=1e-9)     # внутри — ровно плато
    assert st["std"] < 1e-9
    assert st["share_within_band"] == 1.0
    assert st["max_depth"] > 0.2
    # без контакта — пустая сводка
    empty = contact_interior_stats(np.zeros(x.size), quad, q_ref=4.0)
    assert empty["n_interior"] == 0 and empty["share_within_band"] == 0.0


def test_contact_report_fields_consistent():
    """Сводка согласована с прямым счётом (число узлов, пик, сила, площадь, компоненты)."""
    x, y = _grid(n=30)
    r = np.zeros(x.size)
    blob = (x - 0.3) ** 2 + (y - 0.3) ** 2 < 0.12**2
    r[blob] = 1.0 + x[blob]                              # неравномерная реакция
    w = np.full(x.size, (1.0 / 29) ** 2)                # равные веса «квадратуры»
    quad = SimpleNamespace(x=x, y=y, w=w)
    rep = contact_report(r, quad)
    assert rep["n_contact"] == int(blob.sum())
    assert rep["n_components"] == 1
    peak = int(np.argmax(r))
    assert rep["peak_xy"] == (float(x[peak]), float(y[peak]))
    assert np.isclose(rep["r_total"], float(np.sum(w * r)))
    assert 0.0 < rep["contact_fraction"] < 1.0
