r"""Ворота диагностики зоны контакта (diagnostics.py, N5 v0.6.0, §8).

Число связных пятен контакта — граф близости + union–find. Проверяем на
синтетических масках (топология известна точно) и на согласованность сводки
``contact_report`` с прямым счётом.

Мат. обоснование. Порог смежности ``radius = 1.8·s`` (s — медиана шага сетки)
разделяет пятна, отстоящие дальше ``1.8·s``, и связывает соседей внутри пятна
(шаг ~``s``). Для регулярной сетки это корректно при зазорах между пятнами ≥ 2
шагов — что и обеспечивают тесты ниже.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

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
