r"""diagnostics.py — диагностика зоны контакта (§8 ТЗ v0.6.0).

Зона контакта — дискретное подмножество узлов квадратуры ``{i : r_i > 0}`` с
обобщённой реакцией ``r ≥ 0``. Помимо очевидных характеристик (число узлов, пик
реакции, суммарная сила) важна ТОПОЛОГИЯ зоны — число связных пятен контакта:
для многосвязных пластин и профильных препятствий (штамп, §9.2) контакт может
распадаться на несколько несмежных областей.

Число связных компонент считается по ГРАФУ БЛИЗОСТИ (одноуровневая кластеризация
/ перколяция): два контактных узла смежны, если расстояние между ними не
превосходит ``radius``. Порог берётся кратным характерному шагу сетки — медиане
расстояния до ближайшего соседа по ВСЕМ узлам квадратуры (``factor·s``, factor≈1.8):
соседи внутри одного пятна (расстояние ~``s``) связываются, пятна, разделённые
зазором > ``factor·s``, остаются раздельными. Реализация — union–find поверх пар
``scipy.spatial.cKDTree.query_pairs`` (сложность ``O(m log m)``, m — размер зоны).

Диагностика ЧИСТО постобработочная: не влияет на решение, лишь описывает его.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

#: множитель характерного шага сетки для порога смежности графа близости (§8).
_ADJ_FACTOR = 1.8


def _components(n: int, pairs) -> int:
    """Число связных компонент графа на ``n`` вершинах по рёбрам ``pairs`` (union–find)."""
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]           # сжатие пути (полупуть)
            a = parent[a]
        return a

    for i, j in pairs:
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
    return len({find(k) for k in range(n)})


def _node_spacing(x: np.ndarray, y: np.ndarray) -> float:
    """Характерный шаг сетки: медиана расстояния до ближайшего соседа (§8)."""
    pts = np.column_stack([x, y])
    if pts.shape[0] < 2:
        return 0.0
    d, _ = cKDTree(pts).query(pts, k=2)             # d[:,0]=0 (сам узел), d[:,1] — сосед
    return float(np.median(d[:, 1]))


def contact_components(x, y, mask, *, radius: float | None = None) -> int:
    r"""Число связных пятен контакта — топология зоны ``{mask}`` (§8).

    Parameters
    ----------
    x, y : координаты узлов квадратуры.
    mask : булева маска зоны контакта (``r > 0``).
    radius : порог смежности графа близости; по умолчанию ``1.8·s``, где ``s`` —
        медиана расстояния до ближайшего соседа по всем узлам ``(x, y)``.

    Returns
    -------
    int
        Число связных компонент (0 — контакта нет; 1 — одно пятно; ≥2 —
        распавшаяся/многосвязная зона).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.asarray(mask, bool)
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return 0
    if idx.size == 1:
        return 1
    if radius is None:
        s = _node_spacing(x, y)
        radius = _ADJ_FACTOR * s if s > 0.0 else np.inf
    pts = np.column_stack([x[idx], y[idx]])
    pairs = cKDTree(pts).query_pairs(radius)        # множество пар (a,b), a<b — в подмножестве
    return _components(idx.size, pairs)


def contact_report(r_nodes, quad, *, radius: float | None = None) -> dict:
    r"""Сводка по зоне контакта (§8): размер, доля площади, пик, сила, топология.

    Parameters
    ----------
    r_nodes : обобщённая реакция ``r ≥ 0`` в узлах квадратуры.
    quad : квадратура с полями ``x``, ``y`` (координаты) и ``w`` (веса ∫·dA).
    radius : порог смежности для числа компонент (см. :func:`contact_components`).

    Returns
    -------
    dict
        ``n_contact`` — число контактных узлов; ``contact_fraction`` — доля
        площади Ω под контактом (∫_контакт dA / ∫_Ω dA); ``r_max`` — пиковая
        реакция и ``peak_xy`` — её локализация; ``r_total`` — суммарная сила
        реакции ``∫ r dA``; ``n_components`` — число связных пятен контакта.
    """
    r = np.asarray(r_nodes, float)
    x, y, w = np.asarray(quad.x, float), np.asarray(quad.y, float), np.asarray(quad.w, float)
    mask = r > 0.0
    area = float(np.sum(w))
    peak = int(np.argmax(r)) if r.size else 0
    return {
        "n_contact": int(mask.sum()),
        "contact_fraction": (float(np.sum(w[mask])) / area) if area > 0.0 else 0.0,
        "r_max": float(r.max()) if r.size else 0.0,
        "peak_xy": (float(x[peak]), float(y[peak])),
        "r_total": float(np.sum(w * r)),
        "n_components": contact_components(x, y, mask, radius=radius),
    }


__all__ = ["contact_components", "contact_report"]
