r"""diagnostics.py — диагностика зоны контакта (§8 ТЗ v0.6.0).

Зона контакта — дискретное подмножество узлов квадратуры ``{i : r_i > 0}`` с
обобщённой реакцией ``r ≥ 0``. Помимо очевидных характеристик (число узлов, пик
реакции, суммарная сила) важна ТОПОЛОГИЯ зоны — число связных пятен контакта:
для многосвязных пластин и профильных препятствий (штамп, §9.2) контакт может
распадаться на несколько несмежных областей.

Число связных компонент считается ПО РЕШЁТКЕ квадратуры (v0.8.0): узлы
тензорного правила Гаусса–Лежандра отображаются в индексы ``(i, j)`` по осям
(``np.unique`` + ``searchsorted``), и компоненты размечаются 4-связностью
(``scipy.ndimage.label``). Это точно и не зависит от НЕРАВНОМЕРНОСТИ шага:
узлы Гаусса сгущены у кромок bbox, поэтому прежний ГРАФ БЛИЗОСТИ с порогом
``1.8·медиана(ближайший сосед)`` рвал одно пятно на десятки «компонент» в
центре области (на прямоугольнике — по одной на узел; аудит v0.8.0).

Граф близости сохранён как ЗАПАСНОЙ путь: для нерегулярных наборов точек (не
решётка) и при явно заданном ``radius`` смежность определяется через
``scipy.spatial.cKDTree.query_pairs`` + union–find.

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


def _lattice_labels(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> int | None:
    """Число 4-связных компонент по РЕШЁТКЕ тензорной квадратуры (или ``None``).

    Возвращает ``None``, если точки не образуют (подмножество) тензорной сетки:
    признак — координатных значений по оси заметно меньше, чем самих узлов.
    """
    xs, ys = np.unique(x), np.unique(y)
    n = x.size
    if xs.size * 2 > n or ys.size * 2 > n:          # не решётка — запасной путь
        return None
    from scipy import ndimage

    img = np.zeros((xs.size, ys.size), dtype=bool)
    ix = np.searchsorted(xs, x[mask])
    iy = np.searchsorted(ys, y[mask])
    img[ix, iy] = True
    return int(ndimage.label(img)[1])               # 4-связность (структура по умолчанию)


def contact_components(x, y, mask, *, radius: float | None = None) -> int:
    r"""Число связных пятен контакта — топология зоны ``{mask}`` (§8).

    Parameters
    ----------
    x, y : координаты узлов квадратуры.
    mask : булева маска зоны контакта (``r > 0``).
    radius : ЯВНЫЙ порог смежности графа близости. По умолчанию (``None``)
        компоненты размечаются по решётке квадратуры 4-связностью — способ,
        не зависящий от неравномерности шага Гаусса; граф близости с порогом
        ``1.8·s`` (``s`` — медиана расстояния до ближайшего соседа) остаётся
        запасным путём для нерегулярных наборов точек.

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
    if radius is None:                              # штатный путь: решётка квадратуры
        by_lattice = _lattice_labels(x, y, mask)
        if by_lattice is not None:
            return by_lattice
        s = _node_spacing(x, y)                     # запасной путь: граф близости
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


def contact_interior_stats(r_nodes, quad, *, q_ref: float, depth: float = 0.15,
                           band: float = 0.10) -> dict:
    r"""Статистика ВНУТРЕННОСТИ зоны контакта: есть ли плато ``r ≈ q``.

    При малом зазоре зона контакта становится площадной, и в её глубине реакция
    выходит на ПЛАТО ``r ≈ q`` (пластина прижата, нагрузка передаётся основанию
    напрямую); кромочная сингулярность остаётся лишь в узкой полосе. Функция
    отделяет глубину от кромки по расстоянию до ближайшего НЕконтактного узла:
    ``d_i ≥ depth·max(d)``.

    Parameters
    ----------
    r_nodes : реакция в узлах квадратуры; quad : квадратура (x, y, w).
    q_ref : масштаб давления (обычно ``q0``) — знаменатель безразмерного ``r/q``.
    depth : доля максимальной глубины зоны, начиная с которой узел считается
        внутренним (0.15 — по умолчанию).
    band : полуширина полосы вокруг ``r/q = 1`` для доли «на плато» (0.10).

    Returns
    -------
    dict
        ``n_interior`` — число внутренних узлов; ``mean``, ``std`` — среднее и
        разброс ``r/q_ref`` по ним; ``share_within_band`` — доля внутренних
        узлов с ``|r/q_ref − 1| ≤ band``; ``max_depth`` — максимальная глубина
        зоны (в единицах длины). При пустой внутренности — нули и ``nan``.
    """
    r = np.asarray(r_nodes, float)
    x, y = np.asarray(quad.x, float), np.asarray(quad.y, float)
    mask = r > 0.0
    empty = {"n_interior": 0, "mean": float("nan"), "std": float("nan"),
             "share_within_band": 0.0, "max_depth": 0.0}
    if not mask.any() or mask.all():
        # зона пуста или занимает всю область: «кромки» нет — глубина не определена
        return empty if not mask.any() else {
            **empty, "n_interior": int(mask.sum()),
            "mean": float(np.mean(r[mask] / q_ref)),
            "std": float(np.std(r[mask] / q_ref)),
            "share_within_band": float(np.mean(np.abs(r[mask] / q_ref - 1.0) <= band)),
        }
    out = np.column_stack([x[~mask], y[~mask]])
    inside = np.column_stack([x[mask], y[mask]])
    dist, _ = cKDTree(out).query(inside, k=1)        # расстояние до кромки зоны
    d_max = float(np.max(dist))
    deep = dist >= depth * d_max
    if not deep.any():
        return {**empty, "max_depth": d_max}
    vals = r[mask][deep] / q_ref
    return {
        "n_interior": int(deep.sum()),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "share_within_band": float(np.mean(np.abs(vals - 1.0) <= band)),
        "max_depth": d_max,
    }


__all__ = ["contact_components", "contact_interior_stats", "contact_report"]
