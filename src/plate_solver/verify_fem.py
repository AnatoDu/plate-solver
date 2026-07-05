r"""verify_fem.py — независимая сверка изгиба L-формы через scikit-fem (ДВЕ колонки).

На L-форме (входящий угол) сравниваем RFM-решение (расщепление, «мягкий шарнир»)
с двумя НЕЗАВИСИМЫМИ МКЭ-решениями:

  • **FEM-Marcus** — те же две задачи Пуассона (−ΔM=q, M=0; −Δw=M/D, w=0),
    но на треугольной сетке Лагранжем (P2). Это ТА ЖЕ модель, что и RFM ⇒ поля
    должны совпасть в пределах сеточной/полиномиальной точности (~1–2 %). Это и
    есть контрольная колонка верификации численности (тест-ворота test_lshape).

  • **FEM-Kirchhoff** — ИСТИННЫЙ изгиб Кирхгофа (элемент Морли, бигармоника).
    На криволинейных/входящих углах расщепление на две Пуассона (мембранная
    аналогия Маркуса) ≠ Кирхгоф: у входящего угла RFM/Marcus и Кирхгоф расходятся
    (до ~2× у угла, расхождение НЕ убывает с измельчением) — это **парадокс
    Сапонджяна–Бабушки** (NOTES.md §9). Эту колонку выводим в Таблицу 4.2 как
    количественную иллюстрацию модельного эффекта, мотивирующую уточнение (КТН).

scikit-fem импортируется ВНУТРИ функций: пакет грузится и без него (он нужен
только здесь). Сетка L-формы строится структурно, ТОЧНО совпадая с
``geometry.make_L`` (тот же домен `[0,side]² \\ [cut,side]²`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
#  Сетка L-формы, совпадающая с geometry.make_L
# --------------------------------------------------------------------------- #
def lshape_mesh(side: float = 1.0, cut: float = 0.5, m: int = 16, refine: int = 0):
    """Треугольная сетка L-формы ``[0,side]² \\ [cut,side]²`` (MeshTri).

    Структурная сетка m×m по объемлющему квадрату; ячейки с центром в вырезе
    отбрасываются (границы осевые ⇒ домен совпадает с ``make_L`` точно).
    ``refine`` — число равномерных измельчений.
    """
    from skfem import MeshTri

    xs = np.linspace(0.0, side, m + 1)
    h = side / m
    idx = -np.ones((m + 1, m + 1), int)
    pts: list[tuple[float, float]] = []
    tris: list[tuple[int, int, int]] = []

    def node(i: int, j: int) -> int:
        if idx[i, j] < 0:
            idx[i, j] = len(pts)
            pts.append((xs[i], xs[j]))
        return idx[i, j]

    for i in range(m):
        for j in range(m):
            if (xs[i] + h / 2) > cut and (xs[j] + h / 2) > cut:
                continue  # ячейка целиком в вырезе
            a, b = node(i, j), node(i + 1, j)
            c, d = node(i + 1, j + 1), node(i, j + 1)
            tris.append((a, b, c))
            tris.append((a, c, d))

    mesh = MeshTri(np.array(pts, float).T, np.array(tris).T)
    return mesh.refined(refine) if refine else mesh


# --------------------------------------------------------------------------- #
#  МКЭ-решение (две модели)
# --------------------------------------------------------------------------- #
@dataclass
class FemSolution:
    """МКЭ-решение с возможностью вычислить ``w`` в произвольных точках."""

    basis: object
    w: np.ndarray
    model: str

    def at(self, X, Y) -> np.ndarray:
        """Значения ``w`` в точках (X, Y) (интерполяция МКЭ-решения)."""
        P = np.ascontiguousarray(np.vstack([np.ravel(X), np.ravel(Y)]), dtype=float)
        return np.asarray(self.basis.interpolator(self.w)(P), dtype=float)


def annulus_mesh(a: float, b: float, n_r: int = 24, n_t: int = 96):
    """Структурированная триангуляция кольца b < r < a (P3.6 фазы 2).

    Узлы — тензорная сетка ``(n_r+1) × n_t`` по радиусу и углу с замыканием
    по θ; каждая четырёхугольная ячейка делится на два треугольника.
    Граница (обе окружности) — многоугольники: геометрическая ошибка
    ~O(1/n_t²) — учитывать при сверке с гладкой ω-границей RFM.
    """
    from skfem import MeshTri

    if not 0.0 < b < a:
        raise ValueError("Кольцо требует 0 < b < a.")
    r = np.linspace(b, a, n_r + 1)
    t = np.linspace(0.0, 2.0 * np.pi, n_t, endpoint=False)
    R, T = np.meshgrid(r, t, indexing="ij")               # (n_r+1, n_t)
    verts = np.stack([(R * np.cos(T)).ravel(), (R * np.sin(T)).ravel()])

    def vid(i: int, j: int) -> int:
        return i * n_t + (j % n_t)

    tris = []
    for i in range(n_r):
        for j in range(n_t):
            v00, v10 = vid(i, j), vid(i + 1, j)
            v11, v01 = vid(i + 1, j + 1), vid(i, j + 1)
            tris.append((v00, v10, v11))
            tris.append((v00, v11, v01))
    return MeshTri(np.ascontiguousarray(verts, dtype=np.float64),
                   np.ascontiguousarray(np.array(tris).T, dtype=np.int64))


def solve_plate_fem(
    mesh, D: float, q: float, model: str = "kirchhoff", nu: float = 0.3
) -> FemSolution:
    """Решить изгиб пластины на сетке ``mesh`` независимым МКЭ.

    Parameters
    ----------
    model : 'kirchhoff' — истинный Кирхгоф (элемент Морли, бигармоника);
            'marcus'    — две задачи Пуассона (P2-Лагранж) = модель RFM.
    Краевые условия — шарнир (``w = 0`` на ∂Ω).
    """
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriMorley,
        ElementTriP2,
        LinearForm,
        condense,
        solve,
    )
    from skfem.helpers import dd, ddot, dot, grad, trace

    if model == "kirchhoff":
        @BilinearForm
        def plate(u, v, w):
            return D * ((1 - nu) * ddot(dd(u), dd(v)) + nu * trace(dd(u)) * trace(dd(v)))

        @LinearForm
        def load(v, w):
            return q * v

        basis = Basis(mesh, ElementTriMorley())
        # Шарнир: w = 0 в узловых dof 'u' на границе (M_n = 0 — естественное).
        ess = basis.get_dofs().all("u")
        wsol = solve(*condense(plate.assemble(basis), load.assemble(basis), D=ess))
        return FemSolution(basis, wsol, "kirchhoff")

    if model == "marcus":
        @BilinearForm
        def laplace(u, v, w):
            return dot(grad(u), grad(v))

        @LinearForm
        def rhs_q(v, w):
            return q * v

        @LinearForm
        def rhs_MD(v, w):
            return w["Mf"] * v / D

        basis = Basis(mesh, ElementTriP2())
        ess = basis.get_dofs().all()                 # w = 0 на всей границе
        K = laplace.assemble(basis)
        M = solve(*condense(K, rhs_q.assemble(basis), D=ess))           # −ΔM = q
        Mf = basis.interpolate(M)
        wsol = solve(*condense(K, rhs_MD.assemble(basis, Mf=Mf), D=ess))  # −Δw = M/D
        return FemSolution(basis, wsol, "marcus")

    raise ValueError(f"model должен быть 'kirchhoff' или 'marcus', не {model!r}.")


# --------------------------------------------------------------------------- #
#  Сравнение в норме L²
# --------------------------------------------------------------------------- #
def compare_l2(w_rfm_on_pts, w_fem_on_pts, weights=None) -> float:
    """Относительная L²-погрешность ``‖w_rfm − w_fem‖ / ‖w_fem‖`` в ПРОЦЕНТАХ.

    ``weights`` (если заданы) — квадратурные веса в точках (взвешенная L²).
    """
    a = np.asarray(w_rfm_on_pts, float)
    b = np.asarray(w_fem_on_pts, float)
    wts = np.ones_like(b) if weights is None else np.asarray(weights, float)
    num = np.sqrt(np.sum(wts * (a - b) ** 2))
    den = np.sqrt(np.sum(wts * b**2))
    return 100.0 * num / den


@dataclass
class FemComparison:
    """Сводка двухколоночной сверки RFM ↔ МКЭ на L-форме (для Таблицы 4.2)."""

    rel_marcus_pct: float      # RFM ↔ FEM-Marcus (та же модель) — мало
    rel_kirchhoff_pct: float   # RFM ↔ FEM-Kirchhoff — парадокс Сапонджяна
    w_rfm_max: float
    w_marcus_max: float
    w_kirchhoff_max: float
    n_points: int


def compare_rfm_vs_fem(
    cfg, *, side: float = 1.0, cut: float = 0.5, mesh_m: int = 16, refine: int = 3,
    eps: float = 0.02,
) -> FemComparison:
    """Полная сверка: RFM (расщепление) против FEM-Marcus и FEM-Kirchhoff на L-форме.

    Поля сравниваются в общих внутренних точках (узлы квадратуры RFM с запасом
    ``eps`` от границы), взвешенная L²-погрешность — в процентах.
    """
    from . import geometry
    from . import quadrature as quad
    from .plate import PlateBending

    dom = geometry.make_L(side, cut)
    pb = PlateBending.from_config(dom, cfg)
    _, cw = pb.solve_uniform(cfg.q0)

    qn = quad.interior_nodes(dom, cfg.Q)
    keep = dom.omega(qn.x, qn.y) > eps
    X, Y, W = qn.x[keep], qn.y[keep], qn.w[keep]
    w_rfm = pb.deflection(cw, X, Y)

    mesh = lshape_mesh(side, cut, mesh_m, refine)
    sol_m = solve_plate_fem(mesh, cfg.D, cfg.q0, model="marcus")
    sol_k = solve_plate_fem(mesh, cfg.D, cfg.q0, model="kirchhoff", nu=cfg.nu)
    wm, wk = sol_m.at(X, Y), sol_k.at(X, Y)

    return FemComparison(
        rel_marcus_pct=compare_l2(w_rfm, wm, W),
        rel_kirchhoff_pct=compare_l2(w_rfm, wk, W),
        w_rfm_max=float(w_rfm.max()),
        w_marcus_max=float(wm.max()),
        w_kirchhoff_max=float(wk.max()),
        n_points=int(X.size),
    )


__all__ = [
    "lshape_mesh",
    "annulus_mesh",
    "FemSolution",
    "solve_plate_fem",
    "compare_l2",
    "FemComparison",
    "compare_rfm_vs_fem",
]
