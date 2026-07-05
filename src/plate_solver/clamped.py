r"""clamped.py — изгиб ЖЁСТКО ЗАЩЕМЛЁННОЙ пластины: прямой метод Ритца по бигармонике.

В отличие от ``plate.py`` (расщепление бигармоники на две Пуассона, структура
``w = ω·Φ`` — «мягкий шарнир»), здесь берётся структура ЗАЩЕМЛЕНИЯ

    w = ω²·Φ,    Φ = Σ_k c_k T_k,                                   (формула 1.22)

которая ТОЖДЕСТВЕННО даёт ``w = 0`` И ``∂w/∂n = 0`` на ∂Ω (т.к. ``ω = 0`` и
``∇(ω²Φ) = 2ω∇ω·Φ + ω²∇Φ → 0`` при ``ω → 0``). Поэтому КАЖДАЯ пробная функция
удовлетворяет обоим условиям защемления точно — расщепление на две Пуассона НЕ
требуется и НЕ используется.

Энергия защемлённой пластины Кирхгофа:

    U = (D/2) ∫_Ω [ (Δw)² − 2(1−ν)(w_xx w_yy − w_xy²) ] dΩ.

Гауссова кривизна ``∫(w_xx w_yy − w_xy²)`` сводится к контурному интегралу и
ЗАНУЛЯЕТСЯ для любой функции с ``w = 0, ∇w = 0`` на ∂Ω (а значит — для всего
пространства ``ω²Φ``). Поэтому на этом пространстве энергия ТОЧНО равна

    U = (D/2) ∫_Ω (Δw)² dΩ,

и минимизация ``J[w] = U − ∫ q w`` по ``c`` даёт систему Ритца

    (D·S) c = b,   S[k,l] = ∫_Ω Δψ_k Δψ_l dΩ,   b[k] = ∫_Ω q ψ_k dΩ,

где ``ψ_k = ω²·T_k``. Эйлер–Лагранж этой задачи — ИСТИННЫЙ бигармонический
оператор ``D Δ²w = q`` с защемлением (без модельного упрощения шарнира). Ключевое
следствие для доклада: на криволинейной границе (круг) и во входящем угле
(L-форма) расщепление КОРРЕКТНО — парадокс Сапонджяна–Бабушки есть свойство
ШАРНИРНОГО опирания, при защемлении его НЕТ (NOTES.md §§8, 9).

Лапласиан структурной функции (нужен для S):

    Δψ_k = Δ(ω²·T_k)
         = 2(|∇ω|² + ω·Δω)·T_k + 4ω(ω_x T_{k,x} + ω_y T_{k,y}) + ω²·ΔT_k.

Вторые производные ``ω`` берутся СИМВОЛЬНО из ``domain.omega_expr`` (sympy), вторые
производные базиса Чебышёва — через ``numpy.polynomial.chebyshev`` (как в basis.py,
но на порядок выше). Существующие модули не изменяются.
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.chebyshev as _cheb
import scipy.linalg as sla
import sympy as sp

from .basis import ChebyshevBasis
from .geometry import x as _sx
from .geometry import y as _sy
from .quadrature import interior_nodes


# --------------------------------------------------------------------------- #
#  Вторые производные ω (символьно) и базиса Чебышёва (через chebder)
# --------------------------------------------------------------------------- #
class _OmegaHessian:
    """Значения ω, ∇ω и вторых производных ω_xx, ω_yy в точках (символьно → numpy)."""

    def __init__(self, domain):
        w = domain.omega_expr
        self._w = sp.lambdify((_sx, _sy), w, "numpy")
        self._wx = sp.lambdify((_sx, _sy), sp.diff(w, _sx), "numpy")
        self._wy = sp.lambdify((_sx, _sy), sp.diff(w, _sy), "numpy")
        self._wxx = sp.lambdify((_sx, _sy), sp.diff(w, _sx, 2), "numpy")
        self._wyy = sp.lambdify((_sx, _sy), sp.diff(w, _sy, 2), "numpy")

    @staticmethod
    def _ev(fn, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        out = np.asarray(fn(X, Y), float)
        return out if out.shape == X.shape else np.broadcast_to(out, X.shape).astype(float)

    def fields(self, X, Y):
        """-> (ω, ω_x, ω_y, ω_xx, ω_yy) в точках (X, Y)."""
        return (self._ev(self._w, X, Y), self._ev(self._wx, X, Y), self._ev(self._wy, X, Y),
                self._ev(self._wxx, X, Y), self._ev(self._wyy, X, Y))


def _cheb_value_tables(basis: ChebyshevBasis, t: np.ndarray):
    """Таблицы T_n(t), T_n'(t), T_n''(t) (n=0..p) в точках t: три массива (p+1, *t.shape)."""
    p = basis.p
    V0 = np.moveaxis(_cheb.chebvander(t, p), -1, 0)               # T_n(t)
    eye = np.eye(p + 1)
    V1 = np.empty_like(V0)
    V2 = np.empty_like(V0)
    for n in range(p + 1):
        V1[n] = _cheb.chebval(t, _cheb.chebder(eye[n], 1)) if p >= 1 else 0.0
        V2[n] = _cheb.chebval(t, _cheb.chebder(eye[n], 2)) if p >= 2 else 0.0
    return V0, V1, V2


def _basis_second_derivs(basis: ChebyshevBasis, X, Y):
    """ΔT_k = T_{k,xx} + T_{k,yy} тензорного базиса Чебышёва в точках (X, Y): (N, *shape)."""
    xi, eta = basis.to_reference(X, Y)
    xmin, xmax, ymin, ymax = basis.bbox
    dxi_dx = 2.0 / (xmax - xmin)      # ξ_x (ξ линейна ⇒ ξ_xx = 0)
    deta_dy = 2.0 / (ymax - ymin)     # η_y
    Vx0, _, Vx2 = _cheb_value_tables(basis, xi)
    Vy0, _, Vy2 = _cheb_value_tables(basis, eta)
    # φ_xx = T_i''(ξ)ξ_x² · T_j(η);  φ_yy = T_i(ξ) · T_j''(η)η_y².
    phi_xx = (Vx2[:, None, ...] * Vy0[None, :, ...]) * dxi_dx**2
    phi_yy = (Vx0[:, None, ...] * Vy2[None, :, ...]) * deta_dy**2
    lap = (phi_xx + phi_yy).reshape(basis.N, *xi.shape)
    return lap


# --------------------------------------------------------------------------- #
#  Сборка системы Ритца для защемлённой бигармоники
# --------------------------------------------------------------------------- #
def _structure_laplacian(domain, basis, quad):
    r"""ψ = ω²T и Δψ = Δ(ω²T) в узлах квадратуры: матрицы (N, M)."""
    X, Y, W = quad.x, quad.y, quad.w
    om, omx, omy, omxx, omyy = _OmegaHessian(domain).fields(X, Y)
    grad2 = omx**2 + omy**2                       # |∇ω|²
    lap_om = omxx + omyy                          # Δω
    T = basis.values(X, Y)                        # (N, M)
    Tx, Ty = basis.grads(X, Y)                    # (N, M)
    lapT = _basis_second_derivs(basis, X, Y)      # ΔT_k            (N, M)
    psi = (om**2) * T                             # ψ_k = ω²T_k     (N, M)
    # Δψ_k = 2(|∇ω|²+ωΔω)T + 4ω(ω_x T_x + ω_y T_y) + ω²ΔT
    lap_psi = (2.0 * (grad2 + om * lap_om)) * T \
        + 4.0 * om * (omx * Tx + omy * Ty) \
        + (om**2) * lapT
    return psi, lap_psi, W


class ClampedPlate:
    """Изгиб защемлённой пластины: прямой Ритц по ``(D/2)∫(Δw)²`` на ``w = ω²Φ``."""

    def __init__(self, domain, basis, quad, cfg):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        self.cfg = cfg
        self.D = float(cfg.D)
        psi, lap_psi, W = _structure_laplacian(domain, basis, quad)
        self._psi = psi
        self._lap_psi = lap_psi                   # Δψ — кэш для Δw (A3: КТН, напряжения)
        self._W = W
        S = (lap_psi * W) @ lap_psi.T             # S[k,l] = ∫ Δψ_k Δψ_l
        self.S = 0.5 * (S + S.T)                  # подавить несимметрию округления
        # Факторизация ОДИН раз (A3.1): диагональное предобуславливание
        # (нормировка на √S_kk, NOTES §2) + Холецкий; решение произвольной
        # правой части далее — две треугольные подстановки (solve_rhs).
        self._d = 1.0 / np.sqrt(np.diag(self.S))
        Sn = (self.S * self._d).T * self._d       # diag(d) S diag(d)
        try:
            self._chol = sla.cho_factor(Sn * self.D)
            self._Sn_D = None
        except (sla.LinAlgError, np.linalg.LinAlgError):
            self._chol = None                     # потеря ПД — путь МНК
            self._Sn_D = Sn * self.D

    @classmethod
    def from_config(cls, domain, cfg) -> ClampedPlate:
        """Собрать решатель: базис степени ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg)

    @property
    def cond(self) -> float:
        """Число обусловленности cond(S) — диагностика устойчивости (NOTES.md §2)."""
        return float(np.linalg.cond(self.S))

    def solve_rhs(self, f_values) -> np.ndarray:
        r"""Коэффициенты ``c`` прогиба для ПРОИЗВОЛЬНОЙ нагрузки в узлах квадратуры.

        Решается ``(D·S) c = b``, ``b[k] = ∫ f ψ_k``, с диагональным
        предобуславливанием (нормировка на ``√S_kk``, NOTES §2); факторизация
        выполнена один раз в конструкторе — вызов дешёв (две подстановки),
        что делает защемлённый решатель пригодным для итераций МОР (A3).
        """
        b = (self._psi * self._W) @ np.asarray(f_values, float)   # b[k] = ∫ f ψ_k
        bn = b * self._d
        if self._chol is not None:
            cn = sla.cho_solve(self._chol, bn)
        else:
            cn = np.linalg.lstsq(self._Sn_D, bn, rcond=1e-13)[0]
        return cn * self._d                       # c = diag(d) ĉ

    def solve(self, q_values) -> np.ndarray:
        """Прежнее имя: тождественно :meth:`solve_rhs` (A3.2-т1)."""
        return self.solve_rhs(q_values)

    def solve_uniform(self, q: float | None = None) -> np.ndarray:
        """Коэффициенты ``c`` прогиба под равномерной нагрузкой ``q`` (по умолч. cfg.q0)."""
        q = self.cfg.q0 if q is None else float(q)
        return self.solve(np.full(self.quad.x.size, q))

    @staticmethod
    def _solve_spd(A, b) -> np.ndarray:
        """Решение SPD-системы; при потере положительной определённости — лин. МНК."""
        try:
            return sla.cho_solve(sla.cho_factor(A), b)
        except (sla.LinAlgError, np.linalg.LinAlgError):
            return np.linalg.lstsq(A, b, rcond=1e-13)[0]

    def deflection(self, c, X, Y) -> np.ndarray:
        """Прогиб ``w = ω²·Σ c_k T_k`` в точках (X, Y)."""
        Phi = self.basis.values(X, Y)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        om = self.domain.omega(X, Y)
        return om**2 * v

    def deflection_at_quad(self, c) -> np.ndarray:
        """Прогиб в узлах квадратуры через кэш ψ (один GEMV; A3.1)."""
        return np.tensordot(np.asarray(c, float), self._psi, axes=(0, 0))

    def laplacian_at_quad(self, c) -> np.ndarray:
        r"""Кривизна ``Δw = Σ c_k Δψ_k`` в узлах квадратуры (кэш Δψ; A3.1).

        Δψ = Δ(ω²T) собран символикой ω и chebder при построении матрицы S
        (машинерия bending_moments) — численного дифференцирования нет.
        """
        return np.tensordot(np.asarray(c, float), self._lap_psi, axes=(0, 0))

    # -- протокол контакта (общий с PlateBending; A3.3) -------------------- #
    def w_at_quad(self, state) -> np.ndarray:
        """Прогиб в узлах квадратуры по состоянию решения (state = c)."""
        return self.deflection_at_quad(state)

    def lap_w_at_quad(self, state) -> np.ndarray:
        """Кривизна Δw в узлах квадратуры по состоянию решения (state = c)."""
        return self.laplacian_at_quad(state)

    @staticmethod
    def coeffs_w(state) -> np.ndarray:
        """Коэффициенты прогиба из состояния решения (state = c)."""
        return state

    def w_max_on_grid(self, c, grid_n: int = 160) -> float:
        """Максимум прогиба по регулярной сетке bbox (маска ω>0)."""
        x0, x1, y0, y1 = self.domain.bbox
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
        inside = self.domain.omega(Xg, Yg) > 0.0
        return float(self.deflection(c, Xg[inside], Yg[inside]).max())


# --------------------------------------------------------------------------- #
#  Независимый МКЭ-эталон: ЗАЩЕМЛЁННАЯ пластина конформным C¹-элементом Аргириса
# --------------------------------------------------------------------------- #
# Выбор элемента (ТЗ): для защемления нужен КОНФОРМНЫЙ C¹-элемент. В установленной
# версии scikit-fem доступен Аргирис (полный C¹, степень 5) — он и используется.
# Защемление накладывается на ГРАНИЧНЫЕ степени свободы: значение ``u``, обе первые
# производные ``u_x, u_y`` (⇒ ∇w=0: и нормальная, и касательная) в вершинах и
# нормальная производная ``u_n`` на серединах рёбер. Вторые производные оставляем
# свободными (изгибающий момент на крае — часть решения; их фиксация пересжимала бы
# пластину). Проверено: на круге сходится к точной (4.1) ~0.17 %, на квадрате
# совпадает с RFM ~0.06 % (см. tests/test_clamped.py).
class ClampedFem:
    """МКЭ-решение защемлённой пластины (Аргирис); даёт ``w`` в произвольных точках."""

    def __init__(self, basis, w):
        self.basis = basis
        self.w = w

    def at(self, X, Y) -> np.ndarray:
        """Значения ``w`` в точках (X, Y) (интерполяция МКЭ-решения)."""
        P = np.ascontiguousarray(np.vstack([np.ravel(X), np.ravel(Y)]), dtype=float)
        return np.asarray(self.basis.interpolator(self.w)(P), dtype=float)

    def at_point(self, x: float, y: float) -> float:
        """Значение ``w`` в одной точке (скаляр)."""
        return float(np.ravel(self.at(np.array([x]), np.array([y])))[0])

    def w_max_on_grid(self, domain, grid_n: int = 160, eps: float = 1e-3) -> float:
        """Максимум прогиба по сетке домена (маска ω>eps) — сопоставимо с RFM.

        ``eps>0`` даёт отступ от границы: на криволинейной области МКЭ-сетка
        вписана (полигональна), узлы у самой границы могут лежать вне сетки.
        """
        x0, x1, y0, y1 = domain.bbox
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
        inside = domain.omega(Xg, Yg) > eps
        return float(np.max(self.at(Xg[inside], Yg[inside])))


def solve_clamped_fem(mesh, D: float, q: float, nu: float = 0.3) -> ClampedFem:
    """Решить изгиб ЗАЩЕМЛЁННОЙ пластины на сетке ``mesh`` элементом Аргириса (C¹)."""
    from skfem import (
        Basis,
        BilinearForm,
        ElementTriArgyris,
        LinearForm,
        condense,
        solve,
    )
    from skfem.helpers import dd, ddot, trace

    @BilinearForm
    def plate(u, v, w):
        return D * ((1 - nu) * ddot(dd(u), dd(v)) + nu * trace(dd(u)) * trace(dd(v)))

    @LinearForm
    def load(v, w):
        return q * v

    basis = Basis(mesh, ElementTriArgyris())
    dofs = basis.get_dofs()
    # Защемление: w=0 (u), ∇w=0 (u_x,u_y), ∂w/∂n=0 (u_n) на ∂Ω.
    ess = np.concatenate([dofs.all(nm) for nm in ("u", "u_x", "u_y", "u_n")])
    wsol = solve(*condense(plate.assemble(basis), load.assemble(basis), D=ess))
    return ClampedFem(basis, wsol)


def clamped_fem_circle(a: float, D: float, q: float, nu: float = 0.3, nref: int = 4) -> ClampedFem:
    """МКЭ-эталон для защемлённого КРУГА радиуса ``a`` (доп. контроль к формуле 4.1)."""
    from skfem import MeshTri

    mesh = MeshTri.init_circle(nref)        # единичный круг
    if a != 1.0:
        mesh = MeshTri(mesh.p * float(a), mesh.t)   # масштаб до радиуса a
    return solve_clamped_fem(mesh, D, q, nu)


def clamped_fem_lshape(
    D: float, q: float, nu: float = 0.3, *, side: float = 1.0, cut: float = 0.5,
    mesh_m: int = 16, refine: int = 3,
) -> ClampedFem:
    """МКЭ-эталон для защемлённой L-формы (та же сетка, что в verify_fem — единообразие)."""
    from .verify_fem import lshape_mesh

    return solve_clamped_fem(lshape_mesh(side, cut, mesh_m, refine), D, q, nu)


__all__ = [
    "ClampedPlate",
    "ClampedFem",
    "solve_clamped_fem",
    "clamped_fem_circle",
    "clamped_fem_lshape",
]
