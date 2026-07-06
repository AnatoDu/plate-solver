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
следствие: на криволинейной границе (круг) и во входящем угле
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
        self._wxy = sp.lambdify((_sx, _sy), sp.diff(w, _sx, _sy), "numpy")

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

    def fields_full(self, X, Y):
        """-> (ω, ω_x, ω_y, ω_xx, ω_yy, ω_xy) — с СМЕШАННОЙ производной."""
        return (*self.fields(X, Y), self._ev(self._wxy, X, Y))


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




def _structure_second_derivs(domain, basis, quad):
    r"""Все вторые производные структуры ψ = ω²T: (ψ, ψ_xx, ψ_yy, ψ_xy, W).

    Нужны полной билинейной форме: ψ_ab = [ω²]_ab T + [ω²]_a T_b
    + [ω²]_b T_a + ω² T_ab, где [ω²]_a = 2ωω_a, [ω²]_ab = 2(ω_aω_b + ωω_ab).
    """
    from .ladder import _basis_xy

    X, Y, W = quad.x, quad.y, quad.w
    om, omx, omy, omxx, omyy, omxy = _OmegaHessian(domain).fields_full(X, Y)
    T = basis.values(X, Y)
    Tx, Ty = basis.grads(X, Y)
    lapT = _basis_second_derivs(basis, X, Y)          # T_xx + T_yy
    Txy = _basis_xy(basis, X, Y)
    # T_xx и T_yy по отдельности
    xi, eta = basis.to_reference(X, Y)
    xmin, xmax, _ymin, _ymax = basis.bbox
    dxi_dx = 2.0 / (xmax - xmin)
    _, _, Vx2 = _cheb_value_tables(basis, xi)
    Vy0, _, _ = _cheb_value_tables(basis, eta)
    Txx = ((Vx2[:, None, ...] * Vy0[None, :, ...]) * dxi_dx**2).reshape(basis.N, *xi.shape)
    Tyy = lapT - Txx
    g2 = om**2
    gx, gy = 2.0 * om * omx, 2.0 * om * omy
    gxx = 2.0 * (omx**2 + om * omxx)
    gyy = 2.0 * (omy**2 + om * omyy)
    gxy = 2.0 * (omx * omy + om * omxy)
    psi = g2 * T
    psi_xx = gxx * T + 2.0 * gx * Tx + g2 * Txx
    psi_yy = gyy * T + 2.0 * gy * Ty + g2 * Tyy
    psi_xy = gxy * T + gx * Ty + gy * Tx + g2 * Txy
    return psi, psi_xx, psi_yy, psi_xy, W


def assemble_biharmonic_full(domain, basis, quad, nu: float):
    r"""Полная билинейная форма изгиба (NOTES §20):

    .. math:: a(w, v) = D\iint \big[\Delta w\,\Delta v - (1-\nu)
              (w_{xx} v_{yy} + w_{yy} v_{xx} - 2 w_{xy} v_{xy})\big]\, d\Omega

    (без множителя D — он в решателе). На защемлённой части границы гауссов
    член — нуль-лагранжиан (потому clamped-путь с ∫ΔψΔψ корректен);
    на опёртой он даёт естественное условие ИСТИННОГО шарнира M_n = 0.
    Возвращает (ψ, S_full, W).
    """
    psi, pxx, pyy, pxy, W = _structure_second_derivs(domain, basis, quad)
    lap = pxx + pyy
    S = (lap * W) @ lap.T \
        - (1.0 - nu) * ((pxx * W) @ pyy.T + (pyy * W) @ pxx.T
                        - 2.0 * (pxy * W) @ pxy.T)
    return psi, 0.5 * (S + S.T), W


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

    def load_vector(self, f_values) -> np.ndarray:
        """Вектор нагрузки ``b[k] = ∫ f ψ_k`` (внешние правые части, A4)."""
        return (self._psi * self._W) @ np.asarray(f_values, float)

    def solve_from_b(self, b) -> np.ndarray:
        """Решить по ГОТОВОМУ вектору нагрузки (факторизация уже сделана; A4)."""
        bn = np.asarray(b, float) * self._d
        if self._chol is not None:
            cn = sla.cho_solve(self._chol, bn)
        else:
            cn = np.linalg.lstsq(self._Sn_D, bn, rcond=1e-13)[0]
        return cn * self._d

    def structure_at(self, X, Y) -> np.ndarray:
        """Матрица структуры ψ_k = ω²·T_k в произвольных точках: (N, len(X)) (A4)."""
        Phi = self.basis.values(X, Y)
        om = self.domain.omega(X, Y)
        return om**2 * Phi

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

class MixedRectPlate:
    r"""Смешанные КУ на прямоугольнике (v0.3): w = (∏ω_c²)(∏ω_h)·Φ.

    Стороны ``sides = {"x1": ..., "x2": ..., "y1": ..., "y2": ...}`` со
    значениями ``clamped`` (множитель ω² — зануляет w и ∂w/∂n), ``hinge``
    (множитель ω — зануляет только w; статика ИСТИННОГО шарнира M_n = 0
    выходит ЕСТЕСТВЕННО из ПОЛНОЙ билинейной формы, NOTES §20) или ``free``
    (free: БЕЗ множителя — незакреплённая сторона; естественные условия
    полной формы: M_n = 0 и обобщённая перерезывающая Кирхгофа V_n = 0).
    Набор сторон обязан исключать жёсткие смещения (≥ 1 clamped либо
    ≥ 2 hinge). Все
    множители — полиномы (x − x₁), (x₂ − x), … ⇒ домен = bbox, квадратура
    Гаусса точна, изломов ω нет: полная форма работает без квадратурного
    пола маски (контраст с кривыми границами, см. PROGRESS C1).
    """

    def __init__(self, x1: float, x2: float, y1: float, y2: float,
                 sides: dict, cfg):
        import sympy as sp

        from .geometry import x as _gx
        from .geometry import y as _gy
        from .quadrature import interior_nodes

        wanted = {"x1", "x2", "y1", "y2"}
        if set(sides) != wanted or not all(v in ("clamped", "hinge", "free")
                                           for v in sides.values()):
            raise ValueError("sides: нужны все четыре стороны x1|x2|y1|y2 "
                             "со значениями clamped|hinge|free.")
        n_c = sum(1 for v in sides.values() if v == "clamped")
        n_h = sum(1 for v in sides.values() if v == "hinge")
        if n_c == 0 and n_h < 2:
            raise ValueError("sides: жёсткие смещения не исключены — нужно "
                             "≥ 1 clamped либо ≥ 2 hinge (ядро {1, x, y}).")
        self.cfg = cfg
        self.D = float(cfg.D)
        self.sides = dict(sides)
        factors = {"x1": _gx - x1, "x2": x2 - _gx, "y1": _gy - y1, "y2": y2 - _gy}
        g_expr = sp.Integer(1)
        for side, f in factors.items():
            if sides[side] == "clamped":
                g_expr *= f**2
            elif sides[side] == "hinge":
                g_expr *= f
            # free: БЕЗ множителя — сторона не несёт кинематических
            # условий; M_n = 0 и V_n = 0 выходят ЕСТЕСТВЕННО из полной формы
        self._g_expr = g_expr
        # маска/сетка: полиномиальная ω прямоугольника (>0 внутри)
        from .geometry import Domain

        self.domain = Domain(factors["x1"] * factors["x2"]
                             * factors["y1"] * factors["y2"], (x1, x2, y1, y2))
        self.basis = ChebyshevBasis(cfg.p, (x1, x2, y1, y2))
        self.quad = interior_nodes(self.domain, cfg.Q)
        # структура и её вторые производные (sympy → numpy)
        fns = {}
        for name, expr in (("g", g_expr),
                           ("gx", sp.diff(g_expr, _gx)),
                           ("gy", sp.diff(g_expr, _gy)),
                           ("gxx", sp.diff(g_expr, _gx, 2)),
                           ("gyy", sp.diff(g_expr, _gy, 2)),
                           ("gxy", sp.diff(g_expr, _gx, _gy))):
            fns[name] = sp.lambdify((_gx, _gy), expr, "numpy")
        self._fns = fns
        X, Y, W = self.quad.x, self.quad.y, self.quad.w
        g = self._ev("g", X, Y)
        gx = self._ev("gx", X, Y)
        gy = self._ev("gy", X, Y)
        gxx = self._ev("gxx", X, Y)
        gyy = self._ev("gyy", X, Y)
        gxy = self._ev("gxy", X, Y)
        T = self.basis.values(X, Y)
        Tx, Ty = self.basis.grads(X, Y)
        lapT = _basis_second_derivs(self.basis, X, Y)
        from .ladder import _basis_xy

        Txy = _basis_xy(self.basis, X, Y)
        xi, _ = self.basis.to_reference(X, Y)
        bx0, bx1, _, _ = self.basis.bbox
        _, _, Vx2 = _cheb_value_tables(self.basis, xi)
        _, eta = self.basis.to_reference(X, Y)
        Vy0, _, _ = _cheb_value_tables(self.basis, eta)
        Txx = ((Vx2[:, None, ...] * Vy0[None, :, ...])
               * (2.0 / (bx1 - bx0)) ** 2).reshape(self.basis.N, *xi.shape)
        Tyy = lapT - Txx
        psi = g * T
        pxx = gxx * T + 2.0 * gx * Tx + g * Txx
        pyy = gyy * T + 2.0 * gy * Ty + g * Tyy
        pxy = gxy * T + gx * Ty + gy * Tx + g * Txy
        lap = pxx + pyy
        nu = cfg.nu
        S = (lap * W) @ lap.T - (1.0 - nu) * ((pxx * W) @ pyy.T
                                              + (pyy * W) @ pxx.T
                                              - 2.0 * (pxy * W) @ pxy.T)
        self.S = 0.5 * (S + S.T)
        self._psi, self._W = psi, W
        self._lap_psi = lap
        self._d = 1.0 / np.sqrt(np.diag(self.S))
        Sn = (self.S * self._d).T * self._d
        try:
            self._chol = sla.cho_factor(Sn * self.D)
            self._Sn_D = None
        except (sla.LinAlgError, np.linalg.LinAlgError):
            self._chol = None
            self._Sn_D = Sn * self.D

    def _ev(self, name, X, Y):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        out = np.asarray(self._fns[name](X, Y), float)
        return out if out.shape == X.shape else np.broadcast_to(out, X.shape).astype(float)

    @classmethod
    def from_config(cls, x1, x2, y1, y2, sides, cfg) -> MixedRectPlate:
        return cls(x1, x2, y1, y2, sides, cfg)

    @property
    def cond(self) -> float:
        return float(np.linalg.cond(self.S))

    def solve_rhs(self, f_values) -> np.ndarray:
        b = (self._psi * self._W) @ np.asarray(f_values, float)
        bn = b * self._d
        if self._chol is not None:
            cn = sla.cho_solve(self._chol, bn)
        else:
            cn = np.linalg.lstsq(self._Sn_D, bn, rcond=1e-13)[0]
        return cn * self._d

    solve = solve_rhs

    def solve_uniform(self, q: float | None = None) -> np.ndarray:
        q = self.cfg.q0 if q is None else float(q)
        return self.solve_rhs(np.full(self.quad.x.size, q))

    def deflection(self, c, X, Y) -> np.ndarray:
        Phi = self.basis.values(X, Y)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        return self._ev("g", np.asarray(X, float), np.asarray(Y, float)) * v

    def deflection_at_quad(self, c) -> np.ndarray:
        return np.tensordot(np.asarray(c, float), self._psi, axes=(0, 0))

    def w_max_on_grid(self, c, grid_n: int = 160) -> float:
        x0, x1, y0, y1 = self.domain.bbox
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n),
                             np.linspace(y0, y1, grid_n))
        inside = self.domain.omega(Xg, Yg) > 0.0
        return float(np.max(np.abs(self.deflection(c, Xg[inside], Yg[inside]))))

    def moments_at(self, c, X, Y):
        """(Mx, My, Mxy) mixed-структуры g·Φ в точках (Лейбниц по g)."""
        from .ladder import _basis_xy

        c = np.asarray(c, float)
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        T = self.basis.values(X, Y)
        Tx, Ty = self.basis.grads(X, Y)
        lapT = _basis_second_derivs(self.basis, X, Y)
        Txy = _basis_xy(self.basis, X, Y)
        xi, eta = self.basis.to_reference(X, Y)
        bx0, bx1, _, _ = self.basis.bbox
        _, _, Vx2 = _cheb_value_tables(self.basis, xi)
        Vy0, _, _ = _cheb_value_tables(self.basis, eta)
        Txx = ((Vx2[:, None, ...] * Vy0[None, :, ...])
               * (2.0 / (bx1 - bx0)) ** 2).reshape(self.basis.N, *xi.shape)
        Tyy = lapT - Txx
        v = np.tensordot(c, T, axes=(0, 0))
        vx = np.tensordot(c, Tx, axes=(0, 0))
        vy = np.tensordot(c, Ty, axes=(0, 0))
        vxx = np.tensordot(c, Txx, axes=(0, 0))
        vyy = np.tensordot(c, Tyy, axes=(0, 0))
        vxy = np.tensordot(c, Txy, axes=(0, 0))
        g = self._ev("g", X, Y)
        gx = self._ev("gx", X, Y)
        gy = self._ev("gy", X, Y)
        gxx = self._ev("gxx", X, Y)
        gyy = self._ev("gyy", X, Y)
        gxy = self._ev("gxy", X, Y)
        wxx = gxx * v + 2.0 * gx * vx + g * vxx
        wyy = gyy * v + 2.0 * gy * vy + g * vyy
        wxy = gxy * v + gx * vy + gy * vx + g * vxy
        nu = self.cfg.nu
        return (-self.D * (wxx + nu * wyy), -self.D * (wyy + nu * wxx),
                -self.D * (1.0 - nu) * wxy)

    # протокол контакта — на будущее (v0.3 контакт при mixed не включён)
    def w_at_quad(self, state):
        return self.deflection_at_quad(state)

    @staticmethod
    def coeffs_w(state):
        return state

