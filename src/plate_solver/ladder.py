r"""ladder.py — верификационная лестница изгиба пластин (RFM + Ритц), общие кирпичи.

Здесь собраны переиспользуемые элементы лестницы из ТЗ «верификационная лестница»:

  • точные замкнутые решения (1D полоса, синус-нагрузка на прямоугольнике);
  • эталонные ряды Навье (равномерная нагрузка, шарнирный прямоугольник);
  • метод изготовленных решений (MMS): по заданному ``w`` строится ``q = D·Δ²w``;
  • 1D-решатель RFM + Ритц (структура шарнир ``ω·Φ`` / защемление ``ω²·Φ``);
  • вычисление изгибающих моментов из RFM-решения (гессиан структуры ``ω^p·Φ``).

2D-ступени (круг, прямоугольник, L-форма) переиспользуют ГОТОВЫЕ решатели
``plate_solver.plate.PlateBending`` (мягкий шарнир, ``ω·Φ``) и ``plate_solver.clamped.ClampedPlate``
(защемление, ``ω²·Φ``) — без изменений по существу. Контактные ступени (штамп, L)
считаются отдельно (МОР) и сюда не входят.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.polynomial.chebyshev as _cheb


# =========================================================================== #
#  СТУПЕНЬ 1 — точные замкнутые решения 1D (цилиндрический изгиб полосы)
# =========================================================================== #
def strip_hinge_exact(x, a: float, D: float, q: float) -> np.ndarray:
    r"""Шарнир (оба края): ``w = q/(24D)·(x⁴ − 2a x³ + a³ x)``."""
    x = np.asarray(x, float)
    return q / (24.0 * D) * (x**4 - 2.0 * a * x**3 + a**3 * x)


def strip_hinge_wmax(a: float, D: float, q: float) -> float:
    """Центр: ``w_max = 5 q a⁴ / (384 D)``."""
    return 5.0 * q * a**4 / (384.0 * D)


def strip_clamped_exact(x, a: float, D: float, q: float) -> np.ndarray:
    r"""Защемление (оба края): ``w = q/(24D)·x²(a−x)²``."""
    x = np.asarray(x, float)
    return q / (24.0 * D) * x**2 * (a - x) ** 2


def strip_clamped_wmax(a: float, D: float, q: float) -> float:
    """Центр: ``w_max = q a⁴ / (384 D)``."""
    return q * a**4 / (384.0 * D)


# --------------------------------------------------------------------------- #
#  1D Чебышёв: таблицы T_n, T_n', T_n'' в точках ξ ∈ [−1, 1]
# --------------------------------------------------------------------------- #
def _cheb_1d(p: int, xi: np.ndarray):
    """-> (T, T', T'') как массивы (p+1, M) в ξ-пространстве."""
    V0 = np.moveaxis(_cheb.chebvander(xi, p), -1, 0)              # (p+1, M)
    eye = np.eye(p + 1)
    V1 = np.empty_like(V0)
    V2 = np.empty_like(V0)
    for n in range(p + 1):
        V1[n] = _cheb.chebval(xi, _cheb.chebder(eye[n], 1)) if p >= 1 else 0.0
        V2[n] = _cheb.chebval(xi, _cheb.chebder(eye[n], 2)) if p >= 2 else 0.0
    return V0, V1, V2


@dataclass
class Strip1DResult:
    """Решение 1D-полосы: коэффициенты + средства вычисления поля."""

    a: float
    D: float
    p_struct: int
    c: np.ndarray
    cond: float

    def deflection(self, x) -> np.ndarray:
        """``w(x) = ω^p · Σ c_k T_k(ξ)``,  ω = x(a−x)/a,  ξ = (2x−a)/a."""
        x = np.asarray(x, float)
        a = self.a
        xi = (2.0 * x - a) / a
        om = x * (a - x) / a
        T = np.moveaxis(_cheb.chebvander(xi, self.c.size - 1), -1, 0)
        v = np.tensordot(self.c, T, axes=(0, 0))
        return om**self.p_struct * v

    def w_center(self) -> float:
        return float(np.ravel(self.deflection(np.array([self.a / 2.0])))[0])


def solve_strip_1d(a: float, D: float, q: float, support: str, p_deg: int,
                   n_quad: int = 400) -> Strip1DResult:
    r"""1D RFM + Ритц: минимизация ``(D/2)∫(w'')² − ∫ q w`` на ``w = ω^p Φ``.

    Структура: ``ω = x(a−x)/a`` (единичный наклон на краях); ``p=1`` — шарнир
    (краевое ``w=0`` существенное, ``w''=0`` естественное ⇒ истинный шарнир в 1D,
    парадокса нет); ``p=2`` — защемление (``w=0`` и ``w'=0`` существенные).
    Базис ``Φ = Σ c_k T_k(ξ)``, ``ξ=(2x−a)/a``. Решается ``(D·S)c=b``,
    ``S[k,l]=∫ ψ_k'' ψ_l''``, ``b[k]=∫ q ψ_k``.
    """
    p_struct = {"hinge": 1, "clamped": 2}[support]
    t, wt = np.polynomial.legendre.leggauss(n_quad)
    x = 0.5 * a * (t + 1.0)                       # узлы на [0, a]
    W = 0.5 * a * wt
    xi = (2.0 * x - a) / a
    xix = 2.0 / a                                 # ξ_x
    T, Tp_xi, Tpp_xi = _cheb_1d(p_deg, xi)
    Tx = Tp_xi * xix                              # T_k'(x)
    Txx = Tpp_xi * xix**2                         # T_k''(x)

    om = x * (a - x) / a
    omx = 1.0 - 2.0 * x / a
    omxx = np.full_like(x, -2.0 / a)
    ps = p_struct
    om_p = om**ps
    # ψ = ω^p T;  ψ'' = p(p−1)ω^{p−2}ω_x² T + p ω^{p−1}ω_xx T + 2p ω^{p−1}ω_x T_x + ω^p T_xx
    psi = om_p * T
    term_a = (ps * (ps - 1) * om ** (ps - 2) * omx**2 + ps * om ** (ps - 1) * omxx) * T
    term_b = 2.0 * ps * om ** (ps - 1) * omx * Tx
    psi_xx = term_a + term_b + om_p * Txx

    S = (psi_xx * W) @ psi_xx.T
    S = 0.5 * (S + S.T)
    b = (psi * W) @ np.full_like(x, q)
    d = 1.0 / np.sqrt(np.diag(S))
    Sn = (S * d).T * d
    try:
        import scipy.linalg as sla
        cn = sla.cho_solve(sla.cho_factor(Sn * D), b * d)
    except Exception:
        cn = np.linalg.lstsq(Sn * D, b * d, rcond=1e-13)[0]
    c = cn * d
    return Strip1DResult(a=a, D=D, p_struct=ps, c=c, cond=float(np.linalg.cond(S)))


# =========================================================================== #
#  СТУПЕНЬ 4a — синусоидальная нагрузка на шарнирном прямоугольнике (точно)
# =========================================================================== #
def rect_sin_load(X, Y, a: float, b: float, q0: float = 1.0) -> np.ndarray:
    """Нагрузка ``q = q0·sin(πx/a)·sin(πy/b)``."""
    return q0 * np.sin(np.pi * np.asarray(X, float) / a) * np.sin(np.pi * np.asarray(Y, float) / b)


def rect_sin_exact(X, Y, a: float, b: float, D: float, q0: float = 1.0) -> np.ndarray:
    r"""Точное решение SS-прямоугольника под синус-нагрузкой (один член Навье)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    amp = q0 / (np.pi**4 * D * (1.0 / a**2 + 1.0 / b**2) ** 2)
    return amp * np.sin(np.pi * X / a) * np.sin(np.pi * Y / b)


def rect_sin_wmax(a: float, b: float, D: float, q0: float = 1.0) -> float:
    """``w_max`` (центр) синус-нагрузки: ``q0/[π⁴D(1/a²+1/b²)²]``."""
    return q0 / (np.pi**4 * D * (1.0 / a**2 + 1.0 / b**2) ** 2)


# =========================================================================== #
#  СТУПЕНЬ 4b — равномерная нагрузка на шарнирном прямоугольнике (ряд Навье)
# =========================================================================== #
def _navier_odd(n_terms: int):
    return np.arange(1, 2 * n_terms, 2)        # 1, 3, 5, ... (n_terms нечётных)


def navier_uniform(X, Y, a: float, b: float, D: float, q0: float = 1.0, n_terms: int = 50):
    r"""Ряд Навье для равномерной нагрузки на SS-прямоугольнике (прогиб ``w``)."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    m = _navier_odd(n_terms)
    out = np.zeros(np.broadcast(X, Y).shape)
    pref = 16.0 * q0 / (np.pi**6 * D)
    for mm in m:
        sx = np.sin(mm * np.pi * X / a)
        for nn in m:
            denom = mm * nn * ((mm / a) ** 2 + (nn / b) ** 2) ** 2
            out = out + sx * np.sin(nn * np.pi * Y / b) / denom
    return pref * out


def navier_uniform_center(a: float, b: float, D: float, q0: float = 1.0, nu: float = 0.3,
                          n_terms: int = 50):
    r"""Центр прямоугольника: (w_max, M_x) из ряда Навье (контроль табличных констант)."""
    xc, yc = a / 2.0, b / 2.0
    m = _navier_odd(n_terms)
    w = 0.0
    mx = 0.0
    pref_w = 16.0 * q0 / (np.pi**6 * D)
    pref_m = 16.0 * q0 / (np.pi**4)
    for mm in m:
        sx = np.sin(mm * np.pi * xc / a)
        for nn in m:
            sy = np.sin(nn * np.pi * yc / b)
            k = (mm / a) ** 2 + (nn / b) ** 2
            w += sx * sy / (mm * nn * k**2) * pref_w
            # M_x = −D(w_xx + ν w_yy); коэффициент при члене: ((mπ/a)²+ν(nπ/b)²)
            mx += sx * sy * ((mm / a) ** 2 + nu * (nn / b) ** 2) / (mm * nn * k**2) * pref_m
    return float(w), float(mx)


# =========================================================================== #
#  СТУПЕНЬ 3 — метод изготовленных решений (MMS): q = D·Δ²w (символьно)
# =========================================================================== #
def mms_load_and_exact(w_expr, D: float):
    r"""По sympy-выражению ``w(x,y)`` вернуть (q_func, w_func, lap2_func): ``q = D·Δ²w``.

    Бигармониан считается СИМВОЛЬНО (sympy) и компилируется в numpy. Это точный
    эталон при любой ν: подаём ``q`` в защемлённый решатель и сверяем с ``w``.
    """
    import sympy as sp

    from .geometry import x as sx
    from .geometry import y as sy

    lap = sp.diff(w_expr, sx, 2) + sp.diff(w_expr, sy, 2)
    lap2 = sp.diff(lap, sx, 2) + sp.diff(lap, sy, 2)
    q_expr = D * lap2
    w_func = sp.lambdify((sx, sy), w_expr, "numpy")
    q_func = sp.lambdify((sx, sy), q_expr, "numpy")

    def _wrap(fn):
        def g(X, Y):
            X = np.asarray(X, float)
            Y = np.asarray(Y, float)
            out = np.asarray(fn(X, Y), float)
            return out if out.shape == X.shape else np.broadcast_to(out, X.shape).astype(float)
        return g

    return _wrap(q_func), _wrap(w_func)


def mms_clamped_rect_w(a: float, b: float):
    """MMS-поле для защемлённого прямоугольника: ``w = (x²−a²)²(y²−b²)²`` (sympy)."""
    from .geometry import x as sx
    from .geometry import y as sy

    return (sx**2 - a**2) ** 2 * (sy**2 - b**2) ** 2


def mms_clamped_disk_w(R: float, D: float):
    """MMS-поле для защемлённого круга: ``w = (R²−x²−y²)²/(64D)`` (q=1; sympy)."""
    from .geometry import x as sx
    from .geometry import y as sy

    return (R**2 - sx**2 - sy**2) ** 2 / (64.0 * D)


# =========================================================================== #
#  Изгибающие моменты из RFM-решения: гессиан структуры ω^p·Φ
# =========================================================================== #
def _basis_xx_yy(basis, X, Y):
    """φ_{k,xx} и φ_{k,yy} тензорного базиса Чебышёва: два массива (N, *shape)."""
    from .clamped import _cheb_value_tables

    xi, eta = basis.to_reference(X, Y)
    xmin, xmax, ymin, ymax = basis.bbox
    dxi_dx = 2.0 / (xmax - xmin)
    deta_dy = 2.0 / (ymax - ymin)
    Vx0, _, Vx2 = _cheb_value_tables(basis, xi)
    Vy0, _, Vy2 = _cheb_value_tables(basis, eta)
    phi_xx = ((Vx2[:, None, ...] * Vy0[None, :, ...]) * dxi_dx**2).reshape(basis.N, *xi.shape)
    phi_yy = ((Vx0[:, None, ...] * Vy2[None, :, ...]) * deta_dy**2).reshape(basis.N, *xi.shape)
    return phi_xx, phi_yy


def bending_moments(domain, basis, c, p_struct: int, D: float, nu: float, X, Y):
    r"""Изгибающие моменты ``(M_x, M_y)`` RFM-решения ``w = ω^p·Σ c_k φ_k`` в точках.

    ``M_x = −D(w_xx + ν w_yy)``, ``M_y = −D(w_yy + ν w_xx)``. Гессиан структуры:
    ``w_xx = [ω^p]_xx v + 2[ω^p]_x v_x + ω^p v_xx`` (и аналогично по y), где
    ``[ω^p]_x = p ω^{p−1}ω_x``, ``[ω^p]_xx = p(p−1)ω^{p−2}ω_x² + p ω^{p−1}ω_xx``.
    Работает и для мягкого шарнира (p=1), и для защемления (p=2).
    """
    from .clamped import _OmegaHessian

    c = np.asarray(c, float)
    om, omx, omy, omxx, omyy = _OmegaHessian(domain).fields(X, Y)
    T = basis.values(X, Y)
    Tx, Ty = basis.grads(X, Y)
    Txx, Tyy = _basis_xx_yy(basis, X, Y)
    v = np.tensordot(c, T, axes=(0, 0))
    vx = np.tensordot(c, Tx, axes=(0, 0))
    vy = np.tensordot(c, Ty, axes=(0, 0))
    vxx = np.tensordot(c, Txx, axes=(0, 0))
    vyy = np.tensordot(c, Tyy, axes=(0, 0))

    p = p_struct
    om_p = om**p
    om_pm1 = om ** (p - 1)
    quad = (p * (p - 1)) * (om ** (p - 2) if p >= 2 else 0.0)   # коэф. при ω_x² (0 при p=1)
    wxx = (quad * omx**2 + p * om_pm1 * omxx) * v + 2.0 * p * om_pm1 * omx * vx + om_p * vxx
    wyy = (quad * omy**2 + p * om_pm1 * omyy) * v + 2.0 * p * om_pm1 * omy * vy + om_p * vyy
    Mx = -D * (wxx + nu * wyy)
    My = -D * (wyy + nu * wxx)
    return Mx, My


__all__ = [
    "strip_hinge_exact", "strip_hinge_wmax",
    "strip_clamped_exact", "strip_clamped_wmax",
    "Strip1DResult", "solve_strip_1d",
    "rect_sin_load", "rect_sin_exact", "rect_sin_wmax",
    "navier_uniform", "navier_uniform_center",
    "mms_load_and_exact", "mms_clamped_rect_w", "mms_clamped_disk_w",
    "bending_moments",
]
