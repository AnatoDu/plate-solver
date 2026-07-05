r"""radial.py — круглая пластина как ОДНОМЕРНАЯ краевая задача по радиусу (осесимметрия).

«Переход в полярные координаты»: осесимметричная бигармоника по ``w(r)``,

    Δ²w = q/D,   Δw = w'' + (1/r) w' = (1/r)(r w')',

решается 1D методом Ритца по ``r ∈ [0, a]`` (как ступень лестницы, но в радиальной
энергии). Нужна для верификации 1D ↔ 2D ↔ аналитика (доклад по симметрии).

Защемление (``w(a)=0, w'(a)=0``; регулярность в центре) — прямая радиальная энергия

    U = πD ∫₀ᵃ [w''² + w'²/r² + 2ν w''w'/r] r dr,

структура ``w = (a²−r²)²·Φ(r²)`` (зануляет ``w`` и ``w'`` на ``r=a``, чётна по ``r``
⇒ регулярна в центре, ``w'(0)=0``). Точное решение ``w=q(a²−r²)²/(64D)`` лежит в
структуре (Φ=const) ⇒ Ритц воспроизводит его машинно.

Шарнир (опц.) — реализуется КАК В 2D, расщеплением на две радиальные Пуассона
(``−ΔM=q, M(a)=0``; ``−Δw=M/D, w(a)=0``), т.е. «мягкий» шарнир; структура
``(a²−r²)·Φ(r²)``. Так 1D-результат сопоставим с 2D-RFM (мягким) и его эталоном
``w=q(a²−r²)(3a²−r²)/(64D)``.

Базис ``Φ`` — полиномы Чебышёва от ``u = 2(r/a)²−1`` (функции ``r²`` ⇒ чётность и
гладкость в центре). Энергия с весом ``r`` и членами ``1/r, 1/r²`` регулярна при
``w'(0)=0``; интегралы берём Гауссом по внутренним узлам (``r=0`` не попадает).
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.chebyshev as _cheb


def _cheb_even_tables(p: int, r: np.ndarray, a: float):
    """φ_k=T_k(u), u=2(r/a)²−1, и производные по r: (φ, φ', φ'') — массивы (p+1, M)."""
    u = 2.0 * (r / a) ** 2 - 1.0
    up = 4.0 * r / a**2                       # u'(r)
    upp = np.full_like(r, 4.0 / a**2)         # u''(r)
    eye = np.eye(p + 1)
    T = np.moveaxis(_cheb.chebvander(u, p), -1, 0)
    T1 = np.array([_cheb.chebval(u, _cheb.chebder(eye[k], 1)) if p >= 1 else 0.0 * r
                   for k in range(p + 1)])
    T2 = np.array([_cheb.chebval(u, _cheb.chebder(eye[k], 2)) if p >= 2 else 0.0 * r
                   for k in range(p + 1)])
    phi = T
    phip = T1 * up
    phipp = T2 * up**2 + T1 * upp
    return phi, phip, phipp


def _gauss_radial(a: float, nq: int):
    t, wt = np.polynomial.legendre.leggauss(nq)
    return 0.5 * a * (t + 1.0), 0.5 * a * wt      # r, W на [0, a]


class RadialClamped:
    """Защемлённая круглая пластина как 1D-задача: прямая радиальная энергия."""

    def __init__(self, a: float, D: float, p: int = 6, nq: int = 400):
        self.a, self.D, self.p = a, D, p
        r, W = _gauss_radial(a, nq)
        phi, phip, phipp = _cheb_even_tables(p, r, a)
        g = (a**2 - r**2) ** 2                       # множитель структуры (p=2)
        gp = -4.0 * r * (a**2 - r**2)
        gpp = -4.0 * a**2 + 12.0 * r**2
        psi = g * phi
        psip = gp * phi + g * phip
        psipp = gpp * phi + 2.0 * gp * phip + g * phipp
        self._psi, self._r, self._W = psi, r, W
        self._psip, self._psipp = psip, psipp      # ν входит в K на этапе solve

    def solve(self, q: float, nu: float):
        """Коэффициенты под равномерной нагрузкой ``q``; ν входит в энергию."""
        r, W = self._r, self._W
        psip, psipp = self._psip, self._psipp
        # K_jk = D ∫ [ψ''ψ'' r + ψ'ψ'/r + ν(ψ''ψ' + ψ'ψ'')] dr
        integ = (psipp[:, None, :] * psipp[None, :, :] * r
                 + psip[:, None, :] * psip[None, :, :] / r
                 + nu * (psipp[:, None, :] * psip[None, :, :]
                         + psip[:, None, :] * psipp[None, :, :]))
        K = self.D * np.einsum("jkq,q->jk", integ, W)
        f = np.einsum("jq,q->j", self._psi, np.full_like(r, q) * r * W)   # ∫ q ψ r dr
        self.cond = float(np.linalg.cond(K))
        self.coef = np.linalg.solve(K, f)
        return self.coef

    def deflection(self, r) -> np.ndarray:
        r = np.asarray(r, float)
        u = 2.0 * (r / self.a) ** 2 - 1.0
        T = np.moveaxis(_cheb.chebvander(u, self.p), -1, 0)
        return (self.a**2 - r**2) ** 2 * np.tensordot(self.coef, T, axes=(0, 0))


class RadialPoisson:
    """Радиальная Пуассона ``−Δu=f, u(a)=0``; структура ``(a²−r²)·Φ(r²)`` (мягкий шарнир)."""

    def __init__(self, a: float, p: int = 6, nq: int = 400):
        self.a, self.p = a, p
        r, W = _gauss_radial(a, nq)
        phi, phip, _ = _cheb_even_tables(p, r, a)
        g = a**2 - r**2
        gp = -2.0 * r
        psi = g * phi
        psip = gp * phi + g * phip
        # A_jk = ∫ u_j' u_k' r dr  (осесимметричная форма Дирихле)
        self.A = np.einsum("jq,kq,q->jk", psip, psip, r * W)
        self._psi, self._r, self._W = psi, r, W

    def solve(self, f_values) -> np.ndarray:
        b = np.einsum("jq,q->j", self._psi, np.asarray(f_values, float) * self._r * self._W)
        return np.linalg.solve(self.A, b)

    def eval_nodes(self, coef) -> np.ndarray:
        """Значения u в узлах квадратуры (для правой части второй Пуассоны)."""
        return np.tensordot(np.asarray(coef, float), self._psi, axes=(0, 0))

    def deflection(self, coef, r) -> np.ndarray:
        r = np.asarray(r, float)
        u = 2.0 * (r / self.a) ** 2 - 1.0
        T = np.moveaxis(_cheb.chebvander(u, self.p), -1, 0)
        return (self.a**2 - r**2) * np.tensordot(np.asarray(coef, float), T, axes=(0, 0))


def solve_radial_soft_hinge(a: float, D: float, q: float, p: int = 6, nq: int = 400):
    r"""Мягкий шарнир радиально: ``M=Пуассона(q)``, затем ``w=Пуассона(M/D)``.

    Returns ``(rp, cw)``: решатель и коэффициенты ``w``; поле — ``rp.deflection(cw, r)``.
    """
    rp = RadialPoisson(a, p, nq)
    cM = rp.solve(np.full(rp._r.size, q))          # −ΔM=q, M(a)=0
    M_nodes = rp.eval_nodes(cM)
    cw = rp.solve(M_nodes / D)                      # −Δw=M/D, w(a)=0
    return rp, cw


# --------------------------------------------------------------------------- #
#  Обобщение на интервал [b, a], b > 0 — кольцо (P3.3 фазы 2)
# --------------------------------------------------------------------------- #
# На кольце центр не принадлежит области ⇒ чётность по r не нужна: базис —
# обычные Чебышёвы от t = (2r − (a+b))/(a−b) ∈ [−1, 1]. Члены 1/r, 1/r²
# энергии регулярны при r ≥ b > 0. Случай b = 0 — прежний путь
# (RadialClamped / RadialPoisson с чётной структурой и регулярностью в центре).


def _cheb_interval_tables(p: int, r: np.ndarray, a: float, b: float):
    """φ_k = T_k(t), t = (2r − (a+b))/(a−b); производные по r: (φ, φ', φ'')."""
    t = (2.0 * r - (a + b)) / (a - b)
    tp = 2.0 / (a - b)                            # t'(r) — константа
    eye = np.eye(p + 1)
    T = np.moveaxis(_cheb.chebvander(t, p), -1, 0)
    T1 = np.array([_cheb.chebval(t, _cheb.chebder(eye[k], 1)) if p >= 1 else 0.0 * r
                   for k in range(p + 1)])
    T2 = np.array([_cheb.chebval(t, _cheb.chebder(eye[k], 2)) if p >= 2 else 0.0 * r
                   for k in range(p + 1)])
    return T, T1 * tp, T2 * tp**2


def _gauss_interval(a: float, b: float, nq: int):
    t, wt = np.polynomial.legendre.leggauss(nq)
    return 0.5 * (a - b) * (t + 1.0) + b, 0.5 * (a - b) * wt      # r, W на [b, a]


class RadialClampedAnnulus:
    r"""Защемлённое кольцо b < r < a как 1D-задача (прямая радиальная энергия).

    Структура ``w = ω²·Φ(r)``, ω = (a − r)(r − b): тождественно зануляет
    ``w`` и ``w'`` на ОБОИХ краях. Энергия та же, что у :class:`RadialClamped`:

    .. math:: U = \pi D \int_b^a \big[w''^2 + w'^2/r^2 + 2\nu\,w'' w'/r\big]\, r\, dr.

    Решение кольца содержит ln r — на [b, a] (b > 0) он аналитичен, поэтому
    полиномиальный Ритц сходится спектрально (ворота: rel < 1e-8 при p=16).
    """

    def __init__(self, a: float, b: float, D: float, p: int = 16, nq: int = 400):
        if not 0.0 < b < a:
            raise ValueError("Кольцо требует 0 < b < a.")
        self.a, self.b, self.D, self.p = a, b, D, p
        r, W = _gauss_interval(a, b, nq)
        phi, phip, phipp = _cheb_interval_tables(p, r, a, b)
        g = ((a - r) * (r - b)) ** 2                 # ω², ω = (a−r)(r−b)
        om = (a - r) * (r - b)
        omp = a + b - 2.0 * r                        # ω'
        gp = 2.0 * om * omp                          # (ω²)'
        gpp = 2.0 * (omp**2 - 2.0 * om)              # (ω²)'' (ω'' = −2)
        psi = g * phi
        psip = gp * phi + g * phip
        psipp = gpp * phi + 2.0 * gp * phip + g * phipp
        self._psi, self._r, self._W = psi, r, W
        self._psip, self._psipp = psip, psipp

    def solve(self, q: float, nu: float):
        """Коэффициенты под равномерной нагрузкой ``q``; ν входит в энергию."""
        r, W = self._r, self._W
        psip, psipp = self._psip, self._psipp
        integ = (psipp[:, None, :] * psipp[None, :, :] * r
                 + psip[:, None, :] * psip[None, :, :] / r
                 + nu * (psipp[:, None, :] * psip[None, :, :]
                         + psip[:, None, :] * psipp[None, :, :]))
        K = self.D * np.einsum("jkq,q->jk", integ, W)
        f = np.einsum("jq,q->j", self._psi, np.full_like(r, q) * r * W)
        self.cond = float(np.linalg.cond(K))
        self.coef = np.linalg.solve(K, f)
        return self.coef

    def deflection(self, r) -> np.ndarray:
        r = np.asarray(r, float)
        phi, _, _ = _cheb_interval_tables(self.p, r, self.a, self.b)
        return ((self.a - r) * (r - self.b)) ** 2 * np.tensordot(self.coef, phi, axes=(0, 0))


class RadialPoissonAnnulus:
    r"""Радиальная Пуассона на кольце: ``−Δu = f``, ``u(a) = u(b) = 0``.

    Структура ``u = ω·Φ(r)``, ω = (a − r)(r − b); осесимметричная форма
    Дирихле ``A_jk = ∫_b^a ψ_j' ψ_k' r dr`` (кирпич «мягкого шарнира»).
    """

    def __init__(self, a: float, b: float, p: int = 16, nq: int = 400):
        if not 0.0 < b < a:
            raise ValueError("Кольцо требует 0 < b < a.")
        self.a, self.b, self.p = a, b, p
        r, W = _gauss_interval(a, b, nq)
        phi, phip, _ = _cheb_interval_tables(p, r, a, b)
        g = (a - r) * (r - b)
        gp = a + b - 2.0 * r
        psi = g * phi
        psip = gp * phi + g * phip
        self.A = np.einsum("jq,kq,q->jk", psip, psip, r * W)
        self._psi, self._r, self._W = psi, r, W

    def solve(self, f_values) -> np.ndarray:
        b_vec = np.einsum("jq,q->j", self._psi,
                          np.asarray(f_values, float) * self._r * self._W)
        return np.linalg.solve(self.A, b_vec)

    def eval_nodes(self, coef) -> np.ndarray:
        """Значения u в узлах квадратуры (правая часть второй Пуассоны)."""
        return np.tensordot(np.asarray(coef, float), self._psi, axes=(0, 0))

    def deflection(self, coef, r) -> np.ndarray:
        r = np.asarray(r, float)
        phi, _, _ = _cheb_interval_tables(self.p, r, self.a, self.b)
        return ((self.a - r) * (r - self.b)) * np.tensordot(np.asarray(coef, float),
                                                            phi, axes=(0, 0))


def solve_radial_soft_hinge_annulus(a: float, b: float, D: float, q: float,
                                    p: int = 16, nq: int = 400):
    r"""Мягкий шарнир на кольце радиально: две Пуассоны, как в 2D-расщеплении.

    ``−ΔM = q, M(a)=M(b)=0``; затем ``−Δw = M/D, w(a)=w(b)=0``.
    Returns ``(rp, cw)``; поле — ``rp.deflection(cw, r)``.
    """
    rp = RadialPoissonAnnulus(a, b, p, nq)
    cM = rp.solve(np.full(rp._r.size, q))
    M_nodes = rp.eval_nodes(cM)
    cw = rp.solve(M_nodes / D)
    return rp, cw


__all__ = [
    "RadialClamped",
    "RadialPoisson",
    "solve_radial_soft_hinge",
    "RadialClampedAnnulus",
    "RadialPoissonAnnulus",
    "solve_radial_soft_hinge_annulus",
]
