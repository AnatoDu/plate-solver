r"""basis.py — тензорный базис Чебышёва Φ и его частные производные.

Неопределённая компонента структуры ``v = ω·Φ`` ищется в виде
    Φ(x, y) = Σ_k c_k φ_k(x, y),   φ_k = T_i(ξ(x)) · T_j(η(y)),   i, j = 0..p,
где ``T_n`` — полином Чебышёва 1-го рода, а ``ξ, η ∈ [−1, 1]`` — образы координат
при линейном отображении из описанного прямоугольника (bbox):

    ξ = (2x − (x_max + x_min)) / (x_max − x_min),   ξ_x = 2/(x_max − x_min),
    η = (2y − (y_max + y_min)) / (y_max − y_min),   η_y = 2/(y_max − y_min).

Линейный порядок мультииндексов: ``k = i·(p+1) + j`` (сначала i, потом j).
Производные по x, y нужны для сборки ``∇ψ = ∇ω·φ + ω·∇φ`` (assembler.py):
    ∂φ_k/∂x = T_i'(ξ)·ξ_x · T_j(η),   ∂φ_k/∂y = T_i(ξ) · T_j'(η)·η_y.

Значения T_n и их производные берём через ``numpy.polynomial.chebyshev``
(chebvander для значений; матрица дифференцирования chebder — для производных),
без ручной выписки рекуррентных формул. Чебышёв — ради устойчивости при больших p
(NOTES.md §2).
"""

from __future__ import annotations

import numpy as np
import numpy.polynomial.chebyshev as _cheb

BBox = tuple[float, float, float, float]  # (x_min, x_max, y_min, y_max)


class ChebyshevBasis:
    """Тензорный базис ``T_i ⊗ T_j`` на bbox, отображённом в [−1, 1]²."""

    def __init__(self, p: int, bbox: BBox):
        if p < 0:
            raise ValueError("Степень p должна быть неотрицательной.")
        xmin, xmax, ymin, ymax = map(float, bbox)
        if not (xmin < xmax and ymin < ymax):
            raise ValueError("Некорректный bbox: нужно x_min<x_max и y_min<y_max.")
        self.p = int(p)
        self.bbox: BBox = (xmin, xmax, ymin, ymax)
        # Параметры аффинного отображения bbox → [−1, 1]².
        self._bx = (xmax + xmin) / 2.0
        self._ax = (xmax - xmin) / 2.0
        self._by = (ymax + ymin) / 2.0
        self._ay = (ymax - ymin) / 2.0
        self._dxi_dx = 1.0 / self._ax     # ξ_x = 2/(x_max−x_min)
        self._deta_dy = 1.0 / self._ay    # η_y = 2/(y_max−y_min)
        # Матрица дифференцирования Чебышёва: строка n — коэффициенты T_n' по T_m.
        if self.p >= 1:
            eye = np.eye(self.p + 1)
            self._Dcoef = np.array([_cheb.chebder(eye[n]) for n in range(self.p + 1)])
        else:
            self._Dcoef = None

    @property
    def N(self) -> int:
        """Число базисных функций N = (p+1)²."""
        return (self.p + 1) ** 2

    @property
    def index_pairs(self):
        """Список мультииндексов (i, j) в порядке k = i·(p+1)+j."""
        p = self.p
        return [(i, j) for i in range(p + 1) for j in range(p + 1)]

    @classmethod
    def from_domain(cls, domain, p: int) -> ChebyshevBasis:
        """Удобный конструктор: bbox берётся из области (Domain.bbox)."""
        return cls(p, tuple(domain.bbox))

    # -- отображение в [−1, 1]² ------------------------------------------ #
    def to_reference(self, X, Y) -> tuple[np.ndarray, np.ndarray]:
        """Аффинно отобразить (X, Y) из bbox в (ξ, η) ∈ [−1, 1]²."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        return (X - self._bx) / self._ax, (Y - self._by) / self._ay

    # -- значения и производные ------------------------------------------ #
    def values(self, X, Y) -> np.ndarray:
        """Значения φ_k в точках: массив формы ``(N, *X.shape)``."""
        xi, eta = self.to_reference(X, Y)
        Tx = np.moveaxis(_cheb.chebvander(xi, self.p), -1, 0)   # (p+1, *S)
        Ty = np.moveaxis(_cheb.chebvander(eta, self.p), -1, 0)  # (p+1, *S)
        phi = Tx[:, None, ...] * Ty[None, :, ...]               # (p+1, p+1, *S)
        return phi.reshape(self.N, *xi.shape)

    def grads(self, X, Y) -> tuple[np.ndarray, np.ndarray]:
        """Частные производные (∂φ_k/∂x, ∂φ_k/∂y): два массива формы ``(N, *X.shape)``."""
        xi, eta = self.to_reference(X, Y)
        Tx = np.moveaxis(_cheb.chebvander(xi, self.p), -1, 0)   # (p+1, *S)
        Ty = np.moveaxis(_cheb.chebvander(eta, self.p), -1, 0)
        Txd = self._deriv_table(xi)                             # T_i'(ξ): (p+1, *S)
        Tyd = self._deriv_table(eta)                            # T_j'(η)
        dphidx = (Txd[:, None, ...] * Ty[None, :, ...]).reshape(self.N, *xi.shape)
        dphidy = (Tx[:, None, ...] * Tyd[None, :, ...]).reshape(self.N, *xi.shape)
        return dphidx * self._dxi_dx, dphidy * self._deta_dy

    def _deriv_table(self, t: np.ndarray) -> np.ndarray:
        """Значения T_n'(t), n = 0..p, в точках t: массив формы (p+1, *t.shape)."""
        if self.p == 0:
            return np.zeros((1, *t.shape))
        Vd = _cheb.chebvander(t, self.p - 1) @ self._Dcoef.T     # (*S, p+1)
        return np.moveaxis(Vd, -1, 0)                            # (p+1, *S)

    def __repr__(self) -> str:
        return f"ChebyshevBasis(p={self.p}, N={self.N}, bbox={self.bbox})"


__all__ = ["BBox", "ChebyshevBasis"]
