r"""geometry.py — область как R-функция и её ТОЧНЫЙ (символьный) градиент.

Каждая область задаётся функцией ``ω(x, y)``:
    ω > 0 — внутри,  ω = 0 — на границе,  ω < 0 — снаружи.
Структура ``v = ω·Φ`` тождественно зануляет ``v`` на ``∂Ω`` ⇒ однородное условие
Дирихле выполняется точно.

ω строится СИМВОЛЬНО (SymPy) из R-операций и примитивов, после чего ``ω`` и
``∇ω = (ω_x, ω_y)`` компилируются в быстрые векторизованные numpy-функции через
``sympy.lambdify``. Градиент берётся символьным дифференцированием — точно, без
конечных разностей; это важно для корректной сборки матрицы Ритца (assembler.py),
где ``∇(ω·φ) = ∇ω·φ + ω·∇φ``.

R-операции (система R₀ В. Л. Рвачёва):
    f₁ ∧ f₂ = f₁ + f₂ − √(f₁² + f₂²)      (пересечение областей)
    f₁ ∨ f₂ = f₁ + f₂ + √(f₁² + f₂²)      (объединение областей)

Примитивы:
    круг радиуса a:        ω = (a² − x² − y²) / (2a)        (нормирован: |∇ω|=1 на ∂Ω)
    полуплоскость x ≥ x₁:  ω = x − x₁;     x ≤ x₂:  ω = x₂ − x
    полуплоскость y ≥ y₁:  ω = y − y₁;     y ≤ y₂:  ω = y₂ − y
    прямоугольник:         ω = ((x−x₁)∧(x₂−x)) ∧ ((y−y₁)∧(y₂−y))

О нормировке: для круга ``ω`` нормирована точно. Для составных областей
R-операции дают ``ω = 0`` на границе и ``ω > 0`` внутри (этого достаточно для
структуры ``ω·Φ``), но ``|∇ω| = 1`` выполняется лишь приближённо вдали от
границы — на корректность не влияет; у входящего угла L-формы слегка влияет на
точность и компенсируется сгущением базиса/квадратуры (NOTES.md §§2, 4).
"""

from __future__ import annotations

import numpy as np
import sympy as sp

# Символы координат. Доступны наружу, чтобы собирать произвольные ω через r_and/r_or.
x, y = sp.symbols("x y", real=True)

BBox = tuple  # (x_min, x_max, y_min, y_max) — описанный прямоугольник


# --------------------------------------------------------------------------- #
#  R-операции на уровне символьных выражений
# --------------------------------------------------------------------------- #
def r_and(f1: sp.Expr, f2: sp.Expr) -> sp.Expr:
    """R-конъюнкция (пересечение областей): f₁ + f₂ − √(f₁² + f₂²)."""
    return f1 + f2 - sp.sqrt(f1**2 + f2**2)


def r_or(f1: sp.Expr, f2: sp.Expr) -> sp.Expr:
    """R-дизъюнкция (объединение областей): f₁ + f₂ + √(f₁² + f₂²)."""
    return f1 + f2 + sp.sqrt(f1**2 + f2**2)


# --------------------------------------------------------------------------- #
#  Символьные примитивы (возвращают sympy-выражения от x, y)
# --------------------------------------------------------------------------- #
def circle_expr(a: float) -> sp.Expr:
    """ω круга радиуса a с центром (0,0), нормированная до 1-го порядка."""
    return (a**2 - x**2 - y**2) / (2 * a)


def rectangle_expr(x1: float, x2: float, y1: float, y2: float) -> sp.Expr:
    """ω прямоугольника [x1,x2]×[y1,y2] как R-конъюнкция полуплоскостей."""
    wx = r_and(x - x1, x2 - x)
    wy = r_and(y - y1, y2 - y)
    return r_and(wx, wy)


# --------------------------------------------------------------------------- #
#  Область: значение ω и точный градиент ∇ω (через lambdify)
# --------------------------------------------------------------------------- #
class Domain:
    """Область, заданная символьной ``ω(x, y)`` на описанном прямоугольнике bbox.

    Хранит символьные выражения ``ω, ω_x, ω_y`` и их быстрые numpy-реализации.
    Методы :meth:`omega` и :meth:`grad_omega` векторизованы: X, Y — скаляры или
    массивы одинаковой формы; результат имеет форму входа.
    """

    def __init__(self, omega_expr: sp.Expr, bbox: BBox):
        self.omega_expr: sp.Expr = sp.sympify(omega_expr)
        self.bbox: BBox = tuple(map(float, bbox))
        self.dx_expr: sp.Expr = sp.diff(self.omega_expr, x)
        self.dy_expr: sp.Expr = sp.diff(self.omega_expr, y)
        # Компиляция в numpy (точный символьный градиент → быстрые функции).
        self._omega = sp.lambdify((x, y), self.omega_expr, "numpy")
        self._dx = sp.lambdify((x, y), self.dx_expr, "numpy")
        self._dy = sp.lambdify((x, y), self.dy_expr, "numpy")

    @staticmethod
    def _eval(fn, X, Y) -> np.ndarray:
        """Вычислить fn(X, Y) и привести к форме входа (lambdify констант → скаляр)."""
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        out = np.asarray(fn(X, Y), dtype=float)
        if out.shape != X.shape:
            out = np.broadcast_to(out, X.shape).astype(float)
        return out

    def omega(self, X, Y) -> np.ndarray:
        """Значение ω(X, Y), векторизовано."""
        return self._eval(self._omega, X, Y)

    def grad_omega(self, X, Y):
        """Градиент (∂ω/∂x, ∂ω/∂y) в точках (X, Y), векторизовано."""
        return self._eval(self._dx, X, Y), self._eval(self._dy, X, Y)

    def __repr__(self) -> str:
        return f"Domain(omega={self.omega_expr}, bbox={self.bbox})"


# --------------------------------------------------------------------------- #
#  Готовые области
# --------------------------------------------------------------------------- #
def make_circle(a: float = 1.0) -> Domain:
    """Круг радиуса ``a`` с центром (0,0); bbox = (−a, a, −a, a)."""
    if a <= 0:
        raise ValueError("Радиус a должен быть положительным.")
    return Domain(circle_expr(a), (-a, a, -a, a))


def make_rectangle(x1: float, x2: float, y1: float, y2: float) -> Domain:
    """Прямоугольник [x1,x2]×[y1,y2]; bbox совпадает с самим прямоугольником."""
    if not (x1 < x2 and y1 < y2):
        raise ValueError("Требуется x1 < x2 и y1 < y2.")
    return Domain(rectangle_expr(x1, x2, y1, y2), (x1, x2, y1, y2))


def make_L(side: float = 1.0, cut: float = 0.5) -> Domain:
    """L-образная область: квадрат [0,side]² без вырезанного угла [cut,side]².

    Строится как объединение нижней и левой полос через R-дизъюнкцию:
        R1 = [0, side] × [0, cut]   (нижняя полоса)
        R2 = [0, cut]  × [0, side]  (левая полоса)
        ω_L = ω_R1 ∨ ω_R2
    Входящий (реентрантный) угол — в точке (cut, cut).
    """
    if not (0.0 < cut < side):
        raise ValueError("Требуется 0 < cut < side.")
    r1 = rectangle_expr(0.0, side, 0.0, cut)   # нижняя полоса
    r2 = rectangle_expr(0.0, cut, 0.0, side)   # левая полоса
    return Domain(r_or(r1, r2), (0.0, side, 0.0, side))


__all__ = [
    "x",
    "y",
    "BBox",
    "Domain",
    "r_and",
    "r_or",
    "circle_expr",
    "rectangle_expr",
    "make_circle",
    "make_rectangle",
    "make_L",
]
