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


def r_not(f: sp.Expr) -> sp.Expr:
    """R-отрицание (дополнение области): ¬f = −f.

    В системе R₀ отрицание — простая смена знака: ω > 0 внутри становится
    ω < 0, граница ω = 0 сохраняется.
    """
    return -f


def r_diff(f1: sp.Expr, f2: sp.Expr) -> sp.Expr:
    """R-разность областей: f₁ \\ f₂ = f₁ ∧ (¬f₂) = f₁ + (−f₂) − √(f₁² + f₂²).

    Область f₁ с вырезанной областью f₂ (граница выреза — часть ∂Ω).
    """
    return r_and(f1, r_not(f2))


# --------------------------------------------------------------------------- #
#  Символьные примитивы (возвращают sympy-выражения от x, y)
# --------------------------------------------------------------------------- #
def circle_expr(a: float, cx: float = 0.0, cy: float = 0.0) -> sp.Expr:
    """ω круга радиуса a с центром (cx, cy), нормированная до 1-го порядка.

    .. math:: \\omega = \\frac{a^2 - (x-c_x)^2 - (y-c_y)^2}{2a}
    """
    return (a**2 - (x - cx) ** 2 - (y - cy) ** 2) / (2 * a)


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


def make_annulus(a: float = 1.0, b: float = 0.4) -> Domain:
    """Кольцо b < r < a с центром (0,0); bbox = (−a, a, −a, a).

    R-разность двух кругов:

    .. math:: \\omega = \\omega_a \\wedge (-\\omega_b),

    где ω_a, ω_b — нормированные ω кругов радиусов a и b. Внутри кольца
    ω > 0, в дырке (r < b) и снаружи (r > a) ω < 0, обе окружности — ω = 0.
    """
    if not (0.0 < b < a):
        raise ValueError("Требуется 0 < b < a (внутренний радиус меньше внешнего).")
    return Domain(r_diff(circle_expr(a), circle_expr(b)), (-a, a, -a, a))


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


# --------------------------------------------------------------------------- #
#  Мини-язык compose: дерево операций → Domain
# --------------------------------------------------------------------------- #
def make_compose(tree: dict) -> Domain:
    """Составная область из дерева операций (мини-язык case-файла).

    Узел-операция: ``{"op": union|intersect|difference, "children": [...]}``
    (difference строго бинарна: первый операнд минус второй); примитивы:
    ``{"kind": "circle", "a", "cx", "cy"}`` и
    ``{"kind": "rectangle", "x1", "x2", "y1", "y2"}``. Ограда v0.2 (глубина
    ≤ 3, ≤ 7 узлов) проверяется валидатором схемы — единый источник правды
    в problem.py; нарушение — CaseError.

    bbox: union — объединение bbox детей; intersect — их пересечение;
    difference — bbox первого операнда (вырез не расширяет область).
    """
    from .problem import validate_compose_tree

    validate_compose_tree(tree)
    expr, bbox = _compose_node(tree)
    return Domain(expr, bbox)


def _compose_node(node: dict) -> tuple[sp.Expr, BBox]:
    """Рекурсивно построить (ω-выражение, bbox) узла compose-дерева."""
    if "op" in node:
        parts = [_compose_node(ch) for ch in node["children"]]
        exprs = [p[0] for p in parts]
        boxes = [p[1] for p in parts]
        op = node["op"]
        if op == "union":
            expr = exprs[0]
            for e in exprs[1:]:
                expr = r_or(expr, e)
            bbox = (min(b[0] for b in boxes), max(b[1] for b in boxes),
                    min(b[2] for b in boxes), max(b[3] for b in boxes))
        elif op == "intersect":
            expr = exprs[0]
            for e in exprs[1:]:
                expr = r_and(expr, e)
            bbox = (max(b[0] for b in boxes), min(b[1] for b in boxes),
                    max(b[2] for b in boxes), min(b[3] for b in boxes))
            if not (bbox[0] < bbox[1] and bbox[2] < bbox[3]):
                raise ValueError(f"intersect: пересечение bbox пусто: {boxes}")
        else:                                   # difference (бинарна)
            expr = r_diff(exprs[0], exprs[1])
            bbox = boxes[0]
        return expr, bbox
    if node["kind"] == "circle":
        a = float(node["a"])
        cx = float(node.get("cx", 0.0))
        cy = float(node.get("cy", 0.0))
        return circle_expr(a, cx, cy), (cx - a, cx + a, cy - a, cy + a)
    x1, x2 = float(node["x1"]), float(node["x2"])
    y1, y2 = float(node["y1"]), float(node["y2"])
    return rectangle_expr(x1, x2, y1, y2), (x1, x2, y1, y2)


__all__ = [
    "x",
    "y",
    "BBox",
    "Domain",
    "r_and",
    "r_or",
    "r_not",
    "r_diff",
    "circle_expr",
    "rectangle_expr",
    "make_circle",
    "make_rectangle",
    "make_L",
    "make_annulus",
    "make_compose",
]
