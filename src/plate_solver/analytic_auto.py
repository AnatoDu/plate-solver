r"""analytic_auto.py — фабрика аналитических эталонов с самосертификацией (F4).

Для КАНОНИЧЕСКИХ постановок эталон w(x, y) строится автоматически и
выдаётся только вместе с СЕРТИФИКАТОМ: подстановка в уравнение и краевые
условия проверяется символьно (sympy) или численно (< 1e-12; контроль
остатка рядов — удвоением числа членов). Область действия (ограда):

* осесимметрия (circle / annulus): нагрузка — полином по r (uniform —
  частный случай) или точечная сила в центре круга; края независимо
  clamped | soft | true_ss. Построение: частное решение sympy-интегрированием
  осесимметричной бигармоники + C₁ + C₂r² + C₃ln r + C₄r²ln r, константы —
  из системы 2×2 / 4×4 по КУ. «soft» — мягкий шарнир пакета: {w = 0, Δw = 0}
  (модельный предел ν → 1, NOTES §8); «true_ss» — {w = 0, M_r = 0};
* прямоугольник: ряд Навье (SSSS) для uniform | patch | point в произвольной
  точке (синус-коэффициенты в замкнутом виде, контроль остатка удвоением);
  ряд Леви для x-пары hinge и y-пары hinge|clamped (несимметричные пары —
  константы 4×4 на моду);
* 1D-полоса (цилиндрический изгиб): полиномиальная q(x), любые пары КУ
  clamped | hinge — прямое sympy-решение ОДУ D w⁗ = q.

Вне ограды — :class:`FactoryError` (резолвер отказывает как прежде).
Контакт: замкнутые решения существуют лишь при конечномерной свободной
границе (1D; осесимметрия) — произвольные формы принципиально вне ограды.

Ручные функции :mod:`plate_solver.analytic` НЕ заменяются фабрикой
(бит-в-бит сохранность существующих ворот); согласие путей на пересечении
областей действия — тождества 1e-12 в tests/test_analytic_factory.py.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

__all__ = ["CertifiedSolution", "FactoryError", "axisym_solution",
           "axisym_contact_solution", "strip_solution", "navier_solution",
           "levy_solution"]

_R, _X, _Y = sp.symbols("r x y", positive=True)


class FactoryError(ValueError):
    """Постановка вне ограды фабрики (эталон не строится)."""


@dataclass
class CertifiedSolution:
    """Аналитический эталон с сертификатом самопроверки.

    ``w(x, y)`` — прогиб (для осесимметрии аргументы декартовы, внутри
    r = √(x²+y²)); ``certificate`` — словарь невязок: PDE и каждое КУ
    (символьный нуль записывается как 0.0). Эталон без выполненного
    сертификата не существует: фабрика бросает :class:`FactoryError`.
    """

    w: Callable
    kind: str
    certificate: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    def w_max_on_grid(self, X, Y) -> float:
        vals = np.asarray(self.w(X, Y), float)
        return float(np.nanmax(np.abs(vals)))


def _certify(cert: dict, tol: float = 1e-10) -> None:
    bad = {k: v for k, v in cert.items() if not (abs(v) <= tol)}
    if bad:
        raise FactoryError(f"самосертификация не пройдена: {bad}")


# --------------------------------------------------------------------------- #
#  (а) Осесимметрия: circle / annulus
# --------------------------------------------------------------------------- #
def _axisym_ode_particular(q_poly: sp.Expr, D: float) -> sp.Expr:
    """Частное решение r⁻¹(r·(r⁻¹(r·w′)′)′)′ = q(r)/D интегрированием."""
    r = _R
    expr = q_poly / D
    w = expr
    for _ in range(2):
        w = sp.integrate(r * w, r) / r          # обращение r⁻¹(r·f′)′ = g
        w = sp.integrate(w, r)
    return sp.expand(w)


def _bc_rows(w_expr: sp.Expr, r0: float, bc: str, nu: float):
    """Две строки условий на кромке r = r0 для типа КУ."""
    r = _R
    wp = sp.diff(w_expr, r)
    lap = sp.diff(w_expr, r, 2) + wp / r
    mr = -(sp.diff(w_expr, r, 2) + nu * wp / r)      # M_r/D
    if bc == "clamped":
        return [w_expr.subs(r, r0), wp.subs(r, r0)]
    if bc == "soft":                                  # мягкий шарнир пакета
        return [w_expr.subs(r, r0), lap.subs(r, r0)]
    if bc == "true_ss":
        return [w_expr.subs(r, r0), mr.subs(r, r0)]
    raise FactoryError(f"КУ кромки: получено {bc!r}, ожидалось "
                       "clamped | soft | true_ss")


def axisym_solution(*, a: float, b: float = 0.0, bc_outer: str,
                    bc_inner: str | None = None, D: float, nu: float,
                    q_coeffs: tuple = (), P: float | None = None,
                    ) -> CertifiedSolution:
    r"""Осесимметричный эталон: круг (b = 0) или кольцо (b > 0).

    ``q_coeffs`` — коэффициенты полинома q(r) = Σ q_k·r^k (uniform — (q0,));
    ``P`` — точечная сила в центре (только круг; q_coeffs и P взаимно
    исключены). Однородная часть C₁ + C₂r² + C₃ln r + C₄r²ln r; на круге
    регулярность отбрасывает ln-члены (для P — оставляет r²ln r с известным
    множителем P/(8πD)).
    """
    r = _R
    if b < 0 or b >= a:
        raise FactoryError(f"кольцо: ожидалось 0 ≤ b < a, получено b={b}, a={a}")
    if P is not None and (b > 0 or q_coeffs):
        raise FactoryError("точечная сила: только круг, без распределённой q")
    q_poly = sum(sp.Float(c) * r**k for k, c in enumerate(q_coeffs))
    C = sp.symbols("C1:5")
    if b > 0:                                        # кольцо: полный набор
        basis = [sp.Integer(1), r**2, sp.log(r), r**2 * sp.log(r)]
        n_c = 4
    elif P is not None:                              # круг + сила в центре
        basis = [sp.Integer(1), r**2]
        n_c = 2
    else:                                            # круг, регулярность
        basis = [sp.Integer(1), r**2]
        n_c = 2
    w = _axisym_ode_particular(q_poly, D) if P is None else \
        sp.Float(P) / (8 * sp.pi * D) * r**2 * sp.log(r)
    w = w + sum(C[i] * basis[i] for i in range(n_c))

    rows = _bc_rows(w, a, bc_outer, nu)
    if b > 0:
        if bc_inner is None:
            raise FactoryError("кольцо: нужен bc_inner")
        rows += _bc_rows(w, b, bc_inner, nu)
    sol = sp.solve(rows, C[:n_c], dict=True)
    if not sol:
        raise FactoryError("осесимметрия: система КУ вырождена")
    w_final = sp.expand(w.subs(sol[0]))

    # --- сертификат: PDE и КУ подстановкой -------------------------------- #
    lap_op = lambda f: sp.diff(f, r, 2) + sp.diff(f, r) / r        # noqa: E731
    pde = sp.simplify(D * lap_op(lap_op(w_final)) - q_poly)
    # численная оценка невязки (Float-коэффициенты не дают точного
    # символьного нуля); масштаб — характерная нагрузка
    q_scale = max(abs(float(q_poly.subs(r, a))) if q_coeffs else 0.0,
                  abs(P) / a**2 if P is not None else 0.0, 1e-30)
    r_mid = (a + b) / 2
    cert = {"pde": abs(float(pde.subs(r, r_mid))) / q_scale}
    if P is not None:
        # под точечной силой: перерезывающая Q·2πr → −P при r→0 (баланс силы)
        Q = -D * sp.diff(lap_op(w_final), r)
        cert["point_balance"] = float(sp.limit(2 * sp.pi * r * Q, r, 0) + P) / max(abs(P), 1.0)
    for i, row in enumerate(rows):
        cert[f"bc{i}"] = float(sp.Abs(row.subs(sol[0])))
    scale = max(abs(float(w_final.subs(r, (a + b) / 2))), 1e-30)
    for k in list(cert):
        if k.startswith("bc"):
            cert[k] /= scale
    _certify(cert)

    w_num = sp.lambdify(r, w_final, "numpy")

    def w_xy(X, Y):
        rr = np.sqrt(np.asarray(X, float) ** 2 + np.asarray(Y, float) ** 2)
        rr = np.maximum(rr, 1e-12 * a)               # регуляризация центра для ln
        return w_num(rr)

    return CertifiedSolution(w=w_xy, kind="axisym", certificate=cert,
                             meta={"a": a, "b": b, "bc_outer": bc_outer,
                                   "bc_inner": bc_inner, "P": P,
                                   "w_expr": w_final})


# --------------------------------------------------------------------------- #
#  (в) 1D-полоса: цилиндрический изгиб
# --------------------------------------------------------------------------- #
def strip_solution(*, x1: float, x2: float, bc_left: str, bc_right: str,
                   D: float, q_coeffs: tuple) -> CertifiedSolution:
    r"""Полоса [x1, x2]: D·w⁗ = q(x) (полином), пары КУ clamped | hinge."""
    x = _X
    q_poly = sum(sp.Float(c) * x**k for k, c in enumerate(q_coeffs))
    C = sp.symbols("K1:5")
    w = sp.integrate(sp.integrate(sp.integrate(sp.integrate(
        q_poly / D, x), x), x), x) + C[0] + C[1] * x + C[2] * x**2 + C[3] * x**3

    def bc_rows(x0, bc):
        if bc == "clamped":
            return [w.subs(x, x0), sp.diff(w, x).subs(x, x0)]
        if bc == "hinge":
            return [w.subs(x, x0), sp.diff(w, x, 2).subs(x, x0)]
        raise FactoryError(f"КУ полосы: получено {bc!r}, ожидалось clamped | hinge")

    rows = bc_rows(x1, bc_left) + bc_rows(x2, bc_right)
    sol = sp.solve(rows, C, dict=True)
    if not sol:
        raise FactoryError("полоса: система КУ вырождена")
    w_final = sp.expand(w.subs(sol[0]))
    pde = sp.simplify(D * sp.diff(w_final, x, 4) - q_poly)
    q_scale = max(abs(float(q_poly.subs(x, x2))), 1e-30)
    cert = {"pde": abs(float(pde.subs(x, (x1 + x2) / 2))) / q_scale}
    scale = max(abs(float(w_final.subs(x, (x1 + x2) / 2))), 1e-30)
    for i, row in enumerate(rows):
        cert[f"bc{i}"] = float(sp.Abs(row.subs(sol[0]))) / scale
    _certify(cert)
    w_num = sp.lambdify(x, w_final, "numpy")
    return CertifiedSolution(w=lambda X, Y=None: w_num(np.asarray(X, float)),
                             kind="strip", certificate=cert,
                             meta={"w_expr": w_final})


# --------------------------------------------------------------------------- #
#  (б) Прямоугольник: Навье (SSSS) и Леви (hinge-пара × {hinge|clamped})
# --------------------------------------------------------------------------- #
def _navier_qmn(load: dict, Lx: float, Ly: float, m: int, n: int) -> float:
    """Синус-коэффициент нагрузки в замкнутом виде (координаты от угла)."""
    am, bn = m * np.pi / Lx, n * np.pi / Ly
    if load["type"] == "uniform":
        if m % 2 == 0 or n % 2 == 0:
            return 0.0
        return 16.0 * load["q0"] / (np.pi**2 * m * n)
    if load["type"] == "patch":
        x1, x2, y1, y2 = load["zone"]
        return (4.0 * load["q0"] / (Lx * Ly)
                * (np.cos(am * x1) - np.cos(am * x2)) / am
                * (np.cos(bn * y1) - np.cos(bn * y2)) / bn)
    if load["type"] == "point":
        return 4.0 * load["P"] / (Lx * Ly) * np.sin(am * load["x0"]) * np.sin(bn * load["y0"])
    raise FactoryError(f"Навье: нагрузка {load['type']!r} вне ограды")


def navier_solution(*, x1: float, x2: float, y1: float, y2: float,
                    D: float, load: dict, tol: float = 1e-12,
                    n_max: int = 4096) -> CertifiedSolution:
    r"""Ряд Навье (SSSS): w = Σ q_mn sin(α_m ξ) sin(β_n η) / (D·(α_m²+β_n²)²).

    Каждая мода удовлетворяет уравнению и КУ ТОЧНО (символьный факт —
    tests/test_analytic_factory.py); остаток ряда контролируется удвоением
    числа членов до ``tol`` (сертификат содержит достигнутый остаток).
    """
    Lx, Ly = x2 - x1, y2 - y1
    if Lx <= 0 or Ly <= 0:
        raise FactoryError("Навье: вырожденный прямоугольник")
    ld = dict(load)
    if ld["type"] == "patch":                        # зона → координаты от угла
        zx1, zx2, zy1, zy2 = ld["zone"]
        ld["zone"] = (zx1 - x1, zx2 - x1, zy1 - y1, zy2 - y1)
    if ld["type"] == "point":
        ld["x0"], ld["y0"] = ld["x0"] - x1, ld["y0"] - y1

    def _kmn(N):
        """Матрица коэффициентов ряда K_mn = q_mn/(D(α²+β²)²) — векторно."""
        m = np.arange(1, N + 1)
        n = np.arange(1, N + 1)
        am = m * np.pi / Lx
        bn = n * np.pi / Ly
        den = D * (am[:, None] ** 2 + bn[None, :] ** 2) ** 2
        if ld["type"] == "uniform":
            fm = np.where(m % 2 == 1, 1.0 / m, 0.0)
            fn = np.where(n % 2 == 1, 1.0 / n, 0.0)
            qmn = 16.0 * ld["q0"] / np.pi**2 * fm[:, None] * fn[None, :]
        elif ld["type"] == "patch":
            zx1, zx2, zy1, zy2 = ld["zone"]
            fm = (np.cos(am * zx1) - np.cos(am * zx2)) / am
            fn = (np.cos(bn * zy1) - np.cos(bn * zy2)) / bn
            qmn = 4.0 * ld["q0"] / (Lx * Ly) * fm[:, None] * fn[None, :]
        else:                                        # point
            qmn = (4.0 * ld["P"] / (Lx * Ly)
                   * np.sin(am * ld["x0"])[:, None] * np.sin(bn * ld["y0"])[None, :])
        return am, bn, qmn / den

    def partial(N, X, Y):
        am, bn, K = _kmn(N)
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        shp = np.broadcast(X, Y).shape
        Sx = np.sin(np.multiply.outer(X.ravel(), am))     # P×N
        Sy = np.sin(np.multiply.outer(Y.ravel(), bn))     # P×N
        vals = ((Sx @ K) * Sy).sum(axis=1)                # BLAS-путь
        return vals.reshape(shp) if shp else float(vals[0])

    # контроль остатка в контрольных точках (центр + окрестность максимума)
    xc = np.array([Lx / 2, Lx / 3, 2 * Lx / 5])
    yc = np.array([Ly / 2, Ly / 3, 2 * Ly / 5])
    if ld["type"] == "point":
        xc = np.append(xc, ld["x0"])
        yc = np.append(yc, ld["y0"])
    N = 16
    prev = partial(N, xc, yc)
    resid = np.inf
    while N < n_max:
        N *= 2
        cur = partial(N, xc, yc)
        scale = float(np.max(np.abs(cur))) or 1.0
        resid = float(np.max(np.abs(cur - prev))) / scale
        if resid < tol:
            break
        prev = cur
    cert = {"series_residual": resid if resid < 10 * tol else float("inf")}
    _certify(cert, tol=10 * tol)

    def w_xy(X, Y):
        return partial(N, np.asarray(X, float) - x1, np.asarray(Y, float) - y1)

    return CertifiedSolution(w=w_xy, kind="navier", certificate=cert,
                             meta={"n_terms": N, "load": dict(load)})


def levy_solution(*, x1: float, x2: float, y1: float, y2: float,
                  D: float, q0: float, bc_y1: str, bc_y2: str,
                  n_terms: int = 80) -> CertifiedSolution:
    r"""Ряд Леви: x-пара hinge, y-кромки bc_y1 | bc_y2 ∈ {hinge, clamped}.

    Мода: Y_m(y) = y_p + A·ch(αy) + B·sh(αy) + C·y·ch(αy) + E·y·sh(αy)
    (несимметричные пары — полный набор 4×4 на моду); ОДУ моды
    Y⁗ − 2α²Y″ + α⁴Y = q_m/D — символьный факт; константы — численное 4×4
    по КУ, сертификат — невязки КУ каждой моды.
    """
    for bc in (bc_y1, bc_y2):
        if bc not in ("hinge", "clamped"):
            raise FactoryError(f"Леви: КУ {bc!r} вне ограды (hinge | clamped)")
    Lx, Ly = x2 - x1, y2 - y1
    c = Ly / 2.0

    # экспоненциально нормированный базис (устойчив при больших αc):
    # chs = ch(αy)/e^{αc}, shs = sh(αy)/e^{αc} — при |y| ≤ c аргументы
    # экспонент неположительны, переполнения нет, матрица КУ — O(1)
    def _chs(al, y):
        return 0.5 * (np.exp(al * (y - c)) + np.exp(-al * (y + c)))

    def _shs(al, y):
        return 0.5 * (np.exp(al * (y - c)) - np.exp(-al * (y + c)))

    modes = []
    cert = {}
    for m in range(1, 2 * n_terms, 2):               # нечётные: q_m = 4q0/(πm)
        al = m * np.pi / Lx
        qm = 4.0 * q0 / (np.pi * m)
        yp = qm / (D * al**4)

        def rows_at(eta, bc, al=al, yp=yp):
            ch, sh = _chs(al, eta), _shs(al, eta)
            t = eta / c
            # базис: [chs, shs, (y/c)·chs, (y/c)·shs]
            val = np.array([ch, sh, t * ch, t * sh])
            der = np.array([al * sh, al * ch,
                            ch / c + al * t * sh, sh / c + al * t * ch])
            der2 = np.array([al**2 * ch, al**2 * sh,
                             2 * al * sh / c + al**2 * t * ch,
                             2 * al * ch / c + al**2 * t * sh])
            if bc == "clamped":
                return [(val, -yp), (der, 0.0)]
            return [(val, -yp), (der2 / al**2, 0.0)]  # hinge: Y = Y″ = 0

        rows = rows_at(-c, bc_y1) + rows_at(+c, bc_y2)
        A = np.array([r_[0] for r_ in rows])
        b = np.array([r_[1] for r_ in rows])
        coef = np.linalg.solve(A, b)
        resid = float(np.max(np.abs(A @ coef - b))) / (abs(yp) or 1.0)
        cert[f"m{m}_bc"] = resid
        modes.append((al, yp, coef))
        if len(modes) >= n_terms:
            break
    _certify(cert, tol=1e-9)

    y_mid = 0.5 * (y1 + y2)

    def w_xy(X, Y):
        Xl = np.asarray(X, float) - x1
        eta = np.asarray(Y, float) - y_mid
        out = np.zeros(np.broadcast(Xl, eta).shape)
        for al, yp, (A_, B_, C_, E_) in modes:
            t = eta / c
            Ym = (yp + A_ * _chs(al, eta) + B_ * _shs(al, eta)
                  + C_ * t * _chs(al, eta) + E_ * t * _shs(al, eta))
            out = out + Ym * np.sin(al * Xl)
        return out

    return CertifiedSolution(w=w_xy, kind="levy", certificate=cert,
                             meta={"bc_y1": bc_y1, "bc_y2": bc_y2,
                                   "n_terms": len(modes)})


# --------------------------------------------------------------------------- #
#  Осесимметричный контактный эталон (F4.7): круг + плоское основание
# --------------------------------------------------------------------------- #
def axisym_contact_solution(*, a: float, D: float, q0: float, gap: float,
                            ) -> CertifiedSolution:
    r"""Контакт круга (мягкий шарнир) с плоским жёстким основанием (зазор Δ).

    Классическая конструкция Кирхгофа: центральная зона r ≤ c ложится
    плашмя (w ≡ Δ, распределённая реакция r = q₀) плюс КОЛЬЦЕВАЯ
    сосредоточенная реакция P_c на границе зоны (известный дельта-слой
    контакта пластин Кирхгофа: скачок перерезывающей Q при непрерывных
    w, w′, M). Вне зоны — общее осесимметричное решение с условиями
    w(c) = Δ, w′(c) = 0, w″(c) = 0 (непрерывность M при w″ ≡ 0 в зоне),
    w(a) = 0, Δw(a) = 0 (мягкий шарнир); c — корень по последнему условию
    (единственный на (0, a)). Требование существования: 0 < Δ < w_free(0).

    Сертификат: невязки всех пяти условий сшивки/КУ, непроникновение
    (w ≤ Δ вне зоны), P_c ≥ 0; предел Δ → w_free⁻ согласован с функцией
    Грина: 2πc·P_c ≈ 8πD(w_free − Δ)/a² → 0 (tests).
    """
    from scipy.optimize import brentq

    r, c_s, Dl = sp.symbols("r c Delta_gap", positive=True)
    C = sp.symbols("C1:5")
    w = q0 * r**4 / (64 * D) + C[0] + C[1] * r**2 + C[2] * sp.log(r) \
        + C[3] * r**2 * sp.log(r)
    wp = sp.diff(w, r)
    lap = sp.diff(w, r, 2) + wp / r
    w_free0 = 3.0 * q0 * a**4 / (64.0 * D)          # w_free(0), мягкий шарнир
    if not (0.0 < gap < w_free0):
        raise FactoryError(
            f"контактный эталон: ожидалось 0 < Δ < w_free(0) = {w_free0:.6g}, "
            f"получено Δ = {gap:.6g}")
    conds = [sp.Eq(w.subs(r, c_s), Dl), sp.Eq(wp.subs(r, c_s), 0),
             sp.Eq(w.subs(r, a), 0), sp.Eq(lap.subs(r, a), 0)]
    sol = sp.solve(conds, list(C), dict=True)[0]
    wpp_c = sp.lambdify(c_s, sp.diff(w, r, 2).subs(sol).subs(Dl, gap).subs(r, c_s))
    # корень w″(c) = 0: сканирование с брекетингом (края (0, a) сингулярны)
    cs = np.geomspace(1e-8 * a, a * (1 - 1e-4), 2400)
    with np.errstate(all="ignore"):
        vals = np.array([wpp_c(ci) for ci in cs], dtype=float)
    fin = np.isfinite(vals)
    sign = np.sign(vals)
    idx = np.where(fin[:-1] & fin[1:] & (sign[:-1] != sign[1:]))[0]
    if idx.size == 0:
        raise FactoryError("контактный эталон: корень w″(c) = 0 не найден "
                           f"на (0, a) при Δ = {gap:.6g}")
    c_star = brentq(wpp_c, cs[idx[0]], cs[idx[0] + 1])
    subs = {**{k: v.subs({Dl: gap, c_s: c_star}) for k, v in sol.items()},
            Dl: gap, c_s: c_star}
    w_out = sp.expand(w.subs(subs))
    # кольцевая реакция: P_c = Q(c⁺) (внутри зоны Q ≡ 0)
    Q_out = -D * sp.diff(sp.diff(w_out, r, 2) + sp.diff(w_out, r) / r, r)
    P_ring = float(Q_out.subs(r, c_star))

    # --- сертификат -------------------------------------------------------- #
    scale = gap
    cert = {
        "w_c": abs(float(w_out.subs(r, c_star)) - gap) / scale,
        "wp_c": abs(float(sp.diff(w_out, r).subs(r, c_star))) * a / scale,
        "wpp_c": abs(float(sp.diff(w_out, r, 2).subs(r, c_star))) * a**2 / scale,
        "w_a": abs(float(w_out.subs(r, a))) / scale,
        "lap_a": abs(float((sp.diff(w_out, r, 2)
                            + sp.diff(w_out, r) / r).subs(r, a))) * a**2 / scale,
    }
    rr = np.linspace(c_star, a, 400)
    w_out_num = sp.lambdify(r, w_out, "numpy")
    overshoot = float(np.max(w_out_num(rr) - gap)) / scale
    cert["non_penetration"] = max(overshoot, 0.0)
    cert["ring_force_sign"] = 0.0 if P_ring >= 0 else abs(P_ring)
    _certify(cert, tol=1e-8)

    def w_xy(X, Y):
        rr_ = np.sqrt(np.asarray(X, float) ** 2 + np.asarray(Y, float) ** 2)
        out = np.where(rr_ <= c_star, gap, w_out_num(np.maximum(rr_, c_star)))
        return out

    return CertifiedSolution(
        w=w_xy, kind="axisym_contact", certificate=cert,
        meta={"c": c_star, "P_ring": P_ring,
              "ring_force_total": 2 * np.pi * c_star * P_ring,
              "w_free0": w_free0, "gap": gap, "w_expr_outer": w_out})
