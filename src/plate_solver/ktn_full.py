r"""ktn_full.py — ПОЛНАЯ нелинейная теория Кармана–Тимошенко–Нагди (КТН, §3, §5).

КТН = Карман (`membrane.py`) плюс два добавочных члена во внеплоскостном
уравнении (§3.4) и лицевая постобработка (`faces.py`). Первые два уравнения
системы (§3.1) полностью определяют ``w`` и ``Φ``:

.. math::
    D\,\Delta^2 w = q_n - h_*^2\Delta q_n + (I - h_\psi^2\Delta)\,L(\Phi, w),\qquad
    \tfrac{1}{Eh}\Delta^2\Phi = -\tfrac12 L(w, w),

где ``L(\Phi, w) = N_x w_{,xx} + 2N_{xy} w_{,xy} + N_y w_{,yy}``. Относительно
решателя Кармана (``D\Delta^2 w = q_n + L``) добавляется:

* **(A) нагрузочный** ``-h_*^2\Delta q_n``: под равномерной нагрузкой ``Δq_n = 0``
  (проявляется при неравномерной нагрузке, веха N3);
* **(B) регуляризация связи** ``-h_\psi^2\,\Delta L`` — главный новый оператор.

Слабая форма (B) БЕЗ производных 4-го порядка (ключевой приём §3.5). По формуле
Грина при ЗАЩЕМЛЕНИИ (``v = ∂_n v = 0`` на ∂Ω ⇒ граничный интеграл ноль):

.. math:: -h_\psi^2\!\int_\Omega v\,\Delta L\,dA = -h_\psi^2\!\int_\Omega L\,\Delta v\,dA.

Лапласиан переносится с решения на пробную функцию (``\Delta\psi_j`` уже есть в
структуре), ``L`` — из вторых производных ``w`` и замороженных усилий ``N``.
Перенося w-зависимые члены в левую часть, на каждом шаге Пикара решается

.. math:: \big(D\,S_{bend} + K_{geo}(N) + h_\psi^2 M_2(N)\big)\,c = b - h_*^2\,b_{\Delta q},

где ``M_2[j,i] = \iint (N{:}\nabla\nabla\psi_i)\,\Delta\psi_j\,dA`` (член B),
``K_{geo}`` — мембранная связь Кармана, ``b`` — вектор нагрузки. При
``h_\psi^2 = h_*^2 = 0`` система ТОЖДЕСТВЕННО кармановская — редукция КТН→Карман
до машинной точности (Gate R1); все поправки — порядка ``O(h^2/L^2)`` (Gate R4).

Мягкий шарнир для КТН (граничный член слабой формы по полудеформационным
условиям, §3.5) — задел; в v0.5.0 КТН реализована на ЗАЩЕМЛЕНИИ. Поле сдвига
``ψ`` (§3.6) — односторонняя постобработка, ядро от неё не зависит.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .basis import ChebyshevBasis
from .faces import FaceParams
from .membrane import KarmanPlate
from .quadrature import interior_nodes


def _star_boundary_quad(domain, m_bnd: int = 360):
    r"""Квадратура по ∂Ω для звёздной относительно центра bbox области (§3.5, ЭКСПЕРИМ.).

    Для КАЖДОГО угла θ луч из центра bbox пересекает границу ω = 0 в одной точке
    (бисекция по r). Возвращает ``(Xb, Yb, ds, nx, ny)``: точки границы, элемент
    длины ``ds`` (трапеции по замкнутому контуру) и ВНЕШНЮЮ нормаль
    ``n = −∇ω/‖∇ω‖`` (ω > 0 внутри ⇒ ∇ω внутрь). Годится для звёздных областей
    (круг, эллипс, выпуклые); для невыпуклых/многосвязных — задел.
    """
    x0, x1, y0, y1 = domain.bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if float(domain.omega(cx, cy)) <= 0.0:
        raise ValueError(
            "звёздная квадратура ∂Ω: центр bbox вне материала (ω ≤ 0) — область "
            "не звёздная относительно центра (кольцо/многосвязная/невыпуклая); "
            "реализована для circle | ellipse (выпуклые звёздные)")
    r_hi = float(np.hypot(x1 - x0, y1 - y0))                 # заведомо снаружи
    th = np.linspace(0.0, 2.0 * np.pi, m_bnd, endpoint=False)
    dx, dy = np.cos(th), np.sin(th)
    lo = np.zeros(m_bnd)
    hi = np.full(m_bnd, r_hi)
    for _ in range(60):                                      # бисекция ω = 0 по лучу
        mid = 0.5 * (lo + hi)
        inside = domain.omega(cx + mid * dx, cy + mid * dy) > 0.0
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    r = 0.5 * (lo + hi)
    Xb, Yb = cx + r * dx, cy + r * dy
    gx, gy = domain.grad_omega(Xb, Yb)                       # ∇ω (внутрь)
    gn = np.hypot(gx, gy)
    gn[gn == 0.0] = 1.0
    nx, ny = -gx / gn, -gy / gn                              # внешняя нормаль
    d_next = np.hypot(np.roll(Xb, -1) - Xb, np.roll(Yb, -1) - Yb)
    d_prev = np.hypot(Xb - np.roll(Xb, 1), Yb - np.roll(Yb, 1))
    ds = 0.5 * (d_next + d_prev)                             # элемент длины (замкнутый контур)
    return Xb, Yb, ds, nx, ny


def _contour_boundary_quad(domain, ngrid: int = 800):
    r"""Общая квадратура по ∂Ω (все контуры ω=0, вкл. внутренние) через contourpy.

    Для МНОГОСВЯЗНЫХ областей (кольцо): контуры ``ω = 0`` извлекаются из сетки
    bbox; ``ds`` — длины сегментов, внешняя нормаль ``n = −∇ω/‖∇ω‖`` в серединах
    сегментов (автоматически верна и для внутренних границ — направлена в дырку).
    ⚠️ У ВХОДЯЩИХ (реентрантных) углов точность падает (сингулярность) —
    реализована для гладких границ (кольцо), не для L/compose с вырезом.
    """
    from contourpy import contour_generator

    x0, x1, y0, y1 = domain.bbox
    pad = 0.02 * max(x1 - x0, y1 - y0)
    gx = np.linspace(x0 - pad, x1 + pad, ngrid)
    gy = np.linspace(y0 - pad, y1 + pad, ngrid)
    X, Y = np.meshgrid(gx, gy)
    cg = contour_generator(X, Y, domain.omega(X, Y))
    xs, ys, dss = [], [], []
    for poly in cg.lines(0.0):
        P = np.asarray(poly, float)
        if len(P) < 3:
            continue
        mid = 0.5 * (P[:-1] + P[1:])                         # середины сегментов
        seg = P[1:] - P[:-1]
        xs.append(mid[:, 0])
        ys.append(mid[:, 1])
        dss.append(np.hypot(seg[:, 0], seg[:, 1]))
    if not xs:
        raise ValueError("контурная квадратура ∂Ω: граница ω = 0 не найдена")
    Xb, Yb, ds = np.concatenate(xs), np.concatenate(ys), np.concatenate(dss)
    gxv, gyv = domain.grad_omega(Xb, Yb)
    gn = np.hypot(gxv, gyv)
    gn[gn == 0.0] = 1.0
    return Xb, Yb, ds, -gxv / gn, -gyv / gn


def _polygon_boundary_quad(polygon, n_per_edge: int = 48, eps: float = 1e-8):
    r"""ТОЧНАЯ пореберная квадратура ∂Ω полигональной области (§3.5, v0.6.5).

    Гаусс–Лежандр НА КАЖДОМ ребре полигона с ТОЧНЫМИ нормалями (константа на
    ребре) и точным элементом длины. Узлы никогда не попадают в вершины
    (в т.ч. во входящий угол) — подынтегральное выражение кусочно-гладко по
    рёбрам, квадратура сходится спектрально. Контурная квадратура (contourpy)
    у РЕЕНТРАНТНОГО угла не сходится (~5 % дрейфа вклада §3.5 при измельчении;
    NOTES §9) — точный полигон снимает проблему (тождество Грина на L:
    7.8e-2 → ~2e-4, пол внутренней квадратуры).

    Точки вычисления сдвигаются ВНУТРЬ на ``eps·diam`` вдоль нормали:
    производные R-функции ω имеют устранимые 0/0 РОВНО на линиях границы
    (сдвиг обходит их; подынтегральное выражение непрерывно до границы,
    погрешность O(eps)). ``ds`` и нормали — точные (без сдвига).
    """
    verts = np.asarray(polygon, float)
    x_, y_ = verts[:, 0], verts[:, 1]
    if np.sum(x_ * np.roll(y_, -1) - np.roll(x_, -1) * y_) < 0.0:
        verts = verts[::-1]                                  # ориентация CCW
    diam = float(np.max(verts.max(axis=0) - verts.min(axis=0)))
    t, wt = np.polynomial.legendre.leggauss(n_per_edge)
    Xb, Yb, ds, nx, ny = [], [], [], [], []
    for a, b in zip(verts, np.roll(verts, -1, axis=0), strict=True):
        e = b - a
        ln = float(np.hypot(e[0], e[1]))
        tang = e / ln
        n = (tang[1], -tang[0])                              # CCW ⇒ внешняя нормаль
        s = 0.5 * (t + 1.0) * ln
        Xb.append(a[0] + tang[0] * s - eps * diam * n[0])
        Yb.append(a[1] + tang[1] * s - eps * diam * n[1])
        ds.append(0.5 * ln * wt)
        nx.append(np.full(n_per_edge, n[0]))
        ny.append(np.full(n_per_edge, n[1]))
    return (np.concatenate(Xb), np.concatenate(Yb), np.concatenate(ds),
            np.concatenate(nx), np.concatenate(ny))


def _boundary_quad(domain):
    r"""Квадратура по ∂Ω граничного члена §3.5: полигон | звёздная | контурная.

    Точный полигон границы (``domain.polygon``, напр. L-форма) ⇒ пореберная
    квадратура (единственная сходящаяся у входящего угла). Иначе: центр bbox
    в материале (``ω > 0``) ⇒ область звёздная от центра
    (круг/эллипс/прямоугольник) — быстрая бисекция; иначе (кольцо,
    многосвязная) — общая контурная квадратура.
    """
    poly = getattr(domain, "polygon", None)
    if poly is not None:
        return _polygon_boundary_quad(poly)
    x0, x1, y0, y1 = domain.bbox
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    if float(domain.omega(cx, cy)) > 0.0:
        return _star_boundary_quad(domain)
    return _contour_boundary_quad(domain)


def _lin_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение НЕсимметричной системы (член B несимметричен) с диаг. предобусл.

    Симметричное диагональное масштабирование ``D_s A D_s`` (``s = 1/√|A_kk|``) +
    LU; при потере обусловленности — линейный МНК. Матрица КТН — малое
    (``O(h²)``) несимметричное возмущение SPD-матрицы Кармана.
    """
    d = np.abs(np.diag(A))
    if np.all(d > 0.0):
        s = 1.0 / np.sqrt(d)
        An = (A * s) * s[:, None]                    # D_s A D_s
        try:
            return sla.lu_solve(sla.lu_factor(An), b * s) * s
        except (sla.LinAlgError, np.linalg.LinAlgError):
            pass
    return np.linalg.lstsq(A, b, rcond=1e-13)[0]


class KTNPlate(KarmanPlate):
    r"""Полный нелинейный решатель КТН — слой поверх :class:`KarmanPlate` (§5.1).

    Наследует мембранную сборку, оператор ``L`` и драйвер Пикара
    (Андерсон + шаги по нагрузке) Кармана; переопределяет ТОЛЬКО внеплоскостной
    шаг, добавляя КТН-члены (A), (B) §3.4–3.5. Карман остаётся неизменным и
    служит эталоном редукции (Gate R1).

    Parameters
    ----------
    include_ktn_terms : при ``False`` КТН-члены выключены ⇒ решатель ТОЖДЕСТВЕНЕН
        ``KarmanPlate`` (для Gate R1 — редукция до машинной точности).

    Мягкий шарнир (``bc_type="soft_hinge"``, v0.6.3). Полная КТН на шарнире
    отличается от защемления ОДНИМ граничным членом слабой формы (§3.5): при
    структуре ``ω¹`` ``v = 0`` на ∂Ω, но ``∂v/∂n ≠ 0``, поэтому в формуле Грина
    для ``−h_ψ²∫ v ΔL`` остаётся ``−h_ψ²∮ L·∂ψ/∂n ds`` (для защемления ``ω²``
    даёт ``∂ψ/∂n = 0`` ⇒ член ТОЖДЕСТВЕННО ноль). Граничный член собирается
    звёздной квадратурой по ∂Ω (:func:`_star_boundary_quad`; circle, ellipse,
    выпуклые). Корректность граничного члена проверена ДИСКРЕТНЫМ ТОЖДЕСТВОМ
    ГРИНА (``∫ v ΔL = ∫ L Δv − ∮ L ∂v/∂n`` до точности квадратуры; без члена —
    50 % ошибки), редукциями (члены-выкл → Карман; ``h → 0`` → Кирхгоф) и
    сходимостью по ``p``. Независимого литературного эталона полной КТН нет
    (как и отложенный gate R3 для защемления) — краевая модель §3.5
    предполагается стандартной (``w = 0``, ``M_n = 0``).
    """

    def __init__(self, domain, basis, quad, cfg, *, bc_type="clamped",
                 inplane_bc="immovable", include_ktn_terms=True, sides=None):
        super().__init__(domain, basis, quad, cfg, bc_type=bc_type,
                         inplane_bc=inplane_bc, sides=sides)
        self._include_ktn = bool(include_ktn_terms)
        self._faces = FaceParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
        self._h_psi_sq = self._faces.h_psi_sq
        self._h_star_sq = self._faces.h_star_sq
        # вторые производные структуры прогиба (для члена B) — в узлах квадратуры
        _, _, _, pxx, pyy, pxy = self._wstruct(quad.x, quad.y)
        self._pxx, self._pyy, self._pxy = pxx, pyy, pxy
        self._lap_psi = pxx + pyy
        self._lap_q = None                           # Δq для члена (A); см. set_load_laplacian
        # Граничный член слабой формы для мягкого шарнира (§3.5): нужен ТОЛЬКО
        # когда активны КТН-члены (h_ψ² > 0, ktn_full). Для Кармана (члены выкл)
        # и защемления не собирается.
        self._soft_hinge_bnd = (bc_type == "soft_hinge") and self._include_ktn
        self._bnd = None
        if self._soft_hinge_bnd:
            Xb, Yb, ds_b, nx_b, ny_b = _boundary_quad(domain)
            _, psx_b, psy_b, pxx_b, pyy_b, pxy_b = self._wstruct(Xb, Yb)
            self._bnd = {
                "Xb": Xb, "Yb": Yb, "ds": ds_b,
                "pxx": pxx_b, "pyy": pyy_b, "pxy": pxy_b,
                "dpsidn": psx_b * nx_b + psy_b * ny_b,       # ∂ψ_j/∂n на ∂Ω, (N, M_b)
            }

    def set_load_laplacian(self, lap_q) -> None:
        """Задать Δq в узлах квадратуры для члена КТН (A) ``−h_*²Δq`` (§7).

        ``None`` ⇒ член опущен (равномерная нагрузка даёт Δq ≡ 0; разрывные
        patch/point — Δq не гладкая, член опускается с пометкой в диспетчере).
        """
        self._lap_q = None if lap_q is None else np.asarray(lap_q, float)

    def _load_vector(self, f_values):
        r"""Вектор нагрузки КТН: ``∫q ψ − h_*²∫Δq ψ`` (член A, §3.4/§7).

        Под равномерной нагрузкой второй член нулевой (Δq = 0); под НЕравномерной
        (gaussian) он смещает даже СРЕДИННЫЙ прогиб КТН относительно классики.
        """
        b = super()._load_vector(f_values)
        if self._include_ktn and self._lap_q is not None:
            b = b - self._h_star_sq * (self._psi @ (self._W * self._lap_q))
        return b

    @classmethod
    def from_config(cls, domain, cfg, *, bc_type="clamped", inplane_bc="immovable",
                    include_ktn_terms=True, sides=None):
        """Собрать решатель: базис степени ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg, bc_type=bc_type, inplane_bc=inplane_bc,
                   include_ktn_terms=include_ktn_terms, sides=sides)

    def _ktn_regularization(self, Nx, Ny, Nxy) -> np.ndarray:
        r"""Матрица члена (B): ``M_2[j,i] = ∫ (N:∇∇ψ_i)·Δψ_j dA`` (несимметрична)."""
        W = self._W
        # L, применённый к структуре i: N_x ψ_{i,xx} + 2 N_xy ψ_{i,xy} + N_y ψ_{i,yy}
        LN = self._pxx * Nx + 2.0 * self._pxy * Nxy + self._pyy * Ny
        return (self._lap_psi * W) @ LN.T

    def _boundary_term_from_N(self, Nx_b, Ny_b, Nxy_b) -> np.ndarray:
        r"""Граничная матрица по усилиям НА ГРАНИЦЕ: ``B[j,i] = ∮ (N:∇∇ψ_i)·∂ψ_j/∂n ds``.

        Дискретная реализация граничного интеграла формулы Грина (§3.5). Отделена
        от вычисления ``N``, чтобы верификация (MMS с фиксированным ``N``) могла
        подать усилия напрямую.
        """
        b = self._bnd
        LN = b["pxx"] * Nx_b + 2.0 * b["pxy"] * Nxy_b + b["pyy"] * Ny_b  # L(ψ_i) на ∂Ω, (N, M_b)
        return (b["dpsidn"] * b["ds"]) @ LN.T

    def _ktn_boundary_term(self, cu, cv, cw) -> np.ndarray:
        r"""Граничная матрица (§3.5, ЭКСПЕРИМ.): усилия ``N`` на границе из состояния.

        Возникает в формуле Грина для ``−h_ψ²∫ v ΔL`` при мягком шарнире (``v=0``,
        ``∂v/∂n≠0`` на ∂Ω). ``N`` берутся на границе из текущего решения
        (``membrane_forces_at``). Для защемления ``∂ψ/∂n = 0`` ⇒ матрица нулевая.
        """
        b = self._bnd
        Nx_b, Ny_b, Nxy_b = self.membrane_forces_at(cu, cv, cw, b["Xb"], b["Yb"])
        return self._boundary_term_from_N(Nx_b, Ny_b, Nxy_b)

    def _newton_tangent(self, c, forces) -> np.ndarray:
        r"""Касательный оператор КТН: кармановский ``J`` + производная члена (B).

        ``+ h_ψ²(M_2 + T_{M2})`` — матрица регуляризации и её ∂N-производная
        (``T_{M2,jk} = ∫ Δψ_j (∂N_αβ/∂c_k) w_{,αβ}``, тот же ``∂N/∂c``, что у
        Кармана, но с весом кривизны). Для защемления касательная ТОЧНА
        (проверено КР); на мягком шарнире граничный член добавляется матрицей
        без ∂N-члена (``O(h²)`` — модифицированный Ньютон, сверхлинейно).
        """
        J = super()._newton_tangent(c, forces)
        if not self._include_ktn:
            return J
        a, b, Nx, Ny, Nxy = forces
        dNx, dNy, dNxy = self._dN_dc(c @ self._psi_x, c @ self._psi_y)
        wxx, wyy, wxy = c @ self._pxx, c @ self._pyy, c @ self._pxy
        dL = dNx * wxx[:, None] + 2.0 * dNxy * wxy[:, None] + dNy * wyy[:, None]
        T_M2 = (self._lap_psi * self._W[None, :]) @ dL       # ∂N-член M_2
        J = J + self._h_psi_sq * (self._ktn_regularization(Nx, Ny, Nxy) + T_M2)
        if self._soft_hinge_bnd:
            J = J - self._h_psi_sq * self._ktn_boundary_term(a, b, c)
        return J

    def _nonlinear_operator(self, c, forces) -> np.ndarray:
        r"""Оператор невязки КТН: Карман ``+ h_ψ²(M_2 − B_∂Ω)`` (§3.5), применённый к ``c``."""
        Ac = super()._nonlinear_operator(c, forces)
        if self._include_ktn:
            a, b, Nx, Ny, Nxy = forces
            Ac = Ac + self._h_psi_sq * (self._ktn_regularization(Nx, Ny, Nxy) @ c)
            if self._soft_hinge_bnd:
                Ac = Ac - self._h_psi_sq * (self._ktn_boundary_term(a, b, c) @ c)
        return Ac

    def solve(self, f_values, c0=None):
        r"""Пикар (§5.3) или Ньютон (§5.4, ``ktn_method='newton'``).

        Ньютон использует МОДИФИЦИРОВАННЫЙ касательный оператор — кармановский
        ``J`` (члены КТН порядка ``O(h²/L²)`` — малое возмущение, тангенс почти
        точен ⇒ сверхлинейная сходимость без сборки касательной КТН-членов).
        ``c0`` — тёплый старт (внешний цикл МОР, `contact_nl.py`).
        """
        method = getattr(self.cfg, "ktn_method", "picard")
        if method == "newton":
            return self._solve_newton(f_values, c0=c0)
        if method != "picard":
            raise NotImplementedError(
                f"ktn_method = {method!r}: ожидалось picard | newton")
        return super().solve(f_values, c0=c0)

    def solve_fixed_forces(self, q_values, Nx, Ny, Nxy, lap_q=None) -> np.ndarray:
        r"""ЛИНЕЙНЫЙ решатель КТН при ЗАМОРОЖЕННЫХ мембранных усилиях ``N`` (верификация).

        Усилия ``N`` берутся ГОТОВЫМИ (не пересчитываются из ``w``), поэтому
        внеплоскостной оператор КТН линеен по ``w``:

        .. math::
            \bigl(D\,S_\text{bend} + K_\text{geo}(N)
                  + h_\psi^2 M_2(N) - [h_\psi^2 B_{\partial\Omega}(N)]\bigr)\,c
            = \int q\,\psi - h_*^2\!\int \Delta q\,\psi.

        Граничный член ``B_{\partial\Omega}`` добавляется лишь на мягком шарнире
        (§3.5); при защемлении он тождественно нулевой. Служит методу
        изготовленных решений (MMS, `ladder.mms_ktn_load_and_exact`): по
        согласованным ``q``, ``Δq`` и ``N`` восстанавливает ``w`` в структуре
        решателя (сертификат сборки — все члены КТН одним прогоном). ``N`` —
        скаляр или поле в узлах квадратуры.

        Returns
        -------
        c : коэффициенты решения (``deflection(c, ·)`` — прогиб).
        """
        shape = self.quad.x.shape
        Nx = np.broadcast_to(np.asarray(Nx, float), shape)
        Ny = np.broadcast_to(np.asarray(Ny, float), shape)
        Nxy = np.broadcast_to(np.asarray(Nxy, float), shape)
        A = self.D * self._S_bend + self._geometric_stiffness(Nx, Ny, Nxy)
        if self._include_ktn:
            A = A + self._h_psi_sq * self._ktn_regularization(Nx, Ny, Nxy)
            if self._soft_hinge_bnd:                    # граничный член §3.5 (усилия на ∂Ω)
                b = self._bnd
                nb = b["Xb"].shape
                A = A - self._h_psi_sq * self._boundary_term_from_N(
                    np.broadcast_to(np.asarray(Nx).ravel()[0], nb),
                    np.broadcast_to(np.asarray(Ny).ravel()[0], nb),
                    np.broadcast_to(np.asarray(Nxy).ravel()[0], nb))
        self.set_load_laplacian(lap_q)
        return _lin_solve(A, self._load_vector(q_values))

    def _picard_map(self, c, b_level, theta):
        r"""Шаг Пикара с КТН-членом (B). При выключенных членах — тождественно Карман."""
        if not self._include_ktn:
            return super()._picard_map(c, b_level, theta)
        wx = c @ self._psi_x
        wy = c @ self._psi_y
        a, b, Nx, Ny, Nxy = self._membrane_forces(wx, wy)
        A = (self.D * self._S_bend
             + self._geometric_stiffness(Nx, Ny, Nxy)
             + self._h_psi_sq * self._ktn_regularization(Nx, Ny, Nxy))
        if self._soft_hinge_bnd:
            # мягкий шарнир: +h_ψ²∫ v ΔL = +h_ψ²(M_2 − B_∂Ω) (формула Грина, §3.5)
            A = A - self._h_psi_sq * self._ktn_boundary_term(a, b, c)
        c_raw = _lin_solve(A, b_level)
        g = (1.0 - theta) * c + theta * c_raw
        return g, (a, b, Nx, Ny, Nxy)

    # -- интроспекция и лицевые параметры ------------------------------- #
    @property
    def face_params(self) -> FaceParams:
        """Параметры лицевых величин (§6.3): h_ψ², h_*², h_c²."""
        return self._faces


__all__ = ["KTNPlate"]
