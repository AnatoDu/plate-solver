r"""membrane.py — геометрически-НЕЛИНЕЙНЫЙ изгиб по теории Кармана (Föppl–von Kármán).

Неизвестные — прогиб ``w`` и перемещения срединной плоскости ``u, v``.
Мембранные деформации содержат КВАДРАТИЧНЫЕ по прогибу члены (§3.1):

.. math::
    \varepsilon_x = u_{,x} + \tfrac12 w_{,x}^2, \quad
    \varepsilon_y = v_{,y} + \tfrac12 w_{,y}^2, \quad
    \gamma_{xy} = u_{,y} + v_{,x} + w_{,x} w_{,y};

мембранные усилия (плоское напряжённое, изотропия, ``C = Eh/(1-\nu^2)``):

.. math::
    N_x = C(\varepsilon_x + \nu\varepsilon_y), \quad
    N_y = C(\varepsilon_y + \nu\varepsilon_x), \quad
    N_{xy} = \tfrac{1-\nu}{2} C\,\gamma_{xy}.

Система Фёппля–Кармана — плоское равновесие ``N_{\alpha\beta,\beta}=0`` плюс
внеплоскостное уравнение

.. math:: D\,\Delta^2 w = q + (N_x w_{,xx} + 2 N_{xy} w_{,xy} + N_y w_{,yy}).

При ``N -> 0`` возвращается классика ``D\Delta^2 w = q`` (контроль предела —
Gate L). Слоистая архитектура: КТН-члены полной теории (релиз v0.5.0)
надстраиваются ПОВЕРХ мембранного решателя, не переписывая его (§1.1 ТЗ).

Метод (итерация Пикара по замороженным усилиям, §5.1)
-----------------------------------------------------
Дано ``w_0`` (при ``c=0`` первый шаг даёт линейное решение Кирхгофа). Повторять:

1. по ``w_k`` вычислить нелинейные части деформаций ``½w_x², ½w_y², w_x w_y``;
2. решить ЛИНЕЙНУЮ плоскую задачу для ``(u, v)`` с этими частями как
   предварительные деформации → усилия ``N_k`` в узлах квадратуры;
3. решить внеплоскостное уравнение с ЗАМОРОЖЕННЫМИ ``N_k``:
   ``(D·S_bend + K_geo(N_k)) c = b`` → ``w_new``;
4. недорелаксация ``w_{k+1} = (1-θ)w_k + θ w_new``;
5. останов ``‖w_{k+1}-w_k‖_{L2} / ‖w_{k+1}‖ < karman_tol``.

Геометрическая жёсткость берётся в СИММЕТРИЧНОЙ форме (интегрирование по
частям члена ``N:∇∇w`` при ``v=0`` на ∂Ω, §5.1):

.. math:: K_{geo}[i,j] = \iint_\Omega N_{\alpha\beta}\,\psi_{i,\alpha}\,\psi_{j,\beta}\,dA
          = \iint (N_x \psi_{i,x}\psi_{j,x} + N_{xy}(\psi_{i,x}\psi_{j,y}
          + \psi_{i,y}\psi_{j,x}) + N_y \psi_{i,y}\psi_{j,y})\,dA.

Симметричная форма НЕ требует дискретного равновесия ``N`` (в отличие от
``∫N:(∇∇ψ_i)ψ_j``) и положительно определена при растяжении — мембранное
ужесточение (пластина под нагрузкой с неподвижной кромкой ужестчается).

Мат. обоснование сходимости (кандидат в теорему T5, NOTES.md) — отображение
Пикара ``w_{k+1}=T(w_k)`` есть сжатие при нагрузке ниже порога: нелинейный
член квадратичен по ``w``, ``T`` липшицев с константой ``L(R,q)``, при малой
``q`` (или малом шаге нагрузки) ``L<1``; шаги по нагрузке продолжают решение
по параметру. Валидация — против эталонов РАЗНОЙ природы (Hencky, Way,
Тимошенко, Levy), см. ``benchmarks.py`` и ``tests/test_karman.py``.

Точки расширения (не в v0.4.0): полная нелинейная КТН ``(I-h_Ψ²Δ)L(Φ,w)``
поверх ``N`` (v0.5.0); Ньютон как ускоритель (флаг ``method="newton"``);
Карман на невыпуклых областях и нелинейный контакт.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .basis import ChebyshevBasis
from .clamped import _basis_second_derivs, _cheb_value_tables, _OmegaHessian
from .ladder import _basis_xy
from .quadrature import interior_nodes

# Степень множителя ω в структуре прогиба w = ω^m·Φ по типу изгибной кромки.
_W_POWER = {"clamped": 2, "soft_hinge": 1}


# --------------------------------------------------------------------------- #
#  Структура прогиба ψ = ω^m·T и её первые/вторые производные (символьно ω)
# --------------------------------------------------------------------------- #
def _w_structure(domain, basis: ChebyshevBasis, X, Y, power: int):
    r"""Структура ``ψ = ω^power·T`` и её производные в точках (X, Y).

    Возвращает ``(ψ, ψ_x, ψ_y, ψ_xx, ψ_yy, ψ_xy)`` — массивы ``(N, *X.shape)``.
    Первые производные нужны геометрической жёсткости ``K_geo`` (форма
    ``N_{αβ}ψ_{,α}ψ_{,β}``), вторые — билинейной форме изгиба и моментам.
    Множитель ``g = ω^power`` (power=2 — защемление ``w=∂w/∂n=0``; power=1 —
    мягкий шарнир ``w=0``) дифференцируется по Лейбницу:
    ``[ω²]_a = 2ωω_a``, ``[ω²]_{ab} = 2(ω_aω_b + ωω_{ab})``.
    """
    om, omx, omy, omxx, omyy, omxy = _OmegaHessian(domain).fields_full(X, Y)
    if power == 2:
        g = om**2
        gx, gy = 2.0 * om * omx, 2.0 * om * omy
        gxx = 2.0 * (omx**2 + om * omxx)
        gyy = 2.0 * (omy**2 + om * omyy)
        gxy = 2.0 * (omx * omy + om * omxy)
    elif power == 1:
        g, gx, gy, gxx, gyy, gxy = om, omx, omy, omxx, omyy, omxy
    else:                                                    # pragma: no cover
        raise ValueError(f"power структуры w: ожидалось 1 | 2, получено {power}")
    T = basis.values(X, Y)
    Tx, Ty = basis.grads(X, Y)
    lapT = _basis_second_derivs(basis, X, Y)                 # T_xx + T_yy
    Txy = _basis_xy(basis, X, Y)
    xi, eta = basis.to_reference(X, Y)
    xmin, xmax, _ymin, _ymax = basis.bbox
    dxi_dx = 2.0 / (xmax - xmin)
    _, _, Vx2 = _cheb_value_tables(basis, xi)
    Vy0, _, _ = _cheb_value_tables(basis, eta)
    Txx = ((Vx2[:, None, ...] * Vy0[None, :, ...]) * dxi_dx**2).reshape(basis.N, *xi.shape)
    Tyy = lapT - Txx
    psi = g * T
    psi_x = gx * T + g * Tx
    psi_y = gy * T + g * Ty
    psi_xx = gxx * T + 2.0 * gx * Tx + g * Txx
    psi_yy = gyy * T + 2.0 * gy * Ty + g * Tyy
    psi_xy = gxy * T + gx * Ty + gy * Tx + g * Txy
    return psi, psi_x, psi_y, psi_xx, psi_yy, psi_xy


def _disp_structure(domain, basis: ChebyshevBasis, X, Y, immovable: bool):
    r"""Структура перемещений в плане и её первые производные: ``(P, P_x, P_y)``.

    ``immovable`` (u=v=0 на ∂Ω): ``P = ω·φ`` — тождественно зануляет u, v на
    границе (кромка не втягивается, натяжение максимально). ``movable``
    (``N·n = 0``, естественное): ``P = φ`` (без множителя ω); плоская задача
    тогда имеет жёсткие моды (сдвиги, поворот), правая часть предварительных
    деформаций им ортогональна — решается МНК (минимальная норма).

    ⚠️ Термин ``movable`` здесь — ПОЛНОСТЬЮ свободная в плане кромка
    (``N_n = N_{ns} = 0``, кромка может искривляться). Это НЕ «Edge Compression
    Zero» Levy (§A.5), где кромки остаются ПРЯМЫМИ (жёстче): наш вариант
    компланарно податливее, поэтому даёт бо́льший прогиб при той же нагрузке.
    Оба совпадают в линейном пределе; основной проверяемый режим — ``immovable``
    (лучшие эталоны: Way, Hencky, Levy immovable).
    """
    T = basis.values(X, Y)
    Tx, Ty = basis.grads(X, Y)
    if not immovable:
        return T, Tx, Ty
    om = domain.omega(X, Y)
    omx, omy = domain.grad_omega(X, Y)
    return om * T, omx * T + om * Tx, omy * T + om * Ty


def _spd_solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение симметричной системы с диагональным предобуславливанием.

    Нормировка на ``√A_kk`` (NOTES §2) + Холецкий; при потере положительной
    определённости (сжатие, округление) — линейный МНК. ``b`` — вектор или
    матрица столбцов правых частей.
    """
    d = np.diag(A)
    if np.all(d > 0.0):
        s = 1.0 / np.sqrt(d)
        An = (A * s).T * s
        bn = (b.T * s).T if b.ndim > 1 else b * s
        try:
            xn = sla.cho_solve(sla.cho_factor(An), bn)
            return (xn.T * s).T if b.ndim > 1 else xn * s
        except (sla.LinAlgError, np.linalg.LinAlgError):
            pass
    return np.linalg.lstsq(A, b, rcond=1e-13)[0]


# --------------------------------------------------------------------------- #
#  Результат нелинейного решения
# --------------------------------------------------------------------------- #
@dataclass
class KarmanResult:
    """Итог нелинейного решения Кармана (поля в узлах + история сходимости).

    Attributes
    ----------
    cw : коэффициенты прогиба ``w = ω^m·Σ c_k T_k``.
    w_nodes : прогиб в узлах квадратуры (источник ``w_max``).
    w_max, w_max_classic : max|w| нелинейный и линейный (Кирхгоф, ``N=0``).
    cu, cv : коэффициенты перемещений ``u, v`` в плане.
    Nx, Ny, Nxy : мембранные усилия в узлах квадратуры.
    converged : достигнут ли ``karman_tol`` на ПОСЛЕДНЕМ уровне нагрузки.
    n_iter : суммарное число итераций Пикара по всем уровням.
    history : по одному кортежу ``(доля нагрузки, итераций, финальная невязка)``
        на каждый уровень (диагностика сходимости, §5.2).
    """

    cw: np.ndarray
    w_nodes: np.ndarray
    w_max: float
    w_max_classic: float
    cu: np.ndarray
    cv: np.ndarray
    Nx: np.ndarray
    Ny: np.ndarray
    Nxy: np.ndarray
    converged: bool
    n_iter: int
    history: list = field(default_factory=list)


# --------------------------------------------------------------------------- #
#  Решатель Кармана
# --------------------------------------------------------------------------- #
class KarmanPlate:
    r"""Геометрически-нелинейный изгиб пластины по теории Кармана (§5).

    Parameters
    ----------
    domain : область Ω (R-функция ``ω``); в v0.4.0 — круг или прямоугольник.
    basis : тензорный базис Чебышёва (общий для ``w`` и ``u, v``).
    quad : узлы квадратуры внутри Ω.
    cfg : параметры (``E, nu, h, q0``; итерация — ``n_load_steps``,
        ``karman_tol``, ``karman_max_iter``, ``karman_relax``, ``karman_method``).
    bc_type : изгибная кромка ``clamped`` (``w=ω²Φ``) | ``soft_hinge`` (``w=ωΦ``).
    inplane_bc : закрепление в плане ``immovable`` (u=v=0) | ``movable`` (N·n=0).

    Оператор изгиба ``S_bend`` (полная билинейная форма) и мембранная жёсткость
    ``K_uv`` от прогиба НЕ зависят — факторизуются ОДИН раз; на каждом шаге
    Пикара пересобирается лишь ``K_geo(N_k)`` (мал: ``N=(p+1)²``).
    """

    def __init__(self, domain, basis, quad, cfg, *, bc_type="clamped",
                 inplane_bc="immovable"):
        if bc_type not in _W_POWER:
            raise ValueError(f"bc_type: ожидалось {' | '.join(_W_POWER)}, получено {bc_type!r}")
        if inplane_bc not in ("immovable", "movable"):
            raise ValueError(f"inplane_bc: ожидалось immovable | movable, получено {inplane_bc!r}")
        self.domain, self.basis, self.quad, self.cfg = domain, basis, quad, cfg
        self.bc_type, self.inplane_bc = bc_type, inplane_bc
        self.D = float(cfg.D)
        self.nu = float(cfg.nu)
        self._power = _W_POWER[bc_type]
        X, Y, W = quad.x, quad.y, quad.w
        self._W = W
        # -- структура прогиба и билинейная форма изгиба (полная, NOTES §20) -- #
        psi, psi_x, psi_y, pxx, pyy, pxy = _w_structure(domain, basis, X, Y, self._power)
        self._psi, self._psi_x, self._psi_y = psi, psi_x, psi_y
        lap = pxx + pyy
        S = (lap * W) @ lap.T - (1.0 - self.nu) * (
            (pxx * W) @ pyy.T + (pyy * W) @ pxx.T - 2.0 * (pxy * W) @ pxy.T)
        self._S_bend = 0.5 * (S + S.T)
        # -- мембранная жёсткость плоской задачи (от w не зависит) ----------- #
        C = cfg.E * cfg.h / (1.0 - self.nu**2)
        self._C = C
        immovable = inplane_bc == "immovable"
        P, Px, Py = _disp_structure(domain, basis, X, Y, immovable)
        self._Px, self._Py = Px, Py
        half = 0.5 * (1.0 - self.nu)
        Kaa = C * ((Px * W) @ Px.T + half * (Py * W) @ Py.T)
        Kbb = C * ((Py * W) @ Py.T + half * (Px * W) @ Px.T)
        Kab = C * (self.nu * (Px * W) @ Py.T + half * (Py * W) @ Px.T)
        n = basis.N
        Kuv = np.empty((2 * n, 2 * n))
        Kuv[:n, :n] = Kaa
        Kuv[:n, n:] = Kab
        Kuv[n:, :n] = Kab.T
        Kuv[n:, n:] = Kbb
        self._Kuv = 0.5 * (Kuv + Kuv.T)
        # immovable ⇒ K_uv положительно определена (u=v=0 на ∂Ω снимает жёсткие
        # моды); movable ⇒ ядро жёстких мод, правая часть ему ортогональна,
        # решаем псевдообращением (минимальная норма).
        self._membrane_immovable = immovable
        if not immovable:
            self._Kuv_pinv = np.linalg.pinv(self._Kuv)
        else:
            self._Kuv_pinv = None
        self._n = n

    @classmethod
    def from_config(cls, domain, cfg, *, bc_type="clamped", inplane_bc="immovable"):
        """Собрать решатель: базис степени ``cfg.p``, квадратура ``cfg.Q``."""
        basis = ChebyshevBasis(cfg.p, domain.bbox)
        quad = interior_nodes(domain, cfg.Q)
        return cls(domain, basis, quad, cfg, bc_type=bc_type, inplane_bc=inplane_bc)

    # -- подзадачи итерации ------------------------------------------------- #
    def _membrane_forces(self, wx, wy):
        r"""Усилия ``N`` в узлах по нелинейным деформациям ``½w_x², ½w_y², w_x w_y``.

        Решается ЛИНЕЙНАЯ плоская задача ``K_uv·[a;b] = f_e`` (равновесие
        ``∫δε^T N = 0`` с ``N = C·D_{ps}(ε^{lin} + e)``), затем
        ``N = C·D_{ps}(ε^{lin} + e)`` в узлах. ``D_{ps}`` — матрица плоского
        напряжённого состояния.
        """
        C, nu, W = self._C, self.nu, self._W
        Px, Py = self._Px, self._Py
        e_x, e_y, e_xy = 0.5 * wx**2, 0.5 * wy**2, wx * wy
        # предварительные усилия от нелинейных деформаций
        Ne_x = C * (e_x + nu * e_y)
        Ne_y = C * (e_y + nu * e_x)
        Ne_xy = C * 0.5 * (1.0 - nu) * e_xy
        f_a = -(Px @ (W * Ne_x) + Py @ (W * Ne_xy))
        f_b = -(Py @ (W * Ne_y) + Px @ (W * Ne_xy))
        rhs = np.concatenate([f_a, f_b])
        if self._membrane_immovable:
            sol = _spd_solve(self._Kuv, rhs)
        else:
            sol = self._Kuv_pinv @ rhs
        a, b = sol[:self._n], sol[self._n:]
        ex = a @ Px + e_x                                    # полная деформация ε
        ey = b @ Py + e_y
        gxy = a @ Py + b @ Px + e_xy
        Nx = C * (ex + nu * ey)
        Ny = C * (ey + nu * ex)
        Nxy = C * 0.5 * (1.0 - nu) * gxy
        return a, b, Nx, Ny, Nxy

    def _geometric_stiffness(self, Nx, Ny, Nxy):
        r"""``K_geo[i,j] = ∫ N_{αβ} ψ_{i,α} ψ_{j,β} dΩ`` (симметрична)."""
        W = self._W
        px, py = self._psi_x, self._psi_y
        Kg = (px * (Nx * W)) @ px.T + (py * (Ny * W)) @ py.T \
            + (px * (Nxy * W)) @ py.T + (py * (Nxy * W)) @ px.T
        return 0.5 * (Kg + Kg.T)

    def _bending_solve(self, b_vec):
        """Линейное решение Кирхгофа ``D·S_bend c = b`` (при ``N=0``)."""
        return _spd_solve(self.D * self._S_bend, b_vec)

    def _load_vector(self, f_values):
        """Вектор нагрузки ``b[k] = ∫ f ψ_k`` в узлах квадратуры."""
        return self._psi @ (self._W * np.asarray(f_values, float))

    # -- один шаг Пикара (демпфированный) ----------------------------------- #
    def _picard_map(self, c, b_level, theta):
        r"""Одно применение отображения Пикара ``T`` к коэффициентам ``c``.

        По ``c`` строится ``N(w)``, собирается ``K_geo`` и решается
        ``(D·S_bend + K_geo) c_raw = b``; возвращается демпфированное значение
        ``(1-θ)c + θ c_raw`` (при ``θ=1`` — чистый Пикар) вместе с усилиями.
        Неподвижная точка ``T`` не зависит от ``θ`` — демпфирование лишь
        стабилизирует базовое отображение (T5, NOTES.md).
        """
        wx = c @ self._psi_x
        wy = c @ self._psi_y
        a, b, Nx, Ny, Nxy = self._membrane_forces(wx, wy)
        A = self.D * self._S_bend + self._geometric_stiffness(Nx, Ny, Nxy)
        c_raw = _spd_solve(A, b_level)
        g = (1.0 - theta) * c + theta * c_raw
        return g, (a, b, Nx, Ny, Nxy)

    # -- основной цикл ------------------------------------------------------ #
    def solve(self, f_values, c0=None) -> KarmanResult:
        r"""Итерация Пикара (ускорение Андерсона) с шагами по нагрузке (§5.1–5.2).

        ``f_values`` — интенсивность нагрузки в узлах квадратуры (равномерная
        или зонная). Нагрузка наращивается за ``n_load_steps`` шагов, тёплый
        старт каждого уровня — предыдущим решением; на уровне — неподвижная
        точка отображения Пикара ``T`` (§5.1) с УСКОРЕНИЕМ АНДЕРСОНА
        (смешивание истории невязок ``f=T(c)-c`` по окну ``m``): то же
        отображение, но сходимость из линейной становится сверхлинейной, что
        делает достижимым мембранный предел Gate M (``w/h ~ 6``). При
        ``m=0`` — чистый Пикар. Останов ``‖Δw‖_{L2}/‖w‖ < karman_tol``.

        ``c0`` — тёплый старт коэффициентов прогиба (например, решение с
        предыдущего шага внешнего цикла МОР, `contact_nl.py`): при заданном
        нагрузка НЕ дробится (один уровень на полной нагрузке), т.к. старт уже
        близок к решению — резко экономит итерации.
        """
        cfg = self.cfg
        if getattr(cfg, "karman_method", "picard") != "picard":
            raise NotImplementedError(
                "karman_method = 'newton' — опциональный ускоритель, в v0.4.0 "
                "не реализован (метод по умолчанию — 'picard')")
        f_values = np.asarray(f_values, float)
        b_full = self._load_vector(f_values)
        theta = float(cfg.karman_relax)
        tol = float(cfg.karman_tol)
        max_iter = int(cfg.karman_max_iter)
        n_steps = max(1, int(cfg.n_load_steps))
        m_win = int(getattr(cfg, "karman_anderson", 6))      # окно ускорения Андерсона
        W = self._W
        if c0 is None:
            c = np.zeros(self.basis.N)                       # тёплый старт по уровням
        else:
            c = np.asarray(c0, float).copy()                 # тёплый старт извне (МОР)
            n_steps = 1                                      # старт близок ⇒ полная нагрузка
        history: list = []
        total_iter = 0
        converged = False
        forces = (None, None, np.zeros(W.size), np.zeros(W.size), np.zeros(W.size))
        for step in range(1, n_steps + 1):
            b_level = (step / n_steps) * b_full
            converged = False
            rel = float("nan")
            g_hist: list[np.ndarray] = []
            f_hist: list[np.ndarray] = []
            it = 0
            for it in range(1, max_iter + 1):  # noqa: B007 — it нужен после цикла
                w_old = c @ self._psi
                g, forces = self._picard_map(c, b_level, theta)
                f = g - c
                g_hist.append(g)
                f_hist.append(f)
                if len(f_hist) == 1 or m_win == 0:
                    c = g                                    # первый шаг — чистый Пикар
                else:
                    mk = min(m_win, len(f_hist) - 1)
                    dF = np.column_stack([f_hist[-j] - f_hist[-j - 1] for j in range(1, mk + 1)])
                    dG = np.column_stack([g_hist[-j] - g_hist[-j - 1] for j in range(1, mk + 1)])
                    gamma = np.linalg.lstsq(dF, f, rcond=None)[0]
                    c = g - dG @ gamma
                if len(f_hist) > m_win + 1:                  # ограничить окно памяти
                    del g_hist[0], f_hist[0]
                w_new = c @ self._psi
                denom = np.sqrt(np.sum(W * w_new**2))
                rel = (float(np.sqrt(np.sum(W * (w_new - w_old) ** 2)) / denom)
                       if denom > 0 else 0.0)
                if rel < tol:
                    converged = True
                    break
            total_iter += it
            history.append((step / n_steps, it, rel))
        a, b, Nx, Ny, Nxy = forces
        # финальные поля на достигнутом уровне (полная нагрузка)
        w_nodes = c @ self._psi
        w_max = float(np.max(np.abs(w_nodes)))
        c_lin = self._bending_solve(b_full)                  # линейный Кирхгоф (N=0)
        w_max_classic = float(np.max(np.abs(c_lin @ self._psi)))
        self._cw = c
        return KarmanResult(
            cw=c, w_nodes=w_nodes, w_max=w_max, w_max_classic=w_max_classic,
            cu=a, cv=b, Nx=Nx, Ny=Ny, Nxy=Nxy, converged=converged,
            n_iter=total_iter, history=history)

    def solve_uniform(self, q: float | None = None) -> KarmanResult:
        """Решить под равномерной нагрузкой ``q`` (по умолчанию ``cfg.q0``)."""
        q = self.cfg.q0 if q is None else float(q)
        return self.solve(np.full(self.quad.x.size, q))

    # -- поля решения ------------------------------------------------------- #
    @property
    def cond(self) -> float:
        """Число обусловленности линейного оператора изгиба ``cond(D·S_bend)``."""
        return float(np.linalg.cond(self.D * self._S_bend))

    def deflection(self, c, X, Y) -> np.ndarray:
        """Прогиб ``w = ω^m·Σ c_k T_k`` в точках (X, Y)."""
        Phi = self.basis.values(X, Y)
        v = np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))
        return self.domain.omega(X, Y) ** self._power * v

    def deflection_at_quad(self, c) -> np.ndarray:
        """Прогиб в узлах квадратуры через кэш структуры ψ (один GEMV)."""
        return np.tensordot(np.asarray(c, float), self._psi, axes=(0, 0))

    def structure_at(self, X, Y) -> np.ndarray:
        """Матрица структуры ψ_k = ω^m·T_k в произвольных точках: (N, len(X))."""
        Phi = self.basis.values(X, Y)
        return self.domain.omega(X, Y) ** self._power * Phi

    def moments_at(self, c, X, Y):
        r"""Изгибные моменты (Mx, My, Mxy) в точках (§6: поле для viz).

        Классические моменты Кирхгофа от прогиба:
        ``Mx = -D(w_xx + ν w_yy)``, ``My = -D(w_yy + ν w_xx)``,
        ``Mxy = -D(1-ν) w_xy``. Мембранное поле ``N`` — отдельная величина
        (:attr:`KarmanResult.Nx` …); здесь — изгибная составляющая напряжений.
        """
        c = np.asarray(c, float)
        _, _, _, pxx, pyy, pxy = _w_structure(self.domain, self.basis,
                                               np.asarray(X, float),
                                               np.asarray(Y, float), self._power)
        wxx = np.tensordot(c, pxx, axes=(0, 0))
        wyy = np.tensordot(c, pyy, axes=(0, 0))
        wxy = np.tensordot(c, pxy, axes=(0, 0))
        nu, D = self.nu, self.D
        return (-D * (wxx + nu * wyy), -D * (wyy + nu * wxx), -D * (1.0 - nu) * wxy)

    def membrane_forces_at(self, cu, cv, cw, X, Y):
        r"""Мембранные усилия (Nx, Ny, Nxy) в произвольных точках (для viz/полей).

        По коэффициентам перемещений ``(cu, cv)`` и прогиба ``cw`` восстанавливает
        полные деформации ``ε = ε^{lin}(u,v) + e(w)`` и усилия
        ``N = C·D_{ps}·ε``. Осмысленно только при ``inplane_bc = immovable``
        со структурой ``ω·φ`` (movable — та же формула с ``φ``).
        """
        immovable = self._membrane_immovable
        _, Px, Py = _disp_structure(self.domain, self.basis, np.asarray(X, float),
                                    np.asarray(Y, float), immovable)
        _, wx_s, wy_s, _, _, _ = _w_structure(self.domain, self.basis,
                                              np.asarray(X, float),
                                              np.asarray(Y, float), self._power)
        cu = np.asarray(cu, float)
        cv = np.asarray(cv, float)
        cw = np.asarray(cw, float)
        wx = np.tensordot(cw, wx_s, axes=(0, 0))
        wy = np.tensordot(cw, wy_s, axes=(0, 0))
        ex = np.tensordot(cu, Px, axes=(0, 0)) + 0.5 * wx**2
        ey = np.tensordot(cv, Py, axes=(0, 0)) + 0.5 * wy**2
        gxy = np.tensordot(cu, Py, axes=(0, 0)) + np.tensordot(cv, Px, axes=(0, 0)) + wx * wy
        C, nu = self._C, self.nu
        return (C * (ex + nu * ey), C * (ey + nu * ex), C * 0.5 * (1.0 - nu) * gxy)

    def w_max_on_grid(self, c, grid_n: int = 160) -> float:
        """Максимум |w| по регулярной сетке bbox (маска ω>0)."""
        x0, x1, y0, y1 = self.domain.bbox
        Xg, Yg = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
        inside = self.domain.omega(Xg, Yg) > 0.0
        return float(np.max(np.abs(self.deflection(c, Xg[inside], Yg[inside]))))


__all__ = ["KarmanPlate", "KarmanResult"]
