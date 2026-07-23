r"""radial_ktn.py — НЕЗАВИСИМЫЙ осесимметричный решатель полной КТН (ворота R3, 1D↔2D).

У полной нелинейной КТН нет литературных эталонов, поэтому корректность метода
удостоверяется РЕДУКЦИЕЙ: 2D-решатель (`ktn_full.KTNPlate`) на КРУГЕ при
равномерной нагрузке даёт ОСЕСИММЕТРИЧНОЕ поле ``w = w(r)``, которое обязано
совпасть с решением НЕЗАВИСИМОГО одномерного радиального решателя (иная
дискретизация — радиальный Ритц по ``r``, иная квадратура, свой мембранный
подшаг). Это сильнее MMS (`ladder.mms_ktn_load_and_exact`): MMS проверяет
СБОРКУ оператора при ЗАМОРОЖЕННОМ ``N``, а R3 — ПОЛНУЮ нелинейную связь ``N(w)``.

Редукция полной КТН к оси (защемление ``w(a)=w'(a)=0``, `immovable` ``u(a)=0``,
равномерная ``q``). Обозначим ``Δ = ∂_rr + ∂_r/r``. Сильная форма:

.. math::
    D\,\Delta^2 w - (I - h_\psi^2\Delta)\,L(w) = q,\qquad
    L(w) = N_r\,w'' + N_\theta\,\frac{w'}{r},

где мембранные усилия — из осесимметричной плоской задачи Фёппля–Кармана

.. math::
    \varepsilon_r = u' + \tfrac12 (w')^2,\quad \varepsilon_\theta = \frac{u}{r},\quad
    N_r = C(\varepsilon_r + \nu\varepsilon_\theta),\quad
    N_\theta = C(\varepsilon_\theta + \nu\varepsilon_r),\quad C = \frac{E h}{1-\nu^2},

при равновесии ``(r N_r)' - N_\theta = 0``. Член ``−h_*^2 Δq`` (нагрузочный, A)
при равномерной ``q`` тождественно нулевой (``Δq = 0``) и здесь НЕ проверяется —
ворота R3 нацелены на ПОЛНУЮ нелинейную связь ``N(w)`` (геометрическая жёсткость
и член B ``h_ψ²``); член A и коэффициент ``h_*²`` сертифицируются отдельно MMS
полной КТН (`ladder.mms_ktn_load_and_exact`, замороженное ``N``). Общий с 2D —
лишь скаляр ``h_ψ²`` (из тех же `FaceParams`); всё остальное независимо.

Дискретизация — Ритц. Изгиб: ``w = (a^2-r^2)^m\,\Phi(r^2)`` (``m=2`` защемление —
зануляет ``w, w'`` на ``r=a``; ``m=1`` мягкий шарнир — зануляет ``w``, натуральное
КУ ``M_r=0``; чётность по ``r`` ⇒ регулярность в центре), радиальная энергия пластины

.. math:: U_b = \pi D\!\int_0^a\!\big[w''^2 + w'^2/r^2 + 2\nu\,w'' w'/r\big] r\,dr

(при защемлении ``= \pi D\int(\Delta w)^2 r\,dr``; на мягком шарнире ``\nu``-член
даёт натуральное ``M_r=0``). Геометрическая жёсткость — энергетическая форма
``2\pi\int N_r (w')^2 r\,dr`` (равна ``-L(w)`` в силу равновесия). Член B
(регуляризация ``h_\psi^2``) — слабая форма ``\int \Delta\psi_j\, L(\psi_k)\, r\,dr``;
на мягком шарнире добавляется граничный член §3.5 ``-a\,\psi_j'(a) L(\psi_k)(a)``
(в осесимметрии ``\psi_j'(a)=-2a`` постоянна по ``j`` ⇒ член ВЫРОЖДЕН, вклад в
решение машинно нулевой — §3.5 существен на входящих углах, не на гладком круге;
отдельно сверен тождеством Грина, `tests/test_soft_hinge_ktn.py`). Для защемления
``\psi_j'(a)=0`` ⇒ граничный член тождественно нулевой. Мембрана:
``u = r(a^2-r^2)\,\Psi(r^2)`` (зануляет ``u`` на ``r=0`` и ``r=a``), линейная плоская
задача по ``u`` при заданном ``½(w')^2``. Базис ``\Phi, \Psi`` — Чебышёв от
``2(r/a)^2-1``. Пикар по ``N`` (мягкий шарнир — с недорелаксацией: жёстче связь).

Оценка погрешности: базис аналитичен, решение гладкое ⇒ радиальный Ритц сходится
СПЕКТРАЛЬНО (для этой задачи ``w_max`` стабилен до ~1e-8 уже при ``p=12``).
Поэтому в воротах R3 1D-решение выступает практически ТОЧНЫМ оракулом, а невязка
``1D↔2D`` задаётся дискретизацией 2D-RFM на круге и УБЫВАЕТ с ростом 2D-``p``
(``rel(w_max) ≈ 4.1e-3 → 3.0e-3 → 2.5e-3 → 2.0e-3`` при ``p = 12, 14, 16, 18``,
``P̄=6``, ``h=0.1``) — т.е. редукция ТОЧНА, а согласие ограничено лишь 2D-стороной.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.polynomial.chebyshev as _cheb

from .faces import FaceParams


def _cheb_even_tables(p: int, r: np.ndarray, a: float):
    """φ_k = T_k(u), u = 2(r/a)²−1, и производные по r: (φ, φ', φ'') массивами (p+1, M)."""
    u = 2.0 * (r / a) ** 2 - 1.0
    up = 4.0 * r / a**2
    upp = np.full_like(r, 4.0 / a**2)
    eye = np.eye(p + 1)
    T = np.moveaxis(_cheb.chebvander(u, p), -1, 0)
    T1 = np.array([_cheb.chebval(u, _cheb.chebder(eye[k], 1)) if p >= 1 else 0.0 * r
                   for k in range(p + 1)])
    T2 = np.array([_cheb.chebval(u, _cheb.chebder(eye[k], 2)) if p >= 2 else 0.0 * r
                   for k in range(p + 1)])
    return T, T1 * up, T2 * up**2 + T1 * upp


@dataclass(frozen=True)
class RadialKTNResult:
    """Результат осесимметричного радиального решателя КТН."""
    cw: np.ndarray            # коэффициенты изгибной структуры
    w_max: float              # |w(0)| (максимум прогиба в центре)
    r_nodes: np.ndarray       # узлы радиальной квадратуры
    w_nodes: np.ndarray       # w в узлах
    Nr: np.ndarray            # радиальное усилие N_r в узлах
    Nth: np.ndarray           # окружное усилие N_θ в узлах
    converged: bool
    n_iter: int


class RadialKTN:
    r"""Осесимметричная полная КТН на круге как 1D-задача по радиусу (ворота R3).

    Parameters
    ----------
    a, E, nu, h : геометрия/материал (как в 2D `Config`).
    p : степень изгибного базиса Φ(r²); pm : степень мембранного Ψ(r²).
    nq : число узлов Гаусса по ``r ∈ [0, a]``.
    include_ktn : включить члены КТН (``h_ψ²``); False ⇒ чистый Карман (для R1-контроля).
    """

    def __init__(self, a: float, E: float, nu: float, h: float, *,
                 p: int = 12, pm: int = 12, nq: int = 800, include_ktn: bool = True,
                 bc: str = "clamped"):
        if bc not in ("clamped", "soft_hinge"):
            raise ValueError(f"bc = {bc!r}: ожидалось 'clamped' | 'soft_hinge'")
        self.a, self.nu, self.h, self.bc = a, nu, h, bc
        self.include_ktn = include_ktn
        self.m = 2 if bc == "clamped" else 1           # степень структуры (a²−r²)^m
        # защемление сходится Пикаром при relax=1; мягкий шарнир требует демпфирования
        self.relax_default = 1.0 if bc == "clamped" else 0.5
        self.D = E * h**3 / (12.0 * (1.0 - nu**2))
        self.C = E * h / (1.0 - nu**2)
        fp = FaceParams(E=E, nu=nu, h=h)
        self.h_psi_sq = fp.h_psi_sq
        t, wt = np.polynomial.legendre.leggauss(nq)
        self.r = 0.5 * a * (t + 1.0)
        self.W = 0.5 * a * wt
        r = self.r
        self.nb = p + 1
        # --- изгибной базис: ψ = (a²−r²)^m φ ---
        phi, phip, phipp = _cheb_even_tables(p, r, a)
        g, gp, gpp = self._struct(r, phi.dtype)
        self.psi = g * phi
        self.psip = gp * phi + g * phip
        self.psipp = gpp * phi + 2.0 * gp * phip + g * phipp
        # --- мембранный базис: u = r(a²−r²) χ (u(0)=u(a)=0) ---
        chi, chip, _ = _cheb_even_tables(pm, r, a)
        um = (a**2 - r**2) * r
        ump = a**2 - 3.0 * r**2
        self.u_prime = ump * chi + um * chip          # u_k'(r)
        self.u_over_r = (a**2 - r**2) * chi            # u_k(r)/r
        # --- граничные таблицы при r=a (член §3.5 мягкого шарнира; для защемления
        #     ψ'(a)=0 ⇒ член тождественно нулевой) ---
        rb = np.array([a])
        phb, phbp, phbpp = _cheb_even_tables(p, rb, a)
        gb, gbp, gbpp = self._struct(rb, phb.dtype)
        self.psib_p = (gbp * phb + gb * phbp)[:, 0]            # ψ_k'(a)
        self.psib_pp = (gbpp * phb + 2.0 * gbp * phbp + gb * phbpp)[:, 0]  # ψ_k''(a)
        chib, chibp, _ = _cheb_even_tables(pm, rb, a)
        self.ub_p = ((a**2 - 3.0 * rb**2) * chib + (a**2 - rb**2) * rb * chibp)[:, 0]  # u_k'(a)
        # --- предвычисленная изгибная матрица (полная энергия пластины) ---
        W = self.W
        psp, pspp = self.psip, self.psipp
        self.Kbend = self.D * (
            np.einsum("jq,kq,q->jk", pspp, pspp, r * W)
            + np.einsum("jq,kq,q->jk", psp, psp, W / r)
            + nu * (np.einsum("jq,kq,q->jk", pspp, psp, W)
                    + np.einsum("jq,kq,q->jk", psp, pspp, W)))
        self.lap_psi = pspp + psp / r                  # Δψ

    def _struct(self, r, dtype):
        """Множитель структуры ``g = (a²−r²)^m`` и производные ``g', g''`` (m=1|2)."""
        a, d = self.a, self.a**2 - r**2
        if self.m == 2:
            return d**2, -4.0 * r * d, -4.0 * a**2 + 12.0 * r**2
        return d, -2.0 * r, np.full_like(np.asarray(r, dtype), -2.0)

    def _membrane(self, wprime: np.ndarray, wprime_b: float):
        r"""Усилия ``N_r, N_θ`` в узлах и на границе ``r=a`` при заданном ``w'``.

        На границе ``ε_θ(a) = u(a)/a = 0`` (базис ``u`` зануляется) ⇒
        ``N_r(a) = C·ε_r(a)``, ``N_θ(a) = ν N_r(a)``.
        """
        r, W, nu, C = self.r, self.W, self.nu, self.C
        Br, Bth = self.u_prime, self.u_over_r
        e0 = 0.5 * wprime**2                            # нелинейная деформация ½(w')²
        Km = (np.einsum("jq,kq,q->jk", Br, Br, r * W)
              + np.einsum("jq,kq,q->jk", Bth, Bth, r * W)
              + nu * (np.einsum("jq,kq,q->jk", Br, Bth, r * W)
                      + np.einsum("jq,kq,q->jk", Bth, Br, r * W)))
        fm = -np.einsum("jq,q->j", Br + nu * Bth, e0 * r * W)
        am = np.linalg.solve(Km, fm)
        er, eth = am @ Br + e0, am @ Bth
        Nr, Nth = C * (er + nu * eth), C * (eth + nu * er)
        er_b = am @ self.ub_p + 0.5 * wprime_b**2       # ε_r(a) (ε_θ(a)=0)
        Nr_b = C * er_b
        return Nr, Nth, Nr_b, nu * Nr_b

    def solve(self, q: float, *, tol: float = 1e-11, max_iter: int = 1500,
              relax: float | None = None) -> RadialKTNResult:
        r"""Решить осесимметричную КТН под равномерной ``q`` (итерация Пикара по ``N``).

        ``relax`` — недорелаксация Пикара (``None`` ⇒ ``relax_default``: 1.0 для
        защемления, 0.5 для мягкого шарнира, где обратная связь ``N(w)`` жёстче).
        """
        relax = self.relax_default if relax is None else relax
        r, W, a = self.r, self.W, self.a
        psp, pspp = self.psip, self.psipp
        b = np.einsum("jq,q->j", self.psi, np.full_like(r, q) * r * W)   # ∫ q ψ r dr
        c = np.linalg.solve(self.Kbend, b)             # линейный старт (N = 0)
        w0_prev, converged, used = 1e30, False, max_iter
        for it in range(max_iter):
            Nr, Nth, Nrb, Nthb = self._membrane(c @ psp, float(c @ self.psib_p))
            Kg = np.einsum("jq,kq,q->jk", psp, psp, Nr * r * W)          # 2π∫N_r ψ'ψ' r dr
            A = self.Kbend + Kg
            if self.include_ktn:
                Lpsi = Nr * pspp + Nth * psp / r                        # L(ψ_k)=N_r ψ''+N_θ ψ'/r
                M2 = np.einsum("jq,kq,q->jk", self.lap_psi, Lpsi, r * W)  # ∫Δψ_j L(ψ_k) r dr
                A = A + self.h_psi_sq * M2
                if self.m == 1:                          # граничный член §3.5 (мягкий шарнир)
                    Lpsi_b = Nrb * self.psib_pp + Nthb * self.psib_p / a  # L(ψ_k)(a)
                    A = A - self.h_psi_sq * (a * np.outer(self.psib_p, Lpsi_b))
            c_new = np.linalg.solve(A, b)
            c = (1.0 - relax) * c + relax * c_new
            w0 = float(self._w_center(c))
            if abs(w0 - w0_prev) <= tol * abs(w0):
                converged, used = True, it + 1
                break
            w0_prev = w0
        Nr, Nth, _, _ = self._membrane(c @ psp, float(c @ self.psib_p))
        return RadialKTNResult(cw=c, w_max=abs(float(self._w_center(c))),
                               r_nodes=r, w_nodes=c @ self.psi, Nr=Nr, Nth=Nth,
                               converged=converged, n_iter=used)

    def _w_center(self, c: np.ndarray) -> float:
        """w(0) = a^(2m)·Σ c_k T_k(−1) (m — степень структуры)."""
        T = np.moveaxis(_cheb.chebvander(np.array([-1.0]), self.nb - 1), -1, 0)[:, 0]
        return self.a ** (2 * self.m) * float(c @ T)

    def deflection(self, c: np.ndarray, r) -> np.ndarray:
        """Прогиб ``w(r)`` (структура ``(a²−r²)^m·Φ``)."""
        r = np.atleast_1d(np.asarray(r, float))
        u = 2.0 * (r / self.a) ** 2 - 1.0
        T = np.moveaxis(_cheb.chebvander(u, self.nb - 1), -1, 0)
        return (self.a**2 - r**2) ** self.m * np.tensordot(np.asarray(c, float), T, axes=(0, 0))


__all__ = ["RadialKTN", "RadialKTNResult"]
