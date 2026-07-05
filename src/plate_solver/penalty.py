r"""penalty.py — Блок Р: вклад структуры Рвачёва (``ω^p·Φ``) против ШТРАФНОГО учёта ГУ.

Изолируем вклад R-функций: ОДИН и тот же метод Ритца с ОДНИМ базисом Чебышёва, но
краевые условия учтены двумя способами на одном полигоне (круг, шарнир):

  • (структура) ``w = ω·Φ`` — ГУ ``w=0`` выполнено ТОЖДЕСТВЕННО (готовый
    ``plate_solver.plate.PlateBending``, расщепление на две Пуассона, «мягкий шарнир»);
  • (штраф) ``w = Σ c_i φ_i`` БЕЗ множителя ``ω`` — ГУ НЕ выполнено автоматически,
    к функционалу каждой Пуассона добавляется ``(γ/2)∮_∂Ω u² ds`` с большим ``γ``.

Сравнение КОРРЕКТНО как «Ритц со структурой Рвачёва против Ритца со штрафом» —
оба используют один решатель (Ритц), различается лишь способ учёта ГУ. Ожидание:
структура сходится быстро и устойчиво; штраф — медленнее, с худшей
обусловленностью и «полом» точности, зависящим от ``γ`` (мал ``γ`` — ГУ не
выполнены; велик — плохая обусловленность).

Реализация штрафа — ОТДЕЛЬНЫЕ функции, основной решатель не трогается.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .basis import ChebyshevBasis
from .quadrature import interior_nodes


# --------------------------------------------------------------------------- #
#  Граничная квадратура на окружности (для ∮ u² ds)
# --------------------------------------------------------------------------- #
def circle_boundary_nodes(R: float, n: int):
    """Узлы и веса (ds) на окружности радиуса ``R``: ``x=R cosθ, y=R sinθ, ds=R dθ``."""
    th = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    return R * np.cos(th), R * np.sin(th), np.full(n, R * 2.0 * np.pi / n)


# --------------------------------------------------------------------------- #
#  Штрафной решатель Пуассона: −Δu=f, u≈0 на ∂Ω (через γ∮u² ds), БЕЗ структуры ω
# --------------------------------------------------------------------------- #
class PenaltyPoisson:
    r"""Ритц для ``−Δu=f`` на сыром базисе ``u=Σ c_k φ_k`` со штрафом ГУ ``u=0``.

    ``A = ∫∇φ_k·∇φ_l dΩ + γ ∮_∂Ω φ_k φ_l ds``,  ``b = ∫ f φ_k dΩ``.
    Матрица собирается и факторизуется один раз (общая для обеих Пуассона).
    """

    def __init__(self, domain, basis, quad, boundary, gamma: float):
        self.domain = domain
        self.basis = basis
        self.quad = quad
        self.gamma = float(gamma)
        X, Y, W = quad.x, quad.y, quad.w
        self._phi = basis.values(X, Y)                       # (N, M) сырой базис (без ω)
        self._W = W
        phix, phiy = basis.grads(X, Y)
        A = (phix * W) @ phix.T + (phiy * W) @ phiy.T        # жёсткость
        bx, by, bw = boundary
        pb = basis.values(bx, by)                            # (N, Mb) на границе
        A = A + self.gamma * ((pb * bw) @ pb.T)              # штраф ГУ
        self.A = 0.5 * (A + A.T)
        self._lu = sla.lu_factor(self.A)

    @property
    def cond(self) -> float:
        return float(np.linalg.cond(self.A))

    def solve(self, f_values) -> np.ndarray:
        b = (self._phi * self._W) @ np.asarray(f_values, float)
        return sla.lu_solve(self._lu, b)

    def evaluate(self, c, X, Y) -> np.ndarray:
        """``u = Σ c_k φ_k`` (БЕЗ множителя ω) в точках (X, Y)."""
        Phi = self.basis.values(X, Y)
        return np.tensordot(np.asarray(c, float), Phi, axes=(0, 0))

    def nodal(self, c) -> np.ndarray:
        """Значения ``u`` в узлах квадратуры (для правой части (P2))."""
        return np.tensordot(np.asarray(c, float), self._phi, axes=(0, 0))


def softhinge_penalty_wmax(domain, basis, quad, boundary, gamma: float, q: float,
                           D: float) -> float:
    r"""«Мягкий шарнир» ШТРАФОМ: две Пуассона (−ΔM=q, M≈0; −Δw=M/D, w≈0) → ``w(0,0)``."""
    pp = PenaltyPoisson(domain, basis, quad, boundary, gamma)
    cM = pp.solve(np.full(quad.x.size, float(q)))            # (P1) −ΔM=q, M≈0
    M_nodes = pp.nodal(cM)
    cw = pp.solve(M_nodes / D)                               # (P2) −Δw=M/D, w≈0
    w0 = float(pp.evaluate(cw, np.array([0.0]), np.array([0.0]))[0])
    return w0, pp.cond


# --------------------------------------------------------------------------- #
#  Сводная сверка: структура (ω·Φ) против штрафа на круге (шарнир)
# --------------------------------------------------------------------------- #
def penalty_vs_structure_circle(a: float, q: float, D: float, nu: float, E: float, h: float,
                                p_list, Q: int, gamma: float, n_bnd: int, target_pct: float = 1.0):
    r"""Сверка err(N) структуры и штрафа на круге (шарнир) + N для целевой точности.

    Структура — ``PlateBending`` (ω·Φ); штраф — ``softhinge_penalty_wmax``. Эталон —
    «мягкий» шарнир ``w_soft = 3 q a⁴/(64 D)`` (та же реализованная модель).
    """
    from . import analytic, geometry
    from .config import Config
    from .plate import PlateBending

    dom = geometry.make_circle(a)
    w_soft = float(analytic.circular_plate_soft_hinge_wmax(q, a, D))
    boundary = circle_boundary_nodes(a, n_bnd)

    rows = []
    for p in p_list:
        N = (p + 1) ** 2
        # структура (ω·Φ)
        cfg = Config(a=a, q0=q, nu=nu, h=h, E=E, p=p, Q=Q)
        pb = PlateBending.from_config(dom, cfg)
        _, cw = pb.solve_uniform(q)
        w_struct = float(pb.deflection(cw, 0.0, 0.0))
        err_struct = abs(w_struct - w_soft) / w_soft
        cond_struct = pb.poisson.cond
        # штраф (сырой базис + γ∮u²ds)
        basis = ChebyshevBasis(p, dom.bbox)
        quad = interior_nodes(dom, Q)
        w_pen, cond_pen = softhinge_penalty_wmax(dom, basis, quad, boundary, gamma, q, D)
        err_pen = abs(w_pen - w_soft) / w_soft
        rows.append({
            "p": p, "N": N,
            "err_struct": err_struct, "cond_struct": cond_struct,
            "err_pen": err_pen, "cond_pen": cond_pen,
        })

    def _N_for_target(key_err, key_cond):
        for r in rows:
            if 100.0 * r[key_err] < target_pct:
                return r["N"], r[key_cond]
        return None, None

    N_struct, cond_struct_t = _N_for_target("err_struct", "cond_struct")
    N_pen, cond_pen_t = _N_for_target("err_pen", "cond_pen")
    return {
        "w_soft": w_soft, "rows": rows, "gamma": gamma,
        "N_struct": N_struct, "cond_struct_target": cond_struct_t,
        "N_pen": N_pen, "cond_pen_target": cond_pen_t,
        "target_pct": target_pct,
    }


__all__ = [
    "circle_boundary_nodes",
    "PenaltyPoisson",
    "softhinge_penalty_wmax",
    "penalty_vs_structure_circle",
]
