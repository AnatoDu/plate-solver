r"""fd_contact.py — конечно-разностный МОР на прямоугольнике (верификационный кирпич).

НЕЗАВИСИМАЯ перепроверка контактного тракта ЛОКАЛЬНЫМ базисом (v0.7.0).
Мотивация: у ГЛОБАЛЬНЫХ
гладких базисов (ряд Навье) контактная реакция «звенит» и не сходится —
реакция негладка на свободной границе зоны (излом), и её глобальное
приближение осциллирует (эффект Гиббса). Конечные разности — локальный
базис: реакция живёт поточечно, звона нет. Тот же эффект наблюдается и в
самом plate-solver (кольцо реакции у глобального базиса — NOTES §25).

Схема: на каждом шаге МОР решается
разреженная СЛАУ; здесь матрица факторизуется splu ОДИН раз (правая часть
``q − r`` меняется — две треугольные подстановки на шаг).

Постановка: шарнирный (SS) прямоугольник над жёстким плоским основанием
(зазор ``Δ``), равномерная нагрузка. Для ПРЯМЫХ кромок истинный шарнир
``w = 0, M_n = 0`` эквивалентен ``w = 0, Δw = 0`` (на прямой кромке
``w_tt ≡ 0`` ⇒ ``M_n = −D·w_nn = −D·Δw``), поэтому бигармоника собирается
ТОЧНО как квадрат пятиточечного Лапласиана Дирихле: ``A = D·L·L`` — тот же
приём, что у расщепления Навье/Маркуса, но в локальном базисе. Аппроксимация
``O(h²)``; сходимость к спектральному решению RFM+Ритца — ворота
``tests/test_fd_contact.py``.

Внешний цикл — тот же МОР, что в :class:`~plate_solver.contact.ContactMOR`:
``r ← max(0, r + β_eff·(w − Δ))``, ``β_eff = β/gain``, ``gain`` — податливость
от единичной равномерной нагрузки (теорема 4).

Роль модуля — верификационная (как ``radial_ktn``, ``verify_fem``): рабочий
решатель произвольных очертаний — RFM+Ритц; КР дополняет его на прямоугольнике
как метод-дублёр (сверка прогиба, суммарной реакции, зоны) и как эталон
ГЛАДКОСТИ профиля реакции. Обобщение на овал в лоб КР не берёт — это ниша
R-функций; путь «доопределения до прямоугольника» (фиктивные области) теряет
порядок точности у криволинейной границы.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import splu


@dataclass(frozen=True)
class FDContactResult:
    """Результат КР+МОР: поля на внутренней сетке (ny, nx) + диагностика."""

    X: np.ndarray                 # сетка внутренних узлов (ny, nx)
    Y: np.ndarray
    w: np.ndarray                 # прогиб (ny, nx)
    r: np.ndarray                 # реакция ≥ 0 (ny, nx)
    w_max: float
    r_total: float                # ∫r dΩ (сумма·h²)
    n_contact: int
    iters: int
    converged: bool


class FDPlateSS:
    r"""Шарнирный (SS) прямоугольник конечными разностями: ``D·Δ²w = f``.

    Пятиточечный Лапласиан Дирихле ``L`` на внутренних узлах; бигармоника
    ``A = D·L·L`` (точно для SS на прямых кромках — см. модуль). Факторизация
    splu ОДИН раз; ``solve(f)`` — две подстановки.
    """

    def __init__(self, x1: float, x2: float, y1: float, y2: float,
                 nx: int, ny: int, D: float,
                 springs: list[tuple[float, float, float]] | None = None):
        if nx < 4 or ny < 4:
            raise ValueError("КР-сетка: ожидалось nx, ny ≥ 4")
        self.D = float(D)
        self.hx = (x2 - x1) / (nx - 1)
        self.hy = (y2 - y1) / (ny - 1)
        xs = np.linspace(x1, x2, nx)[1:-1]              # внутренние узлы
        ys = np.linspace(y1, y2, ny)[1:-1]
        self.X, self.Y = np.meshgrid(xs, ys)
        mx, my = nx - 2, ny - 2
        self.shape = (my, mx)
        # 1D вторые разности с Дирихле (трёхдиагональные)
        def d2(m, h):
            return sps.diags([1.0, -2.0, 1.0], [-1, 0, 1],
                             shape=(m, m), format="csr") / h**2
        Ix = sps.identity(mx, format="csr")
        Iy = sps.identity(my, format="csr")
        L = sps.kron(Iy, d2(mx, self.hx)) + sps.kron(d2(my, self.hy), Ix)
        A = (self.D * (L @ L)).tolil()                  # Δ² = L² (SS, прямые кромки)
        self.area_w = self.hx * self.hy                 # вес узла (интегралы)
        # точечные пружины: сила −k·w(P)·δ_P как ПЛОТНОСТЬ = −k·w_ij/(hx·hy)
        # в ячейке ближайшего узла ⇒ диагональная добавка k/(hx·hy); погрешность
        # привязки к узлу O(h) по положению — для кросс-чека брать P в узле сетки
        self.spring_idx: list[int] = []
        for (sx, sy, k) in (springs or []):
            j = int(np.argmin((self.X.ravel() - sx) ** 2 + (self.Y.ravel() - sy) ** 2))
            A[j, j] += float(k) / self.area_w
            self.spring_idx.append(j)
        self._lu = splu(A.tocsc())

    def solve(self, f) -> np.ndarray:
        """Прогиб (ny-2, nx-2) от правой части ``f`` (той же формы или скаляр)."""
        f_arr = np.broadcast_to(np.asarray(f, float), self.shape)
        return self._lu.solve(f_arr.ravel()).reshape(self.shape)


def fd_contact_foundation(x1: float, x2: float, y1: float, y2: float, *,
                          D: float, q0: float, gap,
                          nx: int = 81, ny: int = 81, beta: float = 1.2,
                          max_iter: int = 5000, tol: float = 1e-8) -> FDContactResult:
    r"""КР+МОР: SS-прямоугольник над жёстким основанием (зазор ``Δ``).

    ``gap`` — скаляр (плоское основание) ЛИБО массив формы внутренней сетки
    ``(ny−2, nx−2)`` (профиль штампа/основания ``Δ(x, y)``): МОР-шаг
    ``r ← max(0, r + β_eff·(w − Δ))`` поточечный, обе формы равноправны.
    ``q0`` — скаляр (равномерная нагрузка) ЛИБО массив той же формы
    (произвольная нагрузка ``q(x, y)`` — гауссова, expr; v0.7.0).

    Внешний цикл МОР по поточечной реакции (нормировка усиления — как в
    :class:`~plate_solver.contact.ContactMOR`: ``β_eff = β/gain``,
    ``gain = max w`` от единичной нагрузки). Останов — относительное
    изменение реакции ``‖r_new − r‖/‖r‖ < tol``.
    """
    plate = FDPlateSS(x1, x2, y1, y2, nx, ny, D)
    w_unit = plate.solve(1.0)
    gain = float(np.max(np.abs(w_unit)))
    beta_eff = beta / gain
    q0 = np.broadcast_to(np.asarray(q0, float), plate.shape)
    r = np.zeros(plate.shape)
    converged = False
    it = 0
    for it in range(1, max_iter + 1):  # noqa: B007 — it нужен после цикла
        w = plate.solve(q0 - r)
        r_new = np.maximum(0.0, r + beta_eff * (w - gap))
        dr = float(np.sqrt(np.sum((r_new - r) ** 2)))
        scale = float(np.sqrt(np.sum(r_new**2)))
        r = r_new
        if dr <= tol * max(scale, 1e-30):
            converged = True
            break
    w = plate.solve(q0 - r)
    return FDContactResult(
        X=plate.X, Y=plate.Y, w=w, r=r,
        w_max=float(np.max(np.abs(w))),
        r_total=float(np.sum(r) * plate.area_w),
        n_contact=int(np.count_nonzero(r)),
        iters=it, converged=converged)


__all__ = ["FDPlateSS", "FDContactResult", "fd_contact_foundation"]
