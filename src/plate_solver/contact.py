r"""contact.py — метод обобщённой реакции (МОР) для одностороннего контакта (2D).

Внешний цикл, повторяющий логику ``fix_base2.py`` (1D, классика), но в 2D и с
ПРЕДВЫЧИСЛЕННЫМ оператором: матрица ``A`` в :class:`~plate_solver.plate.PlateBending`
факторизована один раз, поэтому каждая итерация дёшева (тысячи итераций допустимы).

Реакция ``r`` хранится В УЗЛАХ КВАДРАТУРЫ (источник истины, NOTES.md §3); на
фоновую сетку ``grid_n × grid_n`` сэмплируется только для вывода/графиков.

Схема (прямой аналог 1D, условия Синьорини с зазором Δ до жёсткого основания):

    r⁰ = 0
    повторять:
        f = q0 − r                         # нагрузка в узлах квадратуры (q̃)
        _, cw = plate.solve(f)             # расщепление (P1)+(P2), A факторизована
        w = plate.deflection(cw, узлы)     # прогиб в тех же узлах
        r ← r + β_eff·(w − Δ)   в зоне основания;   r ← max(r, 0)
        невязка = ‖r − r_prev‖;  стоп при < tol
    зона контакта = { r > 0 }

Масштаб шага β. В 1D (``fix_base2``) задача безразмерна (множитель ``l⁴/D``
приводит прогиб к O(1)), и ``β`` мало (0.01). В 2D при физических ``D`` прогиб
мал (``w ~ q a⁴/D``), поэтому ``β`` НОРМИРУЕТСЯ на «усиление» оператора
``gain`` = макс. прогиб от единичной равномерной нагрузки (оценка ‖G‖):
``β_eff = cfg.beta / gain``. Тогда ``cfg.beta`` — БЕЗРАЗМЕРНЫЙ коэффициент
релаксации, условие сходимости (теорема 4) принимает вид ``0 < cfg.beta < 2``.
Это сохраняет логику МОР и делает её независимой от единиц.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .config import Config
from .ktn import KTNParams
from .plate import PlateBending


@dataclass
class ContactResult:
    r"""Результат контактной задачи (поля на сетке + источники истины в узлах).

    Attributes
    ----------
    Xg, Yg : координаты фоновой сетки grid_n × grid_n (meshgrid).
    w_grid, r_grid : прогиб и реакция на сетке (NaN вне Ω) — для вывода.
    contact_zone : булева маска сетки, где реакция активна (r > 0 внутри Ω).
    r_nodes, w_nodes : реакция и прогиб в узлах квадратуры (источник истины).
    iters : число выполненных итераций МОР.
    converged : достигнут ли критерий ‖Δr‖ < tol до max_iter.
    residual_history : ‖r_k − r_{k-1}‖ по итерациям.
    peak_xy : координаты узла с максимальной реакцией.
    plate, cw : решатель и коэффициенты прогиба при сошедшейся реакции.
    w_ktn_nodes : КТН-поправленный прогиб в узлах (None для классики).
    comp_residual : безразмерная невязка комплементарности условий Синьорини

        .. math:: \max_i |r_i\,(u_i - \Delta)| \,/\, (q_0\,\Delta),

        где ``u`` — смещение контактной поверхности (классика: ``u = w``;
        КТН: с поправками). В точном решении ``r·(u−Δ) ≡ 0`` (либо нет
        контакта и r=0, либо контакт и u=Δ), поэтому величина — прямая
        мера недосхождения, не зависящая от нормировки нагрузки и зазора.
    gap_overshoot : относительный «перелёт» зазора в зоне контакта

        .. math:: (\max_{i:\,r_i>0} u_i - \Delta) \,/\, \Delta

        (насколько прогиб в контакте превышает зазор; NaN, если контакта нет).
    """

    Xg: np.ndarray
    Yg: np.ndarray
    w_grid: np.ndarray
    r_grid: np.ndarray
    contact_zone: np.ndarray
    r_nodes: np.ndarray
    w_nodes: np.ndarray
    iters: int
    converged: bool
    residual_history: np.ndarray
    peak_xy: tuple
    plate: PlateBending
    cw: np.ndarray
    w_ktn_nodes: np.ndarray | None = None
    comp_residual: float = float("nan")
    gap_overshoot: float = float("nan")


class ContactMOR:
    """Метод обобщённой реакции для контакта пластины с жёстким основанием.

    Parameters
    ----------
    plate : решатель изгиба (с факторизованной A).
    cfg : параметры (q0, beta, Delta, max_iter, tol, grid_n).
    foundation_mask : предикат ``(X, Y) -> bool`` зоны возможного контакта в узлах
        квадратуры; ``None`` ⇒ всё Ω (плоское основание под всей пластиной).
    gap : зазор Δ до основания; ``None`` ⇒ ``cfg.Delta``.
    ktn : параметры поправок КТН (:class:`~plate_solver.ktn.KTNParams`); ``None`` ⇒
        классика (Кирхгоф). При заданных ``ktn`` условие контакта использует
        КТН-смещение контактной поверхности (с кривизной Δw = −M/D).
    """

    def __init__(
        self,
        plate: PlateBending,
        cfg: Config,
        foundation_mask: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
        gap: float | None = None,
        ktn: KTNParams | None = None,
        load_values: np.ndarray | None = None,
    ):
        self.plate = plate
        self.cfg = cfg
        self.ktn = ktn
        # Неравномерная нагрузка (patch/point из диспетчера): значения в узлах
        # квадратуры; None ⇒ равномерная cfg.q0 (путь и арифметика прежние).
        self.load = None if load_values is None else np.asarray(load_values, dtype=float)
        if cfg.stop not in ("dr", "comp"):
            raise ValueError(
                f"Неизвестный критерий останова stop={cfg.stop!r} (ожидается 'dr' или 'comp')."
            )
        self.stop = cfg.stop
        # Зазор: скаляр (путь v0.2, арифметика прежняя) ИЛИ поле Δ(x, y)
        # значениями в узлах квадратуры (фаза 3, A1: неплоский штамп,
        # криволинейное основание, ступени). Нормировка gain не меняется.
        gap_raw = cfg.Delta if gap is None else gap
        if np.ndim(gap_raw) == 0:
            self.gap = float(gap_raw)
        else:
            self.gap = np.asarray(gap_raw, dtype=float)
            if self.gap.shape != (plate.quad.x.size,):
                raise ValueError(
                    f"Поле зазора: ожидается массив длины M = {plate.quad.x.size} "
                    f"(узлы квадратуры), получено {self.gap.shape}."
                )
        self._gap_scalar = np.ndim(self.gap) == 0
        # Масштаб Δ для безразмерных метрик: скаляр — сам зазор; поле —
        # минимальный зазор под основанием (контактно-релевантный масштаб).
        self._gap_ref = float(self.gap) if self._gap_scalar else None
        q = plate.quad
        if foundation_mask is None:
            self.fmask = np.ones(q.x.size, dtype=bool)
        else:
            self.fmask = np.asarray(foundation_mask(q.x, q.y), dtype=bool)
        if self._gap_scalar:
            self._gap_f = self.gap                       # тот же float — путь v0.2
        else:
            self._gap_f = self.gap[self.fmask]           # значения поля под основанием
            self._gap_ref = float(np.min(self._gap_f))
            if self._gap_ref <= 0.0:
                raise ValueError("Поле зазора: min Δ под основанием должно быть > 0.")
        # Усиление оператора: макс. прогиб от единичной равномерной нагрузки (~‖G‖).
        _, cw_unit = plate.solve(np.ones(q.x.size))
        w_unit = plate.poisson.evaluate_at_quad(cw_unit)
        self.gain = float(np.max(np.abs(w_unit)))
        self.beta_eff = cfg.beta / self.gain

    def solve(self, r0: np.ndarray | None = None) -> ContactResult:
        r"""Запустить внешний цикл МОР и вернуть :class:`ContactResult`.

        ``r0`` — тёплый старт реакции (массив в узлах квадратуры; проекция
        на допустимое множество выполняется автоматически: r ≥ 0, нуль вне
        основания). ``None`` — прежний холодный старт r ≡ 0. Тёплый старт
        используется силовым штампом (A2): соседние уровни дают близкие
        реакции, экономя итерации внешнего скалярного уравнения.

        Критерий останова выбирается полем ``cfg.stop`` (порог — ``cfg.tol``):

        * ``"dr"`` (по умолчанию; поведение прежнее) — малость шага реакции
          в квадратурной норме :math:`L_2(\Omega)`:

          .. math:: \|r_k - r_{k-1}\|_{L_2} =
                    \Big(\textstyle\sum_m w_m\,(r_k - r_{k-1})_m^2\Big)^{1/2}
                    < \mathrm{tol},

          где :math:`w_m` — веса квадратуры. Критерий абсолютный (наследует
          размерность реакции), поэтому ``tol`` согласуется с масштабом
          нагрузки ``q0`` конкретной задачи.

        * ``"comp"`` — безразмерная KKT-невязка условий Синьорини для
          состояния :math:`(r_k,\,u(r_k))` (``u`` — смещение контактной
          поверхности; классика: :math:`u = w`):

          .. math:: \eta_k = \max\!\Big(
                    \frac{\max_i |r_i\,(u_i - \Delta)|}{q_0\,\Delta},\;
                    \frac{\max_i (u_i - \Delta)_+}{\Delta}\Big) < \mathrm{tol}.

          Первый член — нарушение комплементарности
          :math:`r\,(u-\Delta) = 0`, второй — проникание (:math:`u \le \Delta`
          на основании). Одной комплементарности недостаточно: она тривиально
          равна нулю при :math:`r \equiv 0`. Критерий не зависит от нормировки
          нагрузки и зазора; остановка сертифицирует приближённое выполнение
          всех условий Синьорини с точностью ``tol``.

        В обоих режимах ``residual_history`` хранит :math:`\|r_k - r_{k-1}\|`
        (диагностика сходимости не меняется).
        """
        cfg, q = self.cfg, self.plate.quad
        f0 = cfg.q0 if self.load is None else self.load       # равномерная или поле
        if r0 is None:
            r = np.zeros(q.x.size)
        else:
            r = np.maximum(np.asarray(r0, dtype=float).copy(), 0.0)
            r[~self.fmask] = 0.0                              # проекция тёплого старта
        hist: list[float] = []
        converged = False
        iters = 0

        for iters in range(1, cfg.max_iter + 1):  # noqa: B007 — iters нужен после цикла
            cM, cw = self.plate.solve(f0 - r)                 # f = q̃ − r → (M, w)
            w = self.plate.poisson.evaluate_at_quad(cw)       # прогиб в узлах (кэш, GEMV)
            disp = self._contact_disp(cM, w, r)               # классика: disp = w
            if self.stop == "comp" and self._kkt_residual(disp, r) < cfg.tol:
                converged = True                              # (r, u(r)) уже KKT-точно
                break
            r_new = r.copy()
            r_new[self.fmask] = r[self.fmask] + self.beta_eff * (disp[self.fmask] - self._gap_f)
            np.maximum(r_new, 0.0, out=r_new)                 # проекция r ≥ 0
            r_new[~self.fmask] = 0.0                          # реакция только под основанием
            res = float(np.sqrt(np.sum(q.w * (r_new - r) ** 2)))
            hist.append(res)
            r = r_new
            if self.stop == "dr" and res < cfg.tol:
                converged = True
                break

        cM, cw = self.plate.solve(f0 - r)                     # финальный прогиб
        w = self.plate.poisson.evaluate_at_quad(cw)
        w_ktn = None
        if self.ktn is not None:
            lap_w = -self.plate.poisson.evaluate_at_quad(cM) / self.plate.D
            w_ktn = self.ktn.corrected_deflection(w, lap_w, cfg.q0, r)
        # Диагностика комплементарности по финальному состоянию (алгоритм не меняется):
        # то же смещение u, что входит в условие контакта (классика: u = w).
        disp = self._contact_disp(cM, w, r)
        comp_residual, gap_overshoot = self._complementarity(disp, r)
        peak = int(np.argmax(r))
        return self._package(r, w, cw, iters, converged, np.array(hist),
                             (q.x[peak], q.y[peak]), w_ktn, comp_residual, gap_overshoot)

    def _contact_disp(self, cM, w, r) -> np.ndarray:
        """Смещение контактной поверхности: классика (w) или КТН (с Δw = −M/D)."""
        if self.ktn is None:
            return w
        lap_w = -self.plate.poisson.evaluate_at_quad(cM) / self.plate.D
        return self.ktn.contact_displacement(w, lap_w, self.cfg.q0, r)

    def _kkt_residual(self, disp, r) -> float:
        r"""Безразмерная KKT-невязка Синьорини состояния (r, u(r)); Δ > 0.

        .. math:: \eta = \max\Big(\frac{\max_i |r_i (u_i - \Delta)|}{q_0 \Delta},\;
                  \frac{\max_i (u_i - \Delta)_+}{\Delta}\Big)

        (комплементарность + проникание; см. докстринг :meth:`solve`).
        """
        comp = float(np.max(np.abs(r * (disp - self.gap))) / (self.cfg.q0 * self._gap_ref))
        pen = float(np.max(np.maximum(disp[self.fmask] - self._gap_f, 0.0), initial=0.0)
                    / self._gap_ref)
        return max(comp, pen)

    def _complementarity(self, disp, r) -> tuple[float, float]:
        r"""Безразмерные метрики Синьорини по финальному состоянию (Δ > 0).

        comp_residual = max|r·(u−Δ)| / (q0·Δ);  gap_overshoot = (max u|_{r>0} − Δ)/Δ.
        """
        comp = float(np.max(np.abs(r * (disp - self.gap))) / (self.cfg.q0 * self._gap_ref))
        contact = r > 0.0
        over = (float(np.max((disp - self.gap)[contact])) / self._gap_ref
                if contact.any() else float("nan"))
        return comp, over

    # -- вывод на сетку ------------------------------------------------- #
    def _package(self, r, w, cw, iters, converged, hist, peak_xy, w_ktn=None,
                 comp_residual=float("nan"), gap_overshoot=float("nan")) -> ContactResult:
        from scipy.interpolate import griddata

        cfg, dom, q = self.cfg, self.plate.domain, self.plate.quad
        x0, x1, y0, y1 = dom.bbox
        gx = np.linspace(x0, x1, cfg.grid_n)
        gy = np.linspace(y0, y1, cfg.grid_n)
        Xg, Yg = np.meshgrid(gx, gy)
        inside = dom.omega(Xg, Yg) > 0.0

        w_grid = np.full(Xg.shape, np.nan)
        w_grid[inside] = self.plate.deflection(cw, Xg[inside], Yg[inside])
        # Реакция известна в узлах квадратуры → интерполируем на сетку (только вывод).
        r_grid = griddata((q.x, q.y), r, (Xg, Yg), method="linear", fill_value=0.0)
        r_grid[~inside] = np.nan
        contact_zone = inside & (np.nan_to_num(r_grid) > 0.0)

        return ContactResult(
            Xg=Xg, Yg=Yg, w_grid=w_grid, r_grid=r_grid, contact_zone=contact_zone,
            r_nodes=r, w_nodes=w, iters=iters, converged=converged,
            residual_history=hist, peak_xy=peak_xy, plate=self.plate, cw=cw,
            w_ktn_nodes=w_ktn, comp_residual=comp_residual, gap_overshoot=gap_overshoot,
        )


def solve_contact(
    cfg: Config,
    domain,
    foundation_mask: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    gap: float | None = None,
    ktn: KTNParams | None = None,
) -> ContactResult:
    """Фасад: собрать PlateBending по конфигу и решить контакт методом МОР."""
    plate = PlateBending.from_config(domain, cfg)
    return ContactMOR(plate, cfg, foundation_mask=foundation_mask, gap=gap, ktn=ktn).solve()


__all__ = ["ContactResult", "ContactMOR", "solve_contact"]
