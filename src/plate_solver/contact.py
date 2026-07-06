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
        # значениями в узлах квадратуры (неплоский штамп,
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
        state_unit = plate.solve(np.ones(q.x.size))
        w_unit = plate.w_at_quad(state_unit)
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
            state = self.plate.solve(f0 - r)                  # f = q̃ − r → состояние
            w = self.plate.w_at_quad(state)                   # прогиб в узлах (кэш, GEMV)
            disp = self._contact_disp(state, w, r)            # классика: disp = w
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

        state = self.plate.solve(f0 - r)                      # финальный прогиб
        w = self.plate.w_at_quad(state)
        w_ktn = None
        if self.ktn is not None:
            lap_w = self.plate.lap_w_at_quad(state)
            w_ktn = self.ktn.corrected_deflection(w, lap_w, cfg.q0, r)
        # Диагностика комплементарности по финальному состоянию (алгоритм не меняется):
        # то же смещение u, что входит в условие контакта (классика: u = w).
        disp = self._contact_disp(state, w, r)
        comp_residual, gap_overshoot = self._complementarity(disp, r)
        peak = int(np.argmax(r))
        return self._package(r, w, self.plate.coeffs_w(state), iters, converged,
                             np.array(hist),
                             (q.x[peak], q.y[peak]), w_ktn, comp_residual, gap_overshoot)

    def _contact_disp(self, state, w, r) -> np.ndarray:
        """Смещение контактной поверхности: классика (w) или КТН (с кривизной Δw).

        Кривизна берётся у решателя (протокол A3.3): расщепление — Δw = −M/D
        из (P1); защемление — Δ(ω²Φ) из кэша вторых производных структуры.
        """
        if self.ktn is None:
            return w
        lap_w = self.plate.lap_w_at_quad(state)
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




# --------------------------------------------------------------------------- #
#  Контакт двух пластин
# --------------------------------------------------------------------------- #
@dataclass
class TwoPlateResult:
    r"""Результат контакта двух пластин (узлы контакта — квадратура ПЕРВОЙ).

    Поля повторяют :class:`ContactResult`, где осмыслено; дополнительно —
    прогиб второй пластины. ``r_nodes`` — реакция взаимодействия (пара ±r):
    первая пластина получает −r, вторая +r.
    """

    Xg: np.ndarray
    Yg: np.ndarray
    w_grid: np.ndarray                  # прогиб первой на сетке (NaN вне Ω₁)
    w2_grid: np.ndarray                 # прогиб второй на сетке (NaN вне Ω₂)
    r_grid: np.ndarray
    contact_zone: np.ndarray
    r_nodes: np.ndarray                 # узлы квадратуры первой пластины
    w_nodes: np.ndarray                 # w₁ в узлах первой
    w2_nodes: np.ndarray                # w₂ в узлах первой (NaN вне ω₂ > 0)
    iters: int
    converged: bool
    residual_history: np.ndarray
    peak_xy: tuple
    plate: object                       # первая пластина (для viz-совместимости)
    plate2: object
    cw: np.ndarray                      # коэффициенты прогиба первой
    cw2: np.ndarray                     # коэффициенты прогиба второй
    comp_residual: float = float("nan")
    gap_overshoot: float = float("nan")
    w_ktn_nodes: np.ndarray | None = None   # КТН для пары — направление развития


class TwoPlateMOR:
    r"""МОР для одностороннего контакта ДВУХ пластин (A4).

    Итерация на разности прогибов (реакция — пара ±r):

    .. math:: r \leftarrow \big[r + \beta_{eff}\,((w_1 - w_2) - \Delta)\big]_+ ,
              \qquad q_1 - r, \quad q_2 + r;

    сходимость — теорема 4 с суммарным оператором G = G₁ + G₂
    (β_eff = β / (gain₁ + gain₂)).

    Узлы контакта — квадратура ПЕРВОЙ пластины, маскированная ω₂ > 0
    (пересечение планформ) и зоной. Межсеточного переноса НЕТ — это
    преимущество метода: (i) прогиб второй пластины вычисляется в узлах
    первой ПРЯМО через её структуру ψ₂(x₁) (глобальное разложение, не
    сетка); (ii) вклад реакции в нагрузку второй интегрируется по РОДНОЙ
    квадратуре реакции: b₂ += ψ₂(x₁)·(W₁·r) — реакция нигде не
    интерполируется.

    Метрики Синьорини нормируются ФИЗИЧЕСКИМ масштабом щели
    ``w_scale = max|w₁_free| + max|w₂_free|`` (свободные прогибы от
    фактических нагрузок — максимальный размах u = w₁ − w₂ без
    взаимодействия; Δ может быть нулевым — касание):
    comp = max|r·(u−Δ)| / (q₀·w_scale), overshoot = max(u−Δ)|контакт / w_scale.
    Нормировка шага β_eff = β/(gain₁+gain₂) отдельна — по теореме 4
    gain берётся от ЕДИНИЧНОЙ нагрузки (оценка ‖G₁+G₂‖).
    """

    def __init__(self, plate1, plate2, cfg: Config,
                 f1_values: np.ndarray | None = None,
                 f2_values: np.ndarray | None = None,
                 q2: float | None = None,
                 gap=0.0,
                 zone_mask: np.ndarray | None = None):
        self.plate1, self.plate2, self.cfg = plate1, plate2, cfg
        if cfg.stop not in ("dr", "comp"):
            raise ValueError(f"Неизвестный критерий останова stop={cfg.stop!r}.")
        self.stop = cfg.stop
        q1 = plate1.quad
        om2 = plate2.domain.omega(q1.x, q1.y)
        mask = om2 > 0.0                                # пересечение планформ
        if zone_mask is not None:
            mask &= np.asarray(zone_mask, dtype=bool)
        if int(mask.sum()) == 0:
            raise ValueError("Контакт двух пластин: пересечение планформ пусто.")
        self.mask = mask
        # кэш структуры второй пластины в контактных узлах первой (N₂ × m)
        self.psi2_c = plate2.structure_at(q1.x[mask], q1.y[mask])
        self.W1m = q1.w[mask]
        # нагрузки
        self.f1 = cfg.q0 if f1_values is None else np.asarray(f1_values, float)
        if f2_values is None:
            q2v = cfg.q0 if q2 is None else float(q2)
            f2 = np.full(plate2.quad.x.size, q2v)
        else:
            f2 = np.asarray(f2_values, float)
        self.b2_base = plate2.load_vector(f2)
        # зазор: скаляр ≥ 0 или поле на узлах первой (берём контактный срез)
        if np.ndim(gap) == 0:
            self.gap_c = float(gap)
            self._gap_full = float(gap)
        else:
            g = np.asarray(gap, dtype=float)
            if g.shape != (q1.x.size,):
                raise ValueError("Поле зазора: ожидается массив узлов первой пластины.")
            self.gap_c = g[mask]
            self._gap_full = g
        # усиление суммарного оператора G = G₁ + G₂ (теорема 4; единичная нагрузка)
        s1 = plate1.solve(np.ones(q1.x.size))
        gain1 = float(np.max(np.abs(plate1.w_at_quad(s1))))
        s2 = plate2.solve(np.ones(plate2.quad.x.size))
        gain2 = float(np.max(np.abs(plate2.w_at_quad(s2))))
        self.gain = gain1 + gain2
        self.beta_eff = cfg.beta / self.gain
        # физический масштаб щели для метрик: свободные прогибы от НАГРУЗОК
        sf1 = plate1.solve(self.f1 if np.ndim(self.f1) else
                           np.full(q1.x.size, self.f1))
        sf2 = plate2.solve_from_b(self.b2_base)
        self.w_scale = (float(np.max(np.abs(plate1.w_at_quad(sf1))))
                        + float(np.max(np.abs(plate2.w_at_quad(sf2)))))

    def _u_contact(self, state1, cw2) -> tuple[np.ndarray, np.ndarray]:
        """(w₁ в узлах первой, u = w₁ − w₂ в контактных узлах)."""
        w1 = self.plate1.w_at_quad(state1)
        w2_c = np.tensordot(np.asarray(cw2, float), self.psi2_c, axes=(0, 0))
        return w1, w1[self.mask] - w2_c

    def _eta(self, u, r) -> float:
        """Безразмерная KKT-невязка пары (нормировка w_scale, Δ может быть 0)."""
        rm = r[self.mask]
        comp = float(np.max(np.abs(rm * (u - self.gap_c))) /
                     (self.cfg.q0 * self.w_scale))
        pen = float(np.max(np.maximum(u - self.gap_c, 0.0), initial=0.0)
                    / self.w_scale)
        return max(comp, pen)

    def solve(self, r0: np.ndarray | None = None) -> TwoPlateResult:
        """Внешний цикл МОР пары пластин; критерии останова — как у ContactMOR."""
        cfg = self.cfg
        q1 = self.plate1.quad
        if r0 is None:
            r = np.zeros(q1.x.size)
        else:
            r = np.maximum(np.asarray(r0, float).copy(), 0.0)
            r[~self.mask] = 0.0
        hist: list[float] = []
        converged = False
        iters = 0
        for iters in range(1, cfg.max_iter + 1):  # noqa: B007 — нужен после цикла
            state1 = self.plate1.solve(self.f1 - r)
            state2 = self.plate2.solve_from_b(
                self.b2_base + self.psi2_c @ (self.W1m * r[self.mask]))
            cw2 = self.plate2.coeffs_w(state2)
            w1, u = self._u_contact(state1, cw2)
            if self.stop == "comp":
                comp = float(np.max(np.abs(r[self.mask] * (u - self.gap_c))) /
                             (cfg.q0 * self.w_scale))
                pen = float(np.max(np.maximum(u - self.gap_c, 0.0), initial=0.0)
                            / self.w_scale)
                if max(comp, pen) < cfg.tol:
                    converged = True
                    break
            r_new = r.copy()
            r_new[self.mask] = r[self.mask] + self.beta_eff * (u - self.gap_c)
            np.maximum(r_new, 0.0, out=r_new)
            r_new[~self.mask] = 0.0
            res = float(np.sqrt(np.sum(q1.w * (r_new - r) ** 2)))
            hist.append(res)
            r = r_new
            if self.stop == "dr" and res < cfg.tol:
                converged = True
                break
        # финальное состояние
        state1 = self.plate1.solve(self.f1 - r)
        state2 = self.plate2.solve_from_b(
            self.b2_base + self.psi2_c @ (self.W1m * r[self.mask]))
        cw2 = self.plate2.coeffs_w(state2)
        w1, u = self._u_contact(state1, cw2)
        rm = r[self.mask]
        comp = float(np.max(np.abs(rm * (u - self.gap_c))) /
                     (cfg.q0 * self.w_scale))
        contact = rm > 0.0
        over = (float(np.max((u - self.gap_c)[contact])) / self.w_scale
                if contact.any() else float("nan"))
        peak = int(np.argmax(r))
        return self._package(r, w1, cw2, self.plate1.coeffs_w(state1),
                             iters, converged, np.array(hist),
                             (q1.x[peak], q1.y[peak]), comp, over)

    def _package(self, r, w1, cw2, cw1, iters, converged, hist, peak_xy,
                 comp, over) -> TwoPlateResult:
        from scipy.interpolate import griddata

        cfg = self.cfg
        q1 = self.plate1.quad
        dom1, dom2 = self.plate1.domain, self.plate2.domain
        x0, x1, y0, y1 = dom1.bbox
        gx = np.linspace(x0, x1, cfg.grid_n)
        gy = np.linspace(y0, y1, cfg.grid_n)
        Xg, Yg = np.meshgrid(gx, gy)
        in1 = dom1.omega(Xg, Yg) > 0.0
        in2 = dom2.omega(Xg, Yg) > 0.0
        w_grid = np.full(Xg.shape, np.nan)
        w_grid[in1] = self.plate1.deflection(cw1, Xg[in1], Yg[in1])
        w2_grid = np.full(Xg.shape, np.nan)
        w2_grid[in2] = self.plate2.deflection(cw2, Xg[in2], Yg[in2])
        r_grid = griddata((q1.x, q1.y), r, (Xg, Yg), method="linear", fill_value=0.0)
        r_grid[~(in1 & in2)] = np.nan
        zone = in1 & in2 & (np.nan_to_num(r_grid) > 0.0)
        w2_nodes = np.full(q1.x.size, np.nan)
        w2_nodes[self.mask] = np.tensordot(np.asarray(cw2, float),
                                           self.psi2_c, axes=(0, 0))
        return TwoPlateResult(
            Xg=Xg, Yg=Yg, w_grid=w_grid, w2_grid=w2_grid, r_grid=r_grid,
            contact_zone=zone, r_nodes=r, w_nodes=w1, w2_nodes=w2_nodes,
            iters=iters, converged=converged, residual_history=hist,
            peak_xy=peak_xy, plate=self.plate1, plate2=self.plate2,
            cw=cw1, cw2=cw2, comp_residual=comp, gap_overshoot=over,
        )


__all__ = ["ContactResult", "ContactMOR", "TwoPlateResult", "TwoPlateMOR",
           "solve_contact"]
