r"""contact_nl.py — НЕЛИНЕЙНЫЙ контакт: МОР поверх полной КТН (§4 ТЗ v0.6.0).

Уникальная связка (в литературе её нет): метод обобщённой реакции (МОР) вокруг
полного нелинейного решателя КТН (:class:`~plate_solver.ktn_solver.KTNSolver`).
Контакт «щупает» ЛИЦЕВУЮ поверхность (условие Синьорини на грани, §4.1):

.. math:: u_c = w + (h_c^2 - h_*^2)\,\Delta w \le z,

где ``z`` — зазор/препятствие. Коэффициент кривизны ``(h_c²−h_*²)``
масштабируется теорией (``TheoryParams.face_curv_coeff``): для ``classic``/
``karman`` он НОЛЬ ⇒ контакт по срединной поверхности (``u_c = w``), для полной
КТН — лицевая кривизна (подпись КТН в контакте: сглаживание реакции сдвигом/
обжатием). Алгебраические q,r-поправки лица (``κ_q q``, ``κ_r D r``) — порядка
``O(h²)``, в v0.6.0 опущены (§14 cut; сохранён доминирующий кривизный член).

Схемы композиции двух итераций (§4.2):

* **вложенная** (``scheme="nested"``, эталон корректности): внешний цикл МОР,
  внутри — полный сходящийся ``KTNSolver`` на каждой итерации (прил. C.1); прост
  и надёжен, дорог;
* **совмещённая** (``scheme="merged"``, рабочая, предмет T7): один шаг МОР на
  один шаг Пикара, общая недорелаксация (прил. C.2; веха N4).

Редукции (точны благодаря единой модели §3): контакт выкл (``r≡0``) → решатель
КТН v0.5; нелинейность выкл → линейный МОР; уточнение выкл → кармановский
контакт (``u_c = w``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import Config
from .diagnostics import contact_components
from .ktn_solver import KTNSolver


@dataclass
class NonlinearContactResult:
    r"""Результат нелинейной контактной задачи (поля в узлах квадратуры).

    Attributes
    ----------
    r_nodes : обобщённая реакция ``r ≥ 0`` в узлах (источник истины).
    w_nodes, u_c_nodes : срединный и лицевой прогибы в узлах.
    cw : коэффициенты прогиба (для отрисовки/полей).
    w_max : max|w| (срединный).
    contact_mask : булева маска зоны контакта (``r > 0``).
    r_max : пиковая реакция; peak_xy — её локализация.
    n_contact : число контактных узлов.
    n_components : число связных пятен контакта (топология зоны, §8).
    iters, converged, residual_history : диагностика внешнего МОР.
    n_inner : суммарное число внутренних нелинейных итераций (вложенная схема).
    scheme : "nested" | "merged".
    """

    r_nodes: np.ndarray
    w_nodes: np.ndarray
    u_c_nodes: np.ndarray
    cw: np.ndarray
    w_max: float
    contact_mask: np.ndarray
    r_max: float
    peak_xy: tuple
    n_contact: int
    n_components: int
    iters: int
    converged: bool
    residual_history: np.ndarray
    n_inner: int
    scheme: str


class NonlinearContactMOR:
    r"""Нелинейный контакт МОР+КТН (§4): реакция ``r`` вокруг решателя КТН.

    Parameters
    ----------
    solver : :class:`~plate_solver.ktn_solver.KTNSolver` (пресет теории).
    cfg : параметры (``q0``, ``beta``, ``max_iter``, ``tol``; см. Config).
    gap : зазор/препятствие ``z`` (скаляр — плоское препятствие; массив длины
        числа узлов квадратуры — профиль штампа, §9.2).
    foundation_mask : предикат ``(X, Y) → bool`` зоны возможного контакта;
        ``None`` ⇒ вся Ω.
    scheme : ``"nested"`` (эталон) | ``"merged"`` (совмещённый, N4).
    """

    def __init__(self, solver: KTNSolver, cfg: Config, *, gap,
                 foundation_mask=None, scheme: str | None = None):
        self.solver = solver
        self.cfg = cfg
        self.scheme = scheme if scheme is not None else getattr(cfg, "contact_scheme", "nested")
        if self.scheme not in ("nested", "merged"):
            raise ValueError(f"scheme: ожидалось nested | merged, получено {self.scheme!r}")
        q = solver.quad
        # зона основания
        if foundation_mask is None:
            self.fmask = np.ones(q.x.size, dtype=bool)
        else:
            self.fmask = np.asarray(foundation_mask(q.x, q.y), dtype=bool)
        # зазор: скаляр или поле в узлах
        if np.ndim(gap) == 0:
            self.gap = float(gap)
            self._gap_f = float(gap)
        else:
            self.gap = np.asarray(gap, dtype=float)
            if self.gap.shape != (q.x.size,):
                raise ValueError("Поле зазора: ожидается массив длины числа узлов квадратуры.")
            self._gap_f = self.gap[self.fmask]
        # усиление оператора — СЕКУЩАЯ податливость в рабочей точке ``w_free/q0``
        # (свободное НЕЛИНЕЙНОЕ решение при q0): для линейной задачи совпадает с
        # податливостью на единичную нагрузку, для нелинейной (жёстче) — меньше,
        # поэтому β_eff = β·q0/w_free крупнее ⇒ МОР сходится быстрее. Свободное
        # решение служит и тёплым стартом первого шага МОР.
        q0 = float(cfg.q0)
        self._free = solver.solve(np.full(q.x.size, q0))
        self.gain = self._free.w_max / q0 if q0 != 0.0 else 1.0
        self.beta_eff = cfg.beta / self.gain
        self.max_iter = int(cfg.max_iter)
        self.tol = float(cfg.tol)
        self._c_curv = solver.params.face_curv_coeff        # масштаб теории (§4.1)

    def _face_deflection(self, cw) -> np.ndarray:
        r"""Лицевой прогиб ``u_c = w + (h_c²−h_*²)Δw`` в узлах (§4.1, масштаб теории)."""
        w = cw @ self.solver._psi
        if self._c_curv == 0.0:
            return w                                        # classic/karman: u_c = w
        lap_w = cw @ self.solver._lap_psi                   # Δw в узлах (кэш структуры)
        return w + self._c_curv * lap_w

    def solve(self) -> NonlinearContactResult:
        """Решить нелинейную контактную задачу выбранной схемой (§4.2)."""
        if self.scheme == "nested":
            return self._solve_nested()
        return self._solve_merged()

    # -- вложенная схема (эталон, прил. C.1) ---------------------------- #
    def _solve_nested(self) -> NonlinearContactResult:
        solver, q = self.solver, self.solver.quad
        q0 = float(self.cfg.q0)
        r = np.zeros(q.x.size)
        hist: list[float] = []
        converged = False
        n_inner = 0
        it = 0
        cw = self._free.cw                                  # тёплый старт — свободное решение
        for it in range(1, self.max_iter + 1):  # noqa: B007 — it нужен после цикла
            # тёплый старт нелинейного решателя предыдущим прогибом: нагрузка
            # q0−r меняется слабо между шагами МОР ⇒ внутренняя итерация дёшева.
            res_k = solver.solve(q0 - r, c0=cw)             # полный нелинейный КТН
            n_inner += res_k.n_iter
            cw = res_k.cw
            u_c = self._face_deflection(cw)
            r_new = r.copy()
            r_new[self.fmask] = (r[self.fmask]
                                 + self.beta_eff * (u_c[self.fmask] - self._gap_f))
            np.maximum(r_new, 0.0, out=r_new)               # проекция r ≥ 0
            r_new[~self.fmask] = 0.0
            # ОТНОСИТЕЛЬНАЯ невязка (нормировка на масштаб реакции): терпит
            # «мерцание» дискретной контактной границы у самого решения.
            dr = float(np.sqrt(np.sum(q.w * (r_new - r) ** 2)))
            r_scale = float(np.sqrt(np.sum(q.w * r_new ** 2)))
            res = dr / r_scale if r_scale > 0.0 else dr
            hist.append(res)
            r = r_new
            if res < self.tol:
                converged = True
                break
        # финальное состояние на сошедшейся реакции
        res_k = solver.solve(q0 - r, c0=cw)
        cw = res_k.cw
        w = res_k.w_nodes
        u_c = self._face_deflection(cw)
        return self._package(r, w, u_c, cw, it, converged, hist, n_inner)

    # -- совмещённая схема (рабочая, предмет T7; прил. C.2) -------------- #
    def _solve_merged(self) -> NonlinearContactResult:
        r"""Совмещённый МОР–КТН: ОДИН шаг Пикара на ОДИН шаг МОР (§4.2, прил. C.2).

        Два итерационных процесса связаны в один цикл с общей недорелаксацией:
        нелинейное состояние (``N``, КТН-члены) и реакция ``r`` обновляются
        совместно, без полного внутреннего решателя ⇒ быстрее вложенной схемы в
        разы. Сходимость — предмет теоремы T7 (композиция сжатий МОР T4 и
        Пикара T5; при малости нагрузки/зазора — сжатие). Гейт R4: совпадает с
        вложенной до допуска.
        """
        solver, q = self.solver, self.solver.quad
        q0 = float(self.cfg.q0)
        theta = float(self.cfg.karman_relax)                # общая недорелаксация
        c = self._free.cw.copy()
        r = np.zeros(q.x.size)
        hist: list[float] = []
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 — it нужен после цикла
            w_old = c @ solver._psi
            b_level = solver._load_vector(q0 - r)           # нагрузка при текущей реакции
            c, _forces = solver._picard_map(c, b_level, theta)  # ОДИН шаг Пикара
            u_c = self._face_deflection(c)
            r_new = r.copy()
            r_new[self.fmask] = (r[self.fmask]
                                 + self.beta_eff * (u_c[self.fmask] - self._gap_f))
            np.maximum(r_new, 0.0, out=r_new)
            r_new[~self.fmask] = 0.0
            # совместная сходимость: реакция И прогиб стабилизировались
            dr = float(np.sqrt(np.sum(q.w * (r_new - r) ** 2)))
            r_scale = float(np.sqrt(np.sum(q.w * r_new ** 2)))
            res_r = dr / r_scale if r_scale > 0.0 else dr
            w_new = c @ solver._psi
            wn = float(np.sqrt(np.sum(q.w * w_new ** 2)))
            res_w = float(np.sqrt(np.sum(q.w * (w_new - w_old) ** 2)) / wn) if wn > 0 else 0.0
            res = max(res_r, res_w)
            hist.append(res)
            r = r_new
            if res < self.tol:
                converged = True
                break
        w = c @ solver._psi
        u_c = self._face_deflection(c)
        return self._package(r, w, u_c, c, it, converged, hist, it)

    def _package(self, r, w, u_c, cw, iters, converged, hist, n_inner):
        q = self.solver.quad
        contact = r > 0.0
        peak = int(np.argmax(r)) if r.size else 0
        return NonlinearContactResult(
            r_nodes=r, w_nodes=w, u_c_nodes=u_c, cw=cw,
            w_max=float(np.max(np.abs(w))), contact_mask=contact,
            r_max=float(r.max()) if r.size else 0.0,
            peak_xy=(float(q.x[peak]), float(q.y[peak])),
            n_contact=int(contact.sum()),
            n_components=contact_components(q.x, q.y, contact),   # топология зоны (§8)
            iters=iters, converged=converged,
            residual_history=np.array(hist), n_inner=n_inner, scheme=self.scheme)


@dataclass
class NonlinearTwoPlateResult:
    r"""Результат взаимного контакта ДВУХ пластин КТН (§9.2).

    Реакция ``r ≥ 0`` — общая для пары (3-й закон Ньютона): пластина 1 несёт
    нагрузку ``f₁ − r``, пластина 2 — ``f₂ + r``. Зазор проверяется РАЗНОСТЬЮ
    лицевых прогибов ``u_c1 − u_c2 ≤ z``. Поля — в общих узлах квадратуры.
    """

    r_nodes: np.ndarray
    w1_nodes: np.ndarray
    w2_nodes: np.ndarray
    u_c1_nodes: np.ndarray
    u_c2_nodes: np.ndarray
    cw1: np.ndarray
    cw2: np.ndarray
    w1_max: float
    w2_max: float
    contact_mask: np.ndarray
    r_max: float
    peak_xy: tuple
    n_contact: int
    n_components: int
    iters: int
    converged: bool
    residual_history: np.ndarray


class NonlinearTwoPlateMOR:
    r"""Взаимный контакт ДВУХ деформируемых пластин КТН (§9.2, N7).

    Обобщение линейной :class:`~plate_solver.contact.TwoPlateMOR` на полную
    нелинейную КТН: реакция ``r`` определяется СОВМЕСТНО двумя связанными
    решателями. Условие непроникания — на ЛИЦЕВЫХ прогибах (условие Синьорини
    на грани, масштаб кривизны от теории каждой пластины):

    .. math:: u_{c1} - u_{c2} \le z,\qquad r\ge 0,\qquad f_1 - r,\ \ f_2 + r,

    где ``z`` — начальный зазор (скаляр или поле). Итерация (совмещённая схема,
    как в :meth:`NonlinearContactMOR._solve_merged`): один шаг Пикара на каждую
    пластину + один шаг МОР по общей реакции, недорелаксация общая. Усиление
    суммарного оператора ``G = G₁ + G₂`` ⇒ ``β_eff = β/(gain₁+gain₂)`` (теорема 4).

    Редукция (гейт): жёсткая вторая пластина (``u_c2 → 0``) ⇒ односторонний
    контакт одной пластины :class:`NonlinearContactMOR` с плоским зазором ``z``.

    Пластины должны быть построены на ОБЩЕЙ квадратуре (одна область, один ``Q``):
    реакция ``r`` определена поузельно на интерфейсе (межсеточного переноса нет).

    Parameters
    ----------
    solver1, solver2 : решатели КТН пары (общая квадратура).
    cfg : параметры (``q0``, ``beta``, ``max_iter``, ``tol``, ``karman_relax``).
    gap : начальный зазор ``z`` — скаляр или поле по узлам.
    f1, f2 : нагрузки пластин (скаляр/поле); по умолчанию ``cfg.q0`` и ``q2``/0.
    q2 : скалярная нагрузка второй пластины (если ``f2`` не задан).
    foundation_mask : предикат зоны возможного контакта; ``None`` ⇒ вся Ω.
    """

    def __init__(self, solver1: KTNSolver, solver2: KTNSolver, cfg: Config, *,
                 gap=0.0, f1=None, f2=None, q2=None, foundation_mask=None):
        q = solver1.quad
        if not (np.array_equal(q.x, solver2.quad.x) and np.array_equal(q.y, solver2.quad.y)):
            raise ValueError("две пластины КТН: требуется ОБЩАЯ квадратура "
                             "(одна область, один Q) — реакция определена поузельно")
        self.s1, self.s2, self.cfg = solver1, solver2, cfg
        n = q.x.size
        # зона основания
        if foundation_mask is None:
            self.fmask = np.ones(n, dtype=bool)
        else:
            self.fmask = np.asarray(foundation_mask(q.x, q.y), dtype=bool)
        # зазор: скаляр или поле
        if np.ndim(gap) == 0:
            self._gap_f = float(gap)
        else:
            g = np.asarray(gap, dtype=float)
            if g.shape != (n,):
                raise ValueError("Поле зазора: ожидается массив длины числа узлов квадратуры.")
            self._gap_f = g[self.fmask]
        # нагрузки пластин
        self.f1 = np.full(n, float(cfg.q0)) if f1 is None else self._as_field(f1, n)
        if f2 is not None:
            self.f2 = self._as_field(f2, n)
        else:
            self.f2 = np.full(n, float(q2) if q2 is not None else 0.0)
        # тёплый старт — свободные решения под собственными нагрузками
        self._free1 = solver1.solve(self.f1)
        self._free2 = solver2.solve(self.f2)
        # усиление СУММАРНОГО оператора G = G₁ + G₂ (теорема 4): секущая
        # податливость КАЖДОЙ пластины при ОПОРНОЙ нагрузке. Важно брать не
        # свободное решение под f_i (вторая пластина может быть без нагрузки,
        # но откликается на реакцию — её податливость ненулевая), а отклик на
        # общий масштаб — иначе β_eff завышен и итерация расходится.
        ref = max(float(np.max(np.abs(self.f1))),
                  float(np.max(np.abs(self.f2))), float(cfg.q0))
        gain1 = solver1.solve(np.full(n, ref)).w_max / ref
        gain2 = solver2.solve(np.full(n, ref)).w_max / ref
        gain = gain1 + gain2
        self.beta_eff = cfg.beta / gain if gain > 0.0 else cfg.beta
        self.max_iter = int(cfg.max_iter)
        self.tol = float(cfg.tol)
        self._c1 = solver1.params.face_curv_coeff
        self._c2 = solver2.params.face_curv_coeff

    @staticmethod
    def _as_field(f, n) -> np.ndarray:
        arr = np.asarray(f, dtype=float)
        return np.full(n, float(arr)) if arr.ndim == 0 else arr

    def _face(self, solver, cw, c_curv) -> np.ndarray:
        """Лицевой прогиб пластины ``u_c = w + c_curv·Δw`` в узлах."""
        w = cw @ solver._psi
        if c_curv == 0.0:
            return w
        return w + c_curv * (cw @ solver._lap_psi)

    def solve(self) -> NonlinearTwoPlateResult:
        """Совмещённый МОР для пары пластин (§9.2): по одному шагу Пикара на пластину."""
        q = self.s1.quad
        theta = float(self.cfg.karman_relax)
        c1, c2 = self._free1.cw.copy(), self._free2.cw.copy()
        r = np.zeros(q.x.size)
        hist: list[float] = []
        converged = False
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 — it нужен после цикла
            w1_old, w2_old = c1 @ self.s1._psi, c2 @ self.s2._psi
            b1 = self.s1._load_vector(self.f1 - r)          # пластина 1: f₁ − r
            b2 = self.s2._load_vector(self.f2 + r)          # пластина 2: f₂ + r
            c1, _ = self.s1._picard_map(c1, b1, theta)      # по шагу Пикара на пластину
            c2, _ = self.s2._picard_map(c2, b2, theta)
            u1 = self._face(self.s1, c1, self._c1)
            u2 = self._face(self.s2, c2, self._c2)
            u = u1 - u2                                     # сближение лицевых
            r_new = r.copy()
            r_new[self.fmask] = r[self.fmask] + self.beta_eff * (u[self.fmask] - self._gap_f)
            np.maximum(r_new, 0.0, out=r_new)               # проекция r ≥ 0
            r_new[~self.fmask] = 0.0
            # совместная сходимость: реакция И оба прогиба стабилизировались
            res = max(_rel_change(q.w, r_new, r),
                      _rel_change(q.w, c1 @ self.s1._psi, w1_old),
                      _rel_change(q.w, c2 @ self.s2._psi, w2_old))
            hist.append(res)
            r = r_new
            if res < self.tol:
                converged = True
                break
        w1, w2 = c1 @ self.s1._psi, c2 @ self.s2._psi
        u1 = self._face(self.s1, c1, self._c1)
        u2 = self._face(self.s2, c2, self._c2)
        contact = r > 0.0
        peak = int(np.argmax(r)) if r.size else 0
        return NonlinearTwoPlateResult(
            r_nodes=r, w1_nodes=w1, w2_nodes=w2, u_c1_nodes=u1, u_c2_nodes=u2,
            cw1=c1, cw2=c2, w1_max=float(np.max(np.abs(w1))),
            w2_max=float(np.max(np.abs(w2))), contact_mask=contact,
            r_max=float(r.max()) if r.size else 0.0,
            peak_xy=(float(q.x[peak]), float(q.y[peak])),
            n_contact=int(contact.sum()),
            n_components=contact_components(q.x, q.y, contact),
            iters=it, converged=converged, residual_history=np.array(hist))


def _rel_change(w, new, old) -> float:
    """Относительное изменение поля в норме L2 по квадратуре (‖new−old‖/‖new‖)."""
    scale = float(np.sqrt(np.sum(w * new ** 2)))
    d = float(np.sqrt(np.sum(w * (new - old) ** 2)))
    return d / scale if scale > 0.0 else d


__all__ = ["NonlinearContactMOR", "NonlinearContactResult",
           "NonlinearTwoPlateMOR", "NonlinearTwoPlateResult"]
