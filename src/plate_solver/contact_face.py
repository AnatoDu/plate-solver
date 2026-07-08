r"""contact_face.py — ЭКСПЕРИМЕНТАЛЬНАЯ (опциональная) постановка: контакт с
ЛИЦЕВОЙ поверхностью ``v = w^{h/2}`` как ОСНОВНЫМ полем.

Альтернатива штатному :mod:`~plate_solver.contact_nl` (там контакт «щупает»
восстановленный ``u_c = w + c_curv·Δw``): здесь лицевой прогиб — самостоятельное
поле, условие Синьорини ставится прямо на нём. Модуль АДДИТИВЕН — существующие
теории/решатели не затрагивает; исследовательский задел.

Гипотеза. Если сделать лицевой прогиб ``v = w^{h/2}`` основным искомым
полем и ставить условие Синьорини ``v ≤ z`` прямо на нём, то контакт становится
классической obstacle-задачей (ограничение на само неизвестное), а не
ограничением через дифференциальный оператор ``Δw``; при этом исчезает численный
артефакт (звон Гиббса) восстановления ``v`` через ``Δw``.

Текущая (эталонная) постановка и реформулировка (ТЗ §1, §2)
----------------------------------------------------------
СРЕДИННЫЙ прогиб ``w`` в обоих режимах решается ИЗ РАВНОВЕСИЯ (Ритц, защемление):
реакция ``r`` входит поперечной нагрузкой ``q_n − r``. Различается ТОЛЬКО способ
вычисления контактной величины — лицевого прогиба ``v`` (формула (9)):

.. math:: v = w + (h_c^2 - h_*^2)\,\Delta w \;+\; c_q q_{\rm eff} \;+\; c_r D r ,

где ``c_q = -\kappa_q``, ``c_r = -\kappa_r`` — замороженные коэффициенты
:class:`~plate_solver.ktn.KTNParams`; ``q_{\rm eff}`` — эффективная поперечная
нагрузка (``q_n`` для классики/линейной КТН — соотношение ТОЧНОЕ; ``q_n+L(Φ,w)``
для нелинейной КТН, ТЗ §2).

* ``face_mode="pointwise"`` (ЭТАЛОН = текущий решатель): ``Δw`` берётся ПОУЗЕЛЬНО
  из спектральной структуры (``\Delta(\omega^2 T)``). Второе дифференцирование
  спектрального поля усиливает высокочастотные осцилляции у кромки пятна
  контакта — тот самый звон.
* ``face_mode="weak"`` (РЕФОРМУЛИРОВКА, вариант B ТЗ §4): ``v`` — самостоятельное
  ГЛАДКОЕ поле ``v = \omega\sum c_i\varphi_i`` (базис лицевой поверхности,
  краевое условие ``v=0`` у защемления встроено множителем ``\omega``), связанное
  с ``w`` в СЛАБОЙ форме. Кривизный член интегрируется по частям (защемление ⇒
  ``\varphi_j=\omega T_j=0`` на ∂Ω ⇒ граничный интеграл нулевой):

  .. math:: \int_\Omega (h_c^2-h_*^2)\,\Delta w\,\varphi_j\,dA
            = -(h_c^2-h_*^2)\int_\Omega \nabla w\cdot\nabla\varphi_j\,dA .

  Вместо РОУГЕНИНГА (второй производной ``\Delta w``) в контактную величину
  входит СГЛАЖИВАЮЩИЙ функционал ``\int\nabla w\cdot\nabla\varphi_j`` — одна
  производная снимается с решения на пробную функцию.

Мат. обоснование метода (``face_mode="weak"``)
----------------------------------------------
``v`` — Галёркинова ``L2``-проекция соотношения (9) на конечномерное пространство
``V_h = \mathrm{span}\{\omega T_j\}`` (степень ``p``): ``G\,c_v = b``,
``G_{ij}=\int\varphi_i\varphi_j`` (масса, SPD, факторизуется однажды),
``b_j=\int w\varphi_j - (h_c^2-h_*^2)\int\nabla w\cdot\nabla\varphi_j
      + \int (c_q q_{\rm eff}+c_r D r)\varphi_j``.
Проекция — ортопроектор в ``L2`` (норма 1); гладкая часть аппроксимируется
спектрально (``p``), но высокочастотный «хвост» ``\Delta w`` НЕ усиливается —
под интегралом стоит первая производная. Классика (``h_c^2-h_*^2=0``) ⇒ ``v``
есть ``L2``-проекция ``w`` (тождественно ``v≈w``).

Это ИССЛЕДОВАТЕЛЬСКАЯ ПРОБА (ТЗ §10), не гарантированное улучшение; критерий —
совпадение с эталоном в зоне контакта при уходе ``Δw``-звона и СОХРАНЕНИИ
кромочной сингулярности реакции (физика жёсткого контакта, Михайловский–Тарасов).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

from .config import Config
from .faces import FaceParams
from .ktn import KTNParams
from .ktn_solver import KTNSolver
from .membrane import _disp_structure


# --------------------------------------------------------------------------- #
#  Гладкий лицевой базис V_h = span{ω T_j} и вычисление лицевого поля v
# --------------------------------------------------------------------------- #
class _FaceBasis:
    r"""Базис лицевой поверхности ``φ_j = ω T_j`` (v=0 у защемления) и масса.

    Структура перемещений в плане ``ω·T`` (immovable) — ровно нужный лицевой
    базис: одно краевое условие ``v=0`` встроено множителем ``ω`` (ТЗ §4).
    Масса ``G_{ij}=∫ φ_i φ_j dA`` (SPD) факторизуется один раз (Холецкий).
    """

    def __init__(self, solver: KTNSolver):
        q = solver.quad
        Phi_v, Phi_vx, Phi_vy = _disp_structure(solver.domain, solver.basis,
                                                q.x, q.y, immovable=True)
        self.Phi_v, self.Phi_vx, self.Phi_vy = Phi_v, Phi_vx, Phi_vy
        W = solver._W
        G = (Phi_v * W) @ Phi_v.T
        self.G_chol = sla.cho_factor(0.5 * (G + G.T))

    def project(self, solver, cw, c_curv, extra) -> tuple[np.ndarray, np.ndarray]:
        r"""Слабое (Галёркин) лицевое поле: ``(c_v, v_узлы)`` из (9) по частям."""
        W = solver._W
        w = cw @ solver._psi
        b = self.Phi_v @ (W * (w + extra))
        if c_curv != 0.0:
            wx = cw @ solver._psi_x
            wy = cw @ solver._psi_y
            # ∫ c_curv Δw φ = −c_curv ∫ ∇w·∇φ  (граничный член ноль: φ=0 на ∂Ω)
            b = b - c_curv * (self.Phi_vx @ (W * wx) + self.Phi_vy @ (W * wy))
        cv = sla.cho_solve(self.G_chol, b)
        return cv, cv @ self.Phi_v


def _face_pointwise(solver, cw, c_curv, extra) -> np.ndarray:
    r"""ЭТАЛОН: ``v = w + c_curv·Δw + extra`` (``Δw`` ПОУЗЕЛЬНО, спектральный Δ)."""
    v = cw @ solver._psi
    if c_curv != 0.0:
        v = v + c_curv * (cw @ solver._lap_psi)
    return v + extra


@dataclass
class FaceContactResult:
    r"""Результат контактной задачи с лицевым полем ``v`` (поля в узлах квадратуры).

    Attributes
    ----------
    r_nodes : обобщённая реакция ``r ≥ 0`` в узлах (источник истины).
    w_nodes : СРЕДИННЫЙ прогиб (из равновесия) в узлах.
    v_nodes : ЛИЦЕВОЙ прогиб ``v = w^{h/2}`` (контактная величина) в узлах.
    cw : коэффициенты срединного прогиба ``w = ω^2·Σ c_k T_k``.
    cv : коэффициенты лицевого поля ``v = ω·Σ c_i φ_i`` (только weak; иначе None).
    w_max, v_max : max|w|, max(v) (лицевой — со знаком, для сравнения с зазором).
    r_max : пиковая реакция; peak_xy — её локализация.
    contact_mask : булева маска зоны контакта (``r > 0``).
    n_contact : число контактных узлов.
    rho_c : эффективный радиус пятна контакта = max ρ узла с ``r>0`` (круг).
    plateau_dev : max|v − z| по узлам контакта (нарушение плато ``v=z``).
    iters, converged : диагностика внешнего МОР (сходимость ПО ПРОГИБУ, ТЗ §5).
    residual_history : относительный сдвиг лицевого прогиба по итерациям.
    face_mode : "pointwise" | "weak".
    """

    r_nodes: np.ndarray
    w_nodes: np.ndarray
    v_nodes: np.ndarray
    cw: np.ndarray
    cv: np.ndarray | None
    w_max: float
    v_max: float
    r_max: float
    peak_xy: tuple
    contact_mask: np.ndarray
    n_contact: int
    rho_c: float
    plateau_dev: float
    iters: int
    converged: bool
    residual_history: np.ndarray
    face_mode: str


def _package(solver, r, w, v, cw, cv, gap, iters, converged, hist,
             face_mode) -> FaceContactResult:
    q = solver.quad
    contact = r > 0.0
    peak = int(np.argmax(r)) if r.size else 0
    rho = np.hypot(q.x, q.y)
    rho_c = float(np.max(rho[contact])) if contact.any() else 0.0
    gap_f = gap if np.ndim(gap) == 0 else gap
    plateau_dev = float(np.max(np.abs((v - gap_f)[contact]))) if contact.any() else 0.0
    return FaceContactResult(
        r_nodes=r, w_nodes=w, v_nodes=v, cw=cw, cv=cv,
        w_max=float(np.max(np.abs(w))), v_max=float(np.max(v)),
        r_max=float(r.max()) if r.size else 0.0,
        peak_xy=(float(q.x[peak]), float(q.y[peak])),
        contact_mask=contact, n_contact=int(contact.sum()),
        rho_c=rho_c, plateau_dev=plateau_dev,
        iters=iters, converged=converged,
        residual_history=np.array(hist), face_mode=face_mode)


# --------------------------------------------------------------------------- #
#  Phase 1: линейный контакт (классика + линейная КТН) — соотношение (9) точное
# --------------------------------------------------------------------------- #
class FacePrimaryContact:
    r"""Контакт МОР с лицевым полем ``v`` для КЛАССИКИ и ЛИНЕЙНОЙ КТН (Phase 1 ТЗ).

    Срединный прогиб ``w`` — линейный защемлённый изгиб (Кирхгоф), общий для
    классики и линейной КТН (они различаются лишь лицевой поправкой). Внешний
    цикл МОР одинаков для обоих режимов ``face_mode``; различается только
    вычисление контактной величины ``v`` — строго контролируемое сравнение
    «эталон vs реформулировка».

    Parameters
    ----------
    solver : линейный защемлённый :class:`~plate_solver.ktn_solver.KTNSolver`
        (пресет ``classic``; поставляет структуру ``ψ=ω^2 T`` и оператор изгиба).
    cfg : параметры (``q0``, ``beta``, ``max_iter``, ``tol``, ``E, nu, h``).
    gap : зазор ``z`` (скаляр — плоское препятствие; массив по узлам — штамп).
    face_mode : ``"pointwise"`` (эталон) | ``"weak"`` (реформулировка).
    refined : учитывать лицевую поправку КТН (``linear KTN``); ``False`` —
        классика (``v = w``).
    include_load_terms : добавлять алгебраические члены ``c_q q_0 + c_r D r``
        (физические коэффициенты :class:`KTNParams`); без дифференцирования —
        звон не создают, входят одинаково в оба режима.
    stop : критерий останова — ``"w"`` (по норме ЛИЦЕВОГО прогиба, ТЗ §5:
        реакция может не сходиться) | ``"dr"`` (по норме реакции, для сверки).
    """

    def __init__(self, solver: KTNSolver, cfg: Config, *, gap,
                 face_mode: str = "weak", refined: bool = True,
                 include_load_terms: bool = True, foundation_mask=None,
                 stop: str = "w"):
        if face_mode not in ("pointwise", "weak"):
            raise ValueError(f"face_mode: ожидалось pointwise | weak, получено {face_mode!r}")
        if stop not in ("w", "dr"):
            raise ValueError(f"stop: ожидалось w | dr, получено {stop!r}")
        self.solver = solver
        self.cfg = cfg
        self.face_mode = face_mode
        self.refined = bool(refined)
        self.stop = stop
        q = solver.quad
        n = q.x.size

        fp = FaceParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
        self.c_curv = fp.c_curv if self.refined else 0.0
        if self.refined and include_load_terms:
            kp = KTNParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
            self._cq, self._cr = kp.cq_contact, kp.cr_contact
        else:
            self._cq = self._cr = 0.0

        if np.ndim(gap) == 0:
            self.gap = float(gap)
        else:
            self.gap = np.asarray(gap, dtype=float)
            if self.gap.shape != (n,):
                raise ValueError("Поле зазора: ожидается массив длины числа узлов квадратуры.")
        if foundation_mask is None:
            self.fmask = np.ones(n, dtype=bool)
        else:
            self.fmask = np.asarray(foundation_mask(q.x, q.y), dtype=bool)
        self._gap_f = self.gap if np.ndim(self.gap) == 0 else self.gap[self.fmask]

        self._fb = _FaceBasis(solver) if face_mode == "weak" else None
        self._W = solver._W

        cw_unit = self._w_solve(np.ones(n))
        self.gain = float(np.max(np.abs(cw_unit @ solver._psi)))
        self.beta_eff = cfg.beta / self.gain
        self.max_iter = int(cfg.max_iter)
        self.tol = float(cfg.tol)

    def _w_solve(self, f_values) -> np.ndarray:
        r"""Коэффициенты ``w`` линейного защемлённого изгиба ``D·S_bend c = ∫fψ``."""
        b = self.solver._load_vector(np.asarray(f_values, float))
        return self.solver._bending_solve(b)

    def _extra(self, r) -> np.ndarray | float:
        r"""Алгебраические члены (9): ``c_q q_0 + c_r D r`` (без дифференцирования)."""
        if self._cq == 0.0 and self._cr == 0.0:
            return 0.0
        return self._cq * float(self.cfg.q0) + self._cr * self.solver.D * r

    def _face(self, cw, r) -> tuple[np.ndarray, np.ndarray | None]:
        extra = self._extra(r)
        if self.face_mode == "weak":
            cv, v = self._fb.project(self.solver, cw, self.c_curv, extra)
            return v, cv
        return _face_pointwise(self.solver, cw, self.c_curv, extra), None

    def solve(self) -> FaceContactResult:
        r"""Внешний цикл МОР (obstacle на ``v``); останов по прогибу (ТЗ §5)."""
        s, q, W = self.solver, self.solver.quad, self._W
        q0 = float(self.cfg.q0)
        n = q.x.size
        r = np.zeros(n)
        cv = None
        hist: list[float] = []
        converged = False
        v = np.zeros(n)
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 — it нужен после цикла
            cw = self._w_solve(q0 - r)
            v_old = v
            v, cv = self._face(cw, r)
            r_new = r.copy()
            r_new[self.fmask] = r[self.fmask] + self.beta_eff * (v[self.fmask] - self._gap_f)
            np.maximum(r_new, 0.0, out=r_new)
            r_new[~self.fmask] = 0.0
            res = _residual(self.stop, W, v, v_old, r_new, r)
            hist.append(res)
            r = r_new
            if it > 1 and res < self.tol:
                converged = True
                break
        cw = self._w_solve(q0 - r)
        v, cv = self._face(cw, r)
        w = cw @ s._psi
        return _package(s, r, w, v, cw, cv, self.gap, it, converged, hist, self.face_mode)


# --------------------------------------------------------------------------- #
#  Phase 2: нелинейный контакт (нелинейная КТН) — эффективная нагрузка q_n + L(Φ,w)
# --------------------------------------------------------------------------- #
class FacePrimaryContactKTN:
    r"""Контакт МОР с лицевым полем ``v`` для НЕЛИНЕЙНОЙ КТН (Phase 2 ТЗ).

    Срединный прогиб ``w`` — ПОЛНАЯ нелинейная КТН (совмещённая схема: один шаг
    Пикара на один шаг МОР, как в :class:`~plate_solver.contact_nl.NonlinearContactMOR`).
    Лицевое соотношение (исправленное, ТЗ §2) использует ЭФФЕКТИВНУЮ нагрузку
    ``q_{\rm eff} = q_n + L(Φ,w)`` в нагрузочном члене обжатия
    (``L = N_x w_{xx} + 2N_{xy} w_{xy} + N_y w_{yy}``, из поперечного равновесия
    (ktn3)); при ``effective_load=False`` — старая (9) с ``q_{\rm eff}=q_n``.

    Parameters
    ----------
    solver : нелинейный :class:`~plate_solver.ktn_solver.KTNSolver` (``ktn_full``).
    cfg, gap, face_mode, foundation_mask, stop : как в :class:`FacePrimaryContact`.
    effective_load : ``True`` — исправленная (9) с ``q_n+L``; ``False`` — старая.
    """

    def __init__(self, solver: KTNSolver, cfg: Config, *, gap,
                 face_mode: str = "weak", effective_load: bool = True,
                 foundation_mask=None, stop: str = "w"):
        if face_mode not in ("pointwise", "weak"):
            raise ValueError(f"face_mode: ожидалось pointwise | weak, получено {face_mode!r}")
        if stop not in ("w", "dr"):
            raise ValueError(f"stop: ожидалось w | dr, получено {stop!r}")
        self.solver = solver
        self.cfg = cfg
        self.face_mode = face_mode
        self.effective_load = bool(effective_load)
        self.stop = stop
        q = solver.quad
        n = q.x.size

        self.c_curv = solver.params.face_curv_coeff     # масштаб теории (физич. для ktn_full)
        kp = KTNParams(E=cfg.E, nu=cfg.nu, h=cfg.h)
        self._cq, self._cr = kp.cq_contact, kp.cr_contact

        if np.ndim(gap) == 0:
            self.gap = float(gap)
        else:
            self.gap = np.asarray(gap, dtype=float)
            if self.gap.shape != (n,):
                raise ValueError("Поле зазора: ожидается массив длины числа узлов квадратуры.")
        if foundation_mask is None:
            self.fmask = np.ones(n, dtype=bool)
        else:
            self.fmask = np.asarray(foundation_mask(q.x, q.y), dtype=bool)
        self._gap_f = self.gap if np.ndim(self.gap) == 0 else self.gap[self.fmask]

        self._fb = _FaceBasis(solver) if face_mode == "weak" else None
        self._W = solver._W

        q0 = float(cfg.q0)
        self._free = solver.solve(np.full(n, q0))       # тёплый старт + усиление
        self.gain = self._free.w_max / q0 if q0 != 0.0 else 1.0
        self.beta_eff = cfg.beta / self.gain
        self.max_iter = int(cfg.max_iter)
        self.tol = float(cfg.tol)

    def _bilinear_L(self, cw, Nx, Ny, Nxy) -> np.ndarray:
        r"""Оператор ``L(Φ,w) = N_x w_{xx} + 2 N_{xy} w_{xy} + N_y w_{yy}`` в узлах."""
        s = self.solver
        return Nx * (cw @ s._pxx) + 2.0 * Nxy * (cw @ s._pxy) + Ny * (cw @ s._pyy)

    def _extra(self, cw, r, forces) -> np.ndarray:
        r"""Алгебраические члены (9): ``c_q q_eff + c_r D r`` (q_eff = q_n[+L])."""
        q0 = float(self.cfg.q0)
        q_eff = q0
        if self.effective_load:
            _, _, Nx, Ny, Nxy = forces
            q_eff = q0 + self._bilinear_L(cw, Nx, Ny, Nxy)
        return self._cq * q_eff + self._cr * self.solver.D * r

    def _face(self, cw, r, forces) -> tuple[np.ndarray, np.ndarray | None]:
        extra = self._extra(cw, r, forces)
        if self.face_mode == "weak":
            cv, v = self._fb.project(self.solver, cw, self.c_curv, extra)
            return v, cv
        return _face_pointwise(self.solver, cw, self.c_curv, extra), None

    def solve(self) -> FaceContactResult:
        r"""Совмещённый МОР–КТН (один шаг Пикара на шаг МОР); останов по прогибу."""
        s, q, W = self.solver, self.solver.quad, self._W
        q0 = float(self.cfg.q0)
        theta = float(self.cfg.karman_relax)
        c = self._free.cw.copy()
        r = np.zeros(q.x.size)
        cv = None
        hist: list[float] = []
        converged = False
        v = np.zeros(q.x.size)
        it = 0
        for it in range(1, self.max_iter + 1):  # noqa: B007 — it нужен после цикла
            b_level = s._load_vector(q0 - r)
            c, forces = s._picard_map(c, b_level, theta)     # ОДИН шаг Пикара КТН
            v_old = v
            v, cv = self._face(c, r, forces)
            r_new = r.copy()
            r_new[self.fmask] = r[self.fmask] + self.beta_eff * (v[self.fmask] - self._gap_f)
            np.maximum(r_new, 0.0, out=r_new)
            r_new[~self.fmask] = 0.0
            res = _residual(self.stop, W, v, v_old, r_new, r)
            hist.append(res)
            r = r_new
            if it > 1 and res < self.tol:
                converged = True
                break
        w = c @ s._psi
        return _package(s, r, w, v, c, cv, self.gap, it, converged, hist, self.face_mode)


def _residual(stop, W, v, v_old, r_new, r) -> float:
    r"""Невязка внешнего цикла: по прогибу (``w``, ТЗ §5) или по реакции (``dr``)."""
    if stop == "w":
        vn = float(np.sqrt(np.sum(W * v ** 2)))
        return (float(np.sqrt(np.sum(W * (v - v_old) ** 2))) / vn) if vn > 0 else 0.0
    dr = float(np.sqrt(np.sum(W * (r_new - r) ** 2)))
    rs = float(np.sqrt(np.sum(W * r_new ** 2)))
    return dr / rs if rs > 0 else dr


__all__ = ["FacePrimaryContact", "FacePrimaryContactKTN", "FaceContactResult"]
