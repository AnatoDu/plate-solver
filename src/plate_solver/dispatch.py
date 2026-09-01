r"""dispatch.py — диспетчер: постановка Problem → расчёт → Result.

Маршрутизация (блок-схема — docs/dispatch_flow.md):

* ``bc.type = soft_hinge``  → :class:`~plate_solver.plate.PlateBending`
  (расщепление бигармоники на две задачи Пуассона);
* ``bc.type = clamped``     → :class:`~plate_solver.clamped.ClampedPlate`
  (структура w = ω²Φ, прямой Ритц по бигармонике);
* ``contact.enabled`` + ``theory = classic | ktn_linear`` →
  :class:`~plate_solver.contact.ContactMOR` (soft_hinge И clamped; ``[contact.zone]``
  → ``foundation_mask = [ω_zone > 0]`` в узлах квадратуры);
* ``contact.enabled`` + ``theory = karman | ktn_full`` (v0.6.3) →
  :class:`~plate_solver.contact_nl.NonlinearContactMOR` (МОР поверх полной КТН,
  лицевое условие Синьорини; защемление, равномерная нагрузка);
  пара — :class:`~plate_solver.contact_nl.NonlinearTwoPlateMOR`;
* ``model.theory = ktn_linear`` → :class:`~plate_solver.ktn.KTNParams`
  (в контакте — смещение контактной поверхности, как в ContactMOR; в чистом
  изгибе — ``corrected_deflection`` при r = 0, кривизна Δw = −M/D из (P1)).

Нагрузки: ``uniform`` — константа q0; ``patch`` — q̃ = q0·[ω_zone > 0]
в узлах; ``point`` — регуляризованный patch: круговое пятно радиуса
eps_eff с q = P/(π·eps_eff²). Защита зон: пятно point автоматически
расширяется до ≥ MIN_ZONE_NODES узлов (факт — в ``Result.warnings``);
зона patch/contact с < MIN_ZONE_NODES узлами — :class:`CaseError`.
"""

from __future__ import annotations

import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from . import exprfield, geometry
from .clamped import ClampedPlate
from .config import Config
from .contact import ContactMOR, ContactResult
from .ktn import KTNParams
from .plate import PlateBending
from .problem import MIN_ZONE_NODES, CaseError, GeometrySpec, Problem

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"


# --------------------------------------------------------------------------- #
#  Геометрия: спецификация → Domain (реестр v0.2)
# --------------------------------------------------------------------------- #
def build_domain(spec: GeometrySpec) -> geometry.Domain:
    """Построить Domain по спецификации из case-файла (реестр геометрий).

    Compose-операции union|intersect|difference могут дать ВЫРОЖДЕННУЮ область
    (пустое пересечение bbox, difference с поглощением, пустая внутренность) —
    структурный валидатор (`_validate_compose_node`) их не ловит. Такие случаи
    отклоняются ЗДЕСЬ понятной `CaseError` (иначе — деление на ноль в сборке и
    падение LAPACK на NaN).
    """
    try:
        if spec.kind == "circle":
            dom = geometry.make_circle(spec.a)
        elif spec.kind == "rectangle":
            dom = geometry.make_rectangle(spec.x1, spec.x2, spec.y1, spec.y2)
        elif spec.kind == "L":
            dom = geometry.make_L(spec.side, spec.cut)
        elif spec.kind == "annulus":
            dom = geometry.make_annulus(spec.a, spec.b)
        elif spec.kind == "ellipse":
            dom = geometry.make_ellipse(spec.a, spec.b)
        elif spec.kind == "compose":
            dom = geometry.make_compose(spec.tree)
        else:
            from .problem import GEOMETRY_KINDS

            raise CaseError(f"geometry.kind: получено {spec.kind!r}, ожидалось "
                            f"{' | '.join(GEOMETRY_KINDS)}, "
                            f"см. {_SCHEMA_DOC}#geometry")
    except CaseError:
        raise
    except ValueError as e:                    # напр. пустое пересечение bbox в compose
        raise CaseError(f"geometry.{spec.kind}: вырожденная область — {e}; "
                        f"см. {_SCHEMA_DOC}#geometry") from e
    # непустота внутренности: difference/intersect могут дать пустое множество
    x0, x1, y0, y1 = dom.bbox
    gx, gy = np.linspace(x0, x1, 129), np.linspace(y0, y1, 129)
    XX, YY = np.meshgrid(gx, gy)
    if not np.any(dom.omega(XX, YY) > 0.0):
        raise CaseError(
            f"geometry.{spec.kind}: ПУСТАЯ область — нет внутренних точек "
            f"(compose difference/intersect не должны давать пустое множество); "
            f"см. {_SCHEMA_DOC}#geometry")
    return dom


# --------------------------------------------------------------------------- #
#  Результат расчёта
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Result:
    """Неизменяемый результат расчёта постановки.

    Скалярные итоги + поля на фоновой сетке (NaN вне Ω) + узловые поля при
    контакте. ``save(dir)`` пишет result.json (снимок Problem и Config,
    git-хеш, версии numpy/scipy/sympy, предупреждения, тайминги) и фигуры
    viz.py (при ``problem.output.figures``).
    """

    problem: Problem
    config: Config
    w_max: float                       # max |w| (для theory=ktn — КТН-поправленный)
    cond: float                        # cond(A) решателя
    Xg: np.ndarray                     # фоновая сетка grid_n × grid_n
    Yg: np.ndarray
    w_grid: np.ndarray                 # прогиб на сетке (NaN вне Ω)
    warnings: tuple[str, ...] = ()
    timings: dict = field(default_factory=dict)     # секунды: build, solve
    # контакт (None, если contact.enabled = false)
    contact: ContactResult | None = None
    delta: float | None = None         # скалярный Δ; для поля зазора — min Δ на основании
    level: float | None = None         # силовой штамп (A2): найденный уровень
    force_total: float | None = None   # ∫r dΩ при решении (силовой режим)
    w_free_max: float | None = None    # max|w| без контакта (опора gap_factor)
    # нагрузка point (после защиты ≥ MIN_ZONE_NODES узлов)
    eps_eff: float | None = None
    # чистый изгиб
    w_nodes: np.ndarray | None = None  # прогиб в узлах квадратуры
    w_max_classic: float | None = None  # классический w_max (когда theory=ktn)
    # собственная задача (§eigen v0.6.4): EigenPair (значения λ_cr/ω + формы)
    eigen: object | None = None
    # точечные опоры (v0.7.0): реакции R_j = k·w(P_j) в порядке supports.points
    support_reactions: tuple | None = None

    def scalars(self) -> dict:
        """Скалярная сводка (то, что уходит в result.json и таблицы)."""
        out = {
            "w_max": self.w_max,
            "cond_A": self.cond,
            "w_max_classic": self.w_max_classic,
            "delta": self.delta,
            "w_free_max": self.w_free_max,
            "eps_eff": self.eps_eff,
            "level": self.level,
            "force_total": self.force_total,
        }
        if self.contact is not None:
            c = self.contact
            out.update({
                "iters": c.iters,
                "converged": c.converged,
                "residual_first": float(c.residual_history[0]),
                "residual_last": float(c.residual_history[-1]),
                # полная история сходимости МОР — для самостоятельных графиков
                # сходимости из result.json (v0.6.6)
                "residual_history": [float(v) for v in c.residual_history],
                "comp_residual": c.comp_residual,
                "gap_overshoot": c.gap_overshoot,
                "r_max": float(c.r_nodes.max()),
                "n_contact": int((c.r_nodes > 0).sum()),
                "n_quad": int(c.r_nodes.size),
            })
        if self.eigen is not None:
            out.update({"eigen_kind": self.eigen.kind,
                        "eigen_values": [float(v) for v in self.eigen.values]})
        if self.support_reactions is not None:
            # ключ добавляется ТОЛЬКО при наличии [supports] — прежние
            # result.json бит-неизменны
            out["support_reactions"] = [float(v) for v in self.support_reactions]
        return out

    def save(self, out_dir: str | Path | None = None,
             fig_formats: tuple = ("png", "pdf"), surface: str = "mid") -> Path:
        """Записать result.json, fields.npz (+ фигуры при output.figures).

        ``surface`` — какую поверхность рисовать на w-фигуре:
        ``mid`` | ``top`` | ``bottom`` (лицевые — NOTES §21).
        """
        out = Path(out_dir if out_dir is not None else self.problem.output.dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = {
            "problem": asdict(self.problem),
            "config": asdict(self.config),
            "scalars": self.scalars(),
            "warnings": list(self.warnings),
            "timings": self.timings,
            "provenance": _provenance(),
        }
        # Строгий JSON: NaN не входит в стандарт — метрики, не
        # определённые на пустой зоне (gap_overshoot и т.п.), пишутся null.
        (out / "result.json").write_text(
            json.dumps(_sanitize_nan(payload), ensure_ascii=False, indent=2,
                       allow_nan=False), encoding="utf-8")
        self.save_fields(out / "fields.npz")
        if self.problem.output.figures:
            self._save_figures(out, formats=fig_formats, surface=surface)
        if getattr(self.problem.output, "vtk", False):   # ParaView-экспорт (v0.6.6)
            from .export import to_vtk

            to_vtk(self, out / "result.vtk")
        return out / "result.json"

    def moments_on_grid(self):
        """Моменты (Mx, My, Mxy) на фоновой сетке (NaN вне Ω).

        Термоизгиб (v0.7.0): полные моменты несут сдвиг ``−M_T`` (Mx, My) —
        ЕДИНАЯ точка сдвига (solver.moments_at/bending_moments_* возвращают
        только упругую часть ``−D·∇∇w``).
        """
        from .ladder import bending_moments_full

        solver = self._plate
        m_t = float(getattr(self.config, "thermal_moment", 0.0))
        inside = np.isfinite(self.w_grid)
        if hasattr(solver, "moments_at"):                   # mixed-структура
            Mx = np.full(self.Xg.shape, np.nan)
            My = np.full(self.Xg.shape, np.nan)
            Mxy = np.full(self.Xg.shape, np.nan)
            mx, my, mxy = solver.moments_at(self._c, self.Xg[inside], self.Yg[inside])
            Mx[inside], My[inside], Mxy[inside] = mx, my, mxy
            if m_t != 0.0:
                Mx[inside] -= m_t
                My[inside] -= m_t
            return Mx, My, Mxy
        p_struct = 2 if hasattr(solver, "S") else 1        # ω²Φ у защемления
        Mx = np.full(self.Xg.shape, np.nan)
        My = np.full(self.Xg.shape, np.nan)
        Mxy = np.full(self.Xg.shape, np.nan)
        # У составных R-областей (r_and/r_or) ВТОРЫЕ производные ω имеют
        # линии излома (f₁ = f₂): гессиан структуры там не определён — точки
        # остаются NaN и маскируются в картах (NOTES §19).
        with np.errstate(invalid="ignore", divide="ignore"):
            if self.config.ortho_D is not None:            # ортотропия (v0.7.0)
                from .ladder import bending_moments_ortho

                mx, my, mxy = bending_moments_ortho(
                    solver.domain, solver.basis, self._c, p_struct,
                    self.config.ortho_D, self.Xg[inside], self.Yg[inside])
            elif getattr(self.config, "h_expr", None) is not None:
                # переменная толщина (v0.7.0): моменты с ЛОКАЛЬНОЙ D(x, y) —
                # формулы линейны по D: считаем с D=1 и домножаем поточечно
                mx, my, mxy = bending_moments_full(
                    solver.domain, solver.basis, self._c, p_struct,
                    1.0, self.config.nu, self.Xg[inside], self.Yg[inside])
                d_loc = self._local_D(self.Xg[inside], self.Yg[inside])
                mx, my, mxy = d_loc * mx, d_loc * my, d_loc * mxy
            else:
                mx, my, mxy = bending_moments_full(
                    solver.domain, solver.basis, self._c, p_struct,
                    self.config.D, self.config.nu, self.Xg[inside], self.Yg[inside])
        Mx[inside], My[inside], Mxy[inside] = mx, my, mxy
        if m_t != 0.0:
            Mx[inside] -= m_t
            My[inside] -= m_t
        return Mx, My, Mxy

    def _local_h(self, X, Y):
        """Толщина h(x, y) в точках (model.h_expr, v0.7.0)."""
        e_h = exprfield.parse_field("model.h_expr", self.config.h_expr)
        h = exprfield.field_values(e_h, np.asarray(X, float),
                                   np.asarray(Y, float), key="model.h_expr")
        return np.broadcast_to(np.asarray(h, float), np.shape(X)).copy()

    def _local_D(self, X, Y):
        """Жёсткость D(x, y) = E·h(x, y)³/12(1−ν²) в точках (v0.7.0)."""
        cfg = self.config
        return cfg.E * self._local_h(X, Y) ** 3 / (12.0 * (1.0 - cfg.nu**2))

    def _q_faces_on_grid(self):
        """(q_top, q_bottom) на сетке: нагрузка сверху, реакция снизу (§19).

        Ветвление по ВСЕМ типам нагрузки — общего else нет (у expr/line
        нет x0, их нельзя трактовать как точечное пятно).
        ``line`` и точная δ (``point exact``) — МЕРЫ, а не поверхностные
        плотности: q_top ≡ 0 (поправка обжатия ν/(1−ν)·q_n к ним неприменима).
        """
        load = self.problem.load
        inside = np.isfinite(self.w_grid)
        q_top = np.zeros(self.Xg.shape)
        if load.type == "uniform":
            q_top[inside] = self.config.q0
        elif load.type == "patch":
            zone = build_domain(load.zone)
            m = inside & (zone.omega(self.Xg, self.Yg) > 0.0)
            q_top[m] = self.config.q0
        elif load.type == "gaussian":
            d2 = (self.Xg - load.x0) ** 2 + (self.Yg - load.y0) ** 2
            q_top[inside] = (float(load.q0)
                             * np.exp(-d2 / (2.0 * load.sigma**2)))[inside]
        elif load.type == "expr":
            e = exprfield.parse_field("load.expr", load.expr)
            v = exprfield.field_values(e, self.Xg[inside], self.Yg[inside],
                                       key="load.expr")
            q_top[inside] = float(load.q0) * np.broadcast_to(
                np.asarray(v, float), self.Xg[inside].shape)
        elif load.type == "point" and not load.exact:    # пятно eps_eff
            eps = self.eps_eff if self.eps_eff is not None else 0.0
            m = inside & (((self.Xg - load.x0) ** 2 + (self.Yg - load.y0) ** 2)
                          <= eps**2)
            q_top[m] = self.config.q0
        # line | point exact: сосредоточенные меры — q_top остаётся нулевым
        q_bot = np.zeros(self.Xg.shape)
        if self.contact is not None:
            q_bot = np.nan_to_num(self.contact.r_grid, nan=0.0)
        return q_top, q_bot

    def faces_on_grid(self):
        """(w_top, w_bot, dh) на сетке — прогибы лицевых поверхностей.

        theory = ktn — КАНОН 21.1 (кинематика КТН, формулы кода, NOTES §21):
        w_bot = u_c = ktn.contact_displacement(w, Δw, q⁺, q⁻) — прогиб
        контактирующей (нижней) лицевой; Δw = −(Mx + My)/(D(1+ν)). Канон
        выделяет формулой только контактирующую лицевую: w_top ≡ w
        (срединная, без смешения канонов), dh = w_bot − w — каноническое
        смещение контактирующей лицевой (в зоне контакта dh < 0).
        theory = classic: обе лицевые ≡ срединной, dh ≡ 0. Классическое
        3D-восстановление — независимая диагностика (NOTES §21.2).
        """
        theory = None if self.problem.model is None else self.problem.model.theory
        if theory not in ("ktn_linear", "ktn_full"):
            return self.w_grid.copy(), self.w_grid.copy(), \
                np.zeros_like(self.w_grid)
        from .faces import FaceParams

        Mx, My, _ = self.moments_on_grid()
        q_top, q_bot = self._q_faces_on_grid()
        lap = -(Mx + My) / (self.config.D * (1.0 + self.config.nu))
        fp = FaceParams.from_config(self.config)
        # u_c контактирующей (нижней) лицевой — та же кинематика (§21.1); для
        # ktn_full прогиб w и кривизна Δw уже несут КТН-регуляризацию.
        w_bot = fp.face_deflection(self.w_grid, lap, q_top, q_bot, surface="bottom")
        return self.w_grid.copy(), w_bot, w_bot - self.w_grid

    def thickness_params(self) -> dict:
        """Интроспекция параметров толщины уточнённой теории (§6.3): h_ψ², h_*², h_c², h/L.

        Первоклассный вывод (faces.py) для ЛЮБОЙ теории: видна зависимость КТН-
        поправки от толщины. Характерный размер ``L`` — полуразмер bbox из сетки
        вывода; ``h/L`` и порядок ``(h/L)²`` показывают, как эффект гаснет при
        утоньшении (Gate R4).
        """
        from .faces import FaceParams

        gx, gy = self.Xg[0, :], self.Yg[:, 0]
        length = 0.5 * min(float(gx[-1] - gx[0]), float(gy[-1] - gy[0]))
        return FaceParams.from_config(self.config).introspection(length=length)

    def regrid(self, grid_n: int) -> Result:
        """Мгновенное уплотнение сетки ВЫВОДА без пересчёта решения.

        Все поля (w, моменты/σ через moments_on_grid, поверхности §21,
        r_grid и зона) пере-оцениваются на новой сетке из удержанных в
        памяти коэффициентов и узловой реакции; МОР НЕ перезапускается
        (contact.iters и residual_history остаются исходными). Числа
        решения (w_max, r_max, комплементарность) от grid_n не зависят —
        grid-зависима только диагностика топологии зоны. Держатель
        коэффициентов в result.json/fields.npz не пишется: у результата,
        восстановленного из файла, перегридовка недоступна — пере-решите
        постановку (`solve(problem, grid_n=…)`).
        """
        import copy as _copy
        import dataclasses as _dc

        if int(grid_n) < 2:
            raise CaseError(f"regrid: получено grid_n = {grid_n}, "
                            f"ожидалось целое ≥ 2, см. {_SCHEMA_DOC}#discretization")
        grid_n = int(grid_n)
        try:
            solver = self._plate
            c = self._c
        except AttributeError:
            raise RuntimeError(
                "regrid недоступен: результат без держателя коэффициентов "
                "(например, восстановлен из файла) — пере-решите постановку: "
                "solve(problem, grid_n=…)") from None
        cfg2 = _copy.copy(self.config)
        cfg2.grid_n = grid_n
        if self.contact is None:
            Xg, Yg, W = _grid_fields(
                solver.domain, cfg2,
                lambda X, Y: solver.deflection(c, X, Y))
            new = _dc.replace(self, config=cfg2, Xg=Xg, Yg=Yg, w_grid=W)
        elif hasattr(self.contact, "w2_grid"):            # пара пластин
            from .contact import sample_pair_fields_on_grid

            cres = self.contact
            Xg, Yg, w1, w2, rg, zone = sample_pair_fields_on_grid(
                cres.plate, cres.plate2, cres.cw, cres.cw2,
                cres.r_nodes, grid_n)
            new_c = _dc.replace(cres, Xg=Xg, Yg=Yg, w_grid=w1, w2_grid=w2,
                                r_grid=rg, contact_zone=zone)
            new = _dc.replace(self, config=cfg2, Xg=Xg, Yg=Yg, w_grid=w1,
                              contact=new_c)
        else:
            from .contact import sample_fields_on_grid

            cres = self.contact
            Xg, Yg, W, rg, zone = sample_fields_on_grid(
                cres.plate, cres.cw, cres.r_nodes, grid_n)
            new_c = _dc.replace(cres, Xg=Xg, Yg=Yg, w_grid=W, r_grid=rg,
                                contact_zone=zone)
            new = _dc.replace(self, config=cfg2, Xg=Xg, Yg=Yg, w_grid=W,
                              contact=new_c)
        for ref in ("_plate_ref", "_c_ref", "_plate2_ref", "_cfg2_ref",
                    "_force_calls", "_force_iters_total"):
            try:
                object.__setattr__(new, ref,
                                   object.__getattribute__(self, ref))
            except AttributeError:
                pass
        return new

    def save_fields(self, path) -> None:
        """fields.npz (версия схемы полей = 3): w, усилия M/N, σ-шестёрка+σ_vm, контакт.

        Схема 3 = схема 2 + мембранные усилия ``Nx, Ny, Nxy`` (нелинейные
        теории; входят и в лицевые σ мембранной частью ``N/h``) + эквивалентное
        ``svm_top, svm_bot`` (фон Мизес). Схема 2 = схема 1 + прогибы лицевых
        (w_top, w_bot, dh — NOTES §21) + для пары пластин поля ВТОРОЙ пластины
        (суффикс «2»; канон §19: у верхней q⁻ = r, у нижней q⁺ = r). Полный
        перечень ключей — docs/CASE_SCHEMA.md#output; всё необходимое для
        перерисовки фигур БЕЗ пересчёта — :func:`plate_solver.viz.replot`.
        """
        from .export import forces_on_grid
        from .ktn import stresses_faces, von_mises

        forces = forces_on_grid(self)
        Mx, My, Mxy = forces["Mx"], forces["My"], forces["Mxy"]
        has_N = "Nx" in forces                          # нелинейные теории (v0.6.6)
        Nz = np.zeros_like(Mx)
        Nx = np.nan_to_num(forces["Nx"], nan=0.0) if has_N else Nz
        Ny = np.nan_to_num(forces["Ny"], nan=0.0) if has_N else Nz
        Nxy = np.nan_to_num(forces["Nxy"], nan=0.0) if has_N else Nz
        q_top, q_bot = self._q_faces_on_grid()
        if self.config.ortho_D is not None:
            # ортотропия (v0.7.0): изотропная поправка обжатия ν/(1−ν)·q_n в
            # σ выведена из изотропного 3D-закона — для ортотропа опускается
            # (σ = 6M/h² от ОРТОТРОПНЫХ моментов)
            q_top = np.zeros_like(q_top)
            q_bot = np.zeros_like(q_bot)
        h_stress = self.config.h
        h_grid = None
        if getattr(self.config, "h_expr", None) is not None:
            # переменная толщина (v0.7.0): σ = 6M/h(x, y)² с ЛОКАЛЬНОЙ h;
            # вне Ω — NaN (выражение может быть не определено за областью)
            inside_h = np.isfinite(self.w_grid)
            h_grid = np.full(self.Xg.shape, np.nan)
            h_grid[inside_h] = self._local_h(self.Xg[inside_h],
                                             self.Yg[inside_h])
            h_stress = h_grid
        s = stresses_faces(Mx, My, Mxy, h=h_stress, nu=self.config.nu,
                           q_top=q_top, q_bottom=q_bot, Nx=Nx, Ny=Ny, Nxy=Nxy)
        w_top, w_bot, dh = self.faces_on_grid()
        payload = {
            "fields_schema": np.int64(3),
            "x": self.Xg[0, :], "y": self.Yg[:, 0],
            "w": self.w_grid, "Mx": Mx, "My": My, "Mxy": Mxy,
            "w_top": w_top, "w_bot": w_bot, "dh": dh,
            "problem_json": np.str_(json.dumps(asdict(self.problem),
                                               ensure_ascii=False)),
            "h": np.float64(self.config.h), "nu": np.float64(self.config.nu),
        }
        if h_grid is not None:
            payload["h_grid"] = h_grid            # опциональный ключ (h_expr)
        payload.update(s)
        payload["svm_top"] = von_mises(s["sx_top"], s["sy_top"], s["txy_top"])
        payload["svm_bot"] = von_mises(s["sx_bot"], s["sy_bot"], s["txy_bot"])
        from .export import shear_forces_on_grid
        payload.update(shear_forces_on_grid(self))       # Qx, Qy (равновесие, v0.6.6)
        if has_N:
            payload["Nx"], payload["Ny"], payload["Nxy"] = (
                forces["Nx"], forces["Ny"], forces["Nxy"])
        if self.contact is not None:
            payload["r"] = np.nan_to_num(self.contact.r_grid, nan=0.0)
            payload["zone"] = self.contact.contact_zone
            w2 = getattr(self.contact, "w2_grid", None)
            if w2 is not None:
                payload["w2"] = w2
                payload.update(self._second_plate_fields())
        if self.eigen is not None:
            # ВСЕ собственные формы (норм. max|·|=1) + значения — раньше в npz
            # попадала только первая форма (w), остальные терялись (v0.6.6)
            payload["eigen_values"] = np.asarray(self.eigen.values, float)
            payload["eigen_modes"] = np.stack(
                [self.eigen.mode_on_grid(i, grid_n=self.Xg.shape[1])[2]
                 for i in range(len(self.eigen.values))])
        np.savez_compressed(path, **payload)

    def _second_plate_fields(self) -> dict:
        """Моменты и σ-шестёрка НИЖНЕЙ пластины пары (канон §19).

        Реакция взаимодействия r приходит на ВЕРХНЮЮ лицевую нижней
        пластины: q⁺₂ = r, q⁻₂ = 0 (основания под второй нет).
        """
        from .ktn import stresses_faces
        from .ladder import bending_moments_full

        solver2, cfg2 = self._plate2, self._cfg2
        if solver2 is None or cfg2 is None:
            return {}
        inside2 = np.isfinite(self.contact.w2_grid)
        Mx2 = np.full(self.Xg.shape, np.nan)
        My2 = np.full(self.Xg.shape, np.nan)
        Mxy2 = np.full(self.Xg.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            if hasattr(solver2, "moments_at"):          # KarmanPlate/KTNSolver: своя структура
                mx, my, mxy = solver2.moments_at(
                    self.contact.cw2, self.Xg[inside2], self.Yg[inside2])
            else:
                p2 = 2 if hasattr(solver2, "S") else 1  # ω²Φ у защемления
                mx, my, mxy = bending_moments_full(
                    solver2.domain, solver2.basis, self.contact.cw2, p2,
                    cfg2.D, cfg2.nu, self.Xg[inside2], self.Yg[inside2])
        Mx2[inside2], My2[inside2], Mxy2[inside2] = mx, my, mxy
        r_fld = np.nan_to_num(self.contact.r_grid, nan=0.0)
        s2 = stresses_faces(Mx2, My2, Mxy2, h=cfg2.h, nu=cfg2.nu,
                            q_top=r_fld, q_bottom=0.0)
        out = {"Mx2": Mx2, "My2": My2, "Mxy2": Mxy2}
        out.update({k + "2": v for k, v in s2.items()})
        from .ktn import von_mises
        out["svm_top2"] = von_mises(s2["sx_top"], s2["sy_top"], s2["txy_top"])
        out["svm_bot2"] = von_mises(s2["sx_bot"], s2["sy_bot"], s2["txy_bot"])
        return out

    def _save_figures(self, out: Path, formats: tuple = ("png", "pdf"),
                      surface: str = "mid") -> None:
        """Фигуры публикационного качества (D1/D2): из fields.npz, png+pdf.

        Имена файлов получают префикс по имени case-файла (title кейса).
        """
        from . import viz

        stem = Path(self.problem.source).stem
        if self.eigen is not None:                       # собственные формы (§eigen)
            for fmt in formats:
                viz.plot_modes(self.eigen, n=len(self.eigen.values),
                               save=str(out / f"{stem or 'eigen'}_modes.{fmt}"))
            return
        paths = viz.replot(out, formats=formats, surface=surface)
        if stem and stem != "<dict>":
            for old in paths:
                new = old.with_name(f"{stem}_{old.name}")
                old.replace(new)
        if self.contact is not None:
            dest = str(out / f"{stem}_contact_summary.png")
            if hasattr(self.contact, "w2_grid"):        # пара пластин
                viz.plot_pair_summary(self.contact, save=dest)
            else:
                viz.plot_contact_summary(self.config, self.contact, save=dest)

    # солвер и коэффициенты для фигур (не сериализуются)
    @property
    def _plate(self):
        return object.__getattribute__(self, "_plate_ref")

    @property
    def _c(self):
        return object.__getattribute__(self, "_c_ref")

    @property
    def _plate2(self):
        try:
            return object.__getattribute__(self, "_plate2_ref")
        except AttributeError:
            return None

    @property
    def _cfg2(self):
        try:
            return object.__getattribute__(self, "_cfg2_ref")
        except AttributeError:
            return None


def _sanitize_nan(obj):
    """NaN/Inf → None рекурсивно (result.json обязан быть строгим JSON)."""
    if isinstance(obj, dict):
        return {k: _sanitize_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_nan(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _provenance() -> dict:
    """git-хеш и версии зависимостей для result.json."""
    import numpy
    import scipy
    import sympy

    from . import __version__

    try:
        git = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git = None
    return {
        "plate_solver": __version__,
        "git": git,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
    }


# --------------------------------------------------------------------------- #
#  Нагрузка и зоны в узлах квадратуры
# --------------------------------------------------------------------------- #
def _zone_mask(spec: GeometrySpec, quad, *, key: str, advice: str) -> np.ndarray:
    """Маска [ω_zone > 0] в узлах квадратуры с защитой ≥ MIN_ZONE_NODES узлов."""
    zone = build_domain(spec)
    mask = zone.omega(quad.x, quad.y) > 0.0
    n = int(mask.sum())
    if n < MIN_ZONE_NODES:
        raise CaseError(
            f"{key}: получено {n} узлов квадратуры в зоне, ожидалось "
            f"≥ {MIN_ZONE_NODES} — {advice}, см. {_SCHEMA_DOC}#load"
        )
    return mask


def _load_values(problem: Problem, dom, quad, warnings: list[str]):
    """Нагрузка в узлах квадратуры: (f_values | None, q0_eff, eps_eff).

    None означает «равномерная cfg.q0» (путь решателей без изменений).
    """
    return _load_values_spec(problem.load, dom, quad, warnings)


def _load_laplacian(load, quad) -> np.ndarray | None:
    """Лапласиан нагрузки Δq в узлах квадратуры для члена КТН (A) ``−h_*²Δq`` (§7).

    Аналитична для гладкой нагрузки: у гауссовой ``q = q0·exp(−r²/(2σ²))``
    ``Δq = q·(r²−2σ²)/σ⁴``; у ``expr`` — СИМВОЛЬНОЕ дифференцирование
    (``exprfield.field_laplacian``, точное). Для равномерной ``Δq ≡ 0``
    (член A исчезает). Разрывные нагрузки (patch/point) и негладкие выражения
    (abs/min/max) гладкой Δq не дают ⇒ ``None`` (член A опущен, отражено
    пометкой).
    """
    if load.type == "gaussian":
        s2 = load.sigma**2
        d2 = (quad.x - load.x0) ** 2 + (quad.y - load.y0) ** 2
        q = float(load.q0) * np.exp(-d2 / (2.0 * s2))
        return q * (d2 - 2.0 * s2) / s2**2
    if load.type == "uniform":
        return np.zeros(quad.x.size)
    if load.type == "expr":
        e = exprfield.parse_field("load.expr", load.expr)
        try:
            v = exprfield.field_values(exprfield.field_laplacian(e),
                                       quad.x, quad.y, key="load.expr (Δq)")
        except (ValueError, NotImplementedError, NameError, TypeError):
            return None                     # Δq негладкая — член A опущен
        if np.ndim(v) == 0:
            return float(load.q0) * float(v) * np.ones(quad.x.size)
        return float(load.q0) * v
    return None


def _load_values_spec(load, dom, quad, warnings: list[str]):
    """То же по произвольной LoadSpec (вторая пластина, A4)."""
    if load.type == "uniform":
        return None, float(load.q0), None

    if load.type == "patch":
        mask = _zone_mask(load.zone, quad, key="load.zone",
                          advice="увеличьте Q или зону нагрузки")
        return float(load.q0) * mask.astype(float), float(load.q0), None

    if load.type == "gaussian":
        # гладкая локализованная q = q0·exp(−r²/(2σ²)) (§7); q0_eff — амплитуда
        d2 = (quad.x - load.x0) ** 2 + (quad.y - load.y0) ** 2
        f = float(load.q0) * np.exp(-d2 / (2.0 * load.sigma**2))
        return f, float(load.q0), None

    if load.type == "expr":
        # нагрузка выражением q = q0·g(x, y) (v0.7.0); q0_eff — амплитуда
        # (семантика поправки ktn_linear и нормировок МОР — как у gaussian)
        e = exprfield.parse_field("load.expr", load.expr)
        try:
            v = exprfield.field_values(e, quad.x, quad.y, key="load.expr")
        except ValueError as err:
            raise CaseError(f"{err}, см. {_SCHEMA_DOC}#load") from None
        f = float(load.q0) * (np.full(quad.x.size, float(v))
                              if np.ndim(v) == 0 else v)
        return f, float(load.q0), None

    # point: регуляризованный patch, круговое пятно радиуса eps_eff
    x0, x1, y0, y1 = dom.bbox
    eps = load.eps if load.eps is not None else 0.05 * min(x1 - x0, y1 - y0)
    d2 = (quad.x - load.x0) ** 2 + (quad.y - load.y0) ** 2
    inside = d2 <= eps**2
    if int(inside.sum()) < MIN_ZONE_NODES:
        if d2.size < MIN_ZONE_NODES:
            raise CaseError(
                f"load.eps: получено {int(inside.sum())} узлов в пятне, ожидалось "
                f"≥ {MIN_ZONE_NODES} — увеличьте Q, см. {_SCHEMA_DOC}#load"
            )
        # авторасширение до наименьшего радиуса с ≥ MIN_ZONE_NODES узлами
        eps_eff = float(np.sqrt(np.partition(d2, MIN_ZONE_NODES - 1)[MIN_ZONE_NODES - 1]))
        eps_eff *= 1.0 + 1e-12                      # включить граничный узел
        warnings.append(
            f"load.eps: пятно расширено {eps:.4g} → {eps_eff:.4g} "
            f"(в исходном пятне < {MIN_ZONE_NODES} узлов квадратуры)"
        )
        eps = eps_eff
        inside = d2 <= eps**2
    q0_eff = float(load.P) / (np.pi * eps**2)       # q = P/(π·eps²)
    return q0_eff * inside.astype(float), q0_eff, float(eps)


# --------------------------------------------------------------------------- #
#  Диспетчер
# --------------------------------------------------------------------------- #
def solve(problem: Problem, grid_n: int | None = None) -> Result:
    """Решить постановку: маршрутизация по bc/contact/theory (см. докстринг модуля).

    ``grid_n`` — программный override сетки ВЫВОДА (эквивалент
    ``problem.with_discretization(grid_n=…)``): на числа решения не
    влияет, меняет только фоновую сетку полей и фигур.
    """
    if grid_n is not None:
        problem = problem.with_discretization(grid_n=grid_n)
    warnings: list[str] = []
    t0 = time.perf_counter()
    cfg = problem.to_config()
    dom = build_domain(problem.geometry)
    if problem.supports.points:                           # точечные опоры (v0.7.0)
        _check_support_points(problem, cfg, dom, warnings)
    if problem.model.h_expr is not None:                  # перем. толщина (v0.7.0)
        _check_h_expr(problem, cfg, dom, warnings)

    if problem.eigen is not None:                         # собственная задача (§eigen)
        t_build = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = _solve_eigen(problem, cfg, dom)
        object.__setattr__(result, "timings",
                           {"build": t_build, "solve": time.perf_counter() - t0})
        object.__setattr__(result, "eps_eff", None)
        return result

    nonlinear_contact = (problem.contact.enabled
                         and problem.model.theory in ("karman", "ktn_full"))
    if nonlinear_contact:                                 # нелинейный контакт МОР+КТН (v0.6.3)
        from .ktn_solver import KTNSolver

        # единый решатель КТН (пресет теории) — вокруг него работает
        # NonlinearContactMOR; bc = clamped (любая R-область) или soft_hinge
        # (circle | ellipse, граничный член §3.5) — гарантировано валидатором.
        solver = KTNSolver.from_theory_name(dom, cfg, problem.model.theory,
                                            bc_type=problem.bc.type,
                                            inplane_bc=problem.model.inplane_bc)
    elif problem.model.theory == "ktn_full":              # полная нелинейная КТН (v0.5)
        from .ktn_full import KTNPlate

        solver = KTNPlate.from_config(dom, cfg, bc_type=problem.bc.type,
                                      inplane_bc=problem.model.inplane_bc)
    elif problem.model.theory == "karman":                # геометрическая нелинейность (v0.4)
        from .membrane import KarmanPlate

        sides = dict(problem.bc.sides) if problem.bc.type == "mixed" else None
        solver = KarmanPlate.from_config(dom, cfg, bc_type=problem.bc.type,
                                         inplane_bc=problem.model.inplane_bc, sides=sides)
    elif problem.bc.type == "clamped":
        solver = ClampedPlate.from_config(dom, cfg)
    elif problem.bc.type == "mixed":                       # mixed (v0.3)
        from .clamped import MixedRectPlate

        g = problem.geometry
        solver = MixedRectPlate(g.x1, g.x2, g.y1, g.y2,
                                dict(problem.bc.sides), cfg)
    else:
        solver = PlateBending.from_config(dom, cfg)
    quad = solver.quad

    if problem.load.type == "line" or (problem.load.type == "point"
                                       and problem.load.exact):
        # линейная нагрузка / точная δ (v0.7.0): не площадная плотность —
        # готовый вектор собирается в трактах; cfg.q0 не трогается
        f_values, q0_eff, eps_eff = None, cfg.q0, None
    else:
        f_values, q0_eff, eps_eff = _load_values(problem, dom, quad, warnings)
    if (problem.model.theory == "ktn_linear" and problem.load.type == "expr"
            and f_values is not None and float(np.min(f_values)) < 0.0):
        # поправка ktn_linear использует СКАЛЯРНУЮ амплитуду q0 (как у
        # gaussian); для знакопеременной нагрузки константный сдвиг
        # cq·q0 не имеет смысла локально
        warnings.append(
            "ktn_linear: нагрузка expr знакопеременна — поправка сдвига/обжатия "
            "использует скалярную амплитуду q0 и для таких нагрузок не "
            "определена локально; интерпретируйте w_max_classic")
    if q0_eff != cfg.q0:
        cfg.q0 = q0_eff                     # эффективная интенсивность (КТН, МОР)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    if problem.contact.enabled:
        result = _solve_contact(problem, cfg, dom, solver, f_values, warnings)
    else:
        result = _solve_bending(problem, cfg, dom, solver, f_values, warnings)
    t_solve = time.perf_counter() - t0

    if problem.supports.points:
        # реакции опор R_j = k·w(P_j) (контакт с опорами отсечён валидатором)
        xs = np.array([pt[0] for pt in problem.supports.points])
        ys = np.array([pt[1] for pt in problem.supports.points])
        w_p = np.asarray(solver.deflection(result._c, xs, ys), float)
        object.__setattr__(result, "support_reactions",
                           tuple(float(cfg.supports_stiffness * v) for v in w_p))

    object.__setattr__(result, "timings", {"build": t_build, "solve": t_solve})
    object.__setattr__(result, "eps_eff", eps_eff)
    return result


def _uniform(cfg: Config, quad) -> np.ndarray:
    return np.full(quad.x.size, float(cfg.q0))


def _grid_fields(dom, cfg: Config, evaluate) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Фоновая сетка grid_n × grid_n: (Xg, Yg, w_grid с NaN вне Ω)."""
    x0, x1, y0, y1 = dom.bbox
    gx = np.linspace(x0, x1, cfg.grid_n)
    gy = np.linspace(y0, y1, cfg.grid_n)
    Xg, Yg = np.meshgrid(gx, gy)
    inside = dom.omega(Xg, Yg) > 0.0
    W = np.full(Xg.shape, np.nan)
    W[inside] = evaluate(Xg[inside], Yg[inside])
    return Xg, Yg, W


def _solve_eigen(problem, cfg, dom) -> Result:
    r"""Собственная задача (§eigen, v0.6.4): устойчивость или колебания.

    Линейный решатель (:func:`~plate_solver.eigenmodes.linear_plate`) собирает
    изгибную жёсткость / геометрическую матрицу / матрицу масс; ``buckling`` или
    ``natural_frequencies`` дают значения (λ_cr / ω) и формы. Первая форма (норм.
    на max|·|=1) кладётся в ``w_grid`` для фигуры; ``Result.eigen`` несёт весь
    :class:`~plate_solver.eigenmodes.EigenPair`.
    """
    from .eigenmodes import buckling, linear_plate, natural_frequencies

    spec = problem.eigen
    plate = linear_plate(dom, cfg, bc_type=problem.bc.type,
                         inplane_bc=problem.model.inplane_bc)
    warn: list[str] = []
    prestress = None
    if spec.prestress:
        # преднапряжение (v0.6.6): кармановское решение под [load] даёт N(w);
        # анализ линеен вокруг напряжённого состояния — K_eff = K + K_geo(N(w))
        kr = plate.solve(np.full(plate.quad.x.size, float(cfg.q0)))
        if not kr.converged:
            warn.append("eigen.prestress: кармановское решение не достигло tol — "
                        "поле N(w) полусошедшееся (увеличьте karman_max_iter)")
        prestress = kr
    try:
        if spec.kind == "buckling":
            if prestress is not None:
                eig = buckling(plate, prestress=prestress, n_modes=spec.n_modes)
            else:
                eig = buckling(plate, Nx=spec.Nx, Ny=spec.Ny, Nxy=spec.Nxy,
                               n_modes=spec.n_modes)
        else:
            eig = natural_frequencies(plate, rho_h=spec.rho_h, n_modes=spec.n_modes,
                                      prestress=prestress)
    except ValueError as e:                          # напр. НЕсжимающее преднапряжение
        raise CaseError(f"eigen: {e} см. {_SCHEMA_DOC}#eigen") from e
    if eig.values.size == 0:
        raise CaseError("eigen: собственных значений не найдено — увеличьте p/Q "
                        f"или проверьте постановку, см. {_SCHEMA_DOC}#eigen")
    if eig.values.size < spec.n_modes:
        warn.append(f"eigen: найдено {eig.values.size} мод из запрошенных "
                    f"{spec.n_modes} (ограничение базиса/дискретизации)")
    Xg, Yg, W = eig.mode_on_grid(0, cfg.grid_n)
    res = Result(problem=problem, config=cfg, w_max=float(eig.values[0]),
                 cond=float(np.linalg.cond(plate.D * plate._S_bend)),
                 Xg=Xg, Yg=Yg, w_grid=W, warnings=tuple(warn), eigen=eig)
    object.__setattr__(res, "_plate_ref", plate)
    object.__setattr__(res, "_c_ref", eig.modes[0])
    return res


def _solve_bending(problem, cfg, dom, solver, f_values, warnings) -> Result:
    f = _uniform(cfg, solver.quad) if f_values is None else f_values
    if problem.model.theory == "ktn_full":
        return _solve_ktn_full(problem, cfg, dom, solver, f, warnings)
    if problem.model.theory == "karman":
        return _solve_karman(problem, cfg, dom, solver, f, warnings)
    q = solver.quad
    # line / точная δ (v0.7.0): готовый вектор вместо площадной плотности
    # (недопустимые комбинации отсечены валидатором)
    line_b = _external_load_vector(problem, dom, solver)
    if problem.bc.type in ("clamped", "mixed"):
        c = solver.solve(f) if line_b is None else solver.solve_from_b(line_b)
        w_nodes = solver.deflection_at_quad(c)
        evaluate = lambda X, Y: solver.deflection(c, X, Y)      # noqa: E731
        cond = float(solver.cond)
        w_ktn = None
        if problem.model.theory == "ktn_linear":
            # ktn_linear при защемлении: кривизна из кэша Δ(ω²Φ) (A3.3)
            lap_w = solver.laplacian_at_quad(c)
            kp = KTNParams.from_config(cfg)
            w_ktn = kp.corrected_deflection(w_nodes, lap_w, cfg.q0,
                                            np.zeros(q.x.size))
    else:
        cM, cw = (solver.solve(f) if line_b is None
                  else solver.solve_from_b(line_b))
        w_nodes = solver.poisson.evaluate_at_quad(cw)
        evaluate = lambda X, Y: solver.deflection(cw, X, Y)     # noqa: E731
        c = cw
        cond = float(solver.poisson.cond)
        w_ktn = None
        if problem.model.theory == "ktn_linear":
            # изгиб без контакта: corrected_deflection при r = 0
            lap_w = -solver.poisson.evaluate_at_quad(cM) / solver.D
            kp = KTNParams.from_config(cfg)
            w_ktn = kp.corrected_deflection(w_nodes, lap_w, cfg.q0,
                                            np.zeros(q.x.size))
    Xg, Yg, W = _grid_fields(dom, cfg, evaluate)
    w_max_classic = float(np.max(np.abs(w_nodes)))
    w_max = w_max_classic if w_ktn is None else float(np.max(np.abs(w_ktn)))
    res = Result(problem=problem, config=cfg, w_max=w_max, cond=cond,
                 Xg=Xg, Yg=Yg, w_grid=W, warnings=tuple(warnings),
                 w_nodes=w_nodes,
                 w_max_classic=w_max_classic if w_ktn is not None else None)
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", c)
    return res


def _solve_karman(problem, cfg, dom, solver, f, warnings) -> Result:
    r"""Геометрически-нелинейный тракт (теория Кармана, §5).

    :class:`~plate_solver.membrane.KarmanPlate` решает систему Фёппля–Кармана
    итерацией Пикара с шагами по нагрузке; ``Result`` несёт нелинейный
    ``w_max``, линейный ``w_max_classic`` (как для ``ktn``) и полный
    :class:`~plate_solver.membrane.KarmanResult` (история сходимости,
    мембранные усилия) в поле ``_karman_ref`` — для ноутбука и анализа.
    Несошедшаяся итерация — предупреждение в ``result.json`` (не ошибка).
    """
    b_ext = _external_load_vector(problem, dom, solver)
    if b_ext is not None:
        # line / точная δ (v0.7.0): нулевая площадная плотность + готовый
        # вектор b_extra (наращивается теми же n_load_steps)
        kr = solver.solve(np.zeros(solver.quad.x.size), b_extra=b_ext)
    else:
        kr = solver.solve(f)
    c = kr.cw
    w_nodes = kr.w_nodes
    warn = list(warnings)
    if not kr.converged:
        last = kr.history[-1][2] if kr.history else float("nan")
        warn.append(
            f"karman: итерация ({getattr(cfg, 'karman_method', 'picard')}) не "
            f"достигла karman_tol за karman_max_iter "
            f"(последняя относительная невязка {last:.2e}); увеличьте "
            f"karman_max_iter или n_load_steps")
    Xg, Yg, W = _grid_fields(dom, cfg, lambda X, Y: solver.deflection(c, X, Y))
    res = Result(problem=problem, config=cfg, w_max=kr.w_max,
                 cond=float(solver.cond), Xg=Xg, Yg=Yg, w_grid=W,
                 warnings=tuple(warn), w_nodes=w_nodes,
                 w_max_classic=kr.w_max_classic)
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", c)
    object.__setattr__(res, "_karman_ref", kr)
    return res


def _solve_ktn_full(problem, cfg, dom, solver, f, warnings) -> Result:
    r"""Полная нелинейная КТН (§3, §5): :class:`~plate_solver.ktn_full.KTNPlate`.

    КТН-члены (A), (B) собираются поверх мембранного решателя Кармана; ``Result``
    несёт нелинейный ``w_max``, линейный ``w_max_classic``, историю сходимости
    (``_karman_ref``) и параметры толщины (``thickness_params``). Лицевые
    величины — ``faces_on_grid``/``save_fields`` (§6). Несошедшаяся итерация —
    предупреждение (не ошибка).
    """
    warn = list(warnings)
    lap_q = _load_laplacian(problem.load, solver.quad)   # член (A): −h_*²Δq (§7)
    solver.set_load_laplacian(lap_q)
    if lap_q is None:
        warn.append(
            f"ktn_full: член −h_*²Δq опущен для load.type = '{problem.load.type}' "
            "(Δq не гладкая); аналитический Δq есть у uniform (Δq=0), gaussian "
            "и гладких expr")
    kr = solver.solve(f)
    c = kr.cw
    if not kr.converged:
        last = kr.history[-1][2] if kr.history else float("nan")
        warn.append(
            f"ktn_full: итерация ({getattr(cfg, 'ktn_method', 'picard')}) не "
            f"достигла порога за karman_max_iter "
            f"(последняя относительная невязка {last:.2e}); увеличьте "
            f"karman_max_iter или n_load_steps")
    Xg, Yg, W = _grid_fields(dom, cfg, lambda X, Y: solver.deflection(c, X, Y))
    res = Result(problem=problem, config=cfg, w_max=kr.w_max,
                 cond=float(solver.cond), Xg=Xg, Yg=Yg, w_grid=W,
                 warnings=tuple(warn), w_nodes=kr.w_nodes,
                 w_max_classic=kr.w_max_classic)
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", c)
    object.__setattr__(res, "_karman_ref", kr)
    return res


def _line_load_vector(load, dom, solver, n_gauss: int = 96) -> np.ndarray:
    r"""Вектор линейной нагрузки (v0.7.0): ``b_i = P·∫_seg ψ_i ds``.

    1D Гаусс–Лежандр по отрезку (подынтегральная — гладкая структура
    ``ω^m·T_k`` вдоль отрезка: n = 32 узла дают уже ~1e-13, 96 — с запасом;
    внутренняя константа, НЕ ключ схемы). Ограды: узел отрезка вне Ω — отказ;
    отрезок целиком на ∂Ω (ψ ≡ 0) — отказ (иначе решение МОЛЧА нулевое);
    концы НА границе допустимы (узлы Гаусса открытые).
    """
    x0, y0 = load.p0
    x1, y1 = load.p1
    t, wg = np.polynomial.legendre.leggauss(n_gauss)
    xs = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * t
    ys = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * t
    length = float(np.hypot(x1 - x0, y1 - y0))
    # ограда «в Ω» — на ПЛОТНОЙ равномерной выборке (узлы Гаусса редки в
    # середине — узкий вырез мог бы проскочить между ними)
    td = np.linspace(-1.0, 1.0, 513)
    om_dense = np.asarray(dom.omega(0.5 * (x0 + x1) + 0.5 * (x1 - x0) * td,
                                    0.5 * (y0 + y1) + 0.5 * (y1 - y0) * td),
                          float)
    om = np.concatenate([np.asarray(dom.omega(xs, ys), float), om_dense[1:-1]])
    bx0, bx1, by0, by1 = dom.bbox
    gx, gy = np.meshgrid(np.linspace(bx0, bx1, 65), np.linspace(by0, by1, 65))
    om_scale = float(np.max(dom.omega(gx, gy)))
    if float(np.min(om)) < 0.0:
        raise CaseError(
            f"load.p0/p1: отрезок [{load.p0} — {load.p1}] выходит за область "
            f"(ω < 0 на узлах), см. {_SCHEMA_DOC}#load")
    if float(np.max(om)) <= 1e-9 * om_scale:
        raise CaseError(
            f"load.p0/p1: отрезок [{load.p0} — {load.p1}] лежит на границе "
            f"(ψ ≡ 0 — прогиб был бы тождественно нулевым), "
            f"см. {_SCHEMA_DOC}#load")
    psi = np.asarray(solver.structure_at(xs, ys), float)
    return float(load.intensity) * (length / 2.0) * (psi @ wg)


def _check_h_expr(problem, cfg, dom, warnings: list) -> None:
    """Проверка толщины h(x, y) (v0.7.0): h > 0 на Ω; предупреждение о
    контрасте (обусловленность ~ (h_max/h_min)³)."""
    e_h = exprfield.parse_field("model.h_expr", problem.model.h_expr)
    x0, x1, y0, y1 = dom.bbox
    gx, gy = np.meshgrid(np.linspace(x0, x1, 97), np.linspace(y0, y1, 97))
    m = np.asarray(dom.omega(gx, gy), float) > 0.0
    try:
        h = exprfield.field_values(e_h, gx[m], gy[m], key="model.h_expr")
    except ValueError as err:
        raise CaseError(f"{err}, см. {_SCHEMA_DOC}#model") from None
    h = np.broadcast_to(np.asarray(h, float), gx[m].shape)
    h_min, h_max = float(np.min(h)), float(np.max(h))
    if h_min <= 0.0:
        j = int(np.argmin(h))
        raise CaseError(
            f"model.h_expr: h = {h_min:.4g} ≤ 0 в точке ({gx[m][j]:.4g}, "
            f"{gy[m][j]:.4g}) — толщина обязана быть положительной, "
            f"см. {_SCHEMA_DOC}#model")
    if h_max / h_min > 3.0:
        warnings.append(
            f"model.h_expr: контраст толщины h_max/h_min = {h_max / h_min:.2f} "
            "> 3 — обусловленность растёт ~ (h_max/h_min)³, следите за cond_A")


def _point_load_vector(load, dom, solver) -> np.ndarray:
    r"""Вектор ТОЧНОЙ δ-силы (v0.7.0): ``b_i = P·ψ_i(x0, y0)``.

    Точечное вычисление ограничено на H² в 2D (дискретизация точная, без
    квадратуры); сходимость решения по базису — алгебраическая, ~p⁻¹·⁵
    в точке приложения (функция Грина ~r²ln r; у точечных ОПОР показатель
    мягче, ~p⁻², — там особенность в реакции, а не в поле нагрузки).
    Точка обязана лежать
    внутри Ω (на границе ψ ≡ 0 — сила ушла бы в опору тождественно).
    """
    om = float(np.asarray(dom.omega(np.array([load.x0]),
                                    np.array([load.y0])), float)[0])
    if not om > 0.0:
        raise CaseError(
            f"load.x0/y0: точка ({load.x0}, {load.y0}) вне области или на "
            f"границе (ω ≤ 0) при exact = true, см. {_SCHEMA_DOC}#load")
    psi = np.asarray(solver.structure_at(np.array([float(load.x0)]),
                                         np.array([float(load.y0)])), float)[:, 0]
    return float(load.P) * psi


def _external_load_vector(problem, dom, solver):
    """Готовый вектор нагрузки вне площадной квадратуры: line | точная δ."""
    if problem.load.type == "line":
        return _line_load_vector(problem.load, dom, solver)
    if problem.load.type == "point" and problem.load.exact:
        return _point_load_vector(problem.load, dom, solver)
    return None


def _check_pair_gap_expr(spec, gap_val) -> None:
    """Зазор пары ≥ 0 для expr-поля (v0.7.0; зеркало скалярного правила).

    У унаследованных kind plane/steps проверки min Δ ≥ 0 в парном тракте
    исторически нет — их поведение сознательно сохранено; новая
    проверка — только для expr.
    """
    if (spec.kind == "expr" and np.ndim(gap_val) > 0
            and float(np.min(gap_val)) < 0.0):
        raise CaseError(
            f"contact.gap_expr: min Δ = {float(np.min(gap_val)):.3g} < 0 "
            f"между пластинами (зазор пары ≥ 0; Δ = 0 — касание), "
            f"см. {_SCHEMA_DOC}#contact")


def _check_support_points(problem, cfg, dom, warnings: list) -> None:
    """Проверка точек [supports] (v0.7.0): внутри Ω; предупреждения у кромки/при
    экстремальной жёсткости (риск потери ПД — МНК-fallback)."""
    pts = problem.supports.points
    xs = np.array([pt[0] for pt in pts])
    ys = np.array([pt[1] for pt in pts])
    om = np.asarray(dom.omega(xs, ys), float)
    x0, x1, y0, y1 = dom.bbox
    gx, gy = np.meshgrid(np.linspace(x0, x1, 65), np.linspace(y0, y1, 65))
    om_max = float(np.max(dom.omega(gx, gy)))
    for i, (o, pt) in enumerate(zip(om, pts, strict=True)):
        if not o > 0.0:
            raise CaseError(
                f"supports.points[{i}]: точка {pt} вне области или на границе "
                f"(ω ≤ 0) — опора на кромке уже содержится в КУ, "
                f"см. {_SCHEMA_DOC}#supports")
        if o < 0.05 * om_max:
            warnings.append(
                f"supports.points[{i}]: опора {pt} у кромки — эффективная "
                f"жёсткость гаснет как k·ω^(2m)(P)")
    k = float(cfg.supports_stiffness)
    scale = cfg.D / min(x1 - x0, y1 - y0) ** 3
    if k >= 1e12 * scale:
        # при k ~ 1e14·D МНК-fallback молча искажает реакции на десятки
        # процентов — жёсткий отказ вместо тихой деградации
        raise CaseError(
            f"supports.stiffness = {k:.3g} ≥ 1e12·D/a³ — матрица теряет "
            "положительную определённость, МНК-fallback искажает реакции; "
            f"жёсткая опора достигается уже при k ≈ 1e6·D/a³, "
            f"см. {_SCHEMA_DOC}#supports")
    if k >= 1e8 * scale:
        warnings.append(
            f"supports.stiffness = {k:.3g} ≥ 1e8·D/a³ — риск потери "
            "положительной определённости (МНК-fallback, потеря точности "
            "реакции ~1e-4); жёсткая опора достигается уже при k ≈ 1e6·D/a³")


def _gap_field_values(spec, quad):
    """Поле зазора Δ(x, y) в узлах квадратуры по GapSpec (A1.2); const → скаляр.

    ``const`` возвращает СКАЛЯР — путь и арифметика тождественны скалярному
    ``gap`` v0.2 (ворота-тождество A1.3-т1). Константный ``expr`` также
    редуцируется в скаляр (тот же путь бит-точно).
    """
    if spec.kind == "const":
        return spec.value
    if spec.kind == "plane":
        return spec.a * quad.x + spec.b * quad.y + spec.c
    if spec.kind == "paraboloid":
        return spec.apex + ((quad.x - spec.cx) ** 2 + (quad.y - spec.cy) ** 2) \
            / (2.0 * spec.r_curv)
    if spec.kind == "expr":                             # Δ = f(x, y) (v0.7.0)
        e = exprfield.parse_field("contact.gap_expr", spec.expr)
        try:
            return exprfield.field_values(e, quad.x, quad.y,
                                          key="contact.gap_expr")
        except ValueError as err:
            raise CaseError(f"{err}, см. {_SCHEMA_DOC}#contact") from None
    g = np.full(quad.x.size, float(spec.base))          # steps
    for zone_geom, value in spec.zones:
        zdom = build_domain(zone_geom)
        g[zdom.omega(quad.x, quad.y) > 0.0] = value
    return g


def _cond_of(solver) -> float:
    """cond(A) решателя: ClampedPlate.cond либо PlateBending.poisson.cond."""
    return float(solver.cond if hasattr(solver, "cond") else solver.poisson.cond)


def _solve_contact(problem, cfg, dom, solver, f_values, warnings) -> Result:
    # Нелинейный контакт МОР+КТН (karman | ktn_full, v0.6.3): отдельный тракт
    # вокруг единого решателя КТН (лицевое условие Синьорини, §4).
    if problem.model.theory in ("karman", "ktn_full"):
        if problem.contact.target == "plate2":
            return _solve_two_plates_nonlinear(problem, cfg, dom, solver, warnings)
        if problem.contact.force is not None:
            return _solve_contact_force_nonlinear(problem, cfg, dom, solver, warnings)
        return _solve_contact_nonlinear(problem, cfg, dom, solver, f_values,
                                        warnings)
    q = solver.quad
    # Δ: скаляр gap | gap_factor·w_free | поле [contact.gap] (A1)
    f = _uniform(cfg, q) if f_values is None else f_values
    state_free = solver.solve(f)
    w_free = float(np.max(np.abs(solver.w_at_quad(state_free))))
    c = problem.contact
    fmask = None
    zone_mask = None
    if problem.contact.zone is not None:
        zone_mask = _zone_mask(problem.contact.zone, q, key="contact.zone",
                               advice="увеличьте Q или зону")
        fmask = lambda X, Y: zone_mask                          # noqa: E731

    if c.target == "plate2":                                    # пара пластин (A4)
        return _solve_two_plates(problem, cfg, dom, solver, f_values, warnings,
                                 w_free, zone_mask)

    if c.force is not None:                                     # силовой штамп (A2)
        return _solve_contact_force(problem, cfg, dom, solver, f_values, warnings,
                                    state_free, w_free, zone_mask, fmask)

    if c.gap is not None:
        delta_val = c.gap
    elif c.gap_factor is not None:
        delta_val = c.gap_factor * w_free
    else:
        delta_val = _gap_field_values(c.gap_field, q)

    if np.ndim(delta_val) == 0:
        delta = float(delta_val)
        if delta <= 0:
            raise CaseError(f"contact.gap: получено Δ = {delta:.3g}, ожидалось > 0, "
                            f"см. {_SCHEMA_DOC}#contact")
    else:
        fm = zone_mask if zone_mask is not None else np.ones(q.x.size, dtype=bool)
        gmin = float(np.min(delta_val[fm]))
        if gmin <= 0:
            raise CaseError(
                f"contact.gap: получено min Δ = {gmin:.3g} на основании, ожидалось "
                f"> 0 (поле зазора не должно пересекать пластину), "
                f"см. {_SCHEMA_DOC}#contact")
        delta = gmin                                     # скаляр для отчёта: min Δ

    ktn = KTNParams.from_config(cfg) if problem.model.theory == "ktn_linear" else None
    mor = ContactMOR(solver, cfg, foundation_mask=fmask, gap=delta_val, ktn=ktn,
                     load_values=f_values)
    cres = mor.solve()
    warn = list(warnings)
    if not cres.converged:
        # несходимость МОР — явное предупреждение (симметрично нелинейным
        # трактам): при r ≡ 0 результат неотличим от свободного решения
        last = float(cres.residual_history[-1]) if len(cres.residual_history) else float("nan")
        warn.append(
            f"контакт: МОР не достиг порога за max_iter (последняя невязка "
            f"{last:.2e}); увеличьте max_iter/подберите beta — при r ≡ 0 "
            "результат может совпадать со СВОБОДНЫМ решением")
    w_nodes = cres.w_ktn_nodes if cres.w_ktn_nodes is not None else cres.w_nodes
    res = Result(problem=problem, config=cfg, w_max=float(np.max(np.abs(w_nodes))),
                 cond=_cond_of(solver), Xg=cres.Xg, Yg=cres.Yg,
                 w_grid=cres.w_grid, warnings=tuple(warn), contact=cres,
                 delta=float(delta), w_free_max=w_free,
                 w_max_classic=float(np.max(np.abs(cres.w_nodes))))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", cres.cw)
    return res


def _contact_metrics_nl(r, disp, gap, gap_ref, q0):
    r"""Безразмерные метрики Синьорини нелинейного контакта (Δ > 0, §4.1).

    Совпадают с :meth:`ContactMOR._complementarity`:
    ``comp = max|r·(u−Δ)| / (q0·Δ_ref)``; ``overshoot = max_{r>0}(u−Δ)/Δ_ref``,
    где ``u`` — ЛИЦЕВОЕ смещение ``u_c`` (условие контакта по грани).
    """
    comp = float(np.max(np.abs(r * (disp - gap))) / (q0 * gap_ref))
    contact = r > 0.0
    over = (float(np.max((disp - gap)[contact])) / gap_ref
            if contact.any() else float("nan"))
    return comp, over


def _solve_contact_nonlinear(problem, cfg, dom, solver, f_values,
                             warnings) -> Result:
    r"""Нелинейный контакт МОР поверх КТН (§4): karman | ktn_full на защемлении.

    ``solver`` — единый :class:`~plate_solver.ktn_solver.KTNSolver` (пресет
    теории). Реакция ``r`` определяется вокруг ПОЛНОГО нелинейного решателя КТН
    (:class:`~plate_solver.contact_nl.NonlinearContactMOR`); контакт «щупает»
    ЛИЦЕВУЮ поверхность (условие Синьорини на грани, ``u_c = w + (h_c²−h_*²)Δw``;
    для ``karman`` коэффициент кривизны ноль ⇒ по срединной). Нагрузка —
    равномерная либо гладкое поле gaussian/expr (v0.7.0; гарантировано
    валидатором; нормировка усиления — по амплитуде cfg.q0, как в классическом
    МОР); зазор Δ(x, y) и зона препятствия — как в линейном контакте. Результат
    адаптируется в :class:`ContactResult` (тот же пайплайн сетки/VTK/фигур).
    """
    from .contact import sample_fields_on_grid
    from .contact_nl import NonlinearContactMOR

    q = solver.quad
    c = problem.contact
    warn = list(warnings)
    if problem.model.theory == "ktn_full" and problem.load.type != "uniform":
        warn.append(
            "ktn_full+контакт: член −h_*²Δq в контактном тракте опущен "
            "(для uniform он тождественно нулевой; для gaussian/expr — "
            "приближение порядка h_*²)")

    # свободное (без контакта) нелинейное решение: w_free для gap_factor/отчёта.
    f_arr = (np.full(q.x.size, float(cfg.q0)) if f_values is None
             else np.asarray(f_values, float))
    free = solver.solve(f_arr)
    w_free = float(np.max(np.abs(free.w_nodes)))

    # зона основания (дефолт — вся Ω); предикат для NonlinearContactMOR.
    zone_mask = None
    fmask = None
    if c.zone is not None:
        zone_mask = _zone_mask(c.zone, q, key="contact.zone",
                               advice="увеличьте Q или зону")
        fmask = lambda X, Y: zone_mask                          # noqa: E731

    # зазор Δ: скаляр gap | gap_factor·w_free | поле [contact.gap] (A1)
    if c.gap is not None:
        gap_val = float(c.gap)
    elif c.gap_factor is not None:
        gap_val = c.gap_factor * w_free
    else:
        gap_val = _gap_field_values(c.gap_field, q)
    if np.ndim(gap_val) == 0:
        delta = float(gap_val)
        if delta <= 0:
            raise CaseError(f"contact.gap: получено Δ = {delta:.3g}, ожидалось > 0, "
                            f"см. {_SCHEMA_DOC}#contact")
    else:
        fm = zone_mask if zone_mask is not None else np.ones(q.x.size, dtype=bool)
        gmin = float(np.min(gap_val[fm]))
        if gmin <= 0:
            raise CaseError(
                f"contact.gap: получено min Δ = {gmin:.3g} на основании, ожидалось "
                f"> 0 (поле зазора не должно пересекать пластину), "
                f"см. {_SCHEMA_DOC}#contact")
        delta = gmin

    mor = NonlinearContactMOR(solver, cfg, gap=gap_val, foundation_mask=fmask,
                              scheme=cfg.contact_scheme, gain_mode=cfg.contact_gain,
                              f_values=f_values)
    nres = mor.solve()
    if not nres.converged:
        last = float(nres.residual_history[-1]) if nres.residual_history.size else float("nan")
        warn.append(
            f"{problem.model.theory}: нелинейный контакт МОР ({nres.scheme}) не "
            f"достиг tol за max_iter (последняя невязка {last:.2e}); увеличьте "
            "contact.max_iter, смените contact.gain на linear или scheme на nested")

    # адаптация NonlinearContactResult → ContactResult (пайплайн вывода)
    Xg, Yg, w_grid, r_grid, zone = sample_fields_on_grid(
        solver, nres.cw, nres.r_nodes, cfg.grid_n)
    comp, overshoot = _contact_metrics_nl(nres.r_nodes, nres.u_c_nodes, gap_val,
                                          delta, float(cfg.q0))
    cres = ContactResult(
        Xg=Xg, Yg=Yg, w_grid=w_grid, r_grid=r_grid, contact_zone=zone,
        r_nodes=nres.r_nodes, w_nodes=nres.w_nodes, iters=nres.iters,
        converged=nres.converged, residual_history=nres.residual_history,
        peak_xy=nres.peak_xy, plate=solver, cw=nres.cw, w_ktn_nodes=None,
        comp_residual=comp, gap_overshoot=overshoot)
    res = Result(problem=problem, config=cfg,
                 w_max=float(np.max(np.abs(nres.w_nodes))),
                 cond=_cond_of(solver), Xg=Xg, Yg=Yg, w_grid=w_grid,
                 warnings=tuple(warn), contact=cres, delta=float(delta),
                 w_free_max=w_free, w_max_classic=float(free.w_max_classic))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", nres.cw)
    if nres.cu is not None:                              # мембрана сошедшегося состояния
        object.__setattr__(res, "_karman_ref",
                           SimpleNamespace(cu=nres.cu, cv=nres.cv, cw=nres.cw))
    return res


def _solve_contact_force_nonlinear(problem, cfg, dom, solver, warnings) -> Result:
    r"""Силовой штамп поверх нелинейной КТН (§4 + A2): задана сила P, ищется уровень.

    ``F(level) = ∫r − P`` монотонно убывает по уровню штампа; brentq на
    ``[касание, отрыв]``. Каждый вызов — полный
    :class:`~plate_solver.contact_nl.NonlinearContactMOR` (стартует со
    свободного нелинейного решения; тёплого старта между уровнями нет). Форма
    штампа — из ``[contact.gap]`` (нет ⇒ плоский). Мягкий шарнир требует
    ``gain = "linear"`` (как позиционный контакт).
    """
    from scipy.optimize import brentq

    from .contact import sample_fields_on_grid
    from .contact_nl import NonlinearContactMOR

    q = solver.quad
    c = problem.contact
    warn = list(warnings)
    P = float(c.force)

    free = solver.solve(np.full(q.x.size, float(cfg.q0)))
    w_free = float(np.max(np.abs(free.w_nodes)))
    w_free_nodes = free.w_nodes

    zone_mask = None
    fmask = None
    if c.zone is not None:
        zone_mask = _zone_mask(c.zone, q, key="contact.zone", advice="увеличьте Q или зону")
        fmask = lambda X, Y: zone_mask                          # noqa: E731
    fm = zone_mask if zone_mask is not None else np.ones(q.x.size, dtype=bool)

    shape = _gap_field_values(c.gap_field, q) if c.gap_field is not None else 0.0
    if np.ndim(shape) == 0:
        shape = float(shape)
    if c.gap is not None or c.gap_factor is not None:
        warn.append("contact.gap: скалярный gap/gap_factor игнорируется в силовом "
                    "режиме (force); форма штампа — [contact.gap]")

    shape_fm = shape[fm] if np.ndim(shape) else shape
    level_hi = float(np.max(w_free_nodes[fm] - shape_fm)) * (1.0 + 1e-9) + 1e-30
    scale = w_free
    min_shape = float(np.min(shape_fm)) if np.ndim(shape) else shape
    level_lo = 1e-8 * scale - min_shape

    st = {"nres": None, "iters": 0, "calls": 0}

    # Андерсон в силовом режиме ОТКЛЮЧАЕТСЯ: смешение истории делает останов
    # МОР полусошедшимся квазислучайно ⇒ скаляр F(level) = ∫r − P зашумлён
    # (~7–10 % P) и brentq садится мимо цели МОЛЧА (аудит v0.6.5). Каждый
    # уровень здесь — свежий холодный прогон, Андерсон выигрыша и не давал.
    if int(getattr(cfg, "mor_anderson", 0)) > 0:
        import dataclasses

        warn.append("contact.mor_anderson: отключён в силовом режиме одиночного "
                    "штампа — смешение Андерсона зашумляет F(level) = ∫r − P и "
                    "срывает точность brentq (аудит v0.6.5)")
        cfg = dataclasses.replace(cfg, mor_anderson=0)

    def F(level: float) -> float:
        gap_val = (level + shape) if np.ndim(shape) else float(level + shape)
        nres = NonlinearContactMOR(solver, cfg, gap=gap_val, foundation_mask=fmask,
                                   scheme=cfg.contact_scheme,
                                   gain_mode=cfg.contact_gain).solve()
        st["nres"] = nres
        st["iters"] += nres.iters
        st["calls"] += 1
        return float(np.sum(q.w * nres.r_nodes)) - P

    if F(level_lo) <= 0.0:
        raise CaseError(
            f"contact.force: получено P = {P:g}, ожидалось 0 < P ≤ "
            f"{float(np.sum(q.w * st['nres'].r_nodes)):.6g} (максимум ∫r при "
            f"касании), см. {_SCHEMA_DOC}#contact")
    level_star = brentq(F, level_lo, level_hi, xtol=1e-8 * scale)
    F(level_star)
    nres = st["nres"]
    force_total = float(np.sum(q.w * nres.r_nodes))
    if not nres.converged:
        warn.append(f"{problem.model.theory}: силовой нелинейный контакт — МОР не "
                    "достиг tol (полусходимость); точность ∫r ограничена")

    Xg, Yg, w_grid, r_grid, zone = sample_fields_on_grid(
        solver, nres.cw, nres.r_nodes, cfg.grid_n)
    gap_star = (level_star + shape) if np.ndim(shape) else float(level_star + shape)
    gap_ref = float(np.min(gap_star)) if np.ndim(gap_star) else abs(gap_star) or 1.0
    comp, over = _contact_metrics_nl(nres.r_nodes, nres.u_c_nodes, gap_star,
                                     gap_ref, float(cfg.q0))
    cres = ContactResult(
        Xg=Xg, Yg=Yg, w_grid=w_grid, r_grid=r_grid, contact_zone=zone,
        r_nodes=nres.r_nodes, w_nodes=nres.w_nodes, iters=nres.iters,
        converged=nres.converged, residual_history=nres.residual_history,
        peak_xy=nres.peak_xy, plate=solver, cw=nres.cw, w_ktn_nodes=None,
        comp_residual=comp, gap_overshoot=over)
    res = Result(problem=problem, config=cfg,
                 w_max=float(np.max(np.abs(nres.w_nodes))),
                 cond=_cond_of(solver), Xg=Xg, Yg=Yg, w_grid=w_grid,
                 warnings=tuple(warn), contact=cres, delta=float(level_star + min_shape),
                 w_free_max=w_free, level=float(level_star), force_total=force_total,
                 w_max_classic=float(free.w_max_classic))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", nres.cw)
    if nres.cu is not None:                              # мембрана сошедшегося состояния
        object.__setattr__(res, "_karman_ref",
                           SimpleNamespace(cu=nres.cu, cv=nres.cv, cw=nres.cw))
    object.__setattr__(res, "_force_calls", st["calls"])
    object.__setattr__(res, "_force_iters_total", st["iters"])
    return res


def _plate2_solver(problem: Problem, cfg: Config, dom, *, nonlinear_theory=None):
    """Построить решатель второй пластины по [plate2] (дефолты — от первой).

    ``nonlinear_theory`` (karman | ktn_full) ⇒ единый :class:`KTNSolver` для
    нелинейной пары (§9.2, защемление); иначе — линейный решатель (v0.3).
    """
    import dataclasses

    p2 = problem.plate2
    dom2 = build_domain(p2.geometry) if p2.geometry is not None else dom
    kw = {}
    model2 = p2.model
    if model2 is not None:
        for attr in ("E", "nu", "h"):
            v = getattr(model2, attr)
            if v is not None:
                kw[attr] = v
    disc2 = p2.discretization
    if disc2 is not None:
        for attr in ("p", "Q", "grid_n"):
            v = getattr(disc2, attr)
            if v is not None:
                kw[attr] = v
    if p2.load.q0 is not None:
        kw["q0"] = p2.load.q0
    cfg2 = dataclasses.replace(cfg, **kw)
    if nonlinear_theory is not None:
        from .ktn_solver import KTNSolver

        inplane2 = model2.inplane_bc if model2 is not None else "immovable"
        # изгибная кромка второй пластины — из [plate2.bc] (по умолч. как у первой);
        # для мягкого шарнира пара живёт на ОБЩЕЙ (звёздной) области, §9.2.
        bc2 = p2.bc.type if p2.bc is not None else problem.bc.type
        solver2 = KTNSolver.from_theory_name(dom2, cfg2, nonlinear_theory,
                                             bc_type=bc2, inplane_bc=inplane2)
        return solver2, cfg2, dom2
    if p2.bc.type == "clamped":
        return ClampedPlate.from_config(dom2, cfg2), cfg2, dom2
    return PlateBending.from_config(dom2, cfg2), cfg2, dom2


def _solve_two_plates(problem, cfg, dom, solver, f_values, warnings,
                      w_free, zone_mask) -> Result:
    r"""Контакт двух пластин (A4): r ← [r + β((w₁ − w₂) − Δ)]₊, нагрузки q₁−r и q₂+r.

    Узлы контакта — квадратура первой пластины, маскированная ω₂ > 0 и зоной;
    межсеточного переноса нет (см. докстринг :class:`TwoPlateMOR`).
    """
    from .contact import TwoPlateMOR

    c = problem.contact
    q1 = solver.quad
    solver2, cfg2, dom2 = _plate2_solver(problem, cfg, dom)
    f2_values, q02_eff, _ = _load_values_spec(problem.plate2.load, dom2,
                                              solver2.quad, warnings)
    if q02_eff != cfg2.q0:
        cfg2.q0 = q02_eff

    # зазор: скаляр (≥ 0, Δ=0 — касание) | gap_factor·w_free₁ | поле на узлах первой
    if c.gap is not None:
        gap_val = float(c.gap)
    elif c.gap_factor is not None:
        gap_val = c.gap_factor * w_free
    elif c.gap_field is not None:
        gap_val = _gap_field_values(c.gap_field, q1)
        _check_pair_gap_expr(c.gap_field, gap_val)
    else:
        gap_val = 0.0

    try:
        mor = TwoPlateMOR(solver, solver2, cfg, f1_values=f_values,
                          f2_values=f2_values, q2=cfg2.q0,
                          gap=gap_val, zone_mask=zone_mask)
    except ValueError as e:
        raise CaseError(f"contact.target: {e} см. {_SCHEMA_DOC}#plate2") from None
    n_nodes = int(mor.mask.sum())
    if n_nodes < MIN_ZONE_NODES:
        raise CaseError(
            f"contact.target: получено {n_nodes} узлов в пересечении планформ, "
            f"ожидалось ≥ {MIN_ZONE_NODES} — увеличьте Q или пересечение, "
            f"см. {_SCHEMA_DOC}#plate2")
    cres = mor.solve()
    delta_repr = (float(np.min(gap_val[mor.mask])) if np.ndim(gap_val)
                  else float(gap_val))
    res = Result(problem=problem, config=cfg,
                 w_max=float(np.max(np.abs(cres.w_nodes))),
                 cond=_cond_of(solver), Xg=cres.Xg, Yg=cres.Yg,
                 w_grid=cres.w_grid, warnings=tuple(warnings), contact=cres,
                 delta=delta_repr, w_free_max=w_free,
                 w_max_classic=float(np.max(np.abs(cres.w_nodes))))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", cres.cw)
    object.__setattr__(res, "_plate2_ref", solver2)
    object.__setattr__(res, "_cfg2_ref", cfg2)
    return res


def _solve_two_plates_nonlinear(problem, cfg, dom, solver, warnings) -> Result:
    r"""Взаимный контакт ДВУХ пластин КТН (§9.2): karman | ktn_full, защемление.

    ``solver`` — единый :class:`KTNSolver` первой пластины;
    :class:`~plate_solver.contact_nl.NonlinearTwoPlateMOR` определяет реакцию
    ``r`` СОВМЕСТНО двумя нелинейными решателями. Условие непроникания — на
    ЛИЦЕВЫХ прогибах ``u_c1 − u_c2 ≤ z`` (масштаб кривизны — от теории каждой
    пластины). Общая квадратура (одна планформа/Q) обязательна; результат
    адаптируется в :class:`TwoPlateResult` (тот же пайплайн вывода).
    """
    from .contact import TwoPlateResult, sample_pair_fields_on_grid
    from .contact_nl import NonlinearTwoPlateMOR

    theory = problem.model.theory
    q1 = solver.quad
    c = problem.contact
    warn = list(warnings)
    solver2, cfg2, dom2 = _plate2_solver(problem, cfg, dom, nonlinear_theory=theory)

    # свободные (без взаимодействия) решения: масштаб щели и gap_factor
    free1 = solver.solve(np.full(q1.x.size, float(cfg.q0)))
    free2 = solver2.solve(np.full(solver2.quad.x.size, float(cfg2.q0)))
    w_free1 = float(np.max(np.abs(free1.w_nodes)))
    w_free2 = float(np.max(np.abs(free2.w_nodes)))
    w_scale = (w_free1 + w_free2) if (w_free1 + w_free2) > 0.0 else 1.0

    # зазор: скаляр (≥ 0, Δ=0 — касание) | gap_factor·w_free₁ | поле на узлах первой
    if c.gap is not None:
        gap_val = float(c.gap)
    elif c.gap_factor is not None:
        gap_val = c.gap_factor * w_free1
    elif c.gap_field is not None:
        gap_val = _gap_field_values(c.gap_field, q1)
        _check_pair_gap_expr(c.gap_field, gap_val)
    else:
        gap_val = 0.0

    zone_mask = None
    fmask = None
    if c.zone is not None:
        zone_mask = _zone_mask(c.zone, q1, key="contact.zone",
                               advice="увеличьте Q или пересечение")
        fmask = lambda X, Y: zone_mask                          # noqa: E731

    force_total = None
    level_star = None
    if c.force is not None:
        if c.gap is not None or c.gap_factor is not None or c.gap_field is not None:
            warn.append("contact.gap: зазор игнорируется в силовом режиме пары "
                        "(force) — начальный зазор z ищется из условия ∫r = P")
        gap_val = 0.0                               # силовой режим: z — искомый уровень
    try:
        mor = NonlinearTwoPlateMOR(solver, solver2, cfg, gap=gap_val,
                                   f1=np.full(q1.x.size, float(cfg.q0)),
                                   q2=float(cfg2.q0), foundation_mask=fmask,
                                   gain_mode=cfg.contact_gain)
    except ValueError as e:
        raise CaseError(f"contact.target: {e} см. {_SCHEMA_DOC}#plate2") from None
    if c.force is not None:
        # СИЛОВОЕ управление парой (v0.6.5): задано суммарное усилие прижатия P;
        # ищется НАЧАЛЬНЫЙ зазор z (z < 0 — натяг) из F(z) = ∫r − P = 0
        # (монотонно убывает по z). Объект МОР строится ОДИН раз (set_gap),
        # каждый уровень стартует ТЁПЛО с предыдущего (c1, c2, r) — иначе
        # каждый вызов F был бы полным парным МОР с нуля (минуты вместо секунд).
        from scipy.optimize import brentq

        P = float(c.force)
        u_free = (mor._face(solver, free1.cw, mor._c1)
                  - mor._face(solver2, free2.cw, mor._c2))    # свободное сближение
        fm = zone_mask if zone_mask is not None else np.ones(q1.x.size, dtype=bool)
        z_hi = float(np.max(u_free[fm])) * (1.0 + 1e-9) + 1e-30   # касание: ∫r → 0
        scale = max(w_scale, 1e-12)
        state: dict = {}

        def F(z: float) -> float:
            mor.set_gap(z)
            res_z = mor.solve(c1_0=state.get("c1"), c2_0=state.get("c2"),
                              r0=state.get("r"))
            if not res_z.converged:                   # затхлый тёплый старт? — холодный повтор
                res_z = mor.solve()
            if not res_z.converged:
                # несошедшийся внутренний МОР сделал бы F(z) мусором и сломал
                # brentq МОЛЧА — честный отказ с советом
                last = (float(res_z.residual_history[-1])
                        if res_z.residual_history.size else float("nan"))
                raise CaseError(
                    f"contact.force: внутренний парный МОР не сошёлся на уровне "
                    f"z = {z:.3e} (невязка {last:.2e}). Рабочая область силового "
                    f"режима пары — умеренное прижатие (z* вблизи касания); при "
                    f"ГЛУБОКОМ натяге совмещённая схема не сходится — УМЕНЬШИТЕ "
                    f"force (вторично: contact.max_iter / mor_anderson), "
                    f"см. {_SCHEMA_DOC}#contact")
            state.update(c1=res_z.cw1, c2=res_z.cw2, r=res_z.r_nodes, res=res_z)
            return float(np.sum(q1.w * res_z.r_nodes)) - P

        # ПРОДОЛЖЕНИЕ по z вниз (натяг растёт постепенно): каждый уровень стартует
        # тёпло с СОСЕДНЕГО (дёшево и надёжно; дальние прыжки срывают МОР). Шаг
        # адаптивен по секущей ∫r(z); мост найден — узкий brentq внутри него.
        z_prev, F_prev = z_hi, -P                     # касание: ∫r = 0 (без прогона)
        step = 0.5 * scale
        z = z_hi - step
        bracket = None
        for _ in range(60):
            F_z = F(z)
            if F_z > 0.0:
                bracket = (z, z_prev)
                break
            if F_z > F_prev:                          # секущая: расстояние до нуля
                slope = (F_z - F_prev) / (z_prev - z)
                need = F_z / slope if slope > 0.0 else step
                step = min(2.0 * step, max(0.5 * step, 1.2 * abs(need)))
            z_prev, F_prev = z, F_z
            if z_hi - z > 50.0 * scale:               # защитный потолок натяга
                break
            z = z - step
        if bracket is None:
            raise CaseError(
                f"contact.force: получено P = {P:g} — прижатие не достигается при "
                f"натяге до 50·w_scale (уменьшите force), см. {_SCHEMA_DOC}#contact")
        level_star = brentq(F, bracket[0], bracket[1], xtol=1e-7 * scale)
        F(level_star)
        nres = state["res"]
        gap_val = float(level_star)                  # для комплементарности/вывода
        force_total = float(np.sum(q1.w * nres.r_nodes))
        # Паре Андерсон НЕОБХОДИМ (без него совмещённый цикл не сходится), но его
        # смешение зашумляет F(z) — контролируем фактическое замыкание ∫r = P
        # и честно предупреждаем при отклонении (аудит v0.6.5).
        if abs(force_total - P) / P > 2e-2:
            warn.append(
                f"contact.force: замыкание ∫r = P с точностью "
                f"{abs(force_total - P) / P:.1%} (шум смешения Андерсона в F(z)); "
                "для точнее — уменьшите contact.tol или mor_anderson")
    else:
        nres = mor.solve()
    if not nres.converged:
        last = float(nres.residual_history[-1]) if nres.residual_history.size else float("nan")
        warn.append(
            f"{theory}: нелинейная пара МОР не достигла tol за max_iter "
            f"(последняя невязка {last:.2e}); увеличьте contact.max_iter или "
            "смените contact.gain на linear")

    # адаптация NonlinearTwoPlateResult → TwoPlateResult (пайплайн вывода)
    Xg, Yg, w1g, w2g, rg, zone = sample_pair_fields_on_grid(
        solver, solver2, nres.cw1, nres.cw2, nres.r_nodes, cfg.grid_n)
    u = nres.u_c1_nodes - nres.u_c2_nodes                       # сближение лицевых
    comp = float(np.max(np.abs(nres.r_nodes * (u - gap_val))) / (float(cfg.q0) * w_scale))
    contact = nres.r_nodes > 0.0
    over = (float(np.max((u - gap_val)[contact])) / w_scale
            if contact.any() else float("nan"))
    cres = TwoPlateResult(
        Xg=Xg, Yg=Yg, w_grid=w1g, w2_grid=w2g, r_grid=rg, contact_zone=zone,
        r_nodes=nres.r_nodes, w_nodes=nres.w1_nodes, w2_nodes=nres.w2_nodes,
        iters=nres.iters, converged=nres.converged,
        residual_history=nres.residual_history, peak_xy=nres.peak_xy,
        plate=solver, plate2=solver2, cw=nres.cw1, cw2=nres.cw2,
        comp_residual=comp, gap_overshoot=over)
    delta_repr = (float(np.min(gap_val)) if np.ndim(gap_val) else float(gap_val))
    res = Result(problem=problem, config=cfg, w_max=float(nres.w1_max),
                 cond=_cond_of(solver), Xg=Xg, Yg=Yg, w_grid=w1g,
                 warnings=tuple(warn), contact=cres, delta=delta_repr,
                 w_free_max=w_free1, w_max_classic=float(nres.w1_max),
                 force_total=force_total, level=level_star)
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", nres.cw1)
    object.__setattr__(res, "_plate2_ref", solver2)
    object.__setattr__(res, "_cfg2_ref", cfg2)
    return res


def _solve_contact_force(problem, cfg, dom, solver, f_values, warnings,
                         state_free, w_free, zone_mask, fmask) -> Result:
    r"""Силовой штамп (A2): задана сила P, ищется уровень штампа.

    Δ(x, y) = level + shape(x, y); ``shape`` — форма из ``[contact.gap]``
    (нет таблицы ⇒ плоский штамп, shape ≡ 0). Скалярное уравнение

    .. math:: F(\mathrm{level}) = \int_\Omega r\, d\Omega - P = 0

    монотонно убывает по level (больший зазор — меньшее прижатие);
    решается brentq на [level_lo, level_hi]: lo — почти касание в нижней
    точке штампа, hi — уровень непроникновения (r ≡ 0 ⇒ F = −P < 0).
    Каждый вызов F — полный МОР с ТЁПЛЫМ стартом r от предыдущего уровня.
    """
    from scipy.optimize import brentq

    c = problem.contact
    q = solver.quad
    P = float(c.force)
    if c.gap is not None or c.gap_factor is not None:
        warnings.append("contact.gap: скалярный gap/gap_factor игнорируется "
                        "в силовом режиме (force); форма штампа — [contact.gap]")
    shape = _gap_field_values(c.gap_field, q) if c.gap_field is not None else 0.0
    if np.ndim(shape) == 0:
        shape = float(shape)                       # плоский штамп или const-форма

    fm = zone_mask if zone_mask is not None else np.ones(q.x.size, dtype=bool)
    w_free_nodes = solver.w_at_quad(state_free)
    shape_fm = shape[fm] if np.ndim(shape) else shape
    # Верхняя граница: level + shape ≥ w_free на основании ⇒ контакта нет.
    level_hi = float(np.max(w_free_nodes[fm] - shape_fm)) * (1.0 + 1e-9) + 1e-30
    # Нижняя: почти касание в нижней точке штампа (min Δ = 1e-8 масштаба).
    scale = float(np.max(np.abs(w_free_nodes)))
    min_shape = float(np.min(shape_fm)) if np.ndim(shape) else shape
    level_lo = 1e-8 * scale - min_shape

    ktn = KTNParams.from_config(cfg) if problem.model.theory == "ktn_linear" else None
    state = {"r": None, "res": None, "iters": 0, "calls": 0}

    def F(level: float) -> float:
        gap_val = (level + shape) if np.ndim(shape) else float(level + shape)
        mor = ContactMOR(solver, cfg, foundation_mask=fmask, gap=gap_val,
                         ktn=ktn, load_values=f_values)
        res = mor.solve(r0=state["r"])
        state.update(r=res.r_nodes, res=res)
        state["iters"] += res.iters
        state["calls"] += 1
        return float(np.sum(q.w * res.r_nodes)) - P

    F_lo = F(level_lo)
    if F_lo <= 0.0:
        raise CaseError(
            f"contact.force: получено P = {P:g}, ожидалось 0 < P ≤ "
            f"{F_lo + P:.6g} (максимум ∫r при касании штампа), "
            f"см. {_SCHEMA_DOC}#contact")
    level_star = brentq(F, level_lo, level_hi, xtol=1e-8 * scale)
    F_star = F(level_star)                          # финальный прогон на level*
    cres = state["res"]
    force_total = F_star + P

    delta_min = float(level_star + min_shape)
    w_nodes = cres.w_ktn_nodes if cres.w_ktn_nodes is not None else cres.w_nodes
    res = Result(problem=problem, config=cfg, w_max=float(np.max(np.abs(w_nodes))),
                 cond=_cond_of(solver), Xg=cres.Xg, Yg=cres.Yg,
                 w_grid=cres.w_grid, warnings=tuple(warnings), contact=cres,
                 delta=delta_min, w_free_max=w_free,
                 level=float(level_star), force_total=force_total,
                 w_max_classic=float(np.max(np.abs(cres.w_nodes))))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", cres.cw)
    object.__setattr__(res, "_force_calls", state["calls"])
    object.__setattr__(res, "_force_iters_total", state["iters"])
    return res


__all__ = ["Result", "build_domain", "solve"]
