r"""dispatch.py — диспетчер: постановка Problem → расчёт → Result.

Маршрутизация (блок-схема — docs/dispatch_flow.md):

* ``bc.type = soft_hinge``  → :class:`~plate_solver.plate.PlateBending`
  (расщепление бигармоники на две задачи Пуассона);
* ``bc.type = clamped``     → :class:`~plate_solver.clamped.ClampedPlate`
  (структура w = ω²Φ, прямой Ритц по бигармонике);
* ``contact.enabled``       → :class:`~plate_solver.contact.ContactMOR`
  (только soft_hinge — гарантировано валидатором); ``[contact.zone]`` →
  ``foundation_mask = [ω_zone > 0]`` в узлах квадратуры;
* ``model.theory = ktn``    → :class:`~plate_solver.ktn.KTNParams`
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

import numpy as np

from . import geometry
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
    """Построить Domain по спецификации из case-файла (реестр геометрий)."""
    if spec.kind == "circle":
        return geometry.make_circle(spec.a)
    if spec.kind == "rectangle":
        return geometry.make_rectangle(spec.x1, spec.x2, spec.y1, spec.y2)
    if spec.kind == "L":
        return geometry.make_L(spec.side, spec.cut)
    if spec.kind == "annulus":
        return geometry.make_annulus(spec.a, spec.b)
    if spec.kind == "compose":
        return geometry.make_compose(spec.tree)
    raise CaseError(f"geometry.kind: получено {spec.kind!r}, ожидалось значение "
                    f"реестра v0.2, см. {_SCHEMA_DOC}#geometry")


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
                "comp_residual": c.comp_residual,
                "gap_overshoot": c.gap_overshoot,
                "r_max": float(c.r_nodes.max()),
                "n_contact": int((c.r_nodes > 0).sum()),
                "n_quad": int(c.r_nodes.size),
            })
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
        return out / "result.json"

    def moments_on_grid(self):
        """Моменты (Mx, My, Mxy) на фоновой сетке (NaN вне Ω)."""
        from .ladder import bending_moments_full

        solver = self._plate
        inside = np.isfinite(self.w_grid)
        if hasattr(solver, "moments_at"):                   # mixed-структура
            Mx = np.full(self.Xg.shape, np.nan)
            My = np.full(self.Xg.shape, np.nan)
            Mxy = np.full(self.Xg.shape, np.nan)
            mx, my, mxy = solver.moments_at(self._c, self.Xg[inside], self.Yg[inside])
            Mx[inside], My[inside], Mxy[inside] = mx, my, mxy
            return Mx, My, Mxy
        p_struct = 2 if hasattr(solver, "S") else 1        # ω²Φ у защемления
        Mx = np.full(self.Xg.shape, np.nan)
        My = np.full(self.Xg.shape, np.nan)
        Mxy = np.full(self.Xg.shape, np.nan)
        # У составных R-областей (r_and/r_or) ВТОРЫЕ производные ω имеют
        # линии излома (f₁ = f₂): гессиан структуры там не определён — точки
        # остаются NaN и маскируются в картах (NOTES §19).
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my, mxy = bending_moments_full(
                solver.domain, solver.basis, self._c, p_struct,
                self.config.D, self.config.nu, self.Xg[inside], self.Yg[inside])
        Mx[inside], My[inside], Mxy[inside] = mx, my, mxy
        return Mx, My, Mxy

    def _q_faces_on_grid(self):
        """(q_top, q_bottom) на сетке: нагрузка сверху, реакция снизу (§19)."""
        load = self.problem.load
        inside = np.isfinite(self.w_grid)
        q_top = np.zeros(self.Xg.shape)
        if load.type == "uniform":
            q_top[inside] = self.config.q0
        elif load.type == "patch":
            zone = build_domain(load.zone)
            m = inside & (zone.omega(self.Xg, self.Yg) > 0.0)
            q_top[m] = self.config.q0
        else:                                            # point: пятно eps_eff
            eps = self.eps_eff if self.eps_eff is not None else 0.0
            m = inside & (((self.Xg - load.x0) ** 2 + (self.Yg - load.y0) ** 2)
                          <= eps**2)
            q_top[m] = self.config.q0
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
        if self.problem.model is None or self.problem.model.theory != "ktn_linear":
            return self.w_grid.copy(), self.w_grid.copy(), \
                np.zeros_like(self.w_grid)
        from .faces import FaceParams

        Mx, My, _ = self.moments_on_grid()
        q_top, q_bot = self._q_faces_on_grid()
        lap = -(Mx + My) / (self.config.D * (1.0 + self.config.nu))
        fp = FaceParams.from_config(self.config)
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
        """fields.npz (версия схемы полей = 2): w, моменты, σ-шестёрка, контакт.

        Схема 2 = схема 1 + прогибы лицевых поверхностей (w_top, w_bot,
        dh — NOTES §21) + для пары пластин моменты и σ-шестёрка ВТОРОЙ
        пластины (суффикс «2»; канон §19: у верхней q⁻ = r, у нижней
        q⁺ = r). Всё необходимое для перерисовки фигур БЕЗ пересчёта —
        :func:`plate_solver.viz.replot`.
        """
        from .ktn import stresses_faces

        Mx, My, Mxy = self.moments_on_grid()
        q_top, q_bot = self._q_faces_on_grid()
        s = stresses_faces(Mx, My, Mxy, h=self.config.h, nu=self.config.nu,
                           q_top=q_top, q_bottom=q_bot)
        w_top, w_bot, dh = self.faces_on_grid()
        payload = {
            "fields_schema": np.int64(2),
            "x": self.Xg[0, :], "y": self.Yg[:, 0],
            "w": self.w_grid, "Mx": Mx, "My": My, "Mxy": Mxy,
            "w_top": w_top, "w_bot": w_bot, "dh": dh,
            "problem_json": np.str_(json.dumps(asdict(self.problem),
                                               ensure_ascii=False)),
            "h": np.float64(self.config.h), "nu": np.float64(self.config.nu),
        }
        payload.update(s)
        if self.contact is not None:
            payload["r"] = np.nan_to_num(self.contact.r_grid, nan=0.0)
            payload["zone"] = self.contact.contact_zone
            w2 = getattr(self.contact, "w2_grid", None)
            if w2 is not None:
                payload["w2"] = w2
                payload.update(self._second_plate_fields())
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
        p2 = 2 if hasattr(solver2, "S") else 1
        Mx2 = np.full(self.Xg.shape, np.nan)
        My2 = np.full(self.Xg.shape, np.nan)
        Mxy2 = np.full(self.Xg.shape, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my, mxy = bending_moments_full(
                solver2.domain, solver2.basis, self.contact.cw2, p2,
                cfg2.D, cfg2.nu, self.Xg[inside2], self.Yg[inside2])
        Mx2[inside2], My2[inside2], Mxy2[inside2] = mx, my, mxy
        r_fld = np.nan_to_num(self.contact.r_grid, nan=0.0)
        s2 = stresses_faces(Mx2, My2, Mxy2, h=cfg2.h, nu=cfg2.nu,
                            q_top=r_fld, q_bottom=0.0)
        out = {"Mx2": Mx2, "My2": My2, "Mxy2": Mxy2}
        out.update({k + "2": v for k, v in s2.items()})
        return out

    def _save_figures(self, out: Path, formats: tuple = ("png", "pdf"),
                      surface: str = "mid") -> None:
        """Фигуры публикационного качества (D1/D2): из fields.npz, png+pdf.

        Имена файлов получают префикс по имени case-файла (title кейса).
        """
        from . import viz

        stem = Path(self.problem.source).stem
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


def _load_values_spec(load, dom, quad, warnings: list[str]):
    """То же по произвольной LoadSpec (вторая пластина, A4)."""
    if load.type == "uniform":
        return None, float(load.q0), None

    if load.type == "patch":
        mask = _zone_mask(load.zone, quad, key="load.zone",
                          advice="увеличьте Q или зону нагрузки")
        return float(load.q0) * mask.astype(float), float(load.q0), None

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

    if problem.model.theory in ("karman", "ktn_full"):    # нелинейные теории
        # ktn_full строит собственный решатель в N2; на вехе N0 — KarmanPlate
        # как носитель quad/структуры, а _solve_bending отвергает ktn_full.
        from .membrane import KarmanPlate

        solver = KarmanPlate.from_config(dom, cfg, bc_type=problem.bc.type,
                                         inplane_bc=problem.model.inplane_bc)
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

    f_values, q0_eff, eps_eff = _load_values(problem, dom, quad, warnings)
    if q0_eff != cfg.q0:
        cfg.q0 = q0_eff                     # эффективная интенсивность (КТН, МОР)
    t_build = time.perf_counter() - t0

    t0 = time.perf_counter()
    if problem.contact.enabled:
        result = _solve_contact(problem, cfg, dom, solver, f_values, warnings)
    else:
        result = _solve_bending(problem, cfg, dom, solver, f_values, warnings)
    t_solve = time.perf_counter() - t0

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


def _solve_bending(problem, cfg, dom, solver, f_values, warnings) -> Result:
    f = _uniform(cfg, solver.quad) if f_values is None else f_values
    if problem.model.theory == "ktn_full":
        return _solve_ktn_full(problem, cfg, dom, solver, f, warnings)
    if problem.model.theory == "karman":
        return _solve_karman(problem, cfg, dom, solver, f, warnings)
    q = solver.quad
    if problem.bc.type in ("clamped", "mixed"):
        c = solver.solve(f)
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
        cM, cw = solver.solve(f)
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
    kr = solver.solve(f)
    c = kr.cw
    w_nodes = kr.w_nodes
    warn = list(warnings)
    if not kr.converged:
        last = kr.history[-1][2] if kr.history else float("nan")
        warn.append(
            f"karman: итерация Пикара не достигла karman_tol за karman_max_iter "
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
    """Полная нелинейная КТН (§3, §5). Веха N2.

    ЗАГЛУШКА N0: таксономия и маршрутизация заведены; решатель
    :class:`~plate_solver.ktn_full.KTNPlate` (члены (A), (B) поверх KarmanPlate
    + лицевая постобработка) реализуется на вехе N2 в ``ktn_full.py``.
    """
    raise NotImplementedError(
        "theory = 'ktn_full': полный нелинейный решатель КТН (KTNPlate) — "
        "веха N2 (src/plate_solver/ktn_full.py)")


def _gap_field_values(spec, quad):
    """Поле зазора Δ(x, y) в узлах квадратуры по GapSpec (A1.2); const → скаляр.

    ``const`` возвращает СКАЛЯР — путь и арифметика тождественны скалярному
    ``gap`` v0.2 (ворота-тождество A1.3-т1).
    """
    if spec.kind == "const":
        return spec.value
    if spec.kind == "plane":
        return spec.a * quad.x + spec.b * quad.y + spec.c
    if spec.kind == "paraboloid":
        return spec.apex + ((quad.x - spec.cx) ** 2 + (quad.y - spec.cy) ** 2) \
            / (2.0 * spec.r_curv)
    g = np.full(quad.x.size, float(spec.base))          # steps
    for zone_geom, value in spec.zones:
        zdom = build_domain(zone_geom)
        g[zdom.omega(quad.x, quad.y) > 0.0] = value
    return g


def _cond_of(solver) -> float:
    """cond(A) решателя: ClampedPlate.cond либо PlateBending.poisson.cond."""
    return float(solver.cond if hasattr(solver, "cond") else solver.poisson.cond)


def _solve_contact(problem, cfg, dom, solver, f_values, warnings) -> Result:
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
    w_nodes = cres.w_ktn_nodes if cres.w_ktn_nodes is not None else cres.w_nodes
    res = Result(problem=problem, config=cfg, w_max=float(np.max(np.abs(w_nodes))),
                 cond=_cond_of(solver), Xg=cres.Xg, Yg=cres.Yg,
                 w_grid=cres.w_grid, warnings=tuple(warnings), contact=cres,
                 delta=float(delta), w_free_max=w_free,
                 w_max_classic=float(np.max(np.abs(cres.w_nodes))))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", cres.cw)
    return res


def _plate2_solver(problem: Problem, cfg: Config, dom):
    """Построить решатель второй пластины по [plate2] (дефолты — от первой)."""
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
