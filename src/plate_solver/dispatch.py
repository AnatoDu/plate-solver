r"""dispatch.py — диспетчер: постановка Problem → расчёт → Result.

Маршрутизация (блок-схема — docs/dispatch_flow.md, P5.3):

* ``bc.type = soft_hinge``  → :class:`~plate_solver.plate.PlateBending`
  (расщепление бигармоники на две задачи Пуассона);
* ``bc.type = clamped``     → :class:`~plate_solver.clamped.ClampedPlate`
  (структура w = ω²Φ, прямой Ритц по бигармонике);
* ``contact.enabled``       → :class:`~plate_solver.contact.ContactMOR`
  (только soft_hinge — гарантировано валидатором); ``[contact.zone]`` →
  ``foundation_mask = [ω_zone > 0]`` в узлах квадратуры;
* ``model.theory = ktn``    → :class:`~plate_solver.ktn.KTNParams`
  (в контакте — смещение контактной поверхности, как в фазе 1; в чистом
  изгибе — ``corrected_deflection`` при r = 0, кривизна Δw = −M/D из (P1)).

Нагрузки: ``uniform`` — константа q0; ``patch`` — q̃ = q0·[ω_zone > 0]
в узлах; ``point`` — регуляризованный patch: круговое пятно радиуса
eps_eff с q = P/(π·eps_eff²). Защита зон (P0.2): пятно point автоматически
расширяется до ≥ MIN_ZONE_NODES узлов (факт — в ``Result.warnings``);
зона patch/contact с < MIN_ZONE_NODES узлами — :class:`CaseError`.
"""

from __future__ import annotations

import json
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
    """Неизменяемый результат расчёта постановки (P2.2).

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
    delta: float | None = None         # разрешённый зазор Δ (gap или gap_factor·w_free)
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

    def save(self, out_dir: str | Path | None = None) -> Path:
        """Записать result.json (+ фигуры при output.figures) в каталог."""
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
        (out / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        if self.problem.output.figures:
            self._save_figures(out)
        return out / "result.json"

    def _save_figures(self, out: Path) -> None:
        from . import viz

        if self.contact is not None:
            viz.plot_contact_summary(self.config, self.contact,
                                     save=str(out / "contact_summary.png"))
        else:
            # plot_deflection_surface duck-типна: нужен .domain и .deflection
            viz.plot_deflection_surface(self.config, self._plate, self._c,
                                        save=str(out / "w_surface.png"))

    # солвер и коэффициенты для фигур (не сериализуются)
    @property
    def _plate(self):
        return object.__getattribute__(self, "_plate_ref")

    @property
    def _c(self):
        return object.__getattribute__(self, "_c_ref")


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
    load = problem.load
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
        # авторасширение до наименьшего радиуса с ≥ MIN_ZONE_NODES узлами (P0.2)
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
def solve(problem: Problem) -> Result:
    """Решить постановку: маршрутизация по bc/contact/theory (см. докстринг модуля)."""
    warnings: list[str] = []
    t0 = time.perf_counter()
    cfg = problem.to_config()
    dom = build_domain(problem.geometry)

    if problem.bc.type == "clamped":
        if problem.model.theory == "ktn":
            raise CaseError(
                "model.theory: получено 'ktn' при bc.type='clamped', ожидалось "
                "classic — в v0.2 поправки КТН требуют кривизны Δw = −M/D из "
                f"расщепления (soft_hinge), см. {_SCHEMA_DOC}#model"
            )
        solver = ClampedPlate.from_config(dom, cfg)
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
    q = solver.quad
    if problem.bc.type == "clamped":
        c = solver.solve(f)
        w_nodes = solver.deflection(c, q.x, q.y)
        evaluate = lambda X, Y: solver.deflection(c, X, Y)      # noqa: E731
        w_ktn = None
        cond = float(solver.cond)
    else:
        cM, cw = solver.solve(f)
        w_nodes = solver.poisson.evaluate_at_quad(cw)
        evaluate = lambda X, Y: solver.deflection(cw, X, Y)     # noqa: E731
        c = cw
        cond = float(solver.poisson.cond)
        w_ktn = None
        if problem.model.theory == "ktn":
            # изгиб без контакта: corrected_deflection при r = 0 (P2.1)
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


def _solve_contact(problem, cfg, dom, solver, f_values, warnings) -> Result:
    q = solver.quad
    # Δ: абсолютный gap либо gap_factor·w_free (свободный прогиб той же нагрузки)
    f = _uniform(cfg, q) if f_values is None else f_values
    _, cw_free = solver.solve(f) if f_values is not None else solver.solve_uniform(cfg.q0)
    w_free = float(np.max(np.abs(solver.poisson.evaluate_at_quad(cw_free))))
    delta = (problem.contact.gap if problem.contact.gap is not None
             else problem.contact.gap_factor * w_free)

    fmask = None
    if problem.contact.zone is not None:
        zone_mask = _zone_mask(problem.contact.zone, q, key="contact.zone",
                               advice="увеличьте Q или зону")
        fmask = lambda X, Y: zone_mask                          # noqa: E731

    ktn = KTNParams.from_config(cfg) if problem.model.theory == "ktn" else None
    mor = ContactMOR(solver, cfg, foundation_mask=fmask, gap=delta, ktn=ktn,
                     load_values=f_values)
    cres = mor.solve()
    w_nodes = cres.w_ktn_nodes if cres.w_ktn_nodes is not None else cres.w_nodes
    res = Result(problem=problem, config=cfg, w_max=float(np.max(np.abs(w_nodes))),
                 cond=float(solver.poisson.cond), Xg=cres.Xg, Yg=cres.Yg,
                 w_grid=cres.w_grid, warnings=tuple(warnings), contact=cres,
                 delta=float(delta), w_free_max=w_free,
                 w_max_classic=float(np.max(np.abs(cres.w_nodes))))
    object.__setattr__(res, "_plate_ref", solver)
    object.__setattr__(res, "_c_ref", cres.cw)
    return res


__all__ = ["Result", "build_domain", "solve"]
