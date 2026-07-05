r"""references.py — верификация как свойство постановки (P3 фазы 2).

Резолвер :func:`resolve_reference` по постановке возвращает список
именованных эталонов; :func:`verify_result` сравнивает с ними результат
диспетчера и строит отчёт-таблицу «эталон | значение | rel | статус».

Правило модельной согласованности (NOTES §8): ``reference = "analytic"``
сравнивает с эталоном РЕАЛИЗОВАННОЙ модели (мягкий шарнир — с аналитикой
расщепления, не с истинным Кирхгофом); ``model_gap = true`` добавляет
НЕ-гейтуемую строку «истинный Кирхгоф» — она документирует модельную
погрешность и в допуск ``tol`` не входит.

Сравниваемая величина v0.2 — ``w_max`` (максимум прогиба; как в таблицах
главы 4). Эталоны fem (P3.6) и mms (P3.8) подключаются позже.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import analytic
from .dispatch import Result
from .problem import CaseError, Problem

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"

# Параметры 1D-Ритца для cross_1d: спектральная сходимость по p, дёшево;
# p=16 покрывает ln r кольца (ворота P3.3: rel < 1e-8).
_CROSS_1D_P = 16
_CROSS_1D_NQ = 400


@dataclass(frozen=True)
class Reference:
    """Именованный эталон: имя, значение w_max, участвует ли в допуске tol."""

    name: str
    kind: str            # analytic | cross_1d | model_gap
    w_max: float
    gated: bool


@dataclass(frozen=True)
class RefRow:
    """Строка отчёта: эталон | значение | относительное отклонение | статус."""

    name: str
    reference: float
    value: float
    rel: float
    gated: bool
    passed: bool | None      # None — строка информационная (вне допуска)


@dataclass(frozen=True)
class VerifyReport:
    """Отчёт верификации постановки (таблица + общий вердикт)."""

    rows: tuple[RefRow, ...]
    tol: float

    @property
    def ok(self) -> bool:
        """Все гейтуемые строки в допуске (информационные не учитываются)."""
        return all(r.passed for r in self.rows if r.gated)

    def table(self) -> str:
        """Markdown-таблица «эталон | значение | rel | статус»."""
        lines = ["| эталон | w_max эталона | w_max расчёта | rel | статус |",
                 "|---|---|---|---|---|"]
        for r in self.rows:
            status = ("PASS" if r.passed else "FAIL") if r.gated else "инфо"
            lines.append(f"| {r.name} | {r.reference:.6e} | {r.value:.6e} "
                         f"| {r.rel:.2e} | {status} |")
        return "\n".join(lines)


def _fail(key: str, got, expected: str, anchor: str = "verify") -> None:
    raise CaseError(f"{key}: получено {got!r}, ожидалось {expected}, "
                    f"см. {_SCHEMA_DOC}#{anchor}")


# --------------------------------------------------------------------------- #
#  Аналитические эталоны (модельно-согласованные)
# --------------------------------------------------------------------------- #
def _analytic_wmax(problem: Problem, cfg) -> float:
    g, bc = problem.geometry, problem.bc.type
    if problem.load.type != "uniform":
        _fail("verify.reference", "analytic",
              "уравномерной нагрузки: point-эталон появится в P3.5, "
              "patch — mms | fem | none")
    if g.kind == "circle":
        if bc == "clamped":
            return float(analytic.clamped_uniform_wmax(g.a, cfg.q0, cfg.D))
        return float(analytic.circular_plate_soft_hinge_wmax(cfg.q0, g.a, cfg.D))
    if g.kind == "annulus":
        abc = "clamped" if bc == "clamped" else "soft"
        return analytic.annulus_uniform_wmax(g.a, g.b, cfg.q0, cfg.D, abc, cfg.nu)
    _fail("verify.reference", "analytic",
          "circle | annulus (для rectangle/L/compose — mms | fem | none)")


def _model_gap_wmax(problem: Problem, cfg) -> float | None:
    """Эталон «истинного Кирхгофа» для строки model_gap (только мягкий шарнир)."""
    if problem.bc.type != "soft_hinge" or problem.load.type != "uniform":
        return None                      # у защемления модельного разрыва нет
    g = problem.geometry
    if g.kind == "circle":
        return float(analytic.simply_supported_uniform_wmax(g.a, cfg.q0, cfg.D, cfg.nu))
    if g.kind == "annulus":
        return analytic.annulus_uniform_wmax(g.a, g.b, cfg.q0, cfg.D, "true_ss", cfg.nu)
    return None


def _cross_1d_wmax(problem: Problem, cfg) -> float:
    """1D-Ритц по радиусу (осесимметричность гарантирована валидатором)."""
    from .radial import (
        RadialClamped,
        RadialClampedAnnulus,
        solve_radial_soft_hinge,
        solve_radial_soft_hinge_annulus,
    )

    g, bc = problem.geometry, problem.bc.type
    if g.kind == "circle":
        r = np.linspace(0.0, g.a, 2001)
        if bc == "clamped":
            s = RadialClamped(g.a, cfg.D, p=_CROSS_1D_P, nq=_CROSS_1D_NQ)
            s.solve(cfg.q0, cfg.nu)
            return float(np.max(np.abs(s.deflection(r))))
        rp, cw = solve_radial_soft_hinge(g.a, cfg.D, cfg.q0,
                                         p=_CROSS_1D_P, nq=_CROSS_1D_NQ)
        return float(np.max(np.abs(rp.deflection(cw, r))))
    # annulus
    r = np.linspace(g.b, g.a, 2001)
    if bc == "clamped":
        s = RadialClampedAnnulus(g.a, g.b, cfg.D, p=_CROSS_1D_P, nq=_CROSS_1D_NQ)
        s.solve(cfg.q0, cfg.nu)
        return float(np.max(np.abs(s.deflection(r))))
    rp, cw = solve_radial_soft_hinge_annulus(g.a, g.b, cfg.D, cfg.q0,
                                             p=_CROSS_1D_P, nq=_CROSS_1D_NQ)
    return float(np.max(np.abs(rp.deflection(cw, r))))


# --------------------------------------------------------------------------- #
#  Резолвер и отчёт
# --------------------------------------------------------------------------- #
def resolve_reference(problem: Problem, cfg=None) -> list[Reference]:
    """Список именованных эталонов постановки (P3.1).

    ``cfg`` — Config расчёта (None ⇒ ``problem.to_config()``; диспетчер
    передаёт свой, с эффективной интенсивностью point-нагрузки).
    """
    cfg = problem.to_config() if cfg is None else cfg
    v = problem.verify
    refs: list[Reference] = []
    if v.reference != "none" and problem.contact.enabled:
        _fail("verify.reference", v.reference,
              "none — эталонов контактной задачи в v0.2 нет "
              "(ворота контакта — инварианты, P3.7)")
    if v.reference == "analytic":
        refs.append(Reference(
            name=f"analytic ({problem.geometry.kind}, {problem.bc.type})",
            kind="analytic", w_max=_analytic_wmax(problem, cfg), gated=True))
    elif v.reference in ("fem", "mms"):
        _fail("verify.reference", v.reference,
              "analytic | none — fem подключается в P3.6, mms — в P3.8")
    if v.cross_1d:
        refs.append(Reference(
            name=f"1D-Ритц по радиусу ({problem.geometry.kind}, {problem.bc.type})",
            kind="cross_1d", w_max=_cross_1d_wmax(problem, cfg), gated=True))
    if v.model_gap:
        gap = _model_gap_wmax(problem, cfg)
        if gap is not None:
            refs.append(Reference(name="истинный Кирхгоф (model_gap, вне допуска)",
                                  kind="model_gap", w_max=gap, gated=False))
    return refs


def verify_result(result: Result) -> VerifyReport:
    """Сравнить Result со всеми эталонами постановки; собрать отчёт."""
    problem = result.problem
    refs = resolve_reference(problem, result.config)
    tol = problem.verify.tol
    rows = []
    for ref in refs:
        rel = abs(result.w_max - ref.w_max) / abs(ref.w_max)
        rows.append(RefRow(name=ref.name, reference=ref.w_max, value=result.w_max,
                           rel=rel, gated=ref.gated,
                           passed=(rel <= tol) if ref.gated else None))
    return VerifyReport(rows=tuple(rows), tol=tol)


__all__ = ["Reference", "RefRow", "VerifyReport", "resolve_reference", "verify_result"]
