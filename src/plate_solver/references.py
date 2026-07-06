r"""references.py — верификация как свойство постановки.

Резолвер :func:`resolve_reference` по постановке возвращает список
именованных эталонов; :func:`verify_result` сравнивает с ними результат
диспетчера и строит отчёт-таблицу «эталон | значение | rel | статус».

Правило модельной согласованности (NOTES §8): ``reference = "analytic"``
сравнивает с эталоном РЕАЛИЗОВАННОЙ модели (мягкий шарнир — с аналитикой
расщепления, не с истинным Кирхгофом); ``model_gap = true`` добавляет
НЕ-гейтуемую строку «истинный Кирхгоф» — она документирует модельную
погрешность и в допуск ``tol`` не входит.

Сравниваемая величина v0.2 — ``w_max`` (максимум прогиба; как в эталонных
таблицах); эталоны fem и mms подключает резолвер.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import analytic
from .dispatch import Result
from .problem import CaseError, Problem

_SCHEMA_DOC = "docs/CASE_SCHEMA.md"

# Параметры 1D-Ритца для cross_1d: спектральная сходимость по p, дёшево;
# p=16 покрывает ln r кольца (ворота кольца: rel < 1e-8).
_CROSS_1D_P = 16
_CROSS_1D_NQ = 400


@dataclass(frozen=True)
class Reference:
    """Именованный эталон: имя, значение w_max, участвует ли в допуске tol.

    ``value`` — сравниваемое значение: None ⇒ ``result.w_max`` (analytic,
    cross_1d, model_gap); для ``mms`` эталон гоняет СВОЙ прогон с
    изготовленной нагрузкой и кладёт сюда его w_max — это сертификат
    метода на данной постановке (та же геометрия и дискретизация).
    """

    name: str
    kind: str            # analytic | cross_1d | model_gap | mms | fem
    w_max: float
    gated: bool
    value: float | None = None
    point: tuple | None = None   # сравнение в ТОЧКЕ (центр прямоугольника):
    #                              снимает несоответствие «max по узлам Гаусса
    #                              против max по плотной сетке эталона»


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
    g, bc, load = problem.geometry, problem.bc.type, problem.load
    if load.type == "point":
        if g.kind == "rectangle":                     # фабрика: Навье-point (F4)
            return _navier_factory_ref(problem, cfg)
        if g.kind != "circle" or (load.x0, load.y0) != (0.0, 0.0):
            _fail("verify.reference", "analytic",
                  "point-эталон: сила в ЦЕНТРЕ круга (x0 = y0 = 0) или "
                  "произвольная точка ПРЯМОУГОЛЬНИКА (ряд Навье); "
                  "иначе — mms | fem | none")
        if bc == "clamped":
            return analytic.circle_point_clamped_wmax(g.a, load.P, cfg.D)
        return analytic.circle_point_soft_wmax(g.a, load.P, cfg.D)
    if load.type == "patch":
        if g.kind == "rectangle":                     # фабрика: Навье-patch (F4)
            return _navier_factory_ref(problem, cfg)
        _fail("verify.reference", "analytic",
              "patch-эталон: только прямоугольник с прямоугольной зоной "
              "(ряд Навье); иначе — mms | fem | none")
    if load.type != "uniform":
        _fail("verify.reference", "analytic",
              "uniform | patch/point на прямоугольнике | point в центре "
              "круга; иначе — mms | fem | none")
    if g.kind == "circle":
        if bc == "clamped":
            return float(analytic.clamped_uniform_wmax(g.a, cfg.q0, cfg.D))
        return float(analytic.circular_plate_soft_hinge_wmax(cfg.q0, g.a, cfg.D))
    if g.kind == "annulus":
        abc = "clamped" if bc == "clamped" else "soft"
        return analytic.annulus_uniform_wmax(g.a, g.b, cfg.q0, cfg.D, abc, cfg.nu)
    if g.kind == "rectangle":
        return _rect_analytic_wmax(problem, cfg)      # (значение, точка)
    _fail("verify.reference", "analytic",
          "circle | annulus | rectangle (для L/compose — mms | fem | none)")


def _rect_analytic_wmax(problem: Problem, cfg) -> float:
    r"""Эталоны прямоугольника: Навье (SSSS) и Леви (SCSC).

    На ПРЯМЫХ краях мягкий шарнир совпадает с истинным (NOTES §8, кривизна
    границы 0) ⇒ bc=soft_hinge гейтится рядом Навье. Для mixed: все hinge —
    Навье; пара hinge/пара clamped по осям — Леви (при clamped по x —
    поворот осей); прочие комбинации аналитики не имеют.
    """
    g, bc = problem.geometry, problem.bc
    xc, yc = 0.5 * (g.x1 + g.x2), 0.5 * (g.y1 + g.y2)   # максимум — в центре
    if bc.type == "soft_hinge":
        w = analytic.navier_rect_uniform(xc, yc, g.x1, g.x2, g.y1, g.y2,
                                         cfg.q0, cfg.D)
        return abs(float(w)), (xc, yc)
    if bc.type != "mixed":
        _fail("verify.reference", "analytic",
              "для rectangle: soft_hinge (Навье) или mixed SSSS/SCSC "
              "(Навье/Леви); для clamped — mms | fem")
    sides = dict(bc.sides)
    kinds = set(sides.values())
    if kinds == {"hinge"}:
        w = analytic.navier_rect_uniform(xc, yc, g.x1, g.x2, g.y1, g.y2,
                                         cfg.q0, cfg.D)
        return abs(float(w)), (xc, yc)
    if (sides["x1"] == sides["x2"] and sides["y1"] == sides["y2"]
            and kinds == {"hinge", "clamped"}):
        if sides["x1"] == "hinge":                     # x-hinge, y-clamped
            w = analytic.levy_rect_uniform(xc, yc, g.x1, g.x2, g.y1, g.y2,
                                           cfg.q0, cfg.D)
        else:                                          # поворот осей
            w = analytic.levy_rect_uniform(yc, xc, g.y1, g.y2, g.x1, g.x2,
                                           cfg.q0, cfg.D)
        return abs(float(w)), (xc, yc)
    # фабрика (F4): НЕСИММЕТРИЧНЫЕ Леви-пары — hinge-пара по одной оси,
    # кромки другой оси любые из {hinge, clamped} (симметричные — ручные)
    from .analytic_auto import levy_solution

    if (sides["x1"] == sides["x2"] == "hinge"
            and {sides["y1"], sides["y2"]} <= {"hinge", "clamped", "free"}):
        sol = levy_solution(x1=g.x1, x2=g.x2, y1=g.y1, y2=g.y2, D=cfg.D,
                            q0=cfg.q0, bc_y1=sides["y1"], bc_y2=sides["y2"],
                            nu=cfg.nu)
        return abs(float(sol.w(xc, yc))), (xc, yc)
    if (sides["y1"] == sides["y2"] == "hinge"
            and {sides["x1"], sides["x2"]} <= {"hinge", "clamped", "free"}):
        sol = levy_solution(x1=g.y1, x2=g.y2, y1=g.x1, y2=g.x2, D=cfg.D,
                            q0=cfg.q0, bc_y1=sides["x1"], bc_y2=sides["x2"],
                            nu=cfg.nu)
        return abs(float(sol.w(yc, xc))), (xc, yc)
    _fail("verify.reference", "analytic",
          "mixed: все hinge (Навье), пары hinge/clamped по осям (Леви), "
          "hinge-пара по одной оси + hinge/clamped/free кромки (Леви, "
          "фабрика; free — SFSF/SFCF); иначе — none | fem")


def _navier_factory_ref(problem: Problem, cfg):
    """Эталон фабрики: ряд Навье для patch | point на SSSS-прямоугольнике.

    Требование КУ: soft_hinge (прямые края ≡ истинный шарнир, NOTES §8)
    или mixed со всеми hinge. Сравнение — В ТОЧКЕ: центр зоны патча /
    точка приложения силы (там прогиб максимален или близок к нему).
    """
    from .analytic_auto import navier_solution

    g, bc, load = problem.geometry, problem.bc, problem.load
    all_hinge = (bc.type == "mixed"
                 and set(dict(bc.sides).values()) == {"hinge"})
    if not (bc.type == "soft_hinge" or all_hinge):
        _fail("verify.reference", "analytic",
              "patch/point-эталон Навье: КУ soft_hinge или mixed все hinge")
    if load.type == "patch":
        z = load.zone
        if z.kind != "rectangle":
            _fail("verify.reference", "analytic",
                  "patch-эталон Навье: только прямоугольная зона")
        ld = {"type": "patch", "q0": cfg.q0, "zone": (z.x1, z.x2, z.y1, z.y2)}
        pt = (0.5 * (z.x1 + z.x2), 0.5 * (z.y1 + z.y2))
        tol = 1e-12
    else:
        ld = {"type": "point", "P": load.P, "x0": load.x0, "y0": load.y0}
        pt = (load.x0, load.y0)
        tol = 1e-6            # остаток ряда ~1/N²; много точнее базиса 2D
    sol = navier_solution(x1=g.x1, x2=g.x2, y1=g.y1, y2=g.y2, D=cfg.D,
                          load=ld, tol=tol)
    return abs(float(sol.w(*pt))), pt


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


def _mms_reference(problem: Problem, cfg) -> Reference:
    r"""MMS-эталон: обёртка MMS-шагов ladder.py для rectangle/circle.

    Изготовленное решение ``w_MMS`` и нагрузка ``q = D·Δ²w_MMS`` (sympy);
    защемлённый решатель на ТОЙ ЖЕ дискретизации (p, Q) обязан воспроизвести
    ``w_MMS``. Для прямоугольника ω берётся ПОЛИНОМИАЛЬНОЙ
    (``(a²−ξ²)(b²−η²)``, НЕ R-функция) — тогда и решение в структуре, и
    квадратура точна ⇒ машинная точность (NOTES §14). Для круга граница
    кривая ⇒ остаток задаёт ступенчатая маска ~1/Q.
    """
    import sympy as sp  # noqa: F401 — символика внутри ladder

    from .clamped import ClampedPlate
    from .geometry import Domain
    from .geometry import x as sx
    from .geometry import y as sy
    from .ladder import mms_clamped_disk_w, mms_load_and_exact

    if problem.bc.type != "clamped":
        _fail("verify.reference", "mms",
              "clamped-постановки (MMS-шаги лестницы — защемление; для "
              "soft_hinge — analytic | fem | none)")
    g = problem.geometry
    if g.kind == "rectangle":
        cx, cy = 0.5 * (g.x1 + g.x2), 0.5 * (g.y1 + g.y2)
        ax, ay = 0.5 * (g.x2 - g.x1), 0.5 * (g.y2 - g.y1)
        w_expr = ((sx - cx) ** 2 - ax**2) ** 2 * ((sy - cy) ** 2 - ay**2) ** 2
        omega = (ax**2 - (sx - cx) ** 2) * (ay**2 - (sy - cy) ** 2)
        dom = Domain(omega, (g.x1, g.x2, g.y1, g.y2))     # полиномиальная ω
    elif g.kind == "circle":
        w_expr = mms_clamped_disk_w(g.a, cfg.D)
        from .dispatch import build_domain

        dom = build_domain(g)
    else:
        _fail("verify.reference", "mms",
              "rectangle | circle (MMS-поля лестницы, NOTES §14)")
    q_func, w_func = mms_load_and_exact(w_expr, cfg.D)
    cp = ClampedPlate.from_config(dom, cfg)
    qn = cp.quad
    c = cp.solve(q_func(qn.x, qn.y))
    w_num = float(np.max(np.abs(cp.deflection(c, qn.x, qn.y))))
    w_ex = float(np.max(np.abs(w_func(qn.x, qn.y))))
    return Reference(name=f"mms (изготовленное решение, {g.kind}, clamped)",
                     kind="mms", w_max=w_ex, gated=True, value=w_num)


# Сетки fem-эталона (константы v0.2; в схему не выносятся — ограда ключей).
_FEM_LSHAPE_M, _FEM_LSHAPE_REFINE = 16, 2
_FEM_CIRCLE_NREF = 4
_FEM_RECT_N = 32
_FEM_RECT_MIXED_M = 48
_FEM_ANNULUS_NR, _FEM_ANNULUS_NT = 24, 96


def _fem_wmax(fem, dom, grid_n: int = 160) -> float:
    """max|w| МКЭ-решения по фоновой сетке строго внутри Ω.

    Отступ от границы — 2 % меньшего размера bbox: полигональная сетка
    ВПИСАНА в гладкую ω-границу, и точки между хордой и дугой лежат вне
    сетки; максимум прогиба — внутренний, отступ на него не влияет.
    """
    x0, x1, y0, y1 = dom.bbox
    eps = 0.02 * min(x1 - x0, y1 - y0)
    X, Y = np.meshgrid(np.linspace(x0, x1, grid_n), np.linspace(y0, y1, grid_n))
    inside = dom.omega(X, Y) > eps
    return float(np.max(np.abs(fem.at(X[inside], Y[inside]))))


def _fem_references(problem: Problem, cfg) -> list[Reference]:
    r"""fem-эталоны: существующие пути + структурированное кольцо.

    * L / soft_hinge — две колонки: FEM-Marcus (та же модель — гейт) и
      FEM-Kirchhoff (Морли; парадокс Сапонджяна — вне допуска, NOTES §9);
    * clamped — Аргирис: круг (clamped_fem_circle), прямоугольник
      (solve_clamped_fem на тензорной сетке), L (clamped_fem_lshape);
    * annulus — НОВОЕ: структурированная сетка n_r × n_θ
      (:func:`~plate_solver.verify_fem.annulus_mesh`); clamped — Аргирис,
      soft — Marcus-P2 (шарнир w=0 на обеих окружностях).
    """
    g, bc = problem.geometry, problem.bc.type
    # Сначала — сочетаемость постановки (объясняется без зависимостей),
    # и только потом требование scikit-fem: несовместимому случаю установка
    # не поможет, а CI без extra fem должен видеть содержательную ошибку.
    if problem.load.type != "uniform":
        _fail("verify.reference", "fem", "равномерной нагрузки (v0.2)")
    if bc == "soft_hinge" and g.kind not in ("L", "annulus"):
        _fail("verify.reference", "fem",
              "для soft_hinge — L | annulus (иначе analytic | mms | none)")
    if bc == "clamped" and g.kind not in ("circle", "rectangle", "L", "annulus"):
        _fail("verify.reference", "fem",
              "circle | rectangle | L | annulus (для compose — mms | none)")
    # mixed (в т.ч. free-стороны, F10.4): прямоугольник гарантирован
    # валидатором; Кирхгоф (Морли) со сторонами по типам
    try:
        import skfem  # noqa: F401
    except ImportError:
        _fail("verify.reference", "fem",
              "установленный scikit-fem: pip install -e \".[fem]\"")
    from .dispatch import build_domain

    dom = build_domain(g)
    D, q0, nu = cfg.D, cfg.q0, cfg.nu

    if bc == "mixed":
        from . import verify_fem as vf

        sides = dict(problem.bc.sides)
        mesh = vf.rect_mesh(g.x1, g.x2, g.y1, g.y2, m=_FEM_RECT_MIXED_M)
        fem = vf.solve_rect_fem_mixed(mesh, D, q0, nu, sides,
                                      (g.x1, g.x2, g.y1, g.y2))
        tag = "".join(s[0].upper() for _, s in sorted(sides.items()))
        # Сравнение В ТОЧКЕ (детерминированной): при free-сторонах максимум
        # лежит НА КРОМКЕ — сопоставление max-областей с внутренним отступом
        # даёт ложную систематику; точка — середина первой free-стороны
        # (наибольший прогиб свободной кромки), иначе центр.
        xc, yc = 0.5 * (g.x1 + g.x2), 0.5 * (g.y1 + g.y2)
        pt = (xc, yc)
        mids = {"x1": (g.x1, yc), "x2": (g.x2, yc),
                "y1": (xc, g.y1), "y2": (xc, g.y2)}
        for side in ("x1", "x2", "y1", "y2"):
            if sides[side] == "free":
                pt = mids[side]
                break
        val = abs(float(fem.at(np.array([pt[0]]), np.array([pt[1]]))[0]))
        return [Reference(name=f"FEM-Кирхгоф, Морли (mixed {tag})",
                          kind="fem", w_max=val, point=pt, gated=True)]

    if bc == "soft_hinge":
        from . import verify_fem as vf

        if g.kind == "L":
            mesh = vf.lshape_mesh(g.side, g.cut, m=_FEM_LSHAPE_M,
                                  refine=_FEM_LSHAPE_REFINE)
        else:                                   # annulus (сочетаемость проверена выше)
            mesh = vf.annulus_mesh(g.a, g.b, _FEM_ANNULUS_NR, _FEM_ANNULUS_NT)
        marcus = vf.solve_plate_fem(mesh, D, q0, "marcus", nu)
        refs = [Reference(name=f"FEM-Marcus, P2 ({g.kind}; та же модель)",
                          kind="fem", w_max=_fem_wmax(marcus, dom), gated=True)]
        if g.kind == "L":
            kirch = vf.solve_plate_fem(mesh, D, q0, "kirchhoff", nu)
            refs.append(Reference(
                name="FEM-Kirchhoff, Морли (парадокс Сапонджяна — вне допуска)",
                kind="fem", w_max=_fem_wmax(kirch, dom), gated=False))
        return refs

    # clamped — Аргирис (существующие пути + кольцо)
    from .clamped import clamped_fem_circle, clamped_fem_lshape, solve_clamped_fem

    if g.kind == "circle":
        fem = clamped_fem_circle(g.a, D, q0, nu, nref=_FEM_CIRCLE_NREF)
    elif g.kind == "rectangle":
        from skfem import MeshTri

        mesh = MeshTri.init_tensor(np.linspace(g.x1, g.x2, _FEM_RECT_N + 1),
                                   np.linspace(g.y1, g.y2, _FEM_RECT_N + 1))
        fem = solve_clamped_fem(mesh, D, q0, nu)
    elif g.kind == "L":
        fem = clamped_fem_lshape(D, q0, nu, mesh_m=_FEM_LSHAPE_M,
                                 refine=_FEM_LSHAPE_REFINE)
    else:                                       # annulus (сочетаемость проверена выше)
        fem = solve_clamped_fem(vf_annulus_mesh(g.a, g.b), D, q0, nu)
    return [Reference(name=f"FEM-Аргирис ({g.kind}, clamped)", kind="fem",
                      w_max=_fem_wmax(fem, dom), gated=True)]


def vf_annulus_mesh(a: float, b: float):
    from .verify_fem import annulus_mesh

    return annulus_mesh(a, b, _FEM_ANNULUS_NR, _FEM_ANNULUS_NT)


# --------------------------------------------------------------------------- #
#  Резолвер и отчёт
# --------------------------------------------------------------------------- #
def resolve_reference(problem: Problem, cfg=None) -> list[Reference]:
    """Список именованных эталонов постановки.

    ``cfg`` — Config расчёта (None ⇒ ``problem.to_config()``; диспетчер
    передаёт свой, с эффективной интенсивностью point-нагрузки).
    """
    cfg = problem.to_config() if cfg is None else cfg
    v = problem.verify
    refs: list[Reference] = []
    if v.reference != "none" and problem.contact.enabled:
        _fail("verify.reference", v.reference,
              "none — эталонов контактной задачи в v0.2 нет "
              "(ворота контакта — инварианты)")
    if v.reference == "analytic":
        ref_val = _analytic_wmax(problem, cfg)
        point = None
        if isinstance(ref_val, tuple):
            ref_val, point = ref_val
        refs.append(Reference(
            name=f"analytic ({problem.geometry.kind}, {problem.bc.type})",
            kind="analytic", w_max=ref_val, gated=True, point=point))
    elif v.reference == "mms":
        refs.append(_mms_reference(problem, cfg))
    elif v.reference == "fem":
        refs.extend(_fem_references(problem, cfg))
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
        if ref.point is not None:
            value = abs(float(result._plate.deflection(result._c, *ref.point)))
        elif ref.value is not None:
            value = ref.value
        else:
            value = result.w_max
        rel = abs(value - ref.w_max) / abs(ref.w_max)
        rows.append(RefRow(name=ref.name, reference=ref.w_max, value=value,
                           rel=rel, gated=ref.gated,
                           passed=(rel <= tol) if ref.gated else None))
    return VerifyReport(rows=tuple(rows), tol=tol)


__all__ = ["Reference", "RefRow", "VerifyReport", "resolve_reference", "verify_result"]
