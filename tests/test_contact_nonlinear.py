"""Ворота нелинейного контакта МОР+КТН через ДИСПЕТЧЕР (v0.6.3).

Метод (:class:`~plate_solver.contact_nl.NonlinearContactMOR`) верифицирован на
уровне API в ``tests/test_contact_ktn.py`` (редукции R1/R3/R4, лицевое условие,
подпись КТН). Здесь — что ШТАТНЫЙ вход (case-файл → ``dispatch.solve``)
маршрутизирует ``theory = karman | ktn_full`` с контактом в этот тракт,
сохраняет инварианты (r ≥ 0, комплементарность, зона) и лицевую специфику,
а несовместимые постановки отклоняются понятной ошибкой (не рантайм-падением).

Все случаи — БЫСТРЫЕ (малая нагрузка ⇒ МОР сходится за сотни итераций
совмещённой схемы); тяжёлые эталоны корректности — в ``test_contact_ktn.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import dispatch
from plate_solver.problem import CaseError, Problem

_BASE = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "{bc}"
[load]
type = "{load}"
q0 = 0.01
[model]
theory = "{theory}"
h = 0.2
n_load_steps = 3
karman_max_iter = 100
karman_tol = 1.0e-6
[contact]
enabled = true
{gap}
{scheme}
{gain}
{force}
beta = 1.5
max_iter = 3000
tol = 3.0e-4
[discretization]
p = 8
Q = 64
grid_n = 16
[verify]
reference = "none"
"""


def _solve(tmp_path, *, theory="ktn_full", bc="clamped", load="uniform",
           gap="gap_factor = 0.5", scheme='scheme = "merged"', gain="", force=""):
    text = _BASE.format(theory=theory, bc=bc, load=load, gap=gap,
                        scheme=scheme, gain=gain, force=force)
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


def _case(tmp_path, **kw):
    """Только валидация постановки (ожидаем CaseError без запуска решателя)."""
    defaults = dict(theory="ktn_full", bc="clamped", load="uniform",
                    gap="gap_factor = 0.5", scheme='scheme = "merged"',
                    gain="", force="")
    defaults.update(kw)
    text = _BASE.format(**defaults)
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    return Problem.from_toml(str(p))


# --------------------------------------------------------------------------- #
#  Маршрутизация и инварианты
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theory", ["karman", "ktn_full"])
def test_nonlinear_contact_engages_and_invariants(tmp_path, theory):
    """Контакт включается: r ≥ 0, непустая зона, малая комплементарность."""
    res = _solve(tmp_path, theory=theory, gap="gap_factor = 0.5")
    c = res.contact
    assert c is not None
    assert (c.r_nodes >= -1e-9).all()                     # реакция неотрицательна
    assert c.r_nodes.max() > 0.0 and (c.r_nodes > 0).sum() > 0   # зона непуста
    assert c.comp_residual < 0.15                         # KKT-невязка Синьорини мала
    assert res.w_free_max is not None and res.w_max <= res.w_free_max * (1 + 1e-6)


@pytest.mark.parametrize("theory", ["karman", "ktn_full"])
def test_r1_huge_gap_reduces_to_free(tmp_path, theory):
    """Редукция R1: зазор → ∞ ⇒ реакция ≡ 0, прогиб = свободному."""
    res = _solve(tmp_path, theory=theory, gap="gap = 100.0",
                 scheme='scheme = "nested"')
    c = res.contact
    assert c.r_nodes.max() == 0.0 and (c.r_nodes > 0).sum() == 0
    assert abs(res.w_max - res.w_free_max) / res.w_free_max < 1e-6


def test_karman_contact_probes_midsurface(tmp_path):
    """Карман: коэффициент лицевой кривизны ноль ⇒ лицевые ≡ срединной (dh ≡ 0)."""
    res = _solve(tmp_path, theory="karman", gap="gap_factor = 0.5")
    _wt, _wb, dh = res.faces_on_grid()
    assert np.nanmax(np.abs(dh)) == 0.0


def test_ktn_full_contact_face_signature(tmp_path):
    """Полная КТН: контакт «щупает» ЛИЦЕВУЮ поверхность ⇒ dh ≠ 0 (подпись КТН)."""
    res = _solve(tmp_path, theory="ktn_full", gap="gap_factor = 0.5")
    _wt, _wb, dh = res.faces_on_grid()
    assert np.nanmax(np.abs(dh)) > 0.0
    assert res.thickness_params()["h_over_L"] > 0.0


def test_gain_linear_runs(tmp_path):
    """Нормировка усиления linear (верхняя грань ‖G‖) — сходится, инварианты держатся."""
    res = _solve(tmp_path, theory="ktn_full", gap="gap_factor = 0.5",
                 gain='gain = "linear"')
    c = res.contact
    assert (c.r_nodes >= -1e-9).all() and c.r_nodes.max() > 0.0


# --------------------------------------------------------------------------- #
#  Отклонения валидатора (понятная ошибка, не рантайм-падение)
# --------------------------------------------------------------------------- #
def test_soft_hinge_nonlinear_contact_supported_on_circle(tmp_path):
    """Мягкий шарнир + КТН-контакт на круге — ПОДДЕРЖАН (v0.6.3, звёздная ∂Ω)."""
    _case(tmp_path, theory="ktn_full", bc="soft_hinge")     # валидация проходит


def test_reject_soft_hinge_nonlinear_contact_nonstar(tmp_path):
    """Мягкий шарнир + КТН-контакт на НЕзвёздной области (L) — понятный отказ."""
    text = """
[geometry]
kind = "L"
side = 1.0
cut = 0.5
[bc]
type = "soft_hinge"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
h = 0.2
[contact]
enabled = true
gap_factor = 0.5
scheme = "merged"
[discretization]
p = 8
Q = 64
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="circle"):
        Problem.from_toml(str(p))


def test_force_nonlinear_contact_supported_single(tmp_path):
    """Силовой штамп поверх нелинейной КТН — поддержан для ОДИНОЧНОЙ пластины (v0.6.3)."""
    _case(tmp_path, theory="ktn_full", gap="", force="force = 1.0e-3")   # валидация проходит


def test_force_nonlinear_pair_validates(tmp_path):
    """Силовое управление НЕЛИНЕЙНОЙ парой — ПОДДЕРЖАНО (v0.6.5, снят задел v0.7).

    Валидация проходит; счёт и замыкание ∫r = P — ``test_force_pair_reaches_target``.
    Классическая пара с force остаётся отклонённой (``test_force_pair_classic_rejected``).
    """
    text = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
h = 0.2
[contact]
enabled = true
target = "plate2"
force = 1.0e-3
scheme = "merged"
[plate2]
[plate2.bc]
type = "clamped"
[plate2.load]
type = "uniform"
q0 = 0.0
[plate2.model]
theory = "ktn_full"
[discretization]
p = 6
Q = 48
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    prob = Problem.from_toml(str(p))                    # валидация ПРОХОДИТ
    assert prob.contact.force == pytest.approx(1.0e-3)


@pytest.mark.big
def test_force_nonlinear_contact_closes(tmp_path):
    """Силовой штамп: ∫r сходится к заданной P (замыкание силового уравнения)."""
    P = 8.0e-4
    text = f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
h = 0.2
n_load_steps = 2
karman_max_iter = 80
karman_tol = 1e-6
[contact]
enabled = true
force = {P}
scheme = "merged"
gain = "linear"
beta = 1.5
max_iter = 1500
tol = 3e-4
[discretization]
p = 6
Q = 48
grid_n = 16
[verify]
reference = "none"
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    res = dispatch.solve(Problem.from_toml(str(p)))
    assert res.level is not None and res.force_total is not None
    assert abs(res.force_total - P) / P < 1e-3                # силовое уравнение замкнуто
    assert (res.contact.r_nodes >= -1e-9).all()


def test_reject_nonuniform_load_nonlinear_contact(tmp_path):
    # валидная точечная нагрузка проходит парсинг, но нелинейный контакт
    # реализован только для равномерной ⇒ понятный отказ на load.type
    text = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "point"
P = 1.0e-3
x0 = 0.0
y0 = 0.0
[model]
theory = "ktn_full"
h = 0.2
[contact]
enabled = true
gap_factor = 0.5
scheme = "merged"
[discretization]
p = 8
Q = 64
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="uniform"):
        Problem.from_toml(str(p))


_PAIR = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "{theory}"
h = 0.2
n_load_steps = 2
karman_max_iter = 80
karman_tol = 1.0e-6
[contact]
enabled = true
target = "plate2"
gap = 0.0
scheme = "merged"
beta = 1.5
max_iter = 3000
tol = 3.0e-4
[plate2]
[plate2.bc]
type = "{bc2}"
[plate2.load]
type = "uniform"
q0 = 0.0
[plate2.model]
theory = "{theory2}"
{e2}
[discretization]
p = 8
Q = 64
grid_n = 16
[verify]
reference = "none"
"""


def _pair(tmp_path, *, theory="ktn_full", theory2=None, bc2="clamped", e2=""):
    text = _PAIR.format(theory=theory, theory2=theory2 or theory, bc2=bc2, e2=e2)
    p = tmp_path / "pair.toml"
    p.write_text(text, encoding="utf-8")
    return Problem.from_toml(str(p))


@pytest.mark.parametrize("theory", ["karman", "ktn_full"])
def test_ktn_pair_engages_and_invariants(tmp_path, theory):
    """Нелинейная пара: r ≥ 0, непустая зона, обе пластины деформируются."""
    res = dispatch.solve(_pair(tmp_path, theory=theory))
    c = res.contact
    assert hasattr(c, "w2_grid")                          # это пара
    assert (c.r_nodes >= -1e-9).all() and c.r_nodes.max() > 0.0
    assert (c.r_nodes > 0).sum() > 0 and c.comp_residual < 0.15
    assert np.nanmax(np.abs(c.w2_grid)) > 0.0             # вторая пластина откликается


def test_ktn_pair_rigid_second_reduces_to_support(tmp_path):
    """Редукция: жёсткая вторая пластина (E2→∞) при Δ=0 ⇒ первая почти не гнётся."""
    soft = dispatch.solve(_pair(tmp_path, theory="ktn_full"))
    stiff = dispatch.solve(_pair(tmp_path, theory="ktn_full", e2="E = 1.0e12"))
    assert stiff.w_max < 1e-3 * soft.w_free_max           # опёрта на жёсткое основание
    assert stiff.contact.r_nodes.max() > 0.0


def test_reject_mixed_theory_pair(tmp_path):
    with pytest.raises(CaseError, match="model.theory"):
        _pair(tmp_path, theory="ktn_full", theory2="classic")


def test_soft_hinge_pair_supported_on_star(tmp_path):
    """Пара на МЯГКОМ ШАРНИРЕ (звёздная область) — ПОДДЕРЖАНА (v0.6.5, снят задел v0.7)."""
    res = dispatch.solve(_pair(tmp_path, theory="ktn_full", bc2="soft_hinge"))
    c = res.contact
    assert (c.r_nodes >= -1e-9).all() and c.r_nodes.max() > 0.0   # инвариант r ≥ 0


def test_reject_soft_hinge_pair_on_reentrant(tmp_path):
    """Пара на мягком шарнире на области с ВХОДЯЩИМ углом (L) — отклонена (§3.5 — v0.7)."""
    text = """
[geometry]
kind = "L"
side = 1.0
cut = 0.5
[bc]
type = "soft_hinge"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
h = 0.2
[contact]
enabled = true
target = "plate2"
gap = 0.0
[plate2]
[plate2.bc]
type = "soft_hinge"
[plate2.load]
type = "uniform"
q0 = 0.0
[plate2.model]
theory = "ktn_full"
[discretization]
p = 8
Q = 64
grid_n = 16
[verify]
reference = "none"
"""
    p = tmp_path / "pair_L.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="circle | ellipse | rectangle | annulus"):
        Problem.from_toml(str(p))


def test_reject_scheme_without_nonlinear_contact(tmp_path):
    # classic + scheme — ключ схемы осмыслен только для нелинейного контакта
    text = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 4.0
[model]
theory = "classic"
h = 0.06
[contact]
enabled = true
gap_factor = 0.5
scheme = "merged"
[discretization]
p = 8
Q = 64
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="scheme"):
        Problem.from_toml(str(p))


# --------------------------------------------------------------------------- #
#  Ускорение Андерсона внешнего цикла МОР (mor_anderson, v0.6.4)
# --------------------------------------------------------------------------- #
def _nested_mor(mor_anderson, *, q0=0.02, h=0.1, p=7, Q=80, tol=1.0e-3):
    """Вложенный МОР (полная КТН, круг) с заданным окном Андерсона; вернуть результат."""
    from plate_solver.config import Config
    from plate_solver.contact_nl import NonlinearContactMOR
    from plate_solver.geometry import make_circle
    from plate_solver.ktn_solver import KTNSolver

    dom = make_circle(1.0)
    cfg = Config(E=1.0, nu=0.3, h=h, q0=q0, a=1.0, p=p, Q=Q, n_load_steps=2,
                 karman_tol=1e-7, karman_max_iter=200, beta=1.2, tol=tol,
                 max_iter=3000, mor_anderson=mor_anderson)
    free = KTNSolver.from_theory_name(dom, cfg, "ktn_full", bc_type="clamped",
                                      inplane_bc="immovable")
    gap = 0.5 * free.solve(np.full(free.quad.x.size, q0)).w_max
    solver = KTNSolver.from_theory_name(dom, cfg, "ktn_full", bc_type="clamped",
                                        inplane_bc="immovable")
    return NonlinearContactMOR(solver, cfg, gap=gap, scheme="nested").solve()


@pytest.mark.big
def test_mor_anderson_accelerates_nested():
    """Проекционный Андерсон ускоряет ВНЕШНИЙ цикл МОР: то же решение, меньше шагов.

    Каждый внешний шаг вложенной схемы — ПОЛНЫЙ нелинейный КТН-решатель, поэтому
    сокращение числа шагов реакции = прямая экономия. Неподвижная точка та же
    (проекция r ≥ 0 после смешения; сходимость по ‖F(r)−r‖).
    """
    r0 = _nested_mor(0)
    r5 = _nested_mor(5)
    assert r0.converged and r5.converged
    assert abs(r5.w_max - r0.w_max) / r0.w_max < 3e-3     # та же неподвижная точка (±мерцание)
    assert r5.iters < 0.6 * r0.iters                      # ощутимо меньше внешних шагов


def test_mor_anderson_default_off_unchanged():
    """mor_anderson=0 (дефолт) — прежний проекционный шаг число-в-число (регресс не сдвинут)."""
    from plate_solver.config import Config
    assert Config().mor_anderson == 0                     # выключено по умолчанию


# --------------------------------------------------------------------------- #
#  Силовое управление ПАРОЙ пластин (v0.6.5, снят задел v0.7)
# --------------------------------------------------------------------------- #
_FORCE_PAIR = """
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "{theory}"
E = 1.0
nu = 0.3
h = 0.2
n_load_steps = 2
karman_max_iter = 80
karman_tol = 1.0e-6
karman_relax = 0.7
[contact]
enabled = true
target = "plate2"
force = {force}
scheme = "merged"
beta = 1.5
max_iter = 4000
tol = 3.0e-4
mor_anderson = 5
[plate2]
[plate2.bc]
type = "clamped"
[plate2.load]
type = "uniform"
q0 = 0.0
[plate2.model]
theory = "{theory2}"
{e2}
[discretization]
p = 7
Q = 48
grid_n = 16
[verify]
reference = "none"
"""


def _force_pair(tmp_path, *, force, theory="ktn_full", theory2=None, e2=""):
    text = _FORCE_PAIR.format(theory=theory, theory2=theory2 or theory,
                              force=force, e2=e2)
    p = tmp_path / "force_pair.toml"
    p.write_text(text, encoding="utf-8")
    return dispatch.solve(Problem.from_toml(str(p)))


def test_force_pair_reaches_target(tmp_path):
    """Силовая пара: ∫r = P (поиск начального зазора z продолжением + brentq).

    Умеренное прижатие (рабочая область силового режима, z* вблизи касания);
    глубокий натяг — за пределами совмещённой схемы (честный CaseError).
    """
    P = 2.0e-3
    res = _force_pair(tmp_path, force=P)
    assert res.force_total is not None and res.level is not None
    assert abs(res.force_total - P) / P < 3e-2            # ∫r = P (допуск brentq+МОР)
    assert res.contact.r_nodes.max() > 0.0                # прижатие реально
    assert (res.contact.r_nodes >= -1e-12).all()          # инвариант r ≥ 0


@pytest.mark.big
def test_force_pair_rigid_second_reduces_to_single_stamp(tmp_path):
    """Редукция: жёсткая 2-я пластина ⇒ силовая пара = силовой одиночный штамп.

    Та же сила P, плоский интерфейс: прогиб и найденный уровень совпадают с
    одиночным силовым режимом (`_solve_contact_force_nonlinear`) до остаточной
    податливости жёсткой пластины (< 2 %).
    """
    P = 3.0e-3
    rp = _force_pair(tmp_path, force=P, e2="E = 1.0e12")
    single = f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
E = 1.0
nu = 0.3
h = 0.2
n_load_steps = 2
karman_max_iter = 80
karman_tol = 1.0e-6
karman_relax = 0.7
[contact]
enabled = true
force = {P}
scheme = "merged"
beta = 1.5
max_iter = 4000
tol = 3.0e-4
mor_anderson = 5
[discretization]
p = 7
Q = 48
grid_n = 16
[verify]
reference = "none"
"""
    p = tmp_path / "single_force.toml"
    p.write_text(single, encoding="utf-8")
    rs = dispatch.solve(Problem.from_toml(str(p)))
    assert abs(rp.w_max - rs.w_max) / rs.w_max < 2e-2
    assert abs(rp.level - rs.level) / abs(rs.level) < 2e-2


def test_force_pair_classic_rejected(tmp_path):
    """Силовое управление КЛАССИЧЕСКОЙ парой — не поддержано (ясная ошибка)."""
    text = _FORCE_PAIR.format(theory="classic", theory2="classic",
                              force=1.0e-3, e2="")
    text = text.replace("n_load_steps = 2\n", "").replace(
        "karman_max_iter = 80\n", "").replace("karman_tol = 1.0e-6\n", "").replace(
        "karman_relax = 0.7\n", "").replace('scheme = "merged"\n', "").replace(
        "mor_anderson = 5\n", "")
    p = tmp_path / "classic_force_pair.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="КЛАССИЧЕСКОЙ"):
        Problem.from_toml(str(p))


# --------------------------------------------------------------------------- #
#  Ограждения аудита v0.6.5: mor_anderson-потолок, nested-пара, plate2-mixed
# --------------------------------------------------------------------------- #
def test_mor_anderson_cap_and_anchor(tmp_path):
    """Окно Андерсона > 20 — отказ (большие окна разваливают МОР, аудит v0.6.5);
    misuse-ошибка анкерована на реально заданный ключ."""
    d = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 0.01},
        "model": {"theory": "karman", "h": 0.2},
        "contact": {"enabled": True, "gap_factor": 0.5, "mor_anderson": 50},
        "discretization": {"p": 6, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    with pytest.raises(CaseError, match="mor_anderson"):
        Problem.from_dict(d)                               # потолок 20
    d["contact"]["mor_anderson"] = 5
    d["model"]["theory"] = "classic"
    with pytest.raises(CaseError, match="contact.mor_anderson"):
        Problem.from_dict(d)                               # анкер — заданный ключ


def test_nested_scheme_rejected_for_pair(tmp_path):
    """scheme = nested для ПАРЫ — отказ (пара реализована только совмещённым циклом)."""
    text = _PAIR.format(theory="ktn_full", theory2="ktn_full", bc2="clamped",
                        e2="").replace('scheme = "merged"', 'scheme = "nested"')
    p = tmp_path / "pair_nested.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(CaseError, match="merged"):
        Problem.from_toml(str(p))


def test_plate2_mixed_bc_rejected(tmp_path):
    """[plate2.bc] mixed — отказ (сторонние sides молча игнорировались бы, аудит v0.6.5)."""
    d = {
        "geometry": {"kind": "rectangle", "x1": -1.0, "x2": 1.0, "y1": -0.6, "y2": 0.6},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 1.0},
        "model": {"theory": "classic"},
        "contact": {"enabled": True, "target": "plate2", "gap": 0.0},
        "plate2": {"bc": {"type": "mixed", "sides": [
            {"side": "x1", "type": "hinge"}, {"side": "x2", "type": "hinge"},
            {"side": "y1", "type": "clamped"}, {"side": "y2", "type": "clamped"}]},
            "load": {"type": "uniform", "q0": 0.0}},
        "discretization": {"p": 6, "Q": 48, "grid_n": 16},
        "verify": {"reference": "none"},
    }
    with pytest.raises(CaseError, match="plate2.bc"):
        Problem.from_dict(d)


@pytest.mark.big
def test_force_stamp_anderson_disabled_and_closure_kept(tmp_path):
    """Силовой штамп + mor_anderson: Андерсон отключается (шумит F(level)) — замыкание машинное.

    Аудит v0.6.5: с Андерсоном brentq садился мимо цели на 7–11 % МОЛЧА;
    теперь ∫r = P восстановлено + честное предупреждение.
    """
    P = 8.0e-4
    text = f"""
[geometry]
kind = "circle"
a = 1.0
[bc]
type = "clamped"
[load]
type = "uniform"
q0 = 0.01
[model]
theory = "ktn_full"
h = 0.2
n_load_steps = 2
karman_max_iter = 80
karman_tol = 1e-6
[contact]
enabled = true
force = {P}
scheme = "merged"
gain = "linear"
beta = 1.5
max_iter = 1500
tol = 3e-4
mor_anderson = 5
[discretization]
p = 6
Q = 48
grid_n = 16
[verify]
reference = "none"
"""
    p = tmp_path / "case.toml"
    p.write_text(text, encoding="utf-8")
    res = dispatch.solve(Problem.from_toml(str(p)))
    assert abs(res.force_total - P) / P < 1e-3            # замыкание НЕ зашумлено
    assert any("mor_anderson" in w for w in res.warnings)  # честное предупреждение
