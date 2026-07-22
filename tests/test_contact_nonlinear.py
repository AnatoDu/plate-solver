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


def test_reject_force_nonlinear_pair(tmp_path):
    """Силовое управление ПАРОЙ пластин — направление развития (отказ)."""
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
    with pytest.raises(CaseError, match="force"):
        Problem.from_toml(str(p))


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


def test_reject_soft_hinge_pair(tmp_path):
    with pytest.raises(CaseError, match="clamped"):
        _pair(tmp_path, theory="ktn_full", bc2="soft_hinge")


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
