"""Поправки КТН: сравнение классика ↔ КТН на контакте L-формы (тест-ворота).

Уточнённая теория (Карман–Тимошенко–Нагди) добавляет поперечный сдвиг и обжатие.
В контактной задаче её главный эффект — член податливости ``−κ_r·r`` в лицевом
условии: он даёт ПОЛОЖИТЕЛЬНУЮ ДИАГОНАЛЬНУЮ добавку к оператору задачи
дополнительности, отчего задача становится строго монотонной. Следствия:

* итерация МОР СХОДИТСЯ (классическая при том же бюджете лишь «полусходится»:
  пик давления у кромки зоны продолжает расти — классическое решение Синьорини
  под жёстким основанием имеет там особенность);
* реакция ниже и распределена шире, поле глаже;
* прогиб больше (теория мягче) ⇒ поправка ``w_max``.

МАСШТАБ ЭФФЕКТА (v0.8.0). Безразмерная мера — ``κ_r/‖G‖ ∝ (h/a)⁴``: поправка
гаснет в тонком пределе как ЧЕТВЁРТАЯ степень относительной толщины. Поэтому
регламент ворот — умеренно ТОЛСТАЯ пластина ``h/a = 0.12``, где эффект
однозначен, а решение КТН успевает сойтись в быстром тесте; для ``h = 0.06``
эффект того же знака, но в 16 раз слабее (проверяется отдельными воротами
масштабирования).

История: до v0.8.0 член реакции умножался на цилиндрическую жёсткость
(``κ_r·D·r``), что раздувало эффект в 41.5 раза при E = 2.1e6, h = 0.06 и делало
модель зависимой от системы единиц (CHANGELOG [0.8.0], NOTES §11, §21;
ворота инвариантности — ``tests/test_units_invariance.py``).
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending

#: регламент ворот: умеренно толстая пластина (h/a = 0.12) и общий бюджет МОР
H_GATE = 0.12
BUDGET = 6000


def _roughness(r: np.ndarray) -> float:
    """Узловая шероховатость поля реакции (норма скачков соседних узлов)."""
    return float(np.sqrt(np.sum(np.diff(r) ** 2)))


def _setup(h: float, max_iter: int = BUDGET):
    dom = geometry.make_L(1.0, 0.5)
    cfg = Config(nu=0.3, q0=4.0, h=h, p=8, Q=36, beta=1.0, max_iter=max_iter,
                 tol=1e-12, grid_n=30)
    pb = PlateBending.from_config(dom, cfg)
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    gap = 0.6 * float(pb.deflection(cw, q.x, q.y).max())
    fmask = lambda x, y: (x + y) < 0.9            # noqa: E731 — частичное основание
    return pb, cfg, gap, fmask


def _pair(h: float, max_iter: int = BUDGET):
    pb, cfg, gap, fmask = _setup(h, max_iter)
    classic = ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap).solve()
    ktn = ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap,
                     ktn=KTNParams.from_config(cfg)).solve()
    return classic, ktn


@pytest.fixture(scope="module")
def classic_vs_ktn():
    return _pair(H_GATE)


def test_gate_ktn_regularizes_contact(classic_vs_ktn):
    """ГЛАВНЫЕ ВОРОТА: КТН регуляризует контакт — сходимость, ниже пик, шире зона.

    Сравнение честно́ по бюджету: КТН при нём СОШЁЛСЯ (KKT-невязка машинная),
    классика — нет, и её пик со временем только растёт, поэтому неравенства
    по пику и по числу узлов — оценки СВЕРХУ (с ростом бюджета лишь усиливаются).
    """
    c, k = classic_vs_ktn
    assert k.converged and k.comp_residual < 1e-9          # КТН сошёлся
    assert not c.converged and c.comp_residual > 1e-3      # классика — нет
    assert k.r_nodes.max() < 0.5 * c.r_nodes.max()         # пик существенно ниже
    assert int((k.r_nodes > 0).sum()) > 2 * int((c.r_nodes > 0).sum())   # зона шире
    assert _roughness(k.r_nodes) < 0.5 * _roughness(c.r_nodes)           # глаже


def test_ktn_reaction_nonnegative_and_converges(classic_vs_ktn):
    _, k = classic_vs_ktn
    assert k.r_nodes.min() >= 0.0                                   # односторонняя связь
    h = k.residual_history
    assert h[-1] < h[0] / 10.0                                      # МОР сходится


def test_gate_ktn_corrects_wmax(classic_vs_ktn):
    """ГЛАВНЫЕ ВОРОТА: КТН даёт измеримую поправку w_max (теория мягче ⇒ больше)."""
    c, k = classic_vs_ktn
    assert k.w_ktn_nodes is not None
    rel = (k.w_ktn_nodes.max() - c.w_nodes.max()) / c.w_nodes.max()
    assert rel > 0.05                                     # > 5 % (при h=0.12 ~ +10 %)


def test_gate_ktn_effect_scales_as_h4():
    """ГЛАВНЫЕ ВОРОТА: безразмерная мера эффекта ``κ_r/‖G‖`` растёт РОВНО как h⁴.

    ``κ_r ∝ h/E`` (податливость лицевой), ``‖G‖ ∝ a⁴/D ∝ a⁴/(E h³)`` ⇒
    отношение ``∝ (h/a)⁴`` и НЕ зависит от E. Именно эта комбинация показывает,
    что поправка КТН гаснет в тонком пределе (Gate R4). Прежний канон с
    множителем D давал ``κ_r·D/‖G‖ ∝ E·h⁷`` — рост на три порядка быстрее и
    зависимость от единиц.
    """
    dom = geometry.make_L(1.0, 0.5)
    ratios = {}
    for h in (0.05, 0.10, 0.20):
        cfg = Config(nu=0.3, q0=4.0, h=h, p=8, Q=36, max_iter=1, tol=1e-12)
        pb = PlateBending.from_config(dom, cfg)
        state = pb.solve(np.ones(pb.quad.x.size))
        gain = float(np.max(np.abs(pb.w_at_quad(state))))          # ‖G‖ ~ податливость
        ratios[h] = KTNParams.from_config(cfg).kappa_r / gain
    assert ratios[0.10] / ratios[0.05] == pytest.approx(16.0, rel=1e-10)
    assert ratios[0.20] / ratios[0.10] == pytest.approx(16.0, rel=1e-10)
    # и не зависит от модуля упругости (проверка размерной согласованности)
    cfg_soft = Config(nu=0.3, q0=4.0, h=0.10, E=2.1e3, p=8, Q=36, max_iter=1)
    pb_s = PlateBending.from_config(dom, cfg_soft)
    gain_s = float(np.max(np.abs(pb_s.w_at_quad(pb_s.solve(np.ones(pb_s.quad.x.size))))))
    assert KTNParams.from_config(cfg_soft).kappa_r / gain_s == pytest.approx(
        ratios[0.10], rel=1e-10)


def test_gate_mor_step_normalized_by_face_operator():
    """ГЛАВНЫЕ ВОРОТА: шаг МОР нормируется по ТОЙ поверхности, по которой стоит
    условие Синьорини (v0.8.0).

    Итерация МОР «щупает» лицевой прогиб ``u_c``, значит и оценка нормы
    оператора (``gain``) обязана быть откликом ЛИЦЕВОЙ поверхности на единичную
    нагрузку плюс диагональная податливость ``κ_r``. Прежде нормировка бралась
    по СРЕДИННОМУ прогибу: на L-форме при h/a = 0.3 это занижало оценку в 1.6
    раза (у входящего угла кривизна лицевой велика), т.е. фактический множитель
    сжатия выходил за границу теоремы 4 при формально допустимом β.

    Классический тракт (``ktn=None``) нормируется как прежде — бит-в-бит.
    """
    dom = geometry.make_L(1.0, 0.5)
    cfg = Config(nu=0.3, q0=4.0, h=0.3, p=8, Q=36, beta=1.2, max_iter=10, tol=1e-12)
    pb = PlateBending.from_config(dom, cfg)
    st = pb.solve(np.ones(pb.quad.x.size))
    w_unit, lap_unit = pb.w_at_quad(st), pb.lap_w_at_quad(st)
    gain_mid = float(np.max(np.abs(w_unit)))
    kp = KTNParams.from_config(cfg)
    face_unit = kp.contact_displacement(w_unit, lap_unit, 0.0, 0.0)
    expected = float(np.max(np.abs(face_unit))) + kp.kappa_r

    mor = ContactMOR(pb, cfg, gap=1e-4, ktn=kp)
    assert mor.gain == pytest.approx(expected, rel=1e-14)
    assert mor.beta_eff * mor.gain == pytest.approx(cfg.beta, rel=1e-14)
    assert mor.gain > 1.5 * gain_mid                  # различие существенно при h/a = 0.3
    # классика: нормировка по срединному прогибу, бит-в-бит как прежде
    assert ContactMOR(pb, cfg, gap=1e-4).gain == gain_mid


def test_ktn_effect_vanishes_for_thin_plate():
    """Редукция: при вчетверо меньшей толщине эффект КТН падает на порядки.

    Ворота записаны как СРАВНЕНИЕ двух толщин (не как абсолютный порог):
    при h = 0.03 поправка w_max и снижение пика практически исчезают.
    """
    c_thin, k_thin = _pair(0.03, max_iter=2000)
    rel_thin = (k_thin.w_ktn_nodes.max() - c_thin.w_nodes.max()) / c_thin.w_nodes.max()
    assert 0.0 < rel_thin < 0.01                                  # < 1 % при h = 0.03
    assert k_thin.r_nodes.max() > 0.9 * c_thin.r_nodes.max()      # пик почти классический


def test_ktn_params_reduce_to_small_for_thin_plate():
    """Согласованность: при h→0 поправочные длины → 0 (КТН → классика)."""
    thin = KTNParams(E=2.1e6, nu=0.3, h=1e-3)
    assert thin.c_curv == pytest.approx(0.0, abs=1e-6)
    assert thin.h_psi2 > 0 and thin.h_star2 > 0
    # κ-податливости положительны и линейны по h (см. формулу (9))
    thick = KTNParams(E=2.1e6, nu=0.3, h=2e-3)
    assert thin.kappa_r > 0 and thick.kappa_r == pytest.approx(2 * thin.kappa_r, rel=1e-14)
    assert thin.kappa_q > 0 and thick.kappa_q == pytest.approx(2 * thin.kappa_q, rel=1e-14)
