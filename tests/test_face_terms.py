r"""Лестница слагаемых лицевого условия (``[model.face_terms]``, v0.8.0).

Лицевой прогиб уточнённой теории — сумма трёх вкладов (формула (9), §21.1):

.. math:: u_c = w + \underbrace{c_{curv}\Delta w}_{curvature}
          - \underbrace{\kappa_q q^+}_{load} - \underbrace{\kappa_r r}_{reaction}

Переключатели позволяют включать их ПО ОТДЕЛЬНОСТИ в одном и том же тракте
решателя и тем самым отвечать на вопрос «какой механизм уточнённой теории
меняет контакт». Измеренная лестница (L-форма, h/a = 0.12, бюджет 6000):

===============  ========  ======  ==========
режим            r_max     узлов   сходимость
===============  ========  ======  ==========
классика           78.54       6   нет
все выключены      78.54       6   нет  (бит-в-бит классика)
только curvature   61.59       9   нет
только load        78.07       6   нет
только reaction    15.49      26   да (4005)
curvature+reaction 15.48      28   да (4220)
полное условие     15.48      28   да (4232)
===============  ========  ======  ==========

Вывод (важен для NOTES §11, §18): регуляризацию контакта даёт член ПОДАТЛИВОСТИ
``−κ_r·r`` — положительная диагональная добавка к оператору задачи
дополнительности; кривизный член лишь перераспределяет реакцию (при h = 0.06 он
даже ПОВЫШАЕТ пик на 10 %), а член нагрузки — почти постоянный сдвиг лицевой
(< 1 %). До v0.8.0 сглаживание ошибочно приписывалось кривизному члену.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.faces import FaceTerms
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending

H = 0.12
BUDGET = 6000


@pytest.fixture(scope="module")
def setup():
    dom = geometry.make_L(1.0, 0.5)
    cfg = Config(nu=0.3, q0=4.0, h=H, p=8, Q=36, beta=1.0, max_iter=BUDGET,
                 tol=1e-12, grid_n=30)
    pb = PlateBending.from_config(dom, cfg)
    q = pb.quad
    _, cw = pb.solve_uniform(cfg.q0)
    gap = 0.6 * float(pb.deflection(cw, q.x, q.y).max())
    fmask = lambda x, y: (x + y) < 0.9            # noqa: E731
    return pb, cfg, gap, fmask


def _run(setup, terms, ktn=True):
    pb, cfg, gap, fmask = setup
    kp = KTNParams.from_config(cfg) if ktn else None
    return ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap, ktn=kp,
                      face_terms=terms).solve()


def test_all_terms_off_is_bitwise_classic(setup):
    """ВОРОТА РЕДУКЦИИ: все слагаемые выключены ⇒ решение КЛАССИКИ бит-в-бит."""
    classic = _run(setup, None, ktn=False)
    off = _run(setup, FaceTerms(False, False, False))
    assert np.array_equal(classic.r_nodes, off.r_nodes)
    assert classic.iters == off.iters
    assert float(classic.w_nodes.max()) == float(off.w_nodes.max())


def test_all_terms_on_is_bitwise_default(setup):
    """Явный полный набор ≡ отсутствие ключа (штатный путь не сдвигается)."""
    default = _run(setup, None)
    explicit = _run(setup, FaceTerms())
    assert np.array_equal(default.r_nodes, explicit.r_nodes)
    assert default.iters == explicit.iters


def test_reaction_term_regularizes_contact(setup):
    """ГЛАВНЫЕ ВОРОТА: регуляризацию даёт член ``−κ_r·r``, а не кривизный.

    С членом реакции итерация СХОДИТСЯ и пик падает в разы; без него (кривизна
    и/или нагрузка) при том же бюджете сходимости нет и пик остаётся
    классического порядка.
    """
    classic = _run(setup, None, ktn=False)
    only_r = _run(setup, FaceTerms(False, False, True))
    only_curv = _run(setup, FaceTerms(True, False, False))
    only_q = _run(setup, FaceTerms(False, True, False))

    assert only_r.converged and only_r.comp_residual < 1e-9
    assert only_r.r_nodes.max() < 0.3 * classic.r_nodes.max()
    assert int((only_r.r_nodes > 0).sum()) > 3 * int((classic.r_nodes > 0).sum())

    assert not only_curv.converged and not only_q.converged
    # член нагрузки — почти постоянный сдвиг лицевой: пик меняется < 2 %
    assert abs(only_q.r_nodes.max() - classic.r_nodes.max()) < 0.02 * classic.r_nodes.max()


def test_curvature_plus_reaction_close_to_full(setup):
    """Полное условие ≈ (кривизна + реакция): вклад члена нагрузки мал."""
    full = _run(setup, None)
    cr = _run(setup, FaceTerms(True, False, True))
    assert cr.converged and full.converged
    assert abs(cr.r_nodes.max() - full.r_nodes.max()) < 1e-3 * full.r_nodes.max()
    assert int((cr.r_nodes > 0).sum()) == int((full.r_nodes > 0).sum())


def test_terms_scale_gain_consistently(setup):
    """Нормировка шага учитывает ТОЛЬКО включённые слагаемые.

    Выключение члена реакции убирает диагональ ``κ_r`` из оценки ``‖G‖``,
    выключение кривизного — возвращает нормировку к срединному прогибу
    (классическое значение бит-в-бит).
    """
    pb, cfg, gap, fmask = setup
    kp = KTNParams.from_config(cfg)
    gain_classic = ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap).gain

    def gain(terms):
        return ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap, ktn=kp,
                          face_terms=terms).gain

    assert gain(FaceTerms(False, False, False)) == gain_classic
    assert gain(FaceTerms(False, True, False)) == gain_classic     # нагрузка не в операторе
    assert gain(FaceTerms(False, False, True)) == pytest.approx(
        gain_classic + kp.kappa_r, rel=1e-14)
    assert gain(FaceTerms(True, False, False)) > gain_classic      # кривизна лицевой
    assert gain(None) == pytest.approx(gain(FaceTerms(True, True, True)), rel=1e-14)


def test_case_schema_routes_face_terms(tmp_path):
    """Ключи ``[model.face_terms]`` доходят до решателя через case-файл."""
    from plate_solver.dispatch import solve
    from plate_solver.problem import CaseError, Problem

    case = {
        "geometry": {"kind": "circle", "a": 1.0},
        "bc": {"type": "clamped"},
        "load": {"type": "uniform", "q0": 4.0},
        "model": {"theory": "ktn_linear", "h": 0.15},
        "discretization": {"p": 6, "Q": 32, "grid_n": 16},
        "contact": {"enabled": True, "gap_factor": 0.5, "max_iter": 800,
                    "tol": 1e-10},
    }
    full = solve(Problem.from_dict(case))
    case_off = {**case, "model": {**case["model"],
                                  "face_terms": {"reaction": False}}}
    off = solve(Problem.from_dict(case_off))
    assert off.config.face_terms == (True, True, False)

    # ТОЖДЕСТВО МАРШРУТА: тот же расчёт через API даёт те же числа бит-в-бит
    from plate_solver.clamped import ClampedPlate

    cfg_off = off.config
    pb = ClampedPlate.from_config(geometry.make_circle(cfg_off.a), cfg_off)
    api = ContactMOR(pb, cfg_off, gap=off.delta,          # тот же зазор (gap_factor)
                     ktn=KTNParams.from_config(cfg_off),
                     face_terms=FaceTerms(True, True, False)).solve()
    assert np.array_equal(off.contact.r_nodes, api.r_nodes)

    # направление эффекта: без члена реакции контакт жёстче (пик выше)
    assert off.contact.r_nodes.max() > full.contact.r_nodes.max()
    # ограда: для классики слагаемых нет по построению
    bad = {**case, "model": {"theory": "classic", "face_terms": {"load": False}}}
    with pytest.raises(CaseError, match="face_terms"):
        Problem.from_dict(bad)
