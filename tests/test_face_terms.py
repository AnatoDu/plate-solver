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
дополнительности; кривизный член лишь перераспределяет реакцию (пика он не
снижает: при h = 0.06 и том же бюджете 6000 — +0.4 %), а член нагрузки —
почти постоянный сдвиг лицевой (< 1 %). До v0.8.0 сглаживание ошибочно
приписывалось кривизному члену.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.faces import FaceTerms
from plate_solver.ktn import KTNParams
from plate_solver.plate import PlateBending

_ROOT = Path(__file__).resolve().parents[1]

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


def test_ladder_matches_frozen_snapshot(setup):
    """Регресс ЛЕСТНИЦЫ: семь режимов против снимка в cases/baselines.json.

    Снимок фиксирует то, что показывает лестница как ИЗМЕРЕНИЕ: какой член
    даёт сходимость, какой перераспределяет реакцию, какой почти не влияет.
    Гейтуются ОТНОШЕНИЯ и факт сходимости (кросс-платформенно устойчивые), а
    не пятый знак; сами числа снимка — для сверки при разборе регресса.
    """
    import json

    base = json.loads((_ROOT / "cases" / "baselines.json").read_text(encoding="utf-8"))
    ref = base["lshape_ktn_terms_ladder"]["режимы"]
    got = {
        "classic": _run(setup, None, ktn=False),
        "off": _run(setup, FaceTerms(False, False, False)),
        "curvature": _run(setup, FaceTerms(True, False, False)),
        "load": _run(setup, FaceTerms(False, True, False)),
        "reaction": _run(setup, FaceTerms(False, False, True)),
        "curvature+reaction": _run(setup, FaceTerms(True, False, True)),
        "full": _run(setup, None),
    }
    base_peak = got["classic"].r_nodes.max()
    for name, res in got.items():
        exp = ref[name]
        assert res.converged == exp["converged"], f"{name}: сходимость разошлась со снимком"
        assert int((res.r_nodes > 0).sum()) == exp["n_contact"], name
        assert float(res.r_nodes.max()) == pytest.approx(exp["r_max"], rel=2e-2), name
        # отношение к классике — главная измеряемая величина лестницы
        assert float(res.r_nodes.max()) / base_peak == pytest.approx(
            exp["r_max"] / ref["classic"]["r_max"], rel=2e-2), name
    # ключевые качественные выводы лестницы
    assert got["off"].r_nodes.max() == got["classic"].r_nodes.max()      # бит-в-бит
    assert got["reaction"].converged and not got["curvature"].converged


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


def test_exported_face_matches_mor_constraint_for_ktn_full():
    """ВОРОТА СОГЛАСОВАННОСТИ (v0.8.0): экспортируемая w_bot = величина условия МОР.

    Нелинейный тракт держал непроникание по УСЕЧЁННОЙ лицевой поверхности
    (только кривизный член), а поля ``faces_on_grid`` считались по ПОЛНОМУ
    канону с κ_q, κ_r: в зоне контакта экспортируемая ``w_bot`` отстояла от
    препятствия на десятки зазоров (аудит O04/D09). После включения κ-членов в
    условие обе величины — одна и та же: в зоне ``w_bot ≈ Δ`` с точностью шага
    МОР, тогда как СРЕДИННЫЙ прогиб от Δ отличается на порядок больше.
    """
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem

    res = solve(Problem.from_toml(
        _ROOT / "cases" / "ci" / "ktn_full_circle_clamped_contact.toml"))
    _, w_bot, _ = res.faces_on_grid()
    zone = res.contact.contact_zone
    assert zone.any()
    delta = float(res.delta)
    err_face = float(np.nanmax(np.abs(w_bot[zone] - delta))) / delta
    err_mid = float(np.nanmax(np.abs(res.w_grid[zone] - delta))) / delta
    assert err_face < 1e-2, "экспортируемая лицевая не совпадает с условием МОР"
    assert err_face < err_mid, "лицевая обязана лежать к препятствию ближе срединной"


def test_local_pressure_and_kkt_scale_for_nonuniform_load(setup):
    r"""Неравномерная нагрузка: в формулу (9) идёт ДАВЛЕНИЕ В ТОЧКЕ (v0.8.0).

    Формула (9) содержит локальное давление ``q⁺(x, y)``; до v0.8.0
    подставлялась скалярная амплитуда ``cfg.q0``, что давало постоянный
    нефизический сдвиг лицевой ВНЕ пятна нагрузки. Проверяется тремя
    независимыми признаками:

    * ``_q_load()`` возвращает ПОЛЕ, а не скаляр, когда задан ``load_values``;
    * лицевой прогиб отличается от «скалярного» варианта именно там, где поле
      отличается от амплитуды, и ровно на ``κ_q·(q₀ − q(x, y))``;
    * масштаб KKT-метрик — ``max|q|`` поля, а не ``cfg.q0`` (для гауссианы это
      разные числа; при ``q₀ = 0`` прежняя нормировка давала NaN).
    """
    pb, cfg, gap, fmask = setup
    q = pb.quad
    kp = KTNParams.from_config(cfg)
    # локализованная гауссиана с ПИКОМ ниже амплитуды cfg.q0
    f = 0.5 * cfg.q0 * np.exp(-((q.x - 0.3) ** 2 + (q.y - 0.3) ** 2) / 0.02)

    mor_field = ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap, ktn=kp,
                           load_values=f)
    mor_scalar = ContactMOR(pb, cfg, foundation_mask=fmask, gap=gap, ktn=kp)

    assert isinstance(mor_field._q_load(), np.ndarray)          # поле, не скаляр
    assert mor_scalar._q_load() == cfg.q0
    assert mor_field._q_ref == pytest.approx(float(np.max(np.abs(f))))
    assert mor_field._q_ref < abs(cfg.q0)                       # пик поля ниже амплитуды

    # разность лицевых при одном и том же прогибе — ровно κ_q·(q₀ − q(x, y))
    state = pb.solve(f)
    w = pb.w_at_quad(state)
    r0 = np.zeros(q.x.size)
    u_field = mor_field._contact_disp(state, w, r0)
    u_scalar = mor_scalar._contact_disp(state, w, r0)
    assert np.allclose(u_scalar - u_field, kp.kappa_q * (f - cfg.q0), rtol=1e-12,
                       atol=1e-18)
    assert np.max(np.abs(u_scalar - u_field)) > 0.0             # ворота невакуумны
