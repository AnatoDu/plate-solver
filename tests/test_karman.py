r"""Ворота теории Кармана (v0.4.0): переключатель теорий + верификация.

Структура файла:

* K0 — инфраструктура: лестница ``classic | karman | ktn``, закрепление
  кромки в плане ``inplane_bc``, параметры нелинейной итерации, задел
  ``ktn_full`` (:class:`NotImplementedError`), ограда рамок v0.4.0.
* Gate L/M/K/B и воспроизводимые модульные тесты (Hencky, Тимошенко) —
  добавляются на вехах K1/K2 (числа эталонов — из ``benchmarks.py``).

Правило ``CLAUDE.md``: каждый численный метод — с тестом И мат.
обоснованием; эталоны подобраны РАЗНОЙ математической природы
(аналитика Hencky, ряды Way, энергия Ритца Тимошенко, ряды Фурье Levy).
"""

from __future__ import annotations

import copy
from pathlib import Path

import pytest

from plate_solver import benchmarks as bm
from plate_solver.config import Config
from plate_solver.geometry import make_circle, make_rectangle
from plate_solver.membrane import KarmanPlate
from plate_solver.problem import THEORIES, CaseError, Problem

# --------------------------------------------------------------------------- #
#  K0 — инфраструктура переключателя теорий
# --------------------------------------------------------------------------- #
_MIN_KARMAN = {
    "geometry": {"kind": "circle", "a": 1.0},
    "bc": {"type": "clamped"},
    "load": {"type": "uniform", "q0": 4.0},
    "model": {"theory": "karman"},
}


def _case(**sections) -> dict:
    """Минимальный валидный karman-случай с переопределением секций."""
    d = copy.deepcopy(_MIN_KARMAN)
    for name, content in sections.items():
        d[name] = content
    return d


def _expect_error(data: dict, *fragments: str) -> str:
    with pytest.raises(CaseError) as e:
        Problem.from_dict(data)
    msg = str(e.value)
    assert "ожидалось" in msg and "CASE_SCHEMA.md#" in msg, msg
    for frag in fragments:
        assert frag in msg, (frag, msg)
    return msg


def test_theories_ladder():
    """Лестница моделей (v0.5): classic | karman | ktn_linear | ktn_full."""
    assert THEORIES == ("classic", "karman", "ktn_linear", "ktn_full")


def test_karman_accepted_with_defaults():
    """theory = karman на круге/защемлении принимается; inplane_bc по умолч. immovable."""
    p = Problem.from_dict(_MIN_KARMAN)
    assert p.model.theory == "karman"
    assert p.model.inplane_bc == "immovable"
    # параметры нелинейной итерации не заданы ⇒ наследуют дефолты Config
    assert p.model.n_load_steps is None and p.model.karman_relax is None


def test_inplane_bc_movable_accepted():
    """inplane_bc = movable (N·n = 0) — валиден для karman."""
    p = Problem.from_dict(_case(model={"theory": "karman", "inplane_bc": "movable"}))
    assert p.model.inplane_bc == "movable"


def test_inplane_bc_invalid_value():
    _expect_error(_case(model={"theory": "karman", "inplane_bc": "sliding"}),
                  "model.inplane_bc", "immovable | movable")


def test_inplane_bc_rejected_for_classic():
    """inplane_bc осмыслен только для нелинейных теорий: при classic — отвергается (§4)."""
    _expect_error(_case(model={"theory": "classic", "inplane_bc": "movable"}),
                  "model.inplane_bc", "нелинейных теорий")


def test_karman_iteration_params_to_config():
    """n_load_steps, karman_relax, karman_max_iter, karman_tol, karman_method → Config."""
    data = _case(model={
        "theory": "karman", "inplane_bc": "immovable",
        "n_load_steps": 8, "karman_relax": 0.7, "karman_max_iter": 300,
        "karman_tol": 1e-9, "karman_method": "picard",
    })
    cfg = Problem.from_dict(data).to_config()
    assert cfg.n_load_steps == 8
    assert cfg.karman_relax == pytest.approx(0.7)
    assert cfg.karman_max_iter == 300
    assert cfg.karman_tol == pytest.approx(1e-9)
    assert cfg.karman_method == "picard"


def test_karman_params_rejected_for_linear():
    """Параметры итерации осмысленны только для нелинейных теорий (§4)."""
    _expect_error(_case(model={"theory": "ktn_linear", "h": 0.1, "n_load_steps": 4}),
                  "model.n_load_steps", "нелинейных теорий")


def test_karman_relax_upper_bound():
    """Недорелаксация θ ∈ (0, 1]: θ > 1 отвергается."""
    _expect_error(_case(model={"theory": "karman", "karman_relax": 1.5}),
                  "model.karman_relax", "0 < θ ≤ 1")


def test_karman_method_invalid():
    _expect_error(_case(model={"theory": "karman", "karman_method": "bfgs"}),
                  "model.karman_method", "picard | newton")


def test_ktn_full_accepted_as_theory():
    """theory = ktn_full — валидная теория (v0.5); ktn_method — только для неё."""
    p = Problem.from_dict(_case(model={"theory": "ktn_full", "ktn_method": "picard"}))
    assert p.model.theory == "ktn_full"
    assert p.to_config().ktn_method == "picard"


def test_ktn_alias_deprecated():
    """Устаревший 'ktn' → 'ktn_linear' с DeprecationWarning (§4), поведение сохранено."""
    with pytest.warns(DeprecationWarning, match="ktn_linear"):
        p = Problem.from_dict({
            "geometry": {"kind": "circle", "a": 1.0},
            "bc": {"type": "clamped"},
            "load": {"type": "uniform", "q0": 4.0},
            "model": {"theory": "ktn", "h": 0.1},
        })
    assert p.model.theory == "ktn_linear"          # алиас разрешён в канон


def test_ktn_method_rejected_for_karman():
    """ktn_method осмыслен только для ktn_full (у karman — karman_method)."""
    _expect_error(_case(model={"theory": "karman", "ktn_method": "newton"}),
                  "model.ktn_method", "ktn_full")


def test_karman_geometry_fence():
    """Карман в v0.4.0 — только круг/прямоугольник (§1); кольцо/L отвергаются."""
    _expect_error(_case(geometry={"kind": "annulus", "a": 1.0, "b": 0.4}),
                  "model.theory", "circle | rectangle")


def test_karman_rejects_contact():
    """Нелинейный контакт (МОР поверх Кармана) — вне рамок v0.4.0."""
    _expect_error(_case(contact={"enabled": True, "gap": 1e-3}),
                  "contact.enabled", "направление развития")


def test_config_karman_defaults():
    """Безопасные дефолты §5.4 в Config."""
    c = Config()
    assert c.n_load_steps == 1 and c.karman_max_iter == 200
    assert c.karman_tol == pytest.approx(1e-8) and c.karman_relax == pytest.approx(1.0)
    assert c.karman_method == "picard"


# --------------------------------------------------------------------------- #
#  K1 — верификация решателя Кармана на КРУГЕ (эталоны разной природы)
# --------------------------------------------------------------------------- #
# Безразмерная постановка: E=h=a=1, ν=0.3 ⇒ P̄ = q0 и w/h = w_max (нормировка B).
def _karman_circle(P_bar, *, bc="clamped", inplane="immovable", Q=160, p=12,
                   n_load_steps=2, tol=1e-6, max_iter=300, relax=1.0):
    cfg = Config(E=1.0, h=1.0, nu=0.3, a=1.0, q0=P_bar, p=p, Q=Q,
                 n_load_steps=n_load_steps, karman_tol=tol,
                 karman_max_iter=max_iter, karman_relax=relax)
    kp = KarmanPlate.from_config(make_circle(1.0), cfg, bc_type=bc, inplane_bc=inplane)
    return kp.solve_uniform()


def test_gate_l_clamped_circle():
    """Gate L: при w/h ≪ 1 Карман → Кирхгоф (защемлённый круг), наклон < 0.5 %."""
    r = _karman_circle(0.02, bc="clamped", Q=256, n_load_steps=1)
    ref = bm.kirchhoff_clamped_circle(0.02, 0.3)
    assert abs(r.w_max - ref) / ref < 5e-3          # ловит ошибку жёсткости/КУ
    assert abs(r.w_max - r.w_max_classic) / r.w_max_classic < 1e-3  # N→0: нелинейность исчезла
    assert r.converged


def test_gate_l_hinge_circle():
    """Gate L: шарнирный круг — полная форма на ωΦ даёт ИСТИННЫЙ шарнир (коэф. 4.077)."""
    r = _karman_circle(0.02, bc="soft_hinge", Q=200, n_load_steps=1)
    ref = bm.kirchhoff_hinge_circle(0.02, 0.3)       # (5+ν)/(1+ν)·p a⁴/(64Dh)
    assert abs(r.w_max - ref) / ref < 5e-3
    assert r.converged


def test_gate_k_way_clamped_circle_low():
    """Gate K (быстрый): решатель ложится на ряды Way при умеренном прогибе < 2 %."""
    for P_bar, w_way in bm.way_clamped_circle()[:2]:  # P̄ = 1.818, 3.196 (w/h ≲ 0.5)
        r = _karman_circle(P_bar, Q=140, n_load_steps=2, tol=1e-6, max_iter=200)
        assert abs(r.w_max - w_way) / w_way < 2e-2
        assert r.converged


def test_timoshenko_solver_beats_one_term():
    """Воспроизводимый (§7): многомодовый решатель ближе к Way, чем одночлен Тимошенко.

    Ожидаемая P̄ и одночленная оценка w/h считаются формулами A.3
    (:func:`benchmarks.timoshenko_clamped_circular_inverse`), таблиц не требуют.
    """
    P_bar, w_way = bm.way_clamped_circle()[1]         # P̄ = 3.196, Way w/h = 0.482
    r = _karman_circle(P_bar, Q=160, n_load_steps=2, tol=1e-7, max_iter=200)
    w_timo = bm.timoshenko_clamped_circular_inverse(P_bar, 0.3)
    assert abs(r.w_max - w_way) < abs(w_timo - w_way)  # многомодовый точнее одночлена


@pytest.mark.big
def test_gate_k_way_full_table():
    """Gate K (полная таблица Way): все точки < 3 % + контроль «ближе к Way»."""
    for P_bar, w_way in bm.way_clamped_circle():
        ns = max(2, int(round(P_bar / 2)))
        r = _karman_circle(P_bar, Q=200, n_load_steps=ns, tol=1e-6, max_iter=400)
        assert abs(r.w_max - w_way) / w_way < 3e-2, (P_bar, r.w_max, w_way)
        w_timo = bm.timoshenko_clamped_circular_inverse(P_bar, 0.3)
        assert abs(r.w_max - w_way) <= abs(w_timo - w_way) + 1e-3


@pytest.mark.big
def test_gate_m_hencky_membrane_asymptote():
    """Gate M: при БОЛЬШОЙ нагрузке (P̄=10³) решатель подходит к Hencky СНИЗУ, < 3 %.

    Ожидаемая асимптота ``0.653·P̄^{1/3}`` считается формулой (§A.1); пластина
    жёстче мембраны ⇒ отношение (w/h)_solver / асимптота → 1 снизу.
    """
    P_bar = 1000.0
    r = _karman_circle(P_bar, Q=80, n_load_steps=20, tol=1e-6, max_iter=250)
    asym = bm.hencky_center_deflection(P_bar, 0.3)
    ratio = r.w_max / asym
    assert 0.97 <= ratio < 1.0, (r.w_max, asym, ratio)   # снизу, отклонение < 3 %


def test_karman_stiffer_than_classic():
    """Мембранное ужесточение: при умеренной нагрузке w_karman < w_classic."""
    r = _karman_circle(3.196, Q=140, n_load_steps=2, tol=1e-6, max_iter=200)
    assert r.w_max < r.w_max_classic                  # загиб «нагрузка–прогиб» вниз


# --------------------------------------------------------------------------- #
#  Воспроизводимые модульные тесты эталонов benchmarks.py (не зависят от таблиц)
# --------------------------------------------------------------------------- #
def test_benchmark_timoshenko_roundtrip():
    """Прямая/обратная формулы Тимошенко взаимно обратны."""
    for w_over_h in (0.5, 1.0, 1.5):
        P_bar = bm.timoshenko_clamped_circular(w_over_h, 0.3)
        assert bm.timoshenko_clamped_circular_inverse(P_bar, 0.3) == pytest.approx(w_over_h)


def test_benchmark_timoshenko_matches_way_within_5pct():
    """Взаимоконтроль эталонов: Тимошенко (одночлен) ↔ Way (ряды) в пределах ~5 %."""
    for P_bar, w_way in bm.way_clamped_circle():
        w_timo = bm.timoshenko_clamped_circular_inverse(P_bar, 0.3)
        assert abs(w_timo - w_way) / w_way < 0.06


def test_benchmark_frozen_constants():
    """Замороженные константы эталонов (§A): значения и источники — в benchmarks.py."""
    assert bm.HENCKY_W_COEFF == pytest.approx(0.653)
    assert bm.HENCKY_SIGMA_COEFF == pytest.approx(0.431)
    assert bm.LEVY_SQUARE_SS_IMMOVABLE == (278.5, 1.83)      # Edge Displacement Zero
    assert bm.LEVY_SQUARE_SS_MOVABLE[0] == 247.0            # Edge Compression Zero


def test_benchmark_normalization_converters():
    """Прил. B: P̄ ↔ p a⁴/(D h); замкнутые формы Gate L согласованы."""
    assert bm.pbar(4.0, 1.0, 2.0, 1.0) == pytest.approx(2.0)
    assert bm.pbar_to_pa4_over_64Dh(1.0, 0.3) == pytest.approx(12 * 0.91 / 64.0)
    # шарнир жёстче/мягче защемления по классике: 4.077× в центре круга
    assert (bm.kirchhoff_hinge_circle(1.0, 0.3)
            / bm.kirchhoff_clamped_circle(1.0, 0.3)) == pytest.approx((5.3) / (1.3), rel=1e-6)


# --------------------------------------------------------------------------- #
#  K2 — квадрат (Levy, ряды Фурье, ν=0.316) и закрепление кромки в плане (Gate B)
# --------------------------------------------------------------------------- #
def _karman_square(P_bar, *, bc="clamped", inplane="immovable", nu=0.316, Q=140,
                   p=12, n_load_steps=8, tol=1e-6, max_iter=400, relax=1.0):
    cfg = Config(E=1.0, h=1.0, nu=nu, a=1.0, q0=P_bar, p=p, Q=Q,
                 n_load_steps=n_load_steps, karman_tol=tol,
                 karman_max_iter=max_iter, karman_relax=relax)
    kp = KarmanPlate.from_config(make_rectangle(0.0, 1.0, 0.0, 1.0), cfg,
                                 bc_type=bc, inplane_bc=inplane)
    return kp.solve_uniform()


def test_gate_l_square_hinge_navier():
    """Gate L: шарнирный квадрат при малой нагрузке → Навье ``0.00406 q a⁴/D`` (ν=0.3)."""
    r = _karman_square(0.02, bc="soft_hinge", nu=0.3, Q=120, n_load_steps=1)
    ref = bm.kirchhoff_hinge_square(0.02, 0.3)
    assert abs(r.w_max - ref) / ref < 5e-3
    assert r.converged


def test_gate_l_square_clamped():
    """Gate L: защемлённый квадрат при малой нагрузке → ``0.001263 p a⁴/D`` (ν=0.316)."""
    r = _karman_square(0.02, bc="clamped", nu=0.316, Q=140, n_load_steps=1)
    ref = bm.kirchhoff_clamped_square(0.02, 0.316)
    assert abs(r.w_max - ref) / ref < 5e-3
    assert r.converged


def test_gate_k_levy_square_clamped_point():
    """Gate K (быстрый): защемлённый квадрат ↔ Levy/IGA — P̄=95 → wc/h=0.912 (< 3 %)."""
    P_bar, wc, _ = bm.levy_square_clamped()[3]         # (95.0, 0.912, 11.1)
    r = _karman_square(P_bar, bc="clamped", nu=0.316, Q=140, n_load_steps=6)
    assert abs(r.w_max - wc) / wc < 3e-2
    assert r.converged


def test_gate_b_inplane_constraint():
    """Gate B: контроль реализации u=v=0 — immovable СИЛЬНО жёстче movable (§3.3).

    Оба совпадают с Навье в линейном пределе; при большом прогибе неподвижная
    кромка даёт мембранное натяжение, подвижная — почти нет. Точное отношение
    зависит от нагрузки (TZ: ≈3.3× при очень больших прогибах, где наш
    полностью-свободный ``movable`` податливее «прямокромочного» Levy); здесь —
    робастный качественный контраст.
    """
    P_bar = 100.0
    ri = _karman_square(P_bar, bc="soft_hinge", inplane="immovable", nu=0.3,
                        Q=120, n_load_steps=6)
    rm = _karman_square(P_bar, bc="soft_hinge", inplane="movable", nu=0.3,
                        Q=120, n_load_steps=6)
    assert ri.converged and rm.converged
    assert rm.w_max > 1.6 * ri.w_max                   # подвижная кромка заметно податливее


@pytest.mark.big
def test_gate_k_levy_square_clamped_table():
    """Gate K (таблица Levy): защемлённый квадрат < 3 % на нескольких P̄ (ряды Фурье)."""
    for P_bar, wc, _ in bm.levy_square_clamped()[3:7]:   # P̄ = 95…245
        ns = max(6, int(round(P_bar / 15)))
        r = _karman_square(P_bar, bc="clamped", nu=0.316, Q=180, n_load_steps=ns,
                           max_iter=500)
        assert abs(r.w_max - wc) / wc < 3e-2, (P_bar, r.w_max, wc)


@pytest.mark.big
def test_gate_k_levy_square_ss_immovable():
    """Gate K (§A.5): шарнирный квадрат immovable, P̄=278.5 → wc/h≈1.83 (< 3 %)."""
    P_bar, wc = bm.levy_square_ss_immovable()
    r = _karman_square(P_bar, bc="soft_hinge", inplane="immovable", nu=0.316,
                       Q=180, n_load_steps=12, max_iter=500)
    assert abs(r.w_max - wc) / wc < 3e-2


@pytest.mark.big
def test_gate_b_ratio_grows_with_load():
    """Gate B (усиление): контраст movable/immovable РАСТЁТ с нагрузкой (эффект НЕнулевой)."""
    ratios = []
    for P_bar in (100.0, 500.0):
        ns = max(6, int(round(P_bar / 40)))
        ri = _karman_square(P_bar, bc="soft_hinge", inplane="immovable", nu=0.3,
                            Q=140, n_load_steps=ns)
        rm = _karman_square(P_bar, bc="soft_hinge", inplane="movable", nu=0.3,
                            Q=140, n_load_steps=ns)
        ratios.append(rm.w_max / ri.w_max)
    assert ratios[1] > ratios[0] > 1.5


# --------------------------------------------------------------------------- #
#  Ladder-кейсы (боевые параметры) через диспетчер ↔ эталоны benchmarks.py
# --------------------------------------------------------------------------- #
_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.big
def test_ladder_karman_hencky_case():
    """cases/ladder/karman_circle_hencky_limit.toml (Gate M) ↔ асимптота Hencky."""
    from plate_solver.dispatch import solve

    res = solve(Problem.from_toml(_ROOT / "cases/ladder/karman_circle_hencky_limit.toml"))
    asym = bm.hencky_center_deflection(1000.0, 0.3)
    assert 0.97 <= res.w_max / asym < 1.0             # снизу, отклонение < 3 %


@pytest.mark.big
def test_ladder_karman_levy_case():
    """cases/ladder/karman_square_levy_sweep.toml ↔ верхняя точка Levy (P̄=402→1.902)."""
    from plate_solver.dispatch import solve

    res = solve(Problem.from_toml(_ROOT / "cases/ladder/karman_square_levy_sweep.toml"))
    assert abs(res.w_max - 1.902) / 1.902 < 3e-2
