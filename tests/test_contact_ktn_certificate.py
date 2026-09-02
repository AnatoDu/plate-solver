r"""Сертификат КТН-контакта: редукция к СЕРТИФИЦИРОВАННОМУ классическому (v0.8.0).

Замкнутого решения контактной задачи для уточнённой теории нет, и до v0.8.0
контактный тракт ``ktn_linear`` не имел абсолютного эталона вовсе (в CHANGELOG
это отмечено как отложенный пункт). Здесь он строится РЕДУКЦИЕЙ — тем же
приёмом, что вся верификационная лестница пакета.

Схема. Берётся постановка с ЗАМКНУТЫМ решением классики: защемлённый круг на
плоском основании (``analytic_auto.axisym_contact_solution`` — центральная
плоская зона плюс кольцевая реакция Кирхгофа). Гейт — прогиб ВНЕ зоны, в точке
``r = (c + a)/2``. Тогда:

* классический тракт отстоит от замкнутого решения на ПОСТОЯННУЮ величину —
  пол дискретизации (измерено 2.68e-3 при p = 8, Q = 128, от толщины НЕ
  зависит: классическая задача в безразмерном виде от ``h`` не зависит);
* уточнённый тракт отстоит на «пол + модельная поправка», и ПРЕВЫШЕНИЕ над
  полом обязано гаснуть как ``h²`` — объявленный порядок теории
  (сдвиг и обжатие дают поправки ``O((h/a)²)``).

Измерено (превышение над полом): 1.20e-4 (h = 0.02), 2.70e-4 (h = 0.03),
1.03e-3 (h = 0.06), 4.97e-3 (h = 0.12); отношения при масштабировании
толщины — 2.25 (×1.5), 3.8 (×2), 4.8 (×2) против теоретических 2.25 и 4.

Ворота проверяют ОБА свойства: сходимость уточнённого решения к
сертифицированному классическому при ``h → 0`` и ПОРЯДОК этой сходимости.
Отдельно фиксируется, что регуляризующая часть (податливость ``κ_r``) — иного
порядка, ``κ_r/‖G‖ ∝ (h/a)⁴``: две поправки уточнённой теории входят в
контакт с разными степенями толщины.
"""

from __future__ import annotations

import numpy as np
import pytest

from plate_solver import geometry
from plate_solver.analytic_auto import axisym_contact_solution
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.contact import ContactMOR
from plate_solver.ktn import KTNParams

_A = 1.0
_GAP_FRACTION = 0.47          # Δ / w_free(0): та же доля, что у ci-случая


def _run(h: float):
    """Классический и уточнённый контакт круга + замкнутый эталон Кирхгофа."""
    cfg = Config(nu=0.3, q0=4.0, h=h, a=_A, p=8, Q=128, beta=1.2,
                 max_iter=4000, tol=1e-9, grid_n=16)
    w_free0 = cfg.q0 * cfg.a**4 / (64.0 * cfg.D)      # защемлённый круг, центр
    gap = _GAP_FRACTION * w_free0
    sol = axisym_contact_solution(a=cfg.a, D=cfg.D, q0=cfg.q0, gap=gap, bc="clamped")
    r_gate = 0.5 * (float(sol.meta["c"]) + cfg.a)     # середина между кромкой зоны и краем
    w_ref = float(sol.w(r_gate, 0.0))

    plate = ClampedPlate.from_config(geometry.make_circle(cfg.a), cfg)
    quad = plate.quad
    out = {}
    for name, ktn in (("classic", None), ("ktn", KTNParams.from_config(cfg))):
        res = ContactMOR(plate, cfg, gap=gap, ktn=ktn).solve()
        state = plate.solve(cfg.q0 - res.r_nodes)
        w_gate = float(abs(plate.deflection(state, np.array([r_gate]),
                                            np.array([0.0]))[0]))
        out[name] = abs(w_gate - w_ref) / abs(w_ref)
    unit = plate.solve(np.ones(quad.x.size))
    out["gain"] = float(np.max(np.abs(plate.w_at_quad(unit))))
    out["kappa_r"] = KTNParams.from_config(cfg).kappa_r
    return out


@pytest.fixture(scope="module")
def series():
    return {h: _run(h) for h in (0.02, 0.03, 0.06, 0.12)}


def test_classic_tract_matches_closed_form_and_is_h_independent(series):
    """Опора сертификата: классический тракт даёт ПОСТОЯННЫЙ пол дискретизации."""
    floors = [v["classic"] for v in series.values()]
    assert max(floors) < 5e-3                       # измерено 2.68e-3
    assert max(floors) - min(floors) < 1e-9         # от h не зависит (задача безразмерна)


def test_ktn_contact_reduces_to_certified_classic(series):
    """ГЛАВНЫЕ ВОРОТА: уточнённый контакт сходится к сертифицированному классическому."""
    floor = series[0.02]["classic"]
    thin = series[0.02]["ktn"] - floor
    thick = series[0.12]["ktn"] - floor
    assert thin > 0.0 and thick > thin              # поправка есть и растёт с h
    assert thin < 0.1 * floor                       # при h/a = 0.02 — ниже пола на порядок
    assert series[0.02]["ktn"] == pytest.approx(floor, rel=0.1)


def test_model_correction_vanishes_as_h_squared(series):
    """ГЛАВНЫЕ ВОРОТА: превышение над полом гаснет как h² — порядок теории.

    Отношение поправок при масштабировании толщины в ``s`` раз обязано быть
    близко к ``s²`` (а не к ``s`` или ``s⁴``): проверяются обе пары.
    """
    floor = series[0.02]["classic"]
    exc = {h: v["ktn"] - floor for h, v in series.items()}
    r_15 = exc[0.03] / exc[0.02]                    # s = 1.5 ⇒ ожидание 2.25
    r_2 = exc[0.06] / exc[0.03]                     # s = 2   ⇒ ожидание 4
    assert r_15 == pytest.approx(2.25, rel=0.35), f"измерено {r_15:.2f}"
    assert r_2 == pytest.approx(4.0, rel=0.35), f"измерено {r_2:.2f}"
    # и это НЕ первая степень: рост заметно быстрее линейного
    assert r_2 > 2.5


def test_regularization_parameter_scales_as_h_fourth(series):
    """Регуляризующая часть — иного порядка: ``κ_r/‖G‖ ∝ (h/a)⁴``.

    Две поправки уточнённой теории входят в контакт с РАЗНЫМИ степенями
    толщины: кривизна лицевой даёт O(h²) в решении, податливость ``κ_r`` —
    O(h⁴) в операторе. Проверяется точное отношение 16 при удвоении.
    """
    ratios = []
    for h1, h2 in ((0.03, 0.06), (0.06, 0.12)):
        k1 = series[h1]["kappa_r"] / series[h1]["gain"]
        k2 = series[h2]["kappa_r"] / series[h2]["gain"]
        ratios.append(k2 / k1)
    for r in ratios:
        assert r == pytest.approx(16.0, rel=1e-9)
