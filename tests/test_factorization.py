r"""Ворота КАСКАДА ФАКТОРИЗАЦИИ (находки аудита 0.8: P03, P02, P04).

Постановка. Матрица Ритца положительно определена ТОЧНО, но численно
вырождается с ростом степени базиса ``p`` на криволинейной границе и во
входящем угле. До правки это проявлялось тремя разными способами:

* **P03** — ``PoissonSolver.__init__`` звал ``cho_factor`` без фолбэка:
  законный case-файл падал сырым ``LinAlgError`` с сообщением LAPACK.
  Измерено на L-форме (эта машина, macOS/Accelerate): ``p=12, Q=160``
  факторизуется (``cond ≈ 1.6e18``), а ``p=12, Q=200`` и ``p=14, Q=160`` —
  отказ; исход платформозависим (зависит от BLAS), поэтому ворота (б)
  ищут отказ по лестнице ``p`` и пропускаются, если на данной машине
  Холецкий выдерживает всю лестницу.
* **P02** — ``ClampedPlate``/``MixedRectPlate`` при потере положительной
  определённости МОЛЧА уходили в МНК: пользователь не узнавал о деградации.
* **P04** — ``ktn_full._lin_solve``: ``except LinAlgError`` вокруг
  ``lu_factor`` — МЁРТВАЯ ветка (LAPACK на вырожденной матрице не бросает
  исключение), решение молча становилось ``inf``/``NaN``.

Единый ответ — каскад :class:`~plate_solver.poisson.SPDFactorization`
(Холецкий → масштабирование Якоби → спектральное псевдообращение) с ЧЕСТНОЙ
отчётностью (атрибуты ``fallback``/``n_dropped`` + ``FactorizationWarning``).

Что проверяется:

(а) ШТАТНЫЙ ПУТЬ БИТ-В-БИТ. Каскад не имеет права сдвинуть ни одного числа
    там, где Холецкий проходит: замороженные эталоны (``results/reference``)
    и golden-прогоны обязаны совпасть до последнего бита. Проверяется прямым
    сравнением с ПРЕЖНЕЙ арифметикой (``cho_solve(cho_factor(A), b)``) на
    нескольких конфигурациях (круг/L-форма, разные ``p``, ``Q``), включая
    сквозной путь ``PlateBending`` (P1→P2) и нормированный путь защемления.
(б) ФОЛБЭК ДАЁТ РАЗУМНОЕ РЕШЕНИЕ. Там, где Холецкий отказывает, каскад
    возвращает конечное решение, отличающееся от решения при ``p = 10`` на
    ПРОЦЕНТЫ, а не на порядки (усечение выбрасывает почти-нулевые моды).
(в) ФОЛБЭК СИГНАЛИЗИРУЕТСЯ — атрибутом и предупреждением.
(г) ``_lin_solve`` НЕ возвращает ``NaN`` на вырожденной матрице (P04), причём
    ворота пиннуют и сам МЕХАНИЗМ: ``lu_factor`` исключения не бросает.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import scipy.linalg as sla

from plate_solver.assembler import assemble_load
from plate_solver.basis import ChebyshevBasis
from plate_solver.clamped import ClampedPlate
from plate_solver.config import Config
from plate_solver.geometry import make_circle, make_L
from plate_solver.ktn_full import _lin_solve
from plate_solver.plate import PlateBending
from plate_solver.poisson import (
    REL_CUTOFF,
    FactorizationWarning,
    PoissonSolver,
    SPDFactorization,
)
from plate_solver.quadrature import interior_nodes

# Конфигурации ШТАТНОГО пути (Холецкий проходит): круг и L-форма, разные p, Q.
_HEALTHY = [("circle", 8, 80), ("circle", 12, 120), ("lshape", 6, 64), ("lshape", 10, 120)]


def _domain(name):
    return make_circle(1.0) if name == "circle" else make_L()


def _rhs(quad):
    """Гладкая несимметричная правая часть — чтобы сравнение не было тривиальным."""
    return np.sin(3.0 * quad.x) * np.cos(2.0 * quad.y) + 1.0


def _cho_ok(A) -> bool:
    """Проходит ли ПРЯМОЙ Холецкий (без каскада) — платформозависимо."""
    try:
        sla.cho_factor(A)
    except (sla.LinAlgError, np.linalg.LinAlgError):
        return False
    return True


# --------------------------------------------------------------------------- #
#  (а) штатный путь — бит-в-бит прежняя арифметика
# --------------------------------------------------------------------------- #
def test_cascade_matches_cho_solve_bit_exact_on_spd():
    """Каскад на здоровой SPD-матрице ТОЖДЕСТВЕН ``cho_solve(cho_factor(A), b)``.

    Базовое требование: ступень 1 каскада — тот же вызов LAPACK, без
    предварительных преобразований, поэтому равенство обязано быть побитовым
    (``array_equal``), а не «в пределах допуска».
    """
    rng = np.random.default_rng(20260902)
    for n in (5, 17, 40):
        B = rng.standard_normal((n, n))
        A = B @ B.T + n * np.eye(n)
        b = rng.standard_normal(n)
        fact = SPDFactorization(A)
        assert fact.fallback is None and fact.n_dropped == 0
        assert np.array_equal(fact.solve(b), sla.cho_solve(sla.cho_factor(A), b))


@pytest.mark.parametrize(("name", "p", "Q"), _HEALTHY)
def test_poisson_solve_bit_exact_on_cholesky_path(name, p, Q):
    """``PoissonSolver.solve`` бит-в-бит воспроизводит прежнюю арифметику."""
    dom = _domain(name)
    basis = ChebyshevBasis(p, dom.bbox)
    quad = interior_nodes(dom, Q)
    sol = PoissonSolver(dom, basis, quad)
    assert _cho_ok(sol.A), "конфигурация задумана как ЗДОРОВАЯ (Холецкий проходит)"
    assert sol.fallback is None and sol.n_dropped == 0
    f = _rhs(quad)
    b = (sol.psiW @ f) if sol.cache_fields else assemble_load(dom, basis, quad, f)
    expected = sla.cho_solve(sla.cho_factor(sol.A), b)      # прежний код один-в-один
    assert np.array_equal(sol.solve(f), expected)
    assert np.array_equal(sol.solve_b(b), expected)


@pytest.mark.parametrize(("name", "p", "Q"), _HEALTHY)
def test_plate_bending_bit_exact_on_cholesky_path(name, p, Q):
    """Сквозной путь изгиба (P1 → P2) бит-в-бит совпадает с прежней арифметикой.

    Именно эта пара решений формирует замороженные эталоны, поэтому проверяется
    не только один ``solve``, но и цепочка ``cM → M_nodes → cw``.
    """
    dom = _domain(name)
    cfg = Config(p=p, Q=Q)
    pb = PlateBending.from_config(dom, cfg)
    assert _cho_ok(pb.poisson.A)
    chol = sla.cho_factor(pb.poisson.A)
    qt = np.full(pb.quad.x.size, float(cfg.q0))
    cM_ref = sla.cho_solve(chol, pb.poisson.psiW @ qt)
    M_ref = pb.poisson.evaluate_at_quad(cM_ref)
    cw_ref = sla.cho_solve(chol, pb.poisson.psiW @ (M_ref / pb.D))
    cM, cw = pb.solve_uniform()
    assert np.array_equal(cM, cM_ref)
    assert np.array_equal(cw, cw_ref)


@pytest.mark.parametrize(("name", "p", "Q"), _HEALTHY)
def test_clamped_bit_exact_on_cholesky_path(name, p, Q):
    """Защемление: нормированный путь ``diag(d)·S·diag(d)`` бит-в-бит прежний.

    Прежний код: ``d = 1/√S_kk``, ``Sn = diag(d) S diag(d)``,
    ``ĉ = cho_solve(cho_factor(D·Sn), d·b)``, ``c = d·ĉ``. Каскад обязан
    воспроизвести его посимвольно, пока Холецкий проходит.
    """
    dom = _domain(name)
    cfg = Config(p=p, Q=Q)
    plate = ClampedPlate.from_config(dom, cfg)
    assert plate.fallback is None and plate.n_dropped == 0
    f = _rhs(plate.quad)
    d = 1.0 / np.sqrt(np.diag(plate.S))
    Sn = (plate.S * d).T * d
    b = (plate._psi * plate._W) @ f
    expected = sla.cho_solve(sla.cho_factor(Sn * plate.D), b * d) * d
    assert np.array_equal(plate.solve_rhs(f), expected)
    assert np.array_equal(plate.solve_from_b(b), expected)


def test_healthy_path_emits_no_warning():
    """На здоровой конфигурации каскад МОЛЧИТ (предупреждение — признак деградации)."""
    dom = make_circle(1.0)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        pb = PlateBending.from_config(dom, Config(p=10, Q=120))
        pb.solve_uniform()
        ClampedPlate.from_config(dom, Config(p=10, Q=120)).solve_uniform()
    assert [w for w in rec if issubclass(w.category, FactorizationWarning)] == []


# --------------------------------------------------------------------------- #
#  (б) фолбэк на L-форме: решение получается и разумно
# --------------------------------------------------------------------------- #
def _first_failing_lshape(ladder=((12, 200), (14, 160), (14, 200), (16, 200))):
    """Первая ступень (p, Q) лестницы, на которой ПРЯМОЙ Холецкий отказывает."""
    dom = make_L()
    with warnings.catch_warnings():                # зонд, а не расчёт — не шумим
        warnings.simplefilter("ignore", FactorizationWarning)
        for p, Q in ladder:
            A = PoissonSolver(dom, ChebyshevBasis(p, dom.bbox), interior_nodes(dom, Q)).A
            if not _cho_ok(A):
                return p, Q
    return None


def test_lshape_pinv_fallback_gives_sane_solution():
    r"""L-форма при ``p``, где Холецкий падает: решение есть и отличается на ПРОЦЕНТЫ.

    Раньше здесь был сырой ``LinAlgError``. Теперь включается спектральное
    псевдообращение: усечение выбрасывает ``n_dropped`` почти-нулевых мод
    структуры ``ω·T`` (линейно зависимых с точностью округления), чей вклад в
    энергию ниже пола сборки. Критерий разумности — сравнение с решением при
    ``p = 10`` на ТОЙ ЖЕ задаче: относительная разница ``w_max`` должна
    остаться в процентах (порог 10 %), а не в порядках.

    Отказ Холецкого платформозависим (BLAS) — если на данной машине вся
    лестница ``p`` проходит, ворота пропускаются (нечего проверять).
    """
    step = _first_failing_lshape()
    if step is None:
        pytest.skip("на этом BLAS Холецкий выдерживает всю лестницу p — фолбэк не включается")
    p, Q = step
    dom = make_L()
    pb_ref = PlateBending.from_config(dom, Config(p=10, Q=120))
    _, cw_ref = pb_ref.solve_uniform()
    x0, x1, y0, y1 = dom.bbox
    Xg, Yg = np.meshgrid(np.linspace(x0, x1, 200), np.linspace(y0, y1, 200))
    ins = dom.omega(Xg, Yg) > 0.0
    w_ref = float(pb_ref.deflection(cw_ref, Xg[ins], Yg[ins]).max())

    with pytest.warns(FactorizationWarning):
        pb = PlateBending.from_config(dom, Config(p=p, Q=Q))
    _, cw = pb.solve_uniform()
    w = pb.deflection(cw, Xg[ins], Yg[ins])
    assert np.all(np.isfinite(w)), "фолбэк обязан давать ЧИСЛА, а не NaN/inf"
    assert pb.poisson.fallback == "pinv"
    assert pb.poisson.n_dropped >= 1
    assert abs(float(w.max()) - w_ref) / w_ref < 0.10


# --------------------------------------------------------------------------- #
#  (в) факт фолбэка доступен вызывающему: атрибут + предупреждение
# --------------------------------------------------------------------------- #
def _singular_spd(n=6, k=2):
    """Симметричная полуопределённая матрица ранга ``n−1`` с ТОЧНЫМ нулём в диагонали.

    Строка/столбец ``k`` нулевые ⇒ у LAPACK ``potrf`` пивот ровно ноль (отказ
    ДЕТЕРМИНИРОВАННЫЙ, без зависимости от округления), а ``diag`` не всюду
    положительна ⇒ ступень Якоби неприменима: каскад обязан дойти до
    спектрального псевдообращения ровно с одним отсечённым направлением.
    """
    A = np.diag(np.arange(1.0, n + 1.0))
    A[k, :] = 0.0
    A[:, k] = 0.0
    return A


def test_fallback_reported_by_attribute_and_warning():
    """Псевдообращение: атрибуты ``fallback``/``n_dropped`` + FactorizationWarning."""
    A = _singular_spd()
    b = np.arange(1.0, A.shape[0] + 1.0)
    with pytest.warns(FactorizationWarning, match="ПРОСТРАНСТВО УСЕЧЕНО"):
        fact = SPDFactorization(A, label="тест")
    assert fact.fallback == "pinv"
    assert fact.degraded is True
    assert fact.n_dropped == 1                     # ровно одно нулевое направление
    assert "тест" in fact.warning_message
    x = fact.solve(b)
    assert np.all(np.isfinite(x))
    # решение — минимально-нормальное на подпространстве span V_+: невязка
    # ортогональна образу, а отсечённая координата обнулена
    assert x[2] == pytest.approx(0.0, abs=1e-14)
    keep = np.array([0, 1, 3, 4, 5])
    assert np.allclose((A @ x)[keep], b[keep])


def test_fallback_can_be_silenced_and_still_recorded():
    """``warn=False`` глушит предупреждение, но НЕ прячет факт: атрибуты остаются."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        fact = SPDFactorization(_singular_spd(), warn=False)
    assert rec == []
    assert fact.fallback == "pinv" and fact.n_dropped == 1
    assert fact.warning_message is not None


def test_relative_cutoff_is_relative_to_lambda_max():
    r"""Отсечка ОТНОСИТЕЛЬНАЯ: масштабирование ``A → αA`` не меняет ``n_dropped``.

    Абсолютный порог был бы привязан к физическим единицам (``D``, размеру
    области) и на безразмерной задаче отсекал бы либо всё, либо ничего.
    """
    A = _singular_spd()
    A[0, 0] = 3.0e-11                              # мода ниже отсечки 1e-12·λ_max? нет
    n0 = SPDFactorization(A, warn=False).n_dropped
    for alpha in (1e-9, 1e9):
        assert SPDFactorization(alpha * A, warn=False).n_dropped == n0
    # мода строго под отсечкой отбрасывается дополнительно
    A[0, 0] = 0.1 * REL_CUTOFF * 6.0
    assert SPDFactorization(A, warn=False).n_dropped == n0 + 1


def test_jacobi_stage_solves_correctly_when_it_fires():
    r"""Ступень 2 (масштабирование Якоби): решение верно, пространство НЕ усечено.

    Ищем в детерминированном семействе плохо отмасштабированных SPD-матриц
    ``A = S M S`` (``M`` с разбросом спектра ``1 … 1e-17``, ``S`` — диагональ
    из степеней десяти) первую, на которой прямой Холецкий падает, а Холецкий
    по ``diag(1/√A_kk)·A·diag(1/√A_kk)`` проходит (теорема ван дер Слёйса:
    такое масштабирование минимизирует ``cond`` среди диагональных с точностью
    до множителя ``n``). Ступень обязана дать ПОЛНОЕ решение (``n_dropped = 0``)
    с малой невязкой. Срабатывание зависит от BLAS — при отсутствии кандидата
    ворота пропускаются.
    """
    lam = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-17])
    for seed in range(400):
        rng = np.random.default_rng(seed)
        V = np.linalg.qr(rng.standard_normal((8, 8)))[0]
        M = V @ np.diag(lam) @ V.T
        s = 10.0 ** rng.integers(-4, 5, size=8)
        A = 0.5 * (M + M.T)
        A = (A * s) * s[:, None]
        if _cho_ok(A) or not np.all(np.diag(A) > 0.0):
            continue
        t = 1.0 / np.sqrt(np.diag(A))
        if not _cho_ok((A * t) * t[:, None]):
            continue
        b = A @ np.ones(8)                          # заведомо совместимая правая часть
        with pytest.warns(FactorizationWarning, match="Якоби"):
            fact = SPDFactorization(A)
        assert fact.fallback == "jacobi"
        assert fact.n_dropped == 0                  # пространство НЕ усечено
        x = fact.solve(b)
        assert np.all(np.isfinite(x))
        assert np.linalg.norm(A @ x - b) <= 1e-6 * np.linalg.norm(b)
        return
    pytest.skip("на этом BLAS ступень Якоби в этом семействе не срабатывает")


def test_clamped_and_mixed_expose_fallback_attributes():
    """P02: молчаливый МНК заменён отчётностью — атрибуты есть у обоих решателей."""
    from plate_solver.clamped import MixedRectPlate

    cfg = Config(p=8, Q=80)
    plate = ClampedPlate.from_config(make_circle(1.0), cfg)
    assert plate.fallback is None and plate.n_dropped == 0
    assert plate.fallback_message is None
    mixed = MixedRectPlate(0.0, 1.0, 0.0, 1.0,
                           {"x1": "clamped", "x2": "hinge",
                            "y1": "clamped", "y2": "hinge"}, cfg)
    assert mixed.fallback is None and mixed.n_dropped == 0
    assert mixed.fallback_message is None


# --------------------------------------------------------------------------- #
#  (г) P04: ktn_full._lin_solve на вырожденной матрице не возвращает NaN
# --------------------------------------------------------------------------- #
def test_lu_factor_does_not_raise_on_singular_matrix():
    """Механизм P04: ``lu_factor`` на вырожденной матрице НЕ бросает исключения.

    Ворота пиннуют причину, по которой прежний ``except LinAlgError`` был
    мёртвым: LAPACK ``getrf`` сообщает о нулевом пивоте кодом ``info > 0``,
    который SciPy отдаёт предупреждением, а не ошибкой; ``lu_solve`` затем
    делит на ноль и выдаёт ``inf``/``NaN``. Поэтому критерием фолбэка обязана
    быть КОНЕЧНОСТЬ решения, а не исключение.
    """
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lu = sla.lu_factor(A)                       # НЕ бросает LinAlgError
        x = sla.lu_solve(lu, np.array([1.0, 1.0]))
    assert not np.all(np.isfinite(x)), "если это стало исключением — ворота пересмотреть"


def test_lin_solve_degenerate_returns_finite_solution():
    """``_lin_solve`` на вырожденной матрице даёт КОНЕЧНОЕ решение и предупреждает."""
    A = np.array([[1.0, 1.0], [1.0, 1.0]])
    b = np.array([1.0, 1.0])
    sink: list = []
    with pytest.warns(FactorizationWarning, match="УСЕЧЕНО"):
        x = _lin_solve(A, b, sink=sink)
    assert np.all(np.isfinite(x)), "P04: раньше здесь молча получались NaN/inf"
    assert np.allclose(A @ x, b)
    assert np.allclose(x, [0.5, 0.5])               # минимально-нормальное решение
    assert sink == [("lstsq", 1)]                   # одно отсечённое направление


def test_lin_solve_healthy_path_untouched():
    """На невырожденной матрице ``_lin_solve`` идёт прежним путём LU и молчит."""
    rng = np.random.default_rng(7)
    A = rng.standard_normal((6, 6)) + 6.0 * np.eye(6)
    b = rng.standard_normal(6)
    sink: list = []
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        x = _lin_solve(A, b, sink=sink)
    assert [w for w in rec if issubclass(w.category, FactorizationWarning)] == []
    assert sink == []
    d = np.abs(np.diag(A))
    s = 1.0 / np.sqrt(d)
    expected = sla.lu_solve(sla.lu_factor((A * s) * s[:, None]), b * s) * s
    assert np.array_equal(x, expected)              # арифметика прежняя, бит-в-бит


def test_lin_solve_raises_on_nonfinite_matrix():
    """NaN в матрице — не обусловленность, а ошибка постановки: явный отказ.

    До правки отсюда прилетал непрозрачный ``ValueError: array must not contain
    infs or NaNs`` из внутренней проверки LAPACK; теперь — ``LinAlgError`` с
    указанием причины (``LinAlgError`` наследует ``ValueError``, так что
    внешние обработчики не ломаются).
    """
    A = np.array([[1.0, np.nan], [0.0, 1.0]])
    with pytest.raises(np.linalg.LinAlgError, match="NaN/Inf"):
        _lin_solve(A, np.ones(2))


def test_ktn_plate_reports_no_fallback_on_healthy_run():
    """Здоровый прогон КТН: ``fallback = None``, ``n_dropped = 0`` (нечего скрывать)."""
    from plate_solver.ktn_full import KTNPlate

    cfg = Config(E=1.0, h=0.1, nu=0.3, a=1.0, q0=1e-4, p=8, Q=80,
                 n_load_steps=1, karman_tol=1e-9, karman_max_iter=60)
    plate = KTNPlate.from_config(make_circle(1.0), cfg, bc_type="clamped",
                                 inplane_bc="immovable")
    res = plate.solve_uniform()
    assert np.all(np.isfinite(res.w_nodes))
    assert plate.fallback is None
    assert plate.n_dropped == 0


# --------------------------------------------------------------------------- #
#  (д) МАТРИЧНАЯ правая часть: k систем на одной факторизации
# --------------------------------------------------------------------------- #
def test_cascade_solves_matrix_rhs_on_every_stage():
    r"""``solve`` принимает правую часть ``(n, k)`` на ВСЕХ трёх ступенях каскада.

    Так решается, например, ``∂N/∂c`` в касательном операторе Ньютона
    (`membrane._dN_dc`): одна факторизация, ``k = N`` правых частей столбцами.
    Диагональные множители ступеней 2 (Якоби) и 3 (спектральной) действуют по
    СТРОКАМ, и без явного разворота в столбец numpy либо транслировал их по
    столбцам (иная арифметика), либо отказывал по формам — при p = 12…16 у
    полной КТН это был обрыв прогона (``ValueError`` из ступени 3).

    Проверяется тождество «матричное решение = поколоночное» на каждой ступени
    (главные ворота), точность на здоровой SPD-матрице и совпадение ступени 3
    с независимо собранным псевдообращением. Невязку на ступени 2 проверять
    бессмысленно: она включается лишь на матрицах с ``cond ~ 1e17``, где
    решение любой арифметикой не имеет верных знаков.
    """
    rng = np.random.default_rng(20260902)
    n, k = 8, 5
    B = rng.standard_normal((n, n))
    spd = B @ B.T + n * np.eye(n)                       # ступень 1: Холецкий
    lam = np.array([1.0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-17])
    V = np.linalg.qr(rng.standard_normal((n, n)))[0]
    M = V @ np.diag(lam) @ V.T
    s = 10.0 ** np.array([-3, 2, -1, 4, 0, -4, 3, 1], dtype=float)
    badly_scaled = ((0.5 * (M + M.T)) * s) * s[:, None]  # ступень 2 (если сработает)
    singular = _singular_spd(n=n, k=2)                   # ступень 3: усечение
    Bm = rng.standard_normal((n, k))
    seen = set()
    for name, A, healthy in (("SPD", spd, True), ("плохо масштабированная", badly_scaled, False),
                             ("вырожденная", singular, False)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FactorizationWarning)
            fact = SPDFactorization(A, warn=False)
            X = fact.solve(Bm)
            cols = np.column_stack([fact.solve(Bm[:, j]) for j in range(k)])
        seen.add(fact.fallback)
        assert X.shape == (n, k)
        assert np.array_equal(X, cols), f"{name}: матричный путь разошёлся с поколоночным"
        if healthy:                                      # здоровая SPD: точное решение
            assert fact.fallback is None
            assert np.allclose(A @ X, Bm, rtol=1e-10, atol=1e-12)
        if fact.fallback == "pinv":                      # независимая сборка A⁺
            lam_s, Vp = np.linalg.eigh(0.5 * (A + A.T))
            keep = lam_s > REL_CUTOFF * lam_s[-1]
            ref = Vp[:, keep] @ ((Vp[:, keep].T @ Bm) / lam_s[keep][:, None])
            assert np.allclose(X, ref, rtol=1e-10, atol=1e-12)
    # ступени 1 и 3 обязаны быть пройдены на любой платформе (ступень 2 —
    # платформозависима: где-то её матрицу берёт ещё прямой Холецкий)
    assert {None, "pinv"} <= seen


def test_cascade_rejects_incompatible_rhs():
    """Несогласованная правая часть — понятный отказ, а не молчаливая трансляция."""
    A = _singular_spd()
    fact = SPDFactorization(A, warn=False)
    with pytest.raises(ValueError, match="несовместима"):
        fact.solve(np.ones(A.shape[0] + 1))
    with pytest.raises(ValueError, match="несовместима"):
        fact.solve(np.ones((2, A.shape[0], 3)))
