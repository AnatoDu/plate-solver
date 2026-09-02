"""Реестр = ворота: параметризованный pytest по cases/ci/*.toml.

Каждый ci-случай гоняется через plate-verify (exit 0). Новый файл в
cases/ci/ автоматически становится тестом. Тяжёлые (Q=1024) ступени —
cases/ladder/ с маркером big в своих тестах.

v0.8.0: для КОНТАКТНЫХ случаев проверка перестала быть формальной. Раньше
при ``reference = "none"`` список эталонов был пуст и plate-verify выносил
PASS, не проверив ничего (расходящийся МОР с ``r ≡ 0`` был неотличим от
решения). Теперь к отчёту добавляются ВОРОТА ИНВАРИАНТОВ, не зависящие от
бюджета итераций: ``r ≥ 0``, проникание ≤ 5 % зазора, существование контакта
при ``w_free > Δ`` и замыкание ``∫r = P`` в силовом режиме
(``references.contact_invariant_rows``). Ниже дополнительно проверяется, что
у контактных случаев эти ворота ДЕЙСТВИТЕЛЬНО есть — иначе «зелёный» реестр
снова стал бы формальностью.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from plate_solver.cli import main_verify

_CI = Path(__file__).resolve().parents[1] / "cases" / "ci"


@pytest.mark.parametrize("case", sorted(_CI.glob("*.toml")), ids=lambda p: p.stem)
def test_ci_case_verifies(case, tmp_path):
    """ВОРОТА: plate-verify по ci-случаю завершается кодом 0."""
    assert main_verify([str(case), "--out", str(tmp_path)]) == 0


def test_every_contact_case_is_actually_gated():
    """ВОРОТА против «вакуумного PASS»: у каждого контактного случая есть гейты.

    Проверяется НАЛИЧИЕ гейтуемых строк (эталон и/или инварианты) — чтобы
    зелёный реестр означал проверку, а не молчание.
    """
    from plate_solver.dispatch import solve
    from plate_solver.problem import Problem
    from plate_solver.references import verify_result

    checked = 0
    for path in sorted(_CI.glob("*.toml")):
        problem = Problem.from_toml(path)
        if not problem.contact.enabled:
            continue
        report = verify_result(solve(problem))
        gated = [r for r in report.rows if r.gated]
        assert gated, f"{path.name}: ни одной гейтуемой строки (вакуумный PASS)"
        assert any("инвариант" in r.name for r in gated), path.name
        assert report.ok, f"{path.name}: {[r.name for r in gated if not r.passed]}"
        checked += 1
    assert checked >= 10, f"контактных ci-случаев найдено {checked}"
