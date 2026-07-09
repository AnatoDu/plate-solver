# Выпуск версии и DOI (действия автора)

## Релиз на GitHub

1. Полный локальный прогон: `pytest` (все маркеры) — однострочник в
   CHANGELOG; `ruff check .`.
2. Сборка и проверка пакета: `python -m build && twine check dist/*`
   (extra dev).
3. Аннотированный тег: `git tag -a vX.Y.Z` (релиз-ноты из CHANGELOG),
   `git push origin main --tags`.
4. GitHub → Releases → Draft new release из тега (текст из CHANGELOG).

## DOI через Zenodo (однократная настройка + на каждый релиз)

1. Однократно: zenodo.org → Log in with GitHub → GitHub-интеграция →
   включить репозиторий plate-solver (метаданные подхватятся из
   `.zenodo.json`).
2. Опубликовать GitHub-релиз (шаг выше) — Zenodo автоматически создаст
   депозит и выдаст DOI версии + concept-DOI.
3. Вписать DOI: бейдж в README (заменить плейсхолдер), поле `doi:` в
   CITATION.cff; закоммитить.

## Публикация на PyPI (на каждый релиз)

Артефакты собраны на шаге «Релиз на GitHub» (`python -m build`; `twine check
dist/*` — оба PASSED).

1. **Репетиция на TestPyPI** (версия на боевом PyPI неизменяема):
   `python -m twine upload --repository testpypi dist/*` — проверить рендер
   README и метаданные на `test.pypi.org/project/plate-solver`.
2. **Боевая публикация** — один из вариантов:
   - токен: `python -m twine upload dist/*` (логин `__token__`, пароль —
     API-токен pypi.org);
   - Trusted Publishing (без токенов): добавить репозиторий как trusted
     publisher на PyPI (OIDC) и публиковать GitHub Action
     `pypa/gh-action-pypi-publish` от того же релиза, что и Zenodo.
3. **Проверка:** `pip install plate-solver` в чистом окружении.

## Госрегистрация ПрЭВМ

Комплект материалов собирает `scripts/make_registration_kit.py`
(в неотслеживаемый `private/registration/`); подача — по чек-листу
`private/REGISTRATION.md`.
