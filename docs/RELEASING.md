# Выпуск версии и DOI (действия автора)

## Релиз на GitHub

1. Поднять версию в `plate_solver.__version__` — ЕДИНСТВЕННЫЙ источник
   истины (pyproject берёт её динамически) — и синхронно в витринах
   `CITATION.cff` (поле `version` и `date-released`), `.zenodo.json`,
   `codemeta.json`; ворота — `pytest tests/test_version_sync.py`.
2. Полный локальный прогон: `pytest` (все маркеры) — однострочник в
   CHANGELOG; `ruff check .`.
3. Сборка и проверка пакета: `python -m build && twine check dist/*`
   (extra dev).
4. Аннотированный тег: `git tag -a vX.Y.Z` (релиз-ноты из CHANGELOG),
   `git push origin main --tags`.
5. GitHub → Releases → Draft new release из тега (текст из CHANGELOG).

### Эталонный отчёт и его спутники

`results/reference/reference_v0.8.md` и его CSV заморожены SHA-256
(`tests/test_reference_hash.py`); оба обновляются ТОЛЬКО осознанным коммитом
(перегенерация `scripts/run_reference.py` + новые строки хешей + обоснование
в CHANGELOG). Отчёт ссылается на `provenance.json` рядом с собой — файл
git-ом НЕ отслеживается намеренно (git-хеш, дата и версии сделали бы отчёт
недетерминированным и сломали бы хеш-ворота): он появляется у того, кто
запускает прогон локально, и в архив релиза не входит.

## DOI через Zenodo (однократная настройка + на каждый релиз)

1. Однократно: zenodo.org → Log in with GitHub → GitHub-интеграция →
   включить репозиторий plate-solver (метаданные подхватятся из
   `.zenodo.json`).
2. Опубликовать GitHub-релиз (шаг выше) — Zenodo автоматически создаст
   депозит и выдаст DOI версии + concept-DOI.
3. Вписать DOI. Разделение полей в `CITATION.cff` — такое:

   * `doi:` и `url:` несут CONCEPT-DOI (все версии, резолвится на
     последнюю) — он один на весь проект и НЕ меняется от релиза к
     релизу; concept-DOI в README тоже стоит с v0.6.2 и не трогается;
   * VERSION-DOI очередного релиза добавляется отдельной записью в
     список `identifiers` — `type: doi`, `value: 10.5281/zenodo.<новый>`,
     `description: Version DOI vX.Y.Z`; там же уже лежит запись
     concept-DOI с пометкой «Concept DOI (all versions)».

   Version-DOI выдаётся Zenodo ПОСЛЕ публикации релиза, поэтому запись
   вносится отдельным коммитом вслед за тегом (так было с v0.7.0).
4. Провенанс верификации: перед релизом обновить блок
   `vvprov:verification` в `codemeta.json` — `vvprov:softwareVersion`
   на новую версию, достигнутые величины новых/изменённых актов — по
   текущим тест-воротам; самопроверка — `pytest tests/test_vvprov.py`
   (артефакты существуют, версия синхронна, статусы из словаря).

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
