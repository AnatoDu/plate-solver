#!/usr/bin/env bash
# make_review_zip.sh — приёмо-архив для ревью (F0.4).
#
# Состав (эталонный фильтр):
#   * отслеживаемые git-файлы с расширениями
#     py|md|toml|cfg|txt|yml|yaml|json|csv|cff|ipynb|mmd + LICENSE;
#   * review_extras/fields.npz — один ЛЁГКИЙ снимок полей
#     (cases/ci/lshape_stamp.toml, штатный путь Result.save);
#   * review_extras/pytest_light.log — свежий прогон лёгкой связки
#     (pytest -m "not big and not fem").
#
# Использование:  scripts/make_review_zip.sh [каталог-назначения]
# По умолчанию архив кладётся в private/review/ (не отслеживается git).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PYTHON:-$ROOT/.venv/bin/python}"
STAMP="$(date +%Y%m%d)"
OUT_DIR="${1:-$ROOT/private/review}"
mkdir -p "$OUT_DIR"
ZIP="$OUT_DIR/plate-solver-review-$STAMP.zip"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "== 1/4 реестр файлов (git ls-files + фильтр расширений) =="
git ls-files \
    | grep -E '(^LICENSE|\.(py|md|toml|cfg|txt|yml|yaml|json|csv|cff|ipynb|mmd))$' \
    > "$WORK/files.txt"
wc -l < "$WORK/files.txt" | xargs echo "файлов:"

echo "== 2/4 лёгкий fields.npz (cases/ci/lshape_stamp.toml) =="
mkdir -p "$WORK/npz" "$WORK/extras"
"$PY" -c "import sys; from plate_solver.cli import main; sys.exit(main(sys.argv[1:]))" \
    "cases/ci/lshape_stamp.toml" --out "$WORK/npz" >/dev/null
cp "$WORK/npz/fields.npz" "$WORK/extras/fields.npz"

echo "== 3/4 лёгкий pytest-лог (not big and not fem) =="
"$PY" -m pytest -m "not big and not fem" 2>&1 | tee "$WORK/extras/pytest_light.log" \
    | tail -1

echo "== 4/4 сборка $ZIP =="
rm -f "$ZIP"
zip -q "$ZIP" -@ < "$WORK/files.txt"
(cd "$WORK" && mv extras review_extras && zip -qr "$ZIP" review_extras)
echo "готово: $ZIP ($(du -h "$ZIP" | cut -f1))"
