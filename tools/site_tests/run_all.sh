#!/usr/bin/env bash
# Full check of the site, the way Render will build it.
#
#   bash tools/site_tests/run_all.sh            # build + all tests
#   SKIP_BUILD=1 bash tools/site_tests/run_all.sh   # reuse the last build
#
# 1. Copies exactly the files git would commit (so ignored files can't hide
#    a broken deploy) into a temp folder and runs render-build.sh there in a
#    clean Python 3.10 environment.
# 2. Runs every PyTorch code cell of the book pages through the in-browser
#    Python runtime (in plain Python).
# 3. Serves the site and runs the browser tests (Playwright + Chromium).
#
# Needs: uv. Optional: PANDOC=/path/to/pandoc (a native pandoc, e.g. on Apple
# Silicon where the bundled one may not run). Screenshots go to $SHOTS_DIR if set.
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
WORK="${WORK:-${TMPDIR:-/tmp}/d2l-site-test}"
PORT="${PORT:-8765}"
export SITE_URL="http://127.0.0.1:$PORT/"

if [ -z "${SKIP_BUILD:-}" ]; then
  echo "== Building from the files git would commit, in $WORK"
  rm -rf "$WORK/src"
  mkdir -p "$WORK/src"
  (cd "$REPO" && git ls-files --cached --others --exclude-standard) \
    | (cd "$REPO" && rsync -a --files-from=- . "$WORK/src/")
  [ -d "$WORK/env" ] || uv venv --python 3.10 "$WORK/env"
  if [ -n "${PANDOC:-}" ]; then export PYPANDOC_PANDOC="$PANDOC"; fi
  (cd "$WORK/src" && source "$WORK/env/bin/activate" && \
    uv pip install pip >/dev/null && bash render-build.sh) > "$WORK/build.log" 2>&1 \
    || { echo "BUILD FAILED, see $WORK/build.log"; tail -30 "$WORK/build.log"; exit 1; }
  echo "   build OK"
fi
SITE="$WORK/src/_build/html"
[ -f "$SITE/index.html" ] || { echo "No built site at $SITE"; exit 1; }

echo "== Running book code cells through the browser runtime"
PAGES=$(cd "$REPO" && ls chapter_preliminaries/{ndarray,pandas,linear-algebra,calculus,autograd,probability}.md \
  chapter_linear-regression/{linear-regression,oo-design,synthetic-regression-data,linear-regression-scratch,linear-regression-concise,generalization,weight-decay}.md \
  chapter_linear-classification/{image-classification-dataset,classification,softmax-regression-scratch,softmax-regression-concise}.md \
  chapter_multilayer-perceptrons/{mlp,mlp-implementation,numerical-stability-and-init,dropout,kaggle-house-price}.md)
FAILED=0
(cd "$REPO" && D2L_DATASETS="$SITE/_static/datasets" "$WORK/env/bin/python" tools/site_tests/run_book_cells.py "$SITE/_static/exercises" $PAGES) \
  > "$WORK/cells.log" 2>&1 || FAILED=1
grep -E "FAILED|cells passed|>>>" "$WORK/cells.log" || true

echo "== Browser tests"
[ -d "$WORK/pw" ] || { uv venv --python 3.11 "$WORK/pw" && uv pip install --python "$WORK/pw/bin/python" playwright \
  && "$WORK/pw/bin/python" -m playwright install chromium; }
python3 -m http.server "$PORT" --bind 127.0.0.1 --directory "$SITE" >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 1

for t in test_exercises test_features test_chapter3 test_chapter4 test_chapter5 test_steps_recall test_sync_notes; do
  echo "-- $t"
  "$WORK/pw/bin/python" "$REPO/tools/site_tests/$t.py" > "$WORK/$t.log" 2>&1 || FAILED=1
  grep -E "^FAIL|checks passed|Error" "$WORK/$t.log" || true
done
[ $FAILED -eq 0 ] && echo "== ALL PASSED" || { echo "== SOME TESTS FAILED"; exit 1; }
