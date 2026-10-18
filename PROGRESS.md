# Progress Tracker

My personal copy of *Dive into Deep Learning*, customized for learning and hosted on Render.

- **Live site:** https://d2l-en.onrender.com
- **Upstream book:** https://github.com/d2l-ai/d2l-en

This file is excluded from the book build (see `exclusions` in `config.ini`).

---

## Hosting setup (Render static site)

| Setting           | Value                  |
|-------------------|------------------------|
| Build Command     | `bash render-build.sh` |
| Publish Directory | `_build/html`          |
| Python            | 3.10 (Render default)  |

### How the build works (`render-build.sh`)

1. Installs d2lbook plus pinned dependencies (see "Pinned versions" below).
2. Fills the navbar link placeholders (`###_..._###`) in `config.ini`.
3. Deletes `_build` so every deploy starts clean.
4. `d2lbook build eval --tab <framework>` for each framework. This converts
   the Markdown to notebooks. `rst --tab all` only merges these; it does not
   create them.
5. `d2lbook build rst --tab all`, then `d2lbook build html --tab all`.
6. Copies the homepage (`static/frontpage/`) into the output.

### Pinned versions (and why)

| Package             | Pin                          | Reason |
|---------------------|------------------------------|--------|
| `pandoc-attributes` | installed before `pypandoc`  | Its `setup.py` calls `pypandoc.convert()`, which current `pypandoc` removed. |
| `setuptools`        | `<81`                        | v81+ dropped `pkg_resources`, which `notedown` imports. |
| `pybtex`            | `==0.24.0`                   | 0.25+ removed `_FakeEntryPoint`, which `sphinxcontrib-bibtex` 2.4 imports. |

### Testing a build locally

Run the full script in a clean Python 3.10 venv before pushing. On Apple
Silicon Macs, the pandoc bundled with `pypandoc_binary` is Intel-only, so point
pypandoc at a native pandoc:

```bash
PYPANDOC_PANDOC=/path/to/arm64/pandoc bash render-build.sh
```

The script edits `config.ini` while it runs. Don't commit that change
(`git checkout config.ini` afterwards).

---

## Customizations made

### 2026-10-18: First deploy
- Set `eval_notebook = False`: code isn't executed during the build, so pages
  show code but **no outputs/plots**.
- Added `render-build.sh` and fixed the dependency/build issues above.

### 2026-10-18: Simplified the site for distraction-free learning
- **Navbar:** removed the GitHub, 中文版 (Chinese version), and MXNet PDF links.
- **Code tabs:** removed MXNet (`tabs = pytorch, jax, tensorflow` in
  `config.ini`), and removed the MXNet eval step from `render-build.sh`.
- **Recommender Systems chapter hidden.** Its code is MXNet-only, so it can't
  be built without MXNet. It's removed from the TOC in `index.md` and listed in
  `exclusions` in `config.ini`. To bring it back, revert both and re-add MXNet.
- **Homepage (`static/frontpage/frontpage.html`):** removed the GitHub star
  button, the Chinese-version/GitHub news sentence, and "NumPy/MXNet" from the
  tagline. Unlinked the two "Recommender Systems" author credits.

---

## Still on the site (possible next cleanups)

- Navbar: "Stable Version" (→ d2l.ai), "PyTorch" PDF, "Notebooks", "Courses".
- JAX and TensorFlow code tabs.
- "Discussions" links at the end of each section (→ discuss.d2l.ai).
- "Open the notebook in Colab / SageMaker" buttons.
- Homepage news items, Twitter follow button, and author/translation sections.
- Code outputs (plots, printed results) are missing because
  `eval_notebook = False`.

## Interactive coding exercises

Write Python inside the book pages and run/grade it, LeetCode-style.
**Phase 1 built and tested on 2026-10-18** (awaiting deploy).

### Decision: Pyodide (Python in the browser) in a Web Worker

| Option | Verdict |
|---|---|
| **Pyodide + Web Worker** | ✅ Chosen. Free, no backend, works on Render static hosting. Has NumPy, SciPy, pandas, scikit-learn, matplotlib. |
| Backend runner (Judge0 / own server on Render) | ❌ Costs money or sleeps on free tier, 512 MB RAM, sandboxing is hard. |
| JupyterLite (full notebook in browser) | ❌ for graded exercises (heavy, no grading UI); maybe later for a "scratchpad". |
| Binder / Thebe (real Jupyter kernel) | ❌ 1–2 min cold starts, unreliable. |
| PyTorch in the browser | ❌ Not possible in Pyodide. PyTorch exercises → "Open in Colab" (phase 3). |

### How it fits together

| File | Role |
|---|---|
| `exercises/<id>/` | One folder per exercise (see below). |
| `static/exercises/exercise.js` | Page UI: editor, buttons, results, auto-save. Added to every page via `include_js` in `config.ini`; does nothing on pages without exercises. |
| `static/exercises/exercise.css` | Styling, via `include_css`. |
| `static/exercises/worker.js` | Web Worker that loads Pyodide and runs code. |
| `static/exercises/grader.py` | Runs code and tests. **Shared** by the browser and the build check, so both grade identically. Includes the test helpers. |
| `tools/build_exercises.py` | Validates every exercise, then writes `_build/html/_static/exercises/<id>.json` plus the worker and grader. Run twice by `render-build.sh` (check early, write at the end). |

Pinned versions: Pyodide **314.0.7** (Python 3.14 in the browser) from
jsDelivr, CodeMirror **5.65.18** from cdnjs. Both are set at the top of
`exercise.js`. Pyodide 314+ must be loaded as a module worker (`import()`);
the old `importScripts()` method fails.

### How to add an exercise

1. Create `exercises/<id>/` (`id` = lowercase words joined by `-`) with:
   - `exercise.yml`:
     ```yaml
     title: Implement Vector Addition
     difficulty: easy            # easy | medium | hard
     timeout: 5                  # seconds of run time (default 10)
     packages: []                # optional; imports are detected automatically
     description: |              # Markdown; $math$ and $$math$$ work
       Write `vector_add(a, b)` ...
     explanation: |              # shown with the solution
       ...
     ```
   - `starter.py`: what you start from (must NOT pass all tests).
   - `solution.py`: reference solution (must pass all tests).
   - `tests.py`: plain Python. Every `test_*` function is a test, and its
     docstring is the description shown:
     ```python
     from submission import vector_add        # the code being graded
     from grader import expect_equal, expect_close, expect_raises

     def test_example():
         """[1, 2] + [3, 4] = [4, 6]"""
         expect_equal(vector_add([1, 2], [3, 4]), [4, 6])
     ```
     Helpers in `grader.py`: `expect_equal`, `expect_close` (tolerance; lists,
     numbers, NumPy arrays), `expect_raises`, `expect_type`, `expect_shape`,
     `expect_true`, and `uses_any(...)` (for "don't use `np.dot`" rules).
2. Embed it in a chapter's `.md` (must be this raw-HTML fence; a plain `<div>`
   loses its `data-exercise` attribute during the book build):
   ````
   ```{=html}
   <div class="d2l-exercise" data-exercise="<id>"></div>
   ```
   ````
3. Check it: `python tools/build_exercises.py` (needs `numpy`, `pyyaml`,
   `mistune`). The deploy fails if a solution fails its tests, a starter
   already passes, or a chapter embeds an unknown id.

### Exercises so far

| Id | Chapter | Tests |
|---|---|---|
| `vector-add` | 2.3 Linear Algebra (after "Basic Properties of Tensor Arithmetic") | 8 |
| `matvec-numpy` | 2.3 Linear Algebra (after "Matrix–Vector Products") | 6 |

### Phases
- [x] **Phase 1: Runner + first exercises.** Lazy Pyodide load in a worker,
      stdout/traceback capture, fresh module per run, timeout + Stop button
      (kills and restarts the worker), one run at a time per page, Run /
      Submit / Reset / Show solution, auto-save + "Solved" badge in
      `localStorage`, plain-textarea fallback if CodeMirror's CDN fails.
- [x] **Phase 1 test (2026-10-18):** clean full build (exit 0, no new Sphinx
      warnings) + 27/27 headless-Chromium checks on the generated page: run,
      print, syntax error, traceback, unsupported import, failing and passing
      submissions, NumPy, banned-`@` rule, infinite-loop timeout + recovery,
      Stop, Reset, show solution, reload persistence, two independent
      exercises, math/tabs intact, phone width, no new JS errors. Build check
      verified to fail on a broken solution and on an unknown embedded id.
- [ ] **Deploy** and try it on the live site.
- [ ] **Phase 2: Content.** Add exercises chapter by chapter as I study.
      Ideas: hints, per-page progress.
- [ ] **Phase 3: PyTorch exercises.** Generate a Colab notebook per exercise
      (starter + tests) from the same exercise folder.

### Known limits (by design)
- Tests/solutions ship to the browser, so "Show solution" is honor-system.
- First run downloads Pyodide (~10 MB; NumPy another ~8 MB), cached after.
- Not a security sandbox; it only runs my own code in my own browser.
- Browser Python is 3.14 while the build check uses Render's 3.10, so keep
  exercise code to features both support.

## Pre-existing site issues (found while testing, not fixed yet)
- **Discussions** box at the end of each section is broken: discuss.d2l.ai
  refuses to be embedded on other domains.
- **Google Analytics** with the original D2L tracking IDs is still on every
  page, so visits are reported to D2L's analytics.
- An empty `<script src="">` in the theme causes a harmless
  "Refused to execute script" console error.

## To-do

- [ ] Deploy Phase 1 and try both exercises on the live site.
- [ ] Decide on the pre-existing issues above (likely remove Discussions + analytics).
