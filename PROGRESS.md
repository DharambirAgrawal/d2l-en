# Progress Tracker

My personal copy of *Dive into Deep Learning*, customized for learning and hosted on Render.

- **Live site:** https://d2l-en.onrender.com
- **Upstream book:** https://github.com/d2l-ai/d2l-en

This file is excluded from the book build (see `exclusions` in `config.ini`).

---

## My requirements (keep following these)

Collected from my requests so they are never forgotten. Every change to the
site should respect them.

1. **Distraction-free.** No GitHub, 中文版, MXNet, Discussions/forum, or
   analytics. Only things that help me learn.
2. **Readable text first.** Only trim real filler. **Never cut explanations,
   examples, math, figures or code**: the technical chapters (2+) keep the
   original wording, because heavy compression made them hard to read
   (2026-10-18 feedback on 2.3). Originals remain in git history.
3. **Hidden answers for every end-of-section exercise.** A "Show answer"
   button, so I can solve first and then check.
4. **Coding questions get an interactive coding exercise** (run + tests).
5. **Progress tracking.** Check off completed sections, with ✓ marks in the
   sidebar.
6. **At least one project per chapter.** It must:
   - use everything learned **up to that chapter** (this one and earlier),
     and nothing beyond;
   - be **small and focused on core problem solving**, not a big build that
     eats study time;
   - be **fresh, not the usual tutorial project**, and solve a real problem
     that companies actually have;
   - be **resume-worthy** for internship applications (clear impact).
7. **One continuing capstone** that grows chapter by chapter, so that by the
   end I have built **my own ChatGPT-like model** from what I learned.
   - **Projects and capstone are requirement specs only** (problem, data,
     milestones, acceptance criteria, README/resume notes). **No in-browser
     coding environment for them**: I write the code myself in my own GitHub
     repos, so I can show people what I built.
8. **Keep tracking** everything I ask for here and in the change log below.
9. **Editable, runnable examples.** The book's code examples should be
   editable and runnable in the page (like a notebook), so I can experiment
   while reading. Where the book's code needs PyTorch (which can't run in the
   browser), give a NumPy/pandas version I can run.
10. **Reusable interactive components** (build once, reuse everywhere):
    graphs/plots, playgrounds with sliders for trying different values, so
    adding interactivity to later chapters stays easy.
11. **Short "notes to remember"**, placed **right where the idea is taught**
    (end of the relevant subsection), not collected at the end of the page,
    so I can write notes while I learn.
12. **Playgrounds must be smooth:** plots update continuously while dragging
    a slider (not only after I let go).
13. **Clean, calm UI:** neutral, readable design; no colorful side borders
    or flashy "AI-looking" decoration.
14. **Project names are brand names, not robotic labels.** Name projects the
    way Apple names products: one short, evocative, human word (like
    "Pulse" or "Lumen"), plus a plain subtitle that says what it does.
    Never "XYZ-Analyzer-v2"-style names.

---

## Hosting setup (Render static site)

| Setting           | Value                  |
|-------------------|------------------------|
| Build Command     | `bash render-build.sh` |
| Publish Directory | `_build/html`          |
| Python            | 3.10 (Render default)  |

### How the build works (`render-build.sh`)

1. Installs d2lbook plus pinned dependencies (see "Pinned versions" below).
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

Test from **exactly what git will commit** (tracked + new, non-ignored
files), because `.gitignore` ignores patterns such as `*.csv` and `*.json`
that a plain folder copy would include:

```bash
git ls-files --cached --others --exclude-standard > /tmp/files.txt
rsync -a --files-from=/tmp/files.txt ./ /tmp/d2l-test/   # then build there
```

Data files are kept by exceptions at the end of `.gitignore`
(`!exercises/*/files/*.csv`, `!static/projects/*.csv`); add one if you add a
new kind of data file.



---

## Customizations made

### 2026-10-18: First deploy
- Set `eval_notebook = False`: code isn't executed during the build, so pages
  show code but no precomputed outputs (use the in-page **▶ Run** instead).
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

### 2026-10-18: Interactive exercises, Phase 1 (live)
- Pyodide runner + two exercises in 2.3 (details below).

### 2026-10-18: Preface → Chapter 2 made into a study course
- **Removed:** "Discussions" tabs/comment boxes at the end of every section
  (all 166 chapter files) and Google Analytics (`config.ini`).
- **Shortened the prose** of the Preface, Installation, Notation,
  Introduction and Chapter 2 (about 28% shorter overall; every code cell,
  equation, figure, label, citation and example kept, checked by script).
  The Preface's forum exercises were replaced with a study-plan exercise.
- **Hidden answers** for all 51 end-of-section exercises (`answers/`).
- **12 new coding exercises** for the coding questions (`exercises/`).
- **Notes to remember** box in each section with key formulas (`notes/`).
- **Runnable book examples:** every PyTorch and shared code cell has
  **▶ Run / ✎ Edit / Restart**, running notebook-style in the browser on
  *torchlite* (a NumPy-based subset of PyTorch with a small autograd).
  All 94 Chapter 2 PyTorch cells run.
- **Playgrounds** with sliders and live plots (`playgrounds/`): matrix as a
  transformation, $\ell_p$ unit balls, secant to tangent, coin-toss
  convergence, Bayes for medical tests.
- **Progress tracking:** "Mark section as complete" button, ✓ in the sidebar,
  per-chapter counts, overall progress bar, Export/Import backup
  (`static/progress/`; stored in this browser's localStorage).
- **Projects (requirement docs only; I code them in my own GitHub repos):**
  - 2.8 *Project: On-Call Alert Triage* (data: `static/projects/metrics.csv`,
    2,000 simulated telemetry windows).
  - 2.9 *Capstone, Part 1: MiniGPT from Counts* (baseline: held-out
    perplexity about 11.4 on *The Time Machine*).

### 2026-10-18: Readability and UI pass (feedback)
- **Chapter 2 text restored to the original wording** (my 28% compression
  cut too much; 2.3 was hard to read). Preface/Introduction stay lightly
  shortened.
- **Notes moved inline**: each notes box sits at the end of the subsection
  it summarizes (`after: <heading>` in `notes/*.yml`), not at page end.
- **Playgrounds redrawn in the browser** (canvas, `static/exercises/playgrounds.js`):
  plots update on every slider movement. Each keeps a runnable
  "Python version".
- **Calmer UI**: neutral cards, thin gray borders, one accent color, no
  colored side stripes; quieter answer/notes/cell controls.
- **Cleanup**: removed the remaining navbar links (Stable Version, PDF,
  Notebooks, Courses), the Colab/SageMaker buttons (`github_repo` emptied in
  `[colab]`/`[sagemaker]`, which also skips generating their notebooks), the
  homepage news list, Twitter button and forum promo.

### 2026-10-18: Progress sync + "My Notes" review page
- **Sync** (sidebar progress box → *Sync*), `static/progress/sync.js`:
  - *Sync link*: copies a link carrying completed sections + solved
    exercises in its `#fragment` (never sent to the server); opening it on
    another device merges them in.
  - *Automatic sync with GitHub*: paste a fine-grained token with only
    "Gists: Read and write" on each device; progress is kept in a secret gist
    `d2l-progress.json` in my account, pulled on page load and pushed ~2 s
    after changes. Merging is a union (syncing never loses a check mark;
    unmarking only applies on that device). The token stays in that
    browser's localStorage; *Disconnect* removes it.
  - Not synced: exercise code drafts (stay per browser).
- **My Notes** page (sidebar, after Notation; `chapter_my-notes/index.md`,
  `static/progress/review.js`): all Notes boxes from completed sections in
  book order, links back to each subsection, "Show all sections", *Copy as
  text*, *Print / save as PDF*.

### 2026-10-18: Chapter 3 (Linear Neural Networks for Regression)
- Original text kept. **Answers** for all 41 exercises, **inline notes** for
  every section, **6 coding exercises** (`mean-vs-median`, `poisson-loss`,
  `reproducible-batches`, `linreg-sgd-numpy`, `huber-loss`,
  `weight-decay-updates`), **4 playgrounds** (learning rate & batch size,
  loss shapes, polynomial over/underfitting, weight decay strength).
- **All Chapter 3 PyTorch examples run in the browser**, including the
  `d2l.Module`/`Trainer` training loops with live loss plots:
  - `tools/build_d2l_book.py` generates the browser's `d2l` library
    (`_static/exercises/d2l_book.py`) from the book's own `#@save` cells,
    exactly like the real `d2l` package (plus `save_hyperparameters`,
    `ProgressBoard.draw` and device helpers from later sections). Add a
    chapter to `CHAPTERS` there when preparing it.
  - torchlite now has `nn.Module`, `nn.Parameter`, `nn.Linear`,
    `nn.LazyLinear` (materialized in place, like PyTorch), `nn.MSELoss`,
    `nn.Sequential`, `nn.init`, `optim.SGD` (weight decay, param groups),
    `utils.data.TensorDataset`/`DataLoader`, `.data`, `normal_`/`fill_`.
  - Verified: the scratch and concise models recover w = [2, -3.4],
    b = 4.2 like the book; weight-decay norms match the book's output.
- **Projects renamed to brand names** (requirement 14): 2.8 *Hush* (was
  On-Call Alert Triage), capstone *Quill* (was MiniGPT).
- **3.8 Project: Tempo**, honest delivery-time (ETA) estimates; data
  `static/projects/deliveries.csv` (4,000 orders, 90 days). Reference test
  MAE: average 9.4 → numeric 6.3 → +time 5.4 → +restaurants 4.2 → +Huber
  4.0 min.
- **3.9 Capstone: Quill, Part 2**, the bigram model trained by minibatch SGD;
  reference: validation perplexity 11.7 after 3 epochs (lr 10, batch 256),
  approaching the count baseline 11.4.

---

## Still on the site (possible next cleanups)

- JAX and TensorFlow code tabs.
- An empty `<script src="">` in the theme causes a harmless
  "Refused to execute script" console error.

---

## How the learning features work

All of these are driven by plain files in the repo and checked at build time
by `tools/build_exercises.py` (the deploy fails if something is broken).

| Feature | Where the content lives | Embed / placement |
|---|---|---|
| Coding exercise | `exercises/<id>/` (exercise.yml, starter.py, solution.py, tests.py, optional `files/`) | ```` ```{=html} ```` + `<div class="d2l-exercise" data-exercise="<id>"></div>` in a chapter, or `exercise:` in an answers file |
| Hidden answers | `answers/<chapter>/<section>.yml`: one entry per item of the section's "## Exercises" list (`answer:` Markdown, optional `exercise:` id) | automatic, under each exercise |
| Notes to remember | `notes/<chapter>/<section>.yml`: `- after: <section heading>` + `notes:` list of Markdown | automatic, at the end of that section |
| Playground | `playgrounds/<id>.yml` (title, description, slider params, optional `python` version) + drawing code `registry['<id>']` in `static/exercises/playgrounds.js` | `<div class="d2l-playground" data-playground="<id>"></div>` |
| Runnable book cells | nothing to add: every `pytorch`-tab / shared cell | automatic |
| Progress, sync, My Notes | nothing to add | automatic (`static/progress/`) |
| Project data | `static/projects/` → `_static/projects/` | link from a project page |

Runtime files (`static/exercises/`): `exercise.js` (all UI), `exercise.css`,
`worker.js` (Pyodide in a Web Worker), `grader.py` (tests; shared with the
build check), `notebook.py` (book cells + playgrounds, notebook sessions),
`torchlite.py` (NumPy-based PyTorch subset: tensors, broadcasting,
reductions, linear algebra, autograd, Multinomial), `d2llite.py` (the book's
plotting helpers). `static/progress/` holds progress tracking.

**Adding a chapter later:** shorten its prose (keep code/math/figures/
examples), write `answers/` + `notes/`, add exercises for coding questions,
add playgrounds where a slider helps, write its project spec page and the
next capstone part, run the chapter's PyTorch cells through torchlite (and
extend torchlite if a cell needs more of PyTorch), then do a full local
build + browser test.

## Interactive coding exercises

**Live since 2026-10-18.**

### Decision: Pyodide (Python in the browser) in a Web Worker

| Option | Verdict |
|---|---|
| **Pyodide + Web Worker** | ✅ Chosen. Free, no backend, works on Render static hosting. Has NumPy, SciPy, pandas, scikit-learn, matplotlib. |
| Backend runner (Judge0 / own server on Render) | ❌ Costs money or sleeps on free tier, 512 MB RAM, sandboxing is hard. |
| JupyterLite (full notebook in browser) | ❌ for graded exercises (heavy, no grading UI). |
| Binder / Thebe (real Jupyter kernel) | ❌ 1–2 min cold starts, unreliable. |
| Real PyTorch in the browser | ❌ Not possible in Pyodide → torchlite for the book's basic examples. |

Pinned versions: Pyodide **314.0.7** (Python 3.14 in the browser) from
jsDelivr, CodeMirror **5.65.18** from cdnjs (top of `exercise.js`).
Pyodide 314+ must be loaded as a module worker (`import()`).

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
   - `starter.py` (must NOT pass all tests), `solution.py` (must pass all),
     `tests.py` (every `test_*` function is a test; its docstring is shown):
     ```python
     from submission import vector_add        # the code being graded
     from grader import expect_equal, expect_close, expect_raises

     def test_example():
         """[1, 2] + [3, 4] = [4, 6]"""
         expect_equal(vector_add([1, 2], [3, 4]), [4, 6])
     ```
     Helpers: `expect_equal`, `expect_close` (tolerance; lists, numbers,
     NumPy arrays), `expect_raises`, `expect_type`, `expect_shape`,
     `expect_true`, `uses_any(...)` (for "don't use `np.dot`" rules).
   - Optional `files/`: data files (CSV, text, helper `.py`) put in the
     working directory. Give helper modules unique names.
2. Embed it in a chapter's `.md` (must be this raw-HTML fence; a plain `<div>`
   loses its `data-exercise` attribute during the book build):
   ````
   ```{=html}
   <div class="d2l-exercise" data-exercise="<id>"></div>
   ```
   ````
   or attach it to an end-of-section exercise via `exercise: <id>` in the
   section's answers file.
3. Check: `python tools/build_exercises.py` (needs `numpy`, `pandas`,
   `matplotlib`, `pyyaml`, `mistune`).

### Exercises so far (14)

| Section | Exercises |
|---|---|
| 2.1 Data Manipulation | `broadcast-shape` |
| 2.2 Data Preprocessing | `missing-report` |
| 2.3 Linear Algebra | `vector-add`, `matvec-numpy` (in text); `row-normalize`, `sum-shapes`, `matmul-chain-order`, `stack-and-slice` (Exercises) |
| 2.4 Calculus | `tangent-line`, `numerical-gradient` |
| 2.5 Automatic Differentiation | `chain-rule-by-hand` |
| 2.6 Probability and Statistics | `coin-estimator-variance`, `coupled-tests-bayes`, `portfolio-stats` |

### Testing record
- 2026-10-18 Phase 1: clean full build + 27/27 browser checks (exercises).
- 2026-10-18 Study-course update: clean full build (exit 0, 61 warnings,
  same as before), all 94 Chapter 2 PyTorch cells pass through torchlite,
  exercise suite 27/27, plus browser checks for runnable cells (auto-running
  earlier cells, shared session, editing, errors, restart, autograd, plots),
  playgrounds (render, slider updates), notes, hidden answers, progress
  (mark, persist, sidebar), project page + CSV download, no Discussions or
  analytics requests, no new JS errors.

### Known limits (by design)
- Tests/solutions ship to the browser, so "Show solution"/answers are
  honor-system.
- First run downloads Pyodide (~10 MB; NumPy ~8 MB; matplotlib more for
  plots), cached after.
- torchlite is a teaching subset: no `nn`, `optim`, GPU, or data loaders.
  Later chapters that use them need torchlite extended or Colab.
- Progress and saved code live in this browser's localStorage (use Export /
  Import in the sidebar to back up or move progress).
- Browser Python is 3.14 while the build check uses Render's 3.10.

## To-do

- [ ] Deploy the study-course update and try it on the live site.
- [x] Chapter 3: answers, notes, exercises, playgrounds, runnable training
      loops, Tempo project, Quill Part 2.
- [ ] Chapter 4 (classification): same treatment, Quill Part 3; torchlite
      needs cross-entropy, softmax, `nn.Flatten`, image data (Fashion-MNIST).
