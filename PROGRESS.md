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
15. **Show my growth in Quill.** Every Quill part shows a progress section:
    the metric for every version so far (held-out perplexity), a real
    generated sample from each version (so improvement is visible), the
    skills each part added, and the next target.
16. **Optional side projects in every chapter.** 2--3 short, fun specs per
    chapter that apply that chapter's ideas to something real and
    interesting (like using minimax to solve tic-tac-toe), and later more
    ambitious ones that companies such as Google or Anthropic would value.
    Brand names, one-line pitch, why it matters, tasks, "done when".
17. **Three kinds of projects:** (a) **Quill**, the book-long capstone;
    (b) **one chapter project per chapter**, built **step by step inside the
    chapter**: a short step at the end of each section where it fits (not
    every section has to have one); (c) optional side projects at the end of
    the chapter.
18. **Steps must revisit earlier topics (spacing + interleaving).** A step
    doesn't use a topic once and drop it: each step has a *Revisit* part that
    brings back earlier sections/chapters, and the last step uses the whole
    chapter, so nothing is forgotten by the end.
19. **Quick recall at the top of every section:** 2--3 retrieval questions on
    earlier material (previous section, older sections, previous chapters),
    answers hidden. Based on the strongest evidence in learning science:
    practice testing and distributed practice (Dunlosky et al., 2013;
    Donoghue & Hattie, 2021). I forget fast and want to truly learn the
    whole book.
20. **Projects must feel current (2026).** Chapter projects should be the
    kind of problem AI companies work on now (LLM evaluation, serving,
    safety, RLHF), while still using only the chapter's concepts. Generic
    textbook scenarios felt dull (2026-10-24).

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

**One command does all of this plus every test** (needs `uv`):

```bash
PANDOC=/path/to/arm64/pandoc bash tools/site_tests/run_all.sh
SKIP_BUILD=1 bash tools/site_tests/run_all.sh   # re-test the last build
```

It builds from the files git would commit in a clean Python 3.10 env, runs
every PyTorch book cell of the covered chapters through the browser runtime
(`run_book_cells.py`), serves the site, and runs the Playwright suites in
`tools/site_tests/`: `test_exercises.py` (coding exercises),
`test_features.py` (runnable cells, playgrounds, notes, answers, progress,
removed distractions), `test_chapter3.py` (training in the browser, Ch 3
pages), `test_chapter4.py` (Fashion-MNIST download and training in the
browser, Ch 4 playgrounds and pages), `test_steps_recall.py` (project steps, Quick recall),
`test_sync_notes.py` (sync link, mocked GitHub gist, My Notes). Logs land in
`$WORK` (default `$TMPDIR/d2l-site-test`); set `SHOTS_DIR` for screenshots.
When adding a chapter, add its pages to `PAGES` in `run_all.sh` and a test
file for its features.



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

### 2026-10-18: Quill progress + optional side projects
- Both Quill pages have **"How Far Quill Has Come"**: perplexity for every
  version (random 70.0 → counts 11.4 → SGD 11.7), rough goals for later parts
  (MLP 6--8, RNN 4--6, Transformer below 4; to be replaced by measured
  numbers), real samples from each version, and the skills each part added.
- **Optional side projects** on each chapter's project page:
  - Ch 2 (on Hush): *Ripple* (PageRank), *Spark* (an autograd engine),
    *Verdict* (honest A/B testing).
  - Ch 3 (on Tempo): *Scout* (hyperparameter search), *Drift* (online
    learning under concept drift), *Watt* (energy demand forecasting).


### 2026-10-24: Projects built step by step, Quick recall, test suite
- **Chapter projects are now built inside the chapter.** `projects/<name>.yml`
  lists steps (`page`, `title`, `build`, `revisit`); each section page shows
  its step (e.g. "Tempo · Step 3 of 7") just before the Exercises, and the
  project page lists all steps with where each belongs. Hush: 6 steps;
  Tempo: 7. Every step after the first has a *Revisit* part, and the last
  step uses the whole chapter.
- **Quick recall** at the top of every Ch 2--3 section: 2--3 questions on
  earlier material (previous sections, older sections, the Introduction),
  answers hidden. Content in `recall/<chapter>/<section>.yml`.
- **Permanent test suite** in `tools/site_tests/` (see "Testing a build
  locally").


### 2026-10-24: Chapter 4 (Linear Neural Networks for Classification)
- **Original text kept**; 7 sections of hidden answers, inline notes, and
  Quick recall (questions from Ch 4, 3 and 2).
- **7 new coding exercises:** `stable-softmax`, `soft-cross-entropy`,
  `exact-validation-loss`, `bayes-decision`, `exp-safe-range`,
  `hoeffding-samples`, `shift-detector`.
- **4 playgrounds:** softmax + temperature, softmax regression learning
  decision regions live, test-set size vs. Hoeffding band, covariate shift
  and reweighting (effective sample size).
- **Fashion-MNIST in the browser:** torchlite gained `softmax`,
  `log_softmax`, `F.cross_entropy`, `nn.CrossEntropyLoss`, `nn.Flatten`,
  `nn.ReLU`, any-dataset `DataLoader`, and a `torchvision` subset
  (`datasets.FashionMNIST`, `transforms.Compose/Resize/ToTensor/Normalize`).
  The full dataset (60k + 10k images, MIT license) is
  `static/datasets/fashion-mnist.npz` (31 MB, uint8), downloaded by the page
  once when a cell mentions `FashionMNIST`. Book cells may now run 180 s.
  The book's training cells reach about 83% validation accuracy, as in the
  book.
- **Project: Relay** (route support tickets; `static/projects/tickets.csv`,
  8,000 tickets, a June product launch causes covariate + label shift),
  7 steps in `projects/relay.yml`. Side projects: *Sieve* (spam filter),
  *Gauge* (calibration, temperature scaling), *Sentinel* (drift alarm).
  Reference results: baseline 25%, softmax regression 96.7% (May), 88%
  (June), 93% after labeling 50 June tickets.
- **Quill Part 3:** softmax regression over the previous $k$ characters
  (measured: perplexity 8.4 at $k = 5$; count trigram 7.3 and 4-gram 5.8 do
  better, motivating the MLP); *The War of the Worlds* shift test (9.2).
  Roadmap renumbered: Part 4 = MLP (Ch 5), 5 = RNN, 6 = Transformer.

### 2026-10-24: 2026-style projects + Chapter 5 (Multilayer Perceptrons)

The user found the first projects dull, so all three chapter projects were
replaced with ones about building and running AI systems in 2026. Same
step-by-step structure, new data (generators in `projects/generators/`,
so the data can be rebuilt exactly):

- **2.8 Podium** (`battles.csv`, 6,831 chatbot-arena votes): rank 12 AI
  models with Bradley--Terry (the RLHF reward-model loss), fix a 52/48
  position bias, bootstrap error bars; the new model's "#1" is a
  statistical tie (P = 0.61). Side projects: *Lens* (vector search over
  this book, the retrieval half of RAG), *Ripple*, *Spark*.
- **3.8 Pace** (`llm_requests.csv`, 7,522 requests): predict an LLM's time
  to first token from prompt length, prefix-cache hits, queue depth, model
  and GPU. Test MAE 514 → 472 → 311 → 196 → 172 (Huber) → 155 ms (+client);
  a P90 promise of 1.19× the prediction. Side projects: *Scale*
  (scaling-law extrapolation), *Scout*, *Drift*.
- **4.8 Shield** (`agent_inbox.csv`, 8,000 messages): a prompt-injection /
  exfiltration / jailbreak / phishing filter for an AI agent. May 97.8%;
  a June injection style drops attacks caught from 97% to 81%; cost-aware
  `decide` gets back to 89%; 50 June labels → 93%. Side projects:
  *Router* (model routing), *Gauge*, *Sentinel*.
- Removed: Hush, Tempo, Relay (and metrics/deliveries/tickets CSVs).

Chapter 5, same treatment as Chapters 3--4:

- **Runs in the browser:** all 36 PyTorch cells. torchlite gained
  `torch.max/min` (elementwise and by dim), `maximum/minimum`,
  `nn.Dropout` / `F.dropout` (respects `train()`/`eval()`), `nn.Sigmoid`,
  `nn.Tanh`, `nn.init.xavier_normal_/kaiming_*`. The house-price CSVs ship in
  `static/datasets/` and `d2l.download` returns the local copy (browser
  stand-in in `tools/build_d2l_book.py`); the worker fetches them when a
  cell mentions `KaggleHouse`. Results match the book (MLP about 85%,
  k-fold log MSE about 0.17).
- **Answers** for all 42 exercises, with numbers measured in real PyTorch
  (hidden units, learning rates, activations, init, dropout vs. weight
  decay, house-price tuning).
- **Notes** (28 sections) and **recall** for all 7 sections. Notes can now
  sit under headings with d2lbook slide marks (`[**Training**]`).
- **7 coding exercises:** `activation-derivatives`, `mlp-forward`,
  `mlp-backprop`, `xavier-init`, `early-stopping`, `dropout-layer`,
  `kfold-split`.
- **3 playgrounds:** `mlp-regions` (hidden units bend the boundary),
  `deep-signal` (vanishing/exploding gradients vs. weight scale and
  activation), `early-stopping` (noisy labels get memorized late).
- **5.8 Project: Taste** (`preferences.csv`, 6,770 labels): an RLHF reward
  model. Linear 68% → MLP 74%; annotator agreement 66% (ceiling about 78%);
  deep sigmoid with tiny init stays at 50%, ReLU + He 73%; audit vs. 441
  expert labels finds a learned length bias. Side projects: *Grok*
  (grokking), *Reel* (feed ranking), *Probe* (probing GPT-2).
- **5.9 Quill Part 4:** character MLP with embeddings (Bengio 2003).
  Measured: k=8, 256 units 5.6; 512 + dropout 5.1; two layers of 512 +
  dropout 0.3 **4.5** (beats the 4-gram's 5.8); next-char accuracy 56%.
- Tests: `tools/site_tests/test_chapter5.py`; run_all.sh runs Chapter 5
  cells too.
- **Browser speed:** an MLP batch takes about 0.1 s in WebAssembly NumPy, so
  from Chapter 5 on the browser trains on the first 10,000 Fashion-MNIST
  images (a printed note says so; about 72--78% instead of 85%). Set in
  `notebook.py` (`_set_train_limit`; `D2L_TRAIN_LIMIT` overrides).
- **pandas 3 in the browser** stores text as a "str" dtype, which broke the
  house-price preprocessing; `notebook.py` sets `future.infer_string=False`.
- `.gitignore` now keeps `static/datasets/*.csv` (the house-price files
  would otherwise never reach Render).
- Full run 2026-10-24: build OK, 206/206 book cells, 192/192 browser
  checks (two flaky checks fixed: float rounding in row sums and in the
  autograd control-flow example).

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
| Chapter project steps | `projects/<name>.yml`: `project`, `subtitle`, `page` (project page), `steps:` list of `page`, `title`, `build`, `revisit` (Markdown) | automatic: a step box before the section's Exercises; all steps where the project page has `<div class="d2l-project-steps"></div>` |
| Quick recall | `recall/<chapter>/<section>.yml`: list of `q`, `a` (Markdown) | automatic, before the section's first subsection |

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

### Exercises so far (34)

| Section | Exercises |
|---|---|
| 2.1 Data Manipulation | `broadcast-shape` |
| 2.2 Data Preprocessing | `missing-report` |
| 2.3 Linear Algebra | `vector-add`, `matvec-numpy` (in text); `row-normalize`, `sum-shapes`, `matmul-chain-order`, `stack-and-slice` (Exercises) |
| 2.4 Calculus | `tangent-line`, `numerical-gradient` |
| 2.5 Automatic Differentiation | `chain-rule-by-hand` |
| 2.6 Probability and Statistics | `coin-estimator-variance`, `coupled-tests-bayes`, `portfolio-stats` |
| Chapter 3 | `mean-vs-median`, `poisson-loss`, `reproducible-batches`, `linreg-sgd-numpy`, `huber-loss`, `weight-decay-updates` |
| Chapter 4 | `stable-softmax`, `soft-cross-entropy`, `exact-validation-loss`, `bayes-decision`, `exp-safe-range`, `hoeffding-samples`, `shift-detector` |
| Chapter 5 | `activation-derivatives`, `mlp-forward`, `mlp-backprop`, `xavier-init`, `early-stopping`, `dropout-layer`, `kfold-split` |

### Testing record
- 2026-10-18 Phase 1: clean full build + 27/27 browser checks (exercises).
- 2026-10-18 Study-course update: clean full build (exit 0, 61 warnings,
  same as before), all 94 Chapter 2 PyTorch cells pass through torchlite,
  exercise suite 27/27, plus browser checks for runnable cells (auto-running
  earlier cells, shared session, editing, errors, restart, autograd, plots),
  playgrounds (render, slider updates), notes, hidden answers, progress
  (mark, persist, sidebar), project page + CSV download, no Discussions or
  analytics requests, no new JS errors.

- 2026-10-24 Chapter 4 + test suite: `tools/site_tests/run_all.sh` clean
  build OK, 170/170 book cells (Ch 2--4), browser suites 27 + 38 + 25 + 29
  + 20 + 24 = 163/163 checks. Fashion-MNIST training (10 epochs) takes about
  9 s in headless Chromium.

### Known limits (by design)
- Tests/solutions ship to the browser, so "Show solution"/answers are
  honor-system.
- First run downloads Pyodide (~10 MB; NumPy ~8 MB; matplotlib more for
  plots), cached after.
- torchlite is a teaching subset (tensors, autograd, the parts of `nn`,
  `optim` and `utils.data` the book has used so far; no GPU). Each new
  chapter extends it as needed.
- Progress and saved code live in this browser's localStorage (use Export /
  Import in the sidebar to back up or move progress).
- Browser Python is 3.14 while the build check uses Render's 3.10.

## To-do

- [ ] Deploy the study-course update and try it on the live site.
- [x] Chapter 3: answers, notes, exercises, playgrounds, runnable training
      loops, Tempo project (now Pace), Quill Part 2.
- [x] Chapter 4 (classification): same treatment, Relay (now Shield), Quill Part 3,
      Fashion-MNIST in the browser.
- [x] Replace the chapter projects with 2026-style ones (Podium, Pace,
      Shield).
- [x] Chapter 5 (MLPs): same treatment, Taste, Quill Part 4 (4.5).
- [ ] Chapter 6 (Builders' Guide): same treatment, Quill Part 5 prep.
