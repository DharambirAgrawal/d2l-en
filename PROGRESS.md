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

## To-do

- [ ] _(add next tasks here)_
