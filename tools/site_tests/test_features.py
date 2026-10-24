"""Chapter 2 features: runnable book cells, playgrounds, inline notes,
hidden answers, progress tracking, project pages, removed distractions."""
import time
import urllib.request

from playwright.sync_api import sync_playwright

from common import (BASE, cell_index, cell_images, cell_output, check, finish, relevant_errors, run_cell, shot,
                    wait_cell_done, watch_errors)


def set_slider(box, i, value):
    box.locator('input[type=range]').nth(i).evaluate(
        '(el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }', value)


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={'width': 1280, 'height': 900})
    page = watch_errors(ctx.new_page())
    requests = []
    page.on('request', lambda r: requests.append(r.url))

    # ---------------- runnable book cells (ndarray)
    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    page.wait_for_selector('.d2l-cell-bar')
    n = page.locator('.d2l-cell-bar').count()
    check('ndarray: runnable cells added', n >= 20, n)
    check('ndarray: only PyTorch/shared cells are runnable', page.evaluate(
        '''() => Array.from(document.querySelectorAll(".d2l-cell-bar")).every(b => {
              const panel = b.closest(".mdl-tabs__panel"); return !panel || panel.id.startsWith("pytorch"); })'''))
    t0 = time.time()
    out, _ = run_cell(page, 'x.numel()')
    print(f'     (first cell incl. Python load: {time.time() - t0:.1f}s)')
    check('running a cell runs earlier cells first (x.numel() -> 12)', out.strip() == '12', out)
    out, _ = run_cell(page, 'x.shape')
    check('cells share one session (x.shape)', 'torch.Size([12])' in out, out)
    k = cell_index(page, 'X = x.reshape(3, 4)')
    page.locator('.d2l-cell-bar').nth(k).locator('.d2l-cell-btn', has_text='Edit').click()
    page.wait_for_selector('.d2l-cell-editor .CodeMirror')
    page.evaluate('() => document.querySelector(".d2l-cell-editor .CodeMirror").CodeMirror.setValue("X = x.reshape(2, 6)\\nX")')
    page.locator('.d2l-cell-bar').nth(k).locator('.d2l-cell-run').click()
    wait_cell_done(page, k)
    out = cell_output(page, k)
    check('edited cell runs the new code', out.count('[') == 3 and '11' in out, out)
    out, _ = run_cell(page, 'a + b')
    check('broadcasting example output', '[[0, 1]' in out.replace('[[0,1]', '[[0, 1]'), out)
    out, _ = run_cell(page, 'id(Y) == before')
    check('id(Y) == before -> False', 'False' in out, out)
    page.evaluate('() => document.querySelector(".d2l-cell-editor .CodeMirror").CodeMirror.setValue("x.reshape(5, 5)")')
    page.locator('.d2l-cell-bar').nth(k).locator('.d2l-cell-run').click()
    wait_cell_done(page, k)
    out = cell_output(page, k)
    check('a failing cell shows its error', 'Error' in out or 'cannot reshape' in out, out)
    page.locator('.d2l-cell-bar').nth(0).locator('.d2l-cell-btn', has_text='Restart').click()
    check('restart clears outputs', page.evaluate(
        '() => Array.from(document.querySelectorAll(".d2l-cell-out")).every(o => o.hidden)'))
    shot(page, 'cells.png')

    # ---------------- autograd
    page.goto(BASE + 'chapter_preliminaries/autograd.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, _ = run_cell(page, 'x.grad == 4 * x')
    check('autograd: x.grad == 4 * x is all True', out.count('True') == 4, out)
    out, _ = run_cell(page, 'a.grad == d / a')
    check('autograd: control-flow example -> True', 'True' in out, out)

    # ---------------- calculus: plot cell + live playground + its Python version
    page.goto(BASE + 'chapter_preliminaries/calculus.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, imgs = run_cell(page, "legend=['f(x)', 'Tangent line")
    check('calculus: plot cell renders an image', imgs == 1, out)
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    check('playground draws immediately', 'exact' in box.locator('.d2l-pg-readout').inner_text(),
          box.locator('.d2l-pg-readout').inner_text())
    set_slider(box, 1, 0.05)
    page.wait_for_timeout(150)
    check('moving a slider redraws live', 'error = 0.15000' in box.locator('.d2l-pg-readout').inner_text(),
          box.locator('.d2l-pg-readout').inner_text())
    set_slider(box, 1, 1.82)
    box.locator('.d2l-link-btn', has_text='Show Python version').click()
    page.wait_for_selector('.d2l-pg-python .CodeMirror')
    box.locator('.d2l-pg-python .d2l-btn', has_text='Run with these values').click()
    page.wait_for_function(
        '() => { const o = document.querySelector(".d2l-pg-python .d2l-cell-out"); return o && !o.hidden && o.childElementCount; }',
        timeout=180000)
    out = box.locator('.d2l-pg-python .d2l-cell-out').inner_text()
    check('playground Python version runs with the slider values', 'error = 5.46000' in out, out)
    check('playground Python version draws a figure', box.locator('.d2l-pg-python .d2l-cell-out img').count() == 1)
    shot(box, 'playground.png')

    # ---------------- linear algebra: inline notes, answers, matrix playground
    page.goto(BASE + 'chapter_preliminaries/linear-algebra.html')
    page.wait_for_selector('.d2l-notes')
    page.wait_for_timeout(500)
    check('11 inline notes boxes', page.locator('.d2l-notes').count() == 11, page.locator('.d2l-notes').count())
    check('first notes box sits inside its section (Scalars)', page.evaluate(
        '() => !!document.querySelector(".d2l-notes").closest("#scalars")'))
    check('notes math typeset', page.locator('.d2l-notes mjx-container').count() > 3)
    page.wait_for_selector('#exercises .d2l-answer-btn')
    n_ans = page.locator('#exercises .d2l-answer-btn').count()
    check('12 Show answer buttons', n_ans == 12, n_ans)
    check('coding exercises embedded in Exercises list', page.locator('#exercises .d2l-ex').count() == 4,
          page.locator('#exercises .d2l-ex').count())
    check('answers hidden by default', not page.locator('#exercises .d2l-answer-body').first.is_visible())
    page.locator('#exercises .d2l-answer-btn').first.click()
    check('Show answer reveals it', page.locator('#exercises .d2l-answer-body').first.is_visible())
    mt = page.locator('.d2l-playground-box').first
    mt.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box canvas')
    check('matrix playground renders', mt.locator('canvas').count() == 1)

    # ---------------- progress
    page.wait_for_selector('.d2l-progress-btn')
    page.locator('.d2l-progress-btn').click()
    check('mark complete -> chip + sidebar check', page.locator('.d2l-progress-chip').is_visible()
          and page.locator('.globaltoc a.d2l-done').count() >= 1)
    page.reload()
    page.wait_for_selector('.d2l-progress-btn')
    check('completion persists', 'Completed' in page.locator('.d2l-progress-btn').inner_text())
    label = page.locator('.d2l-progress-label').first.inner_text()
    check('progress label counts sections', 'Progress: 1 /' in label, label)
    check('chapter count badge', '1/' in page.locator('.globaltoc .d2l-progress-count').first.inner_text())

    # ---------------- project pages
    page.goto(BASE + 'chapter_preliminaries/project-alert-triage.html')
    check('Hush page renders with its name', 'Project: Hush' in page.title(), page.title())
    href = page.locator('a', has_text='metrics.csv').first.get_attribute('href')
    data = urllib.request.urlopen(BASE + 'chapter_preliminaries/' + href).read().decode()
    check('metrics.csv downloads', data.startswith('cpu,latency_ms') and data.count('\n') == 2001, data[:60])
    check('project page has no coding environment', page.locator('.d2l-ex').count() == 0)
    page.goto(BASE + 'chapter_preliminaries/capstone-minigpt-1.html')
    check('Quill Part 1 page renders', 'Quill, Part 1' in page.title(), page.title())

    # ---------------- distractions removed
    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    check('no Colab/SageMaker notebook buttons', page.locator(
        'a[href*="colab.research.google.com"], a[href*="studiolab.sagemaker"], a[href*="console.aws.amazon.com"]').count() == 0)
    externals = page.evaluate('''() => Array.from(document.querySelectorAll("header a, .mdl-layout__header a"))
        .map(a => a.href).filter(h => /github|discuss|zh\\.d2l|courses/.test(h))''')
    check('no external links in the navbar', not externals, externals)
    page.goto(BASE + 'index.html')
    check('homepage has no news/forum promos', 'Announcements' not in page.content() and 'discuss.d2l.ai' not in page.content())
    check('no discussion embeds requested', not any('discuss.d2l.ai' in r for r in requests))
    check('no analytics requested', not any('google-analytics' in r or 'googletagmanager' in r for r in requests))
    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()
finish()
