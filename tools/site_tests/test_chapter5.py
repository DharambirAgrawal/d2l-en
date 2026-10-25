"""Chapter 5: MLPs, dropout and house prices trained in the browser,
playgrounds, notes, answers, recall, exercises, Taste and Quill Part 4."""
import os
import urllib.request

from playwright.sync_api import sync_playwright

from common import BASE as ROOT
from common import REPO, check, finish, relevant_errors, run_cell, shot, watch_errors

BASE = ROOT + 'chapter_multilayer-perceptrons/'


def set_slider(box, i, value):
    box.locator('input[type=range]').nth(i).evaluate(
        '(el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }', value)


def readout(page, box):
    page.wait_for_timeout(300)
    return box.locator('.d2l-pg-readout').inner_text()


def playground(page, url):
    page.goto(BASE + url)
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    return box


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = watch_errors(browser.new_page(viewport={'width': 1280, 'height': 900}))

    # ------------- training in the browser
    page.goto(BASE + 'mlp-implementation.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, imgs = run_cell(page, 'model = MLPScratch(', timeout=300000)
    check('5.2: MLP from scratch trains on Fashion-MNIST (torch.max ReLU) and draws curves', imgs == 1, out[:300])
    page.goto(BASE + 'dropout.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, _ = run_cell(page, "print('dropout_p = 0:'")
    check('5.6: dropout from scratch runs for p = 0, 0.5, 1', 'dropout_p = 0.5' in out and 'dropout_p = 1' in out, out[:300])
    out, imgs = run_cell(page, 'model = DropoutMLP(**hparams)', timeout=300000)
    check('5.6: nn.Dropout model trains', imgs == 1, out[:300])
    page.goto(BASE + 'kaggle-house-price.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, _ = run_cell(page, 'data = KaggleHouse(batch_size=64)', timeout=300000)
    check('5.7: house-price data loads in the browser', '(1460, 81)' in out and '(1459, 80)' in out, out)
    out, _ = run_cell(page, 'models = k_fold(trainer, data, k=5, lr=0.01)', timeout=300000)
    check('5.7: 5-fold cross-validation runs', 'average validation log mse' in out, out[-300:])

    # ------------- playgrounds
    box = playground(page, 'mlp.html')
    set_slider(box, 0, 0)
    set_slider(box, 1, 0)
    lin = readout(page, box)
    set_slider(box, 1, 8)
    mlp = readout(page, box)
    err = lambda t: float(t.split('training error ')[1].split('%')[0])
    check('mlp playground: no hidden layer fails on XOR, 8 ReLU units solve it',
          'straight boundary' in lin and err(lin) > 35 and err(mlp) < 15, (lin, mlp))
    shot(box, 'mlp-regions.png')
    box = playground(page, 'numerical-stability-and-init.html')
    set_slider(box, 1, 0.15)
    he = readout(page, box)
    set_slider(box, 1, -0.5)
    small = readout(page, box)
    check('deep-signal playground: He scale is healthy for ReLU, small weights vanish',
          'healthy' in he and 'VANISH' in small, (he, small))
    box = playground(page, 'generalization-deep.html')
    set_slider(box, 0, 0.2)
    t = readout(page, box)
    best = float(t.split('best validation error ')[1].split('%')[0])
    final = float(t.split('validation ')[-1].split('%')[0])
    check('early-stopping playground: noisy labels make late validation error worse than the best', final > best + 3, t)

    # ------------- notes, answers, recall, steps, exercises
    page.goto(BASE + 'mlp.html')
    page.wait_for_selector('.d2l-notes')
    page.wait_for_timeout(500)
    check('5.1: 8 inline notes', page.locator('.d2l-notes').count() == 8, page.locator('.d2l-notes').count())
    page.wait_for_selector('#exercises .d2l-answer-btn')
    check('5.1: 7 hidden answers', page.locator('#exercises .d2l-answer-btn').count() == 7)
    page.wait_for_selector('.d2l-recall')
    check('5.1: quick recall box', page.locator('.d2l-recall li').count() == 3)
    page.wait_for_selector('.d2l-step')
    check('5.1: Taste step 1', 'Taste · Step 1 of 7' in page.locator('.d2l-step .d2l-card-kicker').text_content())
    page.goto(BASE + 'dropout.html')
    page.wait_for_selector('.d2l-notes')
    page.wait_for_timeout(500)
    check('5.6: notes under "[**Concise Implementation**]" (slide marks in heading)', page.evaluate(
        '''() => Array.from(document.querySelectorAll(".d2l-notes")).some(n => /Concise Implementation/.test(
               n.closest("section, div.section").querySelector("h2, h3").textContent))'''))

    page.goto(BASE + 'backprop.html')
    page.wait_for_selector('#exercises .d2l-ex .CodeMirror', timeout=60000)
    idx = page.evaluate('''() => Array.from(document.querySelectorAll(".d2l-ex")).findIndex(e => e.textContent.includes("Backpropagation by Hand"))''')
    sol = open(os.path.join(REPO, 'exercises/mlp-backprop/solution.py')).read()
    page.evaluate('([i, c]) => document.querySelectorAll(".d2l-ex")[i].querySelector(".CodeMirror").CodeMirror.setValue(c)', [idx, sol])
    ex = page.locator('.d2l-ex').nth(idx)
    ex.locator('.d2l-ex-btn', has_text='Submit').click()
    page.wait_for_function('(i) => /tests passed/.test(document.querySelectorAll(".d2l-ex")[i].querySelector(".d2l-ex-status").textContent)',
                           arg=idx, timeout=120000)
    check('5.3: backprop-by-hand exercise passes with the reference solution',
          'All tests passed' in ex.locator('.d2l-ex-status').inner_text(), ex.locator('.d2l-ex-status').inner_text())

    # ------------- project pages
    page.goto(BASE + 'project-taste.html')
    check('Taste page renders with its name', 'Project: Taste' in page.title(), page.title())
    page.wait_for_selector('.d2l-project-steps .d2l-step')
    check('Taste page lists 7 steps', page.locator('.d2l-project-steps .d2l-step').count() == 7)
    for name in ('Grok', 'Reel', 'Probe'):
        check(f'Taste lists optional side project {name}', page.locator('h3', has_text=name).count() == 1)
    href = page.locator('a', has_text='preferences.csv').first.get_attribute('href')
    data = urllib.request.urlopen(BASE + href).read().decode()
    check('preferences.csv downloads', data.startswith('pair_id,week,category') and data.count('\n') == 6771)
    page.goto(BASE + 'capstone-quill-4.html')
    check('Quill Part 4 page renders', 'Quill, Part 4' in page.title(), page.title())
    progress = page.locator('section', has_text='How Far Quill Has Come').last.inner_text()
    check('Quill Part 4 shows its progress row', '4.5' in progress and 'MLP' in progress, progress[:200])

    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()
finish()
