"""Chapter 3: training loops in the browser, playgrounds, notes, answers,
exercises, Tempo and Quill pages."""
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

from common import BASE as ROOT
from common import REPO, check, finish, relevant_errors, run_cell, shot, watch_errors

BASE = ROOT + 'chapter_linear-regression/'


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = watch_errors(browser.new_page(viewport={'width': 1280, 'height': 900}))

    # ------------- training from scratch, in the browser
    page.goto(BASE + 'linear-regression-scratch.html')
    page.wait_for_selector('.d2l-cell-bar')
    t0 = time.time()
    out, imgs = run_cell(page, 'trainer.fit(model, data)')
    print(f'     (scratch training incl. earlier cells and Python load: {time.time() - t0:.1f}s)')
    check('scratch: Trainer.fit runs and draws the loss plot', imgs == 1, out[:300])
    out, _ = run_cell(page, 'error in estimating w')
    errs = [float(v) for v in out.replace('[', ' ').replace(']', ' ').replace(',', ' ').split() if v.replace('.', '').replace('-', '').replace('e', '').isdigit()]
    check('scratch: learned parameters close to the truth (|error| < 0.5)',
          'error in estimating w' in out and errs and max(abs(e) for e in errs) < 0.5, out)

    # ------------- concise, with nn.LazyLinear + optim.SGD
    page.goto(BASE + 'linear-regression-concise.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, imgs = run_cell(page, 'trainer.fit(model, data)')
    check('concise: training runs with nn.LazyLinear/MSELoss/optim.SGD', imgs == 1, out[:300])
    out, _ = run_cell(page, 'error in estimating w')
    errs = [float(v) for v in out.replace('[', ' ').replace(']', ' ').replace(',', ' ').split() if v.replace('.', '').replace('-', '').replace('e', '').isdigit()]
    check('concise: learned parameters close to the truth (|error| < 0.1)',
          errs and max(abs(e) for e in errs) < 0.1, out)

    # ------------- weight decay page
    page.goto(BASE + 'weight-decay.html')
    page.wait_for_selector('.d2l-cell-bar')
    out0, img0 = run_cell(page, 'train_scratch(0)')
    out3, img3 = run_cell(page, 'train_scratch(3)')
    n0 = float(out0.split('L2 norm of w:')[1].split()[0])
    n3 = float(out3.split('L2 norm of w:')[1].split()[0])
    check('weight decay: lambda=3 shrinks the weights', img0 == 1 and img3 == 1 and n3 < n0 / 2, (n0, n3))

    # ------------- playgrounds
    page.goto(BASE + 'linear-regression-scratch.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    check('SGD playground: converges at lr 0.03', 'w = (' in box.locator('.d2l-pg-readout').inner_text(),
          box.locator('.d2l-pg-readout').inner_text())
    box.locator('input[type=range]').first.evaluate('(el) => { el.value = 2.3; el.dispatchEvent(new Event("input")); }')
    page.wait_for_timeout(300)
    check('SGD playground: diverges at lr 2.3', 'too large' in box.locator('.d2l-pg-readout').inner_text(),
          box.locator('.d2l-pg-readout').inner_text())
    box.locator('input[type=range]').first.evaluate('(el) => { el.value = 0.5; el.dispatchEvent(new Event("input")); }')
    t0 = time.time()
    for k in range(30):
        box.locator('input[type=range]').nth(1).evaluate('(el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }', 8 + k)
        page.evaluate('() => new Promise(r => requestAnimationFrame(() => r()))')
    fps = 30 / (time.time() - t0)
    check(f'SGD playground stays interactive (~{fps:.0f} redraws/s)', fps > 10, fps)
    shot(box, '19-sgd-playground.png')

    page.goto(BASE + 'generalization.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    deg = box.locator('input[type=range]').first
    deg.evaluate('(el) => { el.value = 1; el.dispatchEvent(new Event("input")); }')
    page.wait_for_timeout(150)
    under = box.locator('.d2l-pg-readout').inner_text()
    deg.evaluate('(el) => { el.value = 12; el.dispatchEvent(new Event("input")); }')
    page.wait_for_timeout(150)
    over = box.locator('.d2l-pg-readout').inner_text()
    check('polynomial playground: degree 1 underfits, degree 12 overfits',
          'underfitting' in under and 'overfitting' in over, (under, over))
    shot(box, '20-poly-playground.png')

    page.goto(BASE + 'weight-decay.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')

    def val_err():
        t = box.locator('.d2l-pg-readout').inner_text()
        return float(t.split('validation error')[1].split()[0])
    s = box.locator('input[type=range]').first
    v_none = val_err()
    s.evaluate('(el) => { el.value = -3; el.dispatchEvent(new Event("input")); }')
    page.wait_for_timeout(150)
    v_mid = val_err()
    s.evaluate('(el) => { el.value = 2; el.dispatchEvent(new Event("input")); }')
    page.wait_for_timeout(150)
    v_big = val_err()
    check('weight decay playground: U-shaped validation error', v_mid < v_none and v_mid < v_big, (v_none, v_mid, v_big))

    page.goto(BASE + 'linear-regression-concise.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    check('loss playground renders', 'median' in box.locator('.d2l-pg-readout').inner_text())

    # ------------- notes, answers, exercises
    page.goto(BASE + 'linear-regression.html')
    page.wait_for_selector('.d2l-notes')
    check('3.1: inline notes (7 sections)', page.locator('.d2l-notes').count() == 7, page.locator('.d2l-notes').count())
    page.wait_for_selector('#exercises .d2l-answer-btn')
    check('3.1: 8 hidden answers', page.locator('#exercises .d2l-answer-btn').count() == 8)
    check('3.1: coding exercises under questions', page.locator('#exercises .d2l-ex').count() == 2)
    page.goto(BASE + 'linear-regression-scratch.html')
    page.wait_for_selector('#exercises .d2l-ex .CodeMirror', timeout=60000)
    ex = page.locator('#exercises .d2l-ex', has_text='Hand-Derived').first
    sol = open(os.path.join(REPO, 'exercises/linreg-sgd-numpy/solution.py')).read()
    idx = page.evaluate('''() => Array.from(document.querySelectorAll(".d2l-ex")).findIndex(e => e.textContent.includes("Hand-Derived"))''')
    page.evaluate('([i, c]) => document.querySelectorAll(".d2l-ex")[i].querySelector(".CodeMirror").CodeMirror.setValue(c)', [idx, sol])
    ex.locator('.d2l-ex-btn', has_text='Submit').click()
    page.wait_for_function('(i) => /tests passed/.test(document.querySelectorAll(".d2l-ex")[i].querySelector(".d2l-ex-status").textContent)',
                           arg=idx, timeout=120000)
    check('3.4: SGD-by-hand exercise passes with the reference solution',
          'All tests passed' in ex.locator('.d2l-ex-status').inner_text(), ex.locator('.d2l-ex-status').inner_text())

    # ------------- project pages
    page.goto(BASE + 'project-tempo.html')
    check('Tempo page renders with its name', page.title().startswith('3.8. Project: Tempo'), page.title())
    href = page.locator('a', has_text='deliveries.csv').first.get_attribute('href')
    data = urllib.request.urlopen(BASE + href).read().decode()
    check('deliveries.csv downloads', data.startswith('order_day,weekday') and data.count('\n') == 4001)
    page.goto(BASE + 'capstone-quill-2.html')
    check('Quill Part 2 page renders', 'Quill, Part 2' in page.title(), page.title())
    progress = page.locator('section', has_text='How Far Quill Has Come').last.inner_text()
    check('Quill Part 2 shows progress (perplexity per version + samples)',
          '11.4' in progress and 'Part 1' in progress, progress[:300])
    page.goto(BASE + 'project-tempo.html')
    for name in ('Scout', 'Drift', 'Watt'):
        check(f'Tempo lists optional side project {name}', page.locator('h3', has_text=name).count() == 1)
    page.goto(ROOT + 'chapter_preliminaries/project-alert-triage.html')
    check('2.8 renamed to Hush', 'Project: Hush' in page.title(), page.title())
    page.goto(ROOT + 'chapter_preliminaries/capstone-minigpt-1.html')
    check('2.9 renamed to Quill', 'Quill, Part 1' in page.title() and 'MiniGPT' not in page.content(), page.title())

    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()

finish()
