"""Coding exercises: run, print, errors, grading, timeouts, stop, reset,
solution, persistence, independence, layout."""
import os
import time

from playwright.sync_api import sync_playwright

from common import BASE, REPO, check, finish, relevant_errors, shot, watch_errors

URL = BASE + 'chapter_preliminaries/linear-algebra.html'
VEC_SOLUTION = open(os.path.join(REPO, 'exercises/vector-add/solution.py')).read()
MATVEC_SOLUTION = open(os.path.join(REPO, 'exercises/matvec-numpy/solution.py')).read()
MATVEC_STARTER = open(os.path.join(REPO, 'exercises/matvec-numpy/starter.py')).read()
EDITORS = '.d2l-ex .d2l-ex-editor:not(.d2l-ex-readonly) .CodeMirror'


def ex(page, i):
    return page.locator('.d2l-ex').nth(i)


def set_code(page, i, code):
    page.evaluate('([s, i, c]) => document.querySelectorAll(s)[i].CodeMirror.setValue(c)', [EDITORS, i, code])


def get_code(page, i):
    return page.evaluate('([s, i]) => document.querySelectorAll(s)[i].CodeMirror.getValue()', [EDITORS, i])


def wait_idle(page, i, timeout=120000):
    page.wait_for_function(
        '(i) => { const e = document.querySelectorAll(".d2l-ex")[i]; const b = e.querySelector(".d2l-ex-primary");'
        ' return b.textContent.includes("Run") && !b.disabled && e.querySelector(".d2l-ex-panel:not([hidden])"); }',
        arg=i, timeout=timeout)


def click(page, i, label):
    ex(page, i).locator('.d2l-ex-btn', has_text=label).click()


def panel(page, i):
    return ex(page, i).locator('.d2l-ex-panel').inner_text()


with sync_playwright() as p:
    browser = p.chromium.launch()
    ctx = browser.new_context(viewport={'width': 1280, 'height': 900})
    page = watch_errors(ctx.new_page())
    page.goto(URL)
    page.wait_for_function('document.querySelectorAll(".d2l-ex .CodeMirror").length >= 2', timeout=60000)
    check('exercises render with editors', ex(page, 1).locator('.d2l-ex-title').inner_text().startswith('Matrix'))
    page.wait_for_function('document.querySelectorAll(".d2l-ex-desc mjx-container").length > 0', timeout=30000)
    check('math in exercise description is typeset', True)
    check('chapter math still renders', page.locator('mjx-container').count() > 100)
    check('framework code tabs still present', page.locator('.mdl-tabs__tab', has_text='pytorch').count() > 0)
    check('MXNet tab gone', page.locator('.mdl-tabs__tab', has_text='mxnet').count() == 0)

    t0 = time.time()
    click(page, 0, 'Run'); wait_idle(page, 0)
    print(f'     (first run incl. Python load: {time.time() - t0:.1f}s)')
    check('run executes python + print output', 'None' in panel(page, 0), panel(page, 0))
    set_code(page, 0, 'for i in range(3):\n    print("hello", i)')
    click(page, 0, 'Run'); wait_idle(page, 0)
    check('print() output shown', 'hello 2' in panel(page, 0), panel(page, 0))
    set_code(page, 0, 'def f(:\n  pass')
    click(page, 0, 'Run'); wait_idle(page, 0)
    check('syntax error displayed', 'SyntaxError' in panel(page, 0), panel(page, 0))
    set_code(page, 0, 'x = [1]\nprint(x[3])')
    click(page, 0, 'Run'); wait_idle(page, 0)
    check('runtime traceback displayed', 'IndexError' in panel(page, 0) and 'line 2' in panel(page, 0), panel(page, 0))
    set_code(page, 0, 'import torch')
    click(page, 0, 'Run'); wait_idle(page, 0)
    check('unsupported import explained', 'cannot run in the browser' in panel(page, 0), panel(page, 0))

    set_code(page, 0, 'def vector_add(a, b):\n    return [x - y for x, y in zip(a, b)]')
    click(page, 0, 'Submit'); wait_idle(page, 0)
    out = panel(page, 0)
    check('wrong solution gets failure feedback', 'of 8 tests passed' in out and 'Expected' in out and 'Got' in out, out[:300])
    set_code(page, 0, VEC_SOLUTION)
    click(page, 0, 'Submit'); wait_idle(page, 0)
    check('correct solution passes all tests', 'All 8 tests passed' in panel(page, 0), panel(page, 0)[:300])
    check('solved badge shown', ex(page, 0).locator('.d2l-ex-solved').is_visible())
    shot(ex(page, 0), 'exercise-passing.png')

    set_code(page, 1, MATVEC_SOLUTION)
    click(page, 1, 'Submit'); wait_idle(page, 1)
    check('numpy exercise: solution passes', 'All 6 tests passed' in panel(page, 1), panel(page, 1)[:300])
    set_code(page, 1, 'import numpy as np\ndef matvec(A, x):\n    return A @ x')
    click(page, 1, 'Submit'); wait_idle(page, 1)
    check('numpy exercise: banned @ detected', 'uses @' in panel(page, 1), panel(page, 1)[:400])

    set_code(page, 0, 'while True:\n    pass')
    click(page, 0, 'Run')
    page.wait_for_timeout(500)
    check('other exercise disabled while running', ex(page, 1).locator('.d2l-ex-btn', has_text='Submit').is_disabled())
    wait_idle(page, 0, 60000)
    check('infinite loop times out', 'Stopped after 5 seconds' in panel(page, 0), panel(page, 0))
    set_code(page, 0, 'print(1 + 1)')
    click(page, 0, 'Run'); wait_idle(page, 0)
    check('runner recovers after timeout', '2' in panel(page, 0), panel(page, 0))

    set_code(page, 1, 'while True:\n    pass')
    click(page, 1, 'Run')
    page.wait_for_function('document.querySelectorAll(".d2l-ex")[1].querySelector(".d2l-ex-status").textContent.includes("Running")', timeout=60000)
    page.wait_for_timeout(800)
    t0 = time.time()
    ex(page, 1).locator('.d2l-ex-btn', has_text='Stop').click()
    wait_idle(page, 1, 10000)
    check('stop button halts infinite loop', 'Stopped.' in panel(page, 1) and time.time() - t0 < 3, panel(page, 1))

    page.on('dialog', lambda d: d.accept())
    click(page, 1, 'Reset')
    check('reset restores starter', get_code(page, 1) == MATVEC_STARTER, get_code(page, 1)[:100])
    click(page, 0, 'Show solution')
    sol = ex(page, 0).locator('.d2l-ex-solution')
    check('show solution reveals code + explanation', sol.is_visible() and 'zip' in sol.inner_text())

    set_code(page, 0, '# my saved work\nprint("persist")')
    set_code(page, 1, '# second exercise\n')
    page.wait_for_timeout(800)
    page.reload()
    page.wait_for_function('document.querySelectorAll(".d2l-ex .CodeMirror").length >= 2', timeout=60000)
    check('code persists across reload', get_code(page, 0).startswith('# my saved work'), get_code(page, 0))
    check('exercises saved independently', get_code(page, 1) == '# second exercise\n', get_code(page, 1))
    check('solved badge persists', ex(page, 0).locator('.d2l-ex-solved').is_visible())
    check('restored-code notice shown', 'Restored' in ex(page, 0).locator('.d2l-ex-status').inner_text())

    mob = ctx.new_page()
    mob.set_viewport_size({'width': 390, 'height': 844})
    mob.goto(URL)
    mob.wait_for_selector('.d2l-ex .CodeMirror', timeout=60000)
    box = mob.locator('.d2l-ex').nth(1).bounding_box()
    check('fits on phone width', box['width'] <= 390, box)
    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()
finish()
