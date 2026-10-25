"""Shared helpers for the browser tests (Playwright) of the built site.

The site must be served at BASE (run_all.sh does this). Each test file
prints PASS/FAIL lines and exits non-zero on failure.
"""
import os
import sys

BASE = os.environ.get('SITE_URL', 'http://127.0.0.1:8765/')
SHOTS = os.environ.get('SHOTS_DIR', '')
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Console errors that come from the original theme, not from our code.
KNOWN_ERRORS = ('fonts.gstatic.com', 'Failed to load resource', "Unexpected token '<'")

results = []


def check(name, cond, detail=''):
    results.append((name, bool(cond)))
    print(('PASS ' if cond else 'FAIL ') + name + (f'  [{str(detail)[:300]}]' if detail and not cond else ''),
          flush=True)


def shot(page_or_locator, name):
    if SHOTS:
        os.makedirs(SHOTS, exist_ok=True)
        page_or_locator.screenshot(path=os.path.join(SHOTS, name))


def watch_errors(page):
    page.errors = []
    page.on('console', lambda m: page.errors.append(m.text) if m.type == 'error' else None)
    page.on('pageerror', lambda e: page.errors.append(str(e)))
    return page


def relevant_errors(*pages):
    return [e for p in pages for e in p.errors if not any(k in e for k in KNOWN_ERRORS)]


def cell_index(page, needle):
    return page.evaluate('''(needle) => Array.from(document.querySelectorAll(".d2l-cell-bar"))
        .findIndex(b => b.previousElementSibling.textContent.includes(needle))''', needle)


def wait_cell_done(page, i, timeout=180000):
    page.wait_for_function(
        '(i) => { const b = document.querySelectorAll(".d2l-cell-bar")[i]; const o = b.nextElementSibling;'
        ' if (/Stopped:/.test(b.querySelector(".d2l-cell-status").textContent)) return true;'
        ' return !o.hidden && o.childElementCount > 0 &&'
        ' !/Running|Loading/.test(b.querySelector(".d2l-cell-status").textContent); }',
        arg=i, timeout=timeout)


def cell_output(page, i):
    return page.evaluate('(i) => document.querySelectorAll(".d2l-cell-bar")[i].nextElementSibling.innerText', i)


def cell_images(page, i):
    return page.evaluate(
        '(i) => document.querySelectorAll(".d2l-cell-bar")[i].nextElementSibling.querySelectorAll("img").length', i)


def run_cell(page, needle, timeout=180000):
    """Click Run on the first book cell containing `needle`; return (text, n_images)."""
    i = cell_index(page, needle)
    assert i >= 0, f'no runnable cell containing {needle!r}'
    page.locator('.d2l-cell-bar').nth(i).locator('.d2l-cell-run').click()
    wait_cell_done(page, i, timeout)
    return cell_output(page, i), cell_images(page, i)


def finish():
    failed = [n for n, ok in results if not ok]
    print(f'\n{len(results) - len(failed)}/{len(results)} checks passed', flush=True)
    sys.exit(1 if failed else 0)
