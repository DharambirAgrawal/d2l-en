"""Chapter 4: Fashion-MNIST and softmax regression trained in the browser,
playgrounds, notes, answers, exercises, Shield and Quill Part 3."""
import os
import time
import urllib.request

from playwright.sync_api import sync_playwright

from common import BASE as ROOT
from common import REPO, check, finish, relevant_errors, run_cell, shot, watch_errors

BASE = ROOT + 'chapter_linear-classification/'


def set_slider(box, name_index, value):
    box.locator('input[type=range]').nth(name_index).evaluate(
        '(el, v) => { el.value = v; el.dispatchEvent(new Event("input")); }', value)


def readout(page, box):
    page.wait_for_timeout(200)
    return box.locator('.d2l-pg-readout').inner_text()


with sync_playwright() as p:
    browser = p.chromium.launch()
    page = watch_errors(browser.new_page(viewport={'width': 1280, 'height': 900}))

    # ------------- the dataset, downloaded into the browser
    page.goto(BASE + 'image-classification-dataset.html')
    page.wait_for_selector('.d2l-cell-bar')
    t0 = time.time()
    out, _ = run_cell(page, 'len(data.train), len(data.val)', timeout=300000)
    print(f'     (dataset download + load incl. Python: {time.time() - t0:.1f}s)')
    check('Fashion-MNIST loads in the browser: 60000 / 10000 images', '(60000, 10000)' in out, out)
    out, _ = run_cell(page, 'data.train[0][0].shape')
    check('images resized to 1x32x32', 'torch.Size([1, 32, 32])' in out, out)
    out, _ = run_cell(page, 'print(X.shape, X.dtype, y.shape, y.dtype)')
    check('a minibatch has the right shapes and dtypes',
          'torch.Size([64, 1, 32, 32]) torch.float32 torch.Size([64]) torch.int64' in out, out)
    out, imgs = run_cell(page, 'data.visualize(batch)')
    check('visualize draws the images with labels', imgs == 1, out)

    # ------------- training from scratch and concisely
    page.goto(BASE + 'softmax-regression-scratch.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, _ = run_cell(page, 'X_prob, X_prob.sum(1)')
    sums = [float(v) for v in out.rsplit('tensor(', 1)[1].strip('[]() \n').split(',') if v.strip()]
    check('scratch softmax rows sum to 1', len(sums) == 2 and all(abs(v - 1) < 1e-5 for v in sums), out)
    t0 = time.time()
    out, imgs = run_cell(page, 'trainer.fit(model, data)', timeout=300000)
    print(f'     (10 epochs of softmax regression on Fashion-MNIST: {time.time() - t0:.1f}s)')
    check('scratch: Trainer.fit trains on Fashion-MNIST and draws curves', imgs == 1, out[:300])
    out, _ = run_cell(page, 'preds.shape')
    check('scratch: predictions for a validation batch', 'torch.Size([256])' in out, out)
    out, imgs = run_cell(page, 'data.visualize([X, y], labels=labels)')
    check('scratch: misclassified images shown', imgs == 1, out)
    page.goto(BASE + 'softmax-regression-concise.html')
    page.wait_for_selector('.d2l-cell-bar')
    out, imgs = run_cell(page, 'trainer.fit(model, data)', timeout=300000)
    check('concise: nn.Flatten + LazyLinear + F.cross_entropy train', imgs == 1, out[:300])

    # ------------- playgrounds
    page.goto(BASE + 'softmax-regression.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    check('softmax playground: probabilities sum to 1', 'sum to 1.000' in readout(page, box), readout(page, box))
    set_slider(box, 3, 0.05)
    low_t = readout(page, box)
    set_slider(box, 3, 5)
    high_t = readout(page, box)
    ent = lambda t: float(t.split('entropy ')[1].split()[0])
    check('softmax playground: low temperature = sharp, high = flat', ent(low_t) < 0.05 and ent(high_t) > 1.0,
          (low_t, high_t))
    shot(box, 'softmax-temperature.png')

    page.goto(BASE + 'softmax-regression-scratch.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    acc = lambda t: float(t.split('accuracy ')[1].split('%')[0])
    set_slider(box, 0, 0)
    a0 = acc(readout(page, box))
    set_slider(box, 0, 200)
    a200 = acc(readout(page, box))
    check('regions playground: training raises accuracy', a0 < 50 and a200 > 90, (a0, a200))
    shot(box, 'softmax-regions.png')

    page.goto(BASE + 'generalization-classification.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    band = lambda t: float(t.split('guarantees ±')[1].split()[0])
    set_slider(box, 0, 2)
    b2 = band(readout(page, box))
    set_slider(box, 0, 4)
    b4 = band(readout(page, box))
    check('test-set playground: 100x data = 10x tighter band', abs(b2 / b4 - 10) < 0.5, (b2, b4))

    page.goto(BASE + 'environment-and-distribution-shift.html')
    box = page.locator('.d2l-playground-box').first
    box.scroll_into_view_if_needed()
    page.wait_for_selector('.d2l-playground-box .d2l-pg-readout')
    ess = lambda t: int(t.split('effective sample size after reweighting: ')[1].split()[0])
    set_slider(box, 0, 0)
    e0 = ess(readout(page, box))
    set_slider(box, 0, 2.5)
    e25 = ess(readout(page, box))
    check('shift playground: no shift keeps all samples, big shift collapses them', e0 == 2000 and e25 < 200, (e0, e25))

    # ------------- notes, recall, answers, exercises, steps
    page.goto(BASE + 'softmax-regression.html')
    page.wait_for_selector('.d2l-notes')
    page.wait_for_timeout(500)
    check('4.1: 9 inline notes', page.locator('.d2l-notes').count() == 9, page.locator('.d2l-notes').count())
    page.wait_for_selector('#exercises .d2l-answer-btn')
    check('4.1: 7 hidden answers', page.locator('#exercises .d2l-answer-btn').count() == 7)
    page.wait_for_selector('.d2l-recall')
    check('4.1: quick recall box', page.locator('.d2l-recall li').count() == 3)
    page.wait_for_selector('.d2l-step')
    check('4.1: Shield step 1', 'Shield · Step 1 of 7' in page.locator('.d2l-step .d2l-card-kicker').text_content())
    page.goto(BASE + 'classification.html')
    page.wait_for_selector('.d2l-notes')
    page.wait_for_timeout(500)
    check('4.3: notes under "The Classifier Class" (backticks in heading)', page.evaluate(
        '() => !!document.querySelector(".d2l-notes").closest("section, div.section").querySelector("h2 code")'))

    page.goto(BASE + 'softmax-regression-scratch.html')
    page.wait_for_selector('#exercises .d2l-ex .CodeMirror', timeout=60000)
    idx = page.evaluate('''() => Array.from(document.querySelectorAll(".d2l-ex")).findIndex(e => e.textContent.includes("Never Breaks"))''')
    sol = open(os.path.join(REPO, 'exercises/stable-softmax/solution.py')).read()
    page.evaluate('([i, c]) => document.querySelectorAll(".d2l-ex")[i].querySelector(".CodeMirror").CodeMirror.setValue(c)', [idx, sol])
    ex = page.locator('.d2l-ex').nth(idx)
    ex.locator('.d2l-ex-btn', has_text='Submit').click()
    page.wait_for_function('(i) => /tests passed/.test(document.querySelectorAll(".d2l-ex")[i].querySelector(".d2l-ex-status").textContent)',
                           arg=idx, timeout=120000)
    check('4.4: stable-softmax exercise passes with the reference solution',
          'All tests passed' in ex.locator('.d2l-ex-status').inner_text(), ex.locator('.d2l-ex-status').inner_text())

    # ------------- project pages
    page.goto(BASE + 'project-shield.html')
    check('Shield page renders with its name', 'Project: Shield' in page.title(), page.title())
    page.wait_for_selector('.d2l-project-steps .d2l-step')
    check('Shield page lists 7 steps', page.locator('.d2l-project-steps .d2l-step').count() == 7)
    for name in ('Router', 'Gauge', 'Sentinel'):
        check(f'Shield lists optional side project {name}', page.locator('h3', has_text=name).count() == 1)
    href = page.locator('a', has_text='agent_inbox.csv').first.get_attribute('href')
    data = urllib.request.urlopen(BASE + href).read().decode()
    check('agent_inbox.csv downloads', data.startswith('msg_id,received,source') and data.count('\n') >= 8001)
    page.goto(BASE + 'capstone-quill-3.html')
    check('Quill Part 3 page renders', 'Quill, Part 3' in page.title(), page.title())
    progress = page.locator('section', has_text='How Far Quill Has Come').last.inner_text()
    check('Quill Part 3 shows its progress row', '8.4' in progress and 'softmax regression' in progress, progress[:200])

    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()
finish()
