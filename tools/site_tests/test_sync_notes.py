"""Progress sync (sync link + a mocked GitHub gist API) and the My Notes page."""
import json
import re

from playwright.sync_api import sync_playwright

from common import BASE, check, finish, relevant_errors, shot, watch_errors

TOKEN = 'github_pat_TESTTOKEN'


# ---- a fake GitHub gist API shared by all "devices"
GISTS = {}
API_CALLS = []


def fake_github(route):
    req = route.request
    url = req.url.split('api.github.com')[1]
    API_CALLS.append((req.method, url.split('?')[0]))
    if req.headers.get('authorization') != f'Bearer {TOKEN}':
        return route.fulfill(status=401, body='{}')
    if req.method == 'GET' and url.startswith('/gists?'):
        return route.fulfill(status=200, content_type='application/json',
                             body=json.dumps([{'id': i, 'files': {k: {} for k in g}} for i, g in GISTS.items()]))
    if req.method == 'POST' and url == '/gists':
        body = json.loads(req.post_data)
        gid = f'gist{len(GISTS) + 1}'
        GISTS[gid] = {k: v['content'] for k, v in body['files'].items()}
        return route.fulfill(status=201, content_type='application/json', body=json.dumps({'id': gid}))
    m = re.match(r'/gists/(\w+)$', url)
    if m and m.group(1) in GISTS:
        gid = m.group(1)
        if req.method == 'PATCH':
            for k, v in json.loads(req.post_data)['files'].items():
                GISTS[gid][k] = v['content']
        return route.fulfill(status=200, content_type='application/json', body=json.dumps(
            {'id': gid, 'files': {k: {'content': c, 'truncated': False} for k, c in GISTS[gid].items()}}))
    return route.fulfill(status=404, body='{}')


def device(browser, name):
    ctx = browser.new_context(viewport={'width': 1280, 'height': 900})
    ctx.grant_permissions(['clipboard-read', 'clipboard-write'], origin=BASE.rstrip('/'))
    ctx.route('https://api.github.com/**', fake_github)
    return ctx, watch_errors(ctx.new_page())


def progress(page):
    return page.evaluate('() => window.D2LProgress.load()')


def set_progress(page, data):
    page.evaluate('(d) => localStorage.setItem("d2l-progress:v1", JSON.stringify(d))', data)


with sync_playwright() as p:
    browser = p.chromium.launch()

    # ---------------- My Notes page
    ctx, page = device(browser, 'A')
    page.goto(BASE + 'chapter_my-notes/index.html')
    page.wait_for_selector('.d2l-review-empty, .d2l-review-page', timeout=30000)
    check('My Notes: empty state before completing anything', page.locator('.d2l-review-empty').count() == 1)
    check('My Notes page has no "Mark complete" button', page.locator('.d2l-progress-btn').count() == 0)
    set_progress(page, {'chapter_preliminaries/linear-algebra.html': '2026-10-18',
                        'chapter_preliminaries/calculus.html': '2026-10-18',
                        'chapter_preliminaries/lookup-api.html': '2026-10-18'})
    page.reload()
    page.wait_for_selector('.d2l-review-page')
    titles = page.locator('.d2l-review-page > h2').all_inner_texts()
    check('My Notes: shows completed sections with notes, in book order',
          titles == ['2.3. Linear Algebra', '2.4. Calculus'], titles)
    check('My Notes: subsection headings + notes', page.locator('.d2l-review-page h3').count() == 11 + 4
          and page.locator('.d2l-review-notes li').count() > 15)
    page.wait_for_function('() => document.querySelectorAll(".d2l-review-notes mjx-container").length > 5', timeout=30000)
    check('My Notes: math typeset', True)
    href = page.locator('.d2l-review-page h3 a').nth(8).get_attribute('href')
    check('My Notes: section links point at the right anchor', href.endswith('linear-algebra.html#matrixvector-products'), href)
    page.locator('#d2l-review-all').check()
    check('My Notes: "Show all" shows every section with notes', page.locator('.d2l-review-page').count() == 21,
          page.locator('.d2l-review-page').count())
    page.locator('#d2l-review-all').uncheck()
    page.locator('.d2l-btn', has_text='Copy as text').click()
    text = page.evaluate('() => navigator.clipboard.readText()')
    check('My Notes: copy as text', text.startswith('## 2.3. Linear Algebra') and '### Matrix–Vector Products' in text, text[:120])
    shot(page, '16-my-notes.png')

    # ---------------- sync link: A -> B
    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    page.wait_for_selector('.d2l-progress-sync')
    page.locator('.d2l-progress-sync').first.click()
    page.wait_for_selector('.d2l-sync-dialog[open]')
    shot(page, '17-sync-dialog.png')
    page.locator('.d2l-sync-dialog .d2l-btn', has_text='Copy sync link').click()
    page.wait_for_function('() => document.querySelector(".d2l-sync-dialog .d2l-sync-status").textContent.includes("Copied")')
    link = page.evaluate('() => navigator.clipboard.readText()')
    check('sync link is short and carries data in the #fragment', '#d2l-sync=g' in link and len(link) < 600, (len(link), link[:80]))
    page.locator('.d2l-sync-dialog .d2l-link-btn', has_text='Close').click()

    ctxB, pageB = device(browser, 'B')
    pageB.goto(BASE + 'index.html')
    set_progress(pageB, {'chapter_preface/index.html': '2026-10-17'})
    pageB.goto(link)
    pageB.wait_for_selector('.d2l-sync-toast', timeout=15000)
    check('sync link import: toast shown', '3 new completed' in pageB.locator('.d2l-sync-toast').inner_text(),
          pageB.locator('.d2l-sync-toast').inner_text())
    pB = progress(pageB)
    check('sync link import: merged (union of both devices)', len(pB) == 4 and 'chapter_preface/index.html' in pB, pB)
    check('sync link import: fragment removed from the URL', '#' not in pageB.url, pageB.url)
    check('sidebar updates after import', pageB.locator('.d2l-progress-label').first.inner_text().startswith('Progress: 4 /'))

    # ---------------- GitHub sync with the fake API
    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    page.evaluate('() => localStorage.setItem("d2l-exercise:vector-add:solved", "1")')
    page.locator('.d2l-progress-sync').first.click()
    page.locator('.d2l-sync-input').fill('wrong-token')
    page.locator('.d2l-sync-dialog .d2l-btn', has_text='Connect').click()
    page.wait_for_function('() => document.querySelector(".d2l-sync-dialog").textContent.includes("rejected the token")')
    check('GitHub: a bad token shows a clear error', True)
    page.locator('.d2l-sync-input').fill(TOKEN)
    page.locator('.d2l-sync-dialog .d2l-btn', has_text='Connect').click()
    page.wait_for_function('() => document.querySelector(".d2l-sync-dialog").textContent.includes("Connected.")', timeout=15000)
    check('GitHub: connect creates one secret gist', len(GISTS) == 1 and ('POST', '/gists') in API_CALLS, GISTS)
    gist = json.loads(next(iter(GISTS.values()))['d2l-progress.json'])
    check('GitHub: gist holds progress + solved exercises', len(gist['progress']) == 3 and gist['solved'] == {'vector-add': 1}, gist)
    shot(page, '18-sync-connected.png')

    # Device B connects with the same token: finds the gist, merges both ways.
    pageB.goto(BASE + 'chapter_preliminaries/ndarray.html')
    set_progress(pageB, {**progress(pageB), 'chapter_introduction/index.html': '2026-10-16'})
    pageB.locator('.d2l-progress-sync').first.click()
    pageB.locator('.d2l-sync-input').fill(TOKEN)
    pageB.locator('.d2l-sync-dialog .d2l-btn', has_text='Connect').click()
    pageB.wait_for_function('() => document.querySelector(".d2l-sync-dialog").textContent.includes("Connected.")', timeout=15000)
    check('GitHub: second device reuses the same gist', len(GISTS) == 1)
    check('GitHub: second device received solved exercise',
          pageB.evaluate('() => localStorage.getItem("d2l-exercise:vector-add:solved")') == '1')
    gist = json.loads(next(iter(GISTS.values()))['d2l-progress.json'])
    check('GitHub: gist now has the union of both devices', len(gist['progress']) == 5, gist['progress'])

    # Device A picks up B's extra section on its next page load.
    page.reload()
    page.wait_for_function('() => Object.keys(window.D2LProgress.load()).length === 5', timeout=15000)
    check('GitHub: first device pulls the other device\'s progress on load', True)
    check('sidebar shows synced state', 'Synced' in page.locator('.d2l-progress-tools').first.inner_text(),
          page.locator('.d2l-progress-tools').first.inner_text())

    # Marking a section complete pushes automatically.
    page.goto(BASE + 'chapter_preliminaries/autograd.html')
    page.wait_for_selector('.d2l-progress-btn')
    page.locator('.d2l-progress-btn').click()
    page.wait_for_function('() => document.querySelector(".d2l-progress-tools").textContent.includes("Synced")', timeout=15000)
    page.wait_for_timeout(3000)
    gist = json.loads(next(iter(GISTS.values()))['d2l-progress.json'])
    check('GitHub: completing a section pushes automatically', 'chapter_preliminaries/autograd.html' in gist['progress'], gist['progress'])

    # Disconnect removes the token from this browser.
    page.locator('.d2l-progress-sync').first.click()
    page.locator('.d2l-sync-dialog .d2l-btn', has_text='Disconnect').click()
    check('GitHub: disconnect forgets the token', page.evaluate('() => localStorage.getItem("d2l-sync:github")') is None)

    errs = relevant_errors(page, pageB)
    check('no JavaScript errors', not errs, errs)
    browser.close()
finish()
