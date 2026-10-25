"""Chapter projects built step by step (step box on section pages, all steps
on the project page) and Quick recall boxes at the top of sections."""
from playwright.sync_api import sync_playwright

from common import BASE, check, finish, relevant_errors, shot, watch_errors

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = watch_errors(browser.new_page(viewport={'width': 1280, 'height': 900}))

    # ---------------- step box on a section page
    page.goto(BASE + 'chapter_linear-regression/linear-regression-scratch.html')
    page.wait_for_selector('.d2l-step', timeout=30000)
    step = page.locator('.d2l-step')
    kicker = step.locator('.d2l-card-kicker').text_content()
    check('section page shows its Pace step', step.count() == 1 and kicker.startswith('Pace · Step')
          and 'of 7' in kicker, kicker)
    check('step box sits right before the Exercises', page.evaluate(
        '() => document.querySelector(".d2l-step").nextElementSibling.id === "exercises"'))
    labels = step.locator('.d2l-step-label').all_text_contents()
    check('step has Build and Revisit parts', labels == ['Build', 'Revisit'], labels)
    href = step.locator('.d2l-step-foot a').get_attribute('href')
    check('step links to the project page', href.endswith('chapter_linear-regression/project-pace.html'), href)
    shot(step, 'step.png')

    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    page.wait_for_selector('.d2l-step', timeout=30000)
    kicker = page.locator('.d2l-step .d2l-card-kicker').text_content()
    check('Chapter 2 section shows Podium step 1', kicker.startswith('Podium · Step 1 of 6'), kicker)

    # ---------------- project page lists every step, in order
    for name, path, total in (('Pace', 'chapter_linear-regression/project-pace.html', 7),
                              ('Podium', 'chapter_preliminaries/project-podium.html', 6),
                              ('Shield', 'chapter_linear-classification/project-shield.html', 7)):
        page.goto(BASE + path)
        page.wait_for_selector('.d2l-project-steps .d2l-step', timeout=30000)
        kickers = page.locator('.d2l-project-steps .d2l-card-kicker').all_text_contents()
        check(f'{name} page lists all {total} steps in order',
              kickers == [f'{name} · Step {i} of {total}' for i in range(1, total + 1)], kickers)
        where = page.locator('.d2l-project-steps .d2l-step-where').first.inner_text()
        check(f'{name} steps say which section they follow', where.startswith('Do this after:')
              and len(where) > len('Do this after: ') + 3, where)
        check(f'{name} last step revisits the whole chapter', page.locator(
            '.d2l-project-steps .d2l-step').last.locator('.d2l-step-label', has_text='Revisit').count() == 1)
    shot(page.locator('.d2l-project-steps'), 'project-steps.png')

    # ---------------- Quick recall
    page.goto(BASE + 'chapter_linear-regression/weight-decay.html')
    page.wait_for_selector('.d2l-recall', timeout=30000)
    box = page.locator('.d2l-recall')
    check('Quick recall box shown', box.count() == 1 and 'Quick recall' in box.text_content())
    check('recall sits before the first subsection', page.evaluate(
        '''() => { const r = document.querySelector(".d2l-recall"); const n = r.nextElementSibling;
                   return n && (n.tagName === "SECTION" || n.classList.contains("section")); }'''))
    n = box.locator('li').count()
    check('2-3 recall questions', 2 <= n <= 3, n)
    check('recall answers hidden by default', not box.locator('.d2l-answer-body').first.is_visible())
    box.locator('.d2l-answer-btn').first.click()
    check('Show answer reveals a recall answer', box.locator('.d2l-answer-body').first.is_visible())
    page.wait_for_function('() => document.querySelectorAll(".d2l-recall mjx-container").length > 0', timeout=30000)
    check('recall math typeset', True)
    shot(box, 'recall.png')

    page.goto(BASE + 'chapter_preliminaries/ndarray.html')
    page.wait_for_selector('.d2l-recall', timeout=30000)
    check('first Chapter 2 section recalls the Introduction',
          '1. Introduction' in page.locator('.d2l-recall').inner_text())
    page.goto(BASE + 'chapter_linear-regression/project-pace.html')
    page.wait_for_selector('.d2l-project-steps .d2l-step', timeout=30000)
    check('project pages have no recall box', page.locator('.d2l-recall').count() == 0)

    check('no JavaScript errors', not relevant_errors(page), relevant_errors(page))
    browser.close()
finish()
