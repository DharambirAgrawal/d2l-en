/* Reading progress tracker for the D2L book.
 *
 * Appended to _static/d2l.js through `include_js` in config.ini. Adds:
 *   - a "Mark section as complete" button at the end of every book page,
 *   - a ✓ next to completed pages in the sidebar table of contents, an n/m
 *     count on each chapter, and an overall progress bar,
 *   - Export / Import of progress as a JSON file (backup, or moving to
 *     another browser).
 *
 * Progress lives in this browser's localStorage. Only pages listed in the
 * table of contents can be marked.
 */
// Leading semicolon: this file is appended to other scripts in d2l.js.
;(function () {
  'use strict';

  const STORAGE_KEY = 'd2l-progress:v1';

  function load() {
    try {
      const data = JSON.parse(window.localStorage.getItem(STORAGE_KEY) || '{}');
      return data && typeof data === 'object' ? data : {};
    } catch (e) {
      return {};
    }
  }

  function save(data) {
    try {
      window.localStorage.setItem(STORAGE_KEY, JSON.stringify(data));
      return true;
    } catch (e) {
      return false;
    }
  }

  function siteRoot() {
    const opts = document.getElementById('documentation_options');
    const root = (opts && opts.getAttribute('data-url_root')) || './';
    return new URL(root, window.location.href).pathname;
  }

  /* Page key: path relative to the site root, e.g.
   * "chapter_preliminaries/linear-algebra.html". */
  function pageKey(url, root) {
    let path = new URL(url, window.location.href).pathname;
    if (path.endsWith('/')) path += 'index.html';
    return path.startsWith(root) ? decodeURIComponent(path.slice(root.length)) : null;
  }

  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text != null) node.textContent = text;
    return node;
  }

  function init() {
    const tocs = document.querySelectorAll('.globaltoc nav');
    if (!tocs.length) return;
    const root = siteRoot();
    const here = pageKey(window.location.href, root);
    let done = load();

    // Every TOC entry, keyed by page. The current page links to "#".
    const entries = [];
    tocs.forEach((nav) => {
      nav.querySelectorAll('li > a.reference.internal').forEach((a) => {
        const href = a.getAttribute('href');
        const key = href === '#' ? here : pageKey(a.href.split('#')[0], root);
        if (key) entries.push({a, key, li: a.parentElement});
      });
    });
    const pages = Array.from(new Set(entries.map((e) => e.key)));
    const onTocPage = pages.includes(here);

    // Overall progress box, at the top of each table of contents.
    const boxes = [];
    tocs.forEach((nav) => {
      const box = el('div', 'd2l-progress-box');
      const label = el('div', 'd2l-progress-label');
      const bar = el('div', 'd2l-progress-bar');
      const fill = el('div', 'd2l-progress-fill');
      bar.append(fill);
      const tools = el('div', 'd2l-progress-tools');
      const exportBtn = el('button', 'd2l-progress-link', 'Export');
      const importBtn = el('button', 'd2l-progress-link', 'Import');
      exportBtn.type = importBtn.type = 'button';
      exportBtn.title = 'Download your progress as a file (backup)';
      importBtn.title = 'Load progress from an exported file';
      exportBtn.addEventListener('click', exportProgress);
      importBtn.addEventListener('click', importProgress);
      tools.append(exportBtn, el('span', null, ' · '), importBtn);
      box.append(label, bar, tools);
      nav.prepend(box);
      boxes.push({label, fill});
    });

    // Button at the end of the page, and a chip next to the title.
    let button = null;
    let chip = null;
    if (onTocPage) {
      const body = document.querySelector('.document .section, .document section') ||
        document.querySelector('.document');
      const pager = document.querySelector('.pagenation');
      const wrap = el('div', 'd2l-progress-mark');
      button = el('button', 'd2l-progress-btn');
      button.type = 'button';
      button.addEventListener('click', () => {
        done = load();
        if (done[here]) delete done[here];
        else done[here] = new Date().toISOString().slice(0, 10);
        save(done);
        render();
      });
      wrap.append(button);
      if (pager) pager.before(wrap);
      else if (body) body.append(wrap);

      const title = document.querySelector('.document h1');
      if (title) {
        chip = el('span', 'd2l-progress-chip', '✓ Completed');
        title.append(chip);
      }
    }

    function render() {
      const count = pages.filter((k) => done[k]).length;
      const pct = pages.length ? Math.round((100 * count) / pages.length) : 0;
      boxes.forEach(({label, fill}) => {
        label.textContent = `Progress: ${count} / ${pages.length} sections (${pct}%)`;
        fill.style.width = pct + '%';
      });

      entries.forEach(({a, key, li}) => {
        a.classList.toggle('d2l-done', !!done[key]);
        // Chapters: count the chapter page plus every page nested under it.
        const childList = li.querySelector(':scope > ul');
        let badge = a.querySelector(':scope > .d2l-progress-count');
        if (childList) {
          const keys = [key].concat(Array.from(childList.querySelectorAll('a.reference.internal'))
            .map((c) => (c.getAttribute('href') === '#' ? here : pageKey(c.href.split('#')[0], root)))
            .filter(Boolean));
          const unique = Array.from(new Set(keys));
          const n = unique.filter((k) => done[k]).length;
          if (!badge) {
            badge = el('span', 'd2l-progress-count');
            a.append(badge);
          }
          badge.textContent = `${n}/${unique.length}`;
          badge.classList.toggle('d2l-progress-full', n === unique.length);
          badge.hidden = n === 0;
        }
      });

      if (button) {
        const complete = !!done[here];
        button.textContent = complete
          ? `✓ Completed on ${done[here]} · click to undo`
          : '☐ Mark section as complete';
        button.classList.toggle('d2l-progress-complete', complete);
        if (chip) chip.hidden = !complete;
      }
    }

    function exportProgress() {
      const blob = new Blob([JSON.stringify(load(), null, 2)], {type: 'application/json'});
      const a = el('a');
      a.href = URL.createObjectURL(blob);
      a.download = `d2l-progress-${new Date().toISOString().slice(0, 10)}.json`;
      document.body.append(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(a.href), 1000);
    }

    function importProgress() {
      const input = el('input');
      input.type = 'file';
      input.accept = 'application/json,.json';
      input.addEventListener('change', () => {
        const file = input.files && input.files[0];
        if (!file) return;
        file.text().then((text) => {
          const data = JSON.parse(text);
          if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('bad file');
          // Merge: keep what is already marked here, add what the file has.
          done = Object.assign(load(), data);
          if (!save(done)) throw new Error('storage unavailable');
          render();
          window.alert(`Imported. ${Object.keys(data).length} completed sections in the file.`);
        }).catch(() => window.alert('That file could not be imported.'));
      });
      input.click();
    }

    // Keep several open tabs in sync.
    window.addEventListener('storage', (e) => {
      if (e.key === STORAGE_KEY) {
        done = load();
        render();
      }
    });

    render();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
