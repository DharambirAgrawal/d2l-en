/* "My Notes" review page: every notes box from the sections you have
 * completed, in book order, ready to copy or print.
 *
 * Runs on the page that contains <div class="d2l-review"></div>
 * (chapter_my-notes/index.md). Uses window.D2LProgress (progress.js) and the
 * notes JSON written by tools/build_exercises.py (_static/notes/).
 *
 * Appended to _static/d2l.js through `include_js`, after progress.js.
 */
// Leading semicolon: this file is appended to other scripts in d2l.js.
;(function () {
  'use strict';

  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children || [])) if (c) node.append(c);
    return node;
  }

  function typeset(node) {
    if (window.MathJax && window.MathJax.typesetPromise) window.MathJax.typesetPromise([node]).catch(() => {});
  }

  function sectionAnchor(heading) {
    // Sphinx turns headings into ids like "matrixvector-products".
    return heading.toLowerCase().replace(/--/g, '').replace(/[^a-z0-9\s-]/g, '')
      .trim().replace(/\s+/g, '-');
  }

  async function init() {
    const container = document.querySelector('.d2l-review');
    if (!container || !window.D2LProgress) return;
    const P = window.D2LProgress;
    const base = new URL('_static/notes/', new URL(P.siteRoot(), window.location.href));
    container.textContent = 'Loading your notes…';

    let index;
    try {
      index = await (await fetch(new URL('index.json', base), {cache: 'no-cache'})).json();
    } catch (e) {
      container.textContent = 'Could not load notes.';
      return;
    }
    const withNotes = new Set(index.pages.map((p) => p + '.html'));
    const pages = P.tocPages().filter((p) => withNotes.has(p.key));
    const notes = {};
    await Promise.all(pages.map(async (p) => {
      const res = await fetch(new URL(p.key.replace(/\.html$/, '.json'), base), {cache: 'no-cache'});
      if (res.ok) notes[p.key] = await res.json();
    }));

    let showAll = false;
    const list = h('div', {class: 'd2l-review-list'});
    const summary = h('p', {class: 'd2l-review-summary'});
    const toggle = h('input', {type: 'checkbox', id: 'd2l-review-all'});
    toggle.addEventListener('change', () => { showAll = toggle.checked; render(); });

    function visiblePages() {
      const done = P.load();
      return pages.filter((p) => showAll || done[p.key]);
    }

    function asText() {
      return visiblePages().map((p) => {
        const parts = [`## ${p.title}`];
        for (const s of notes[p.key].sections) parts.push(`### ${s.heading.replace(/--/g, '–')}`, s.text);
        return parts.join('\n\n');
      }).join('\n\n');
    }

    function render() {
      const shown = visiblePages();
      const done = P.load();
      const completed = pages.filter((p) => done[p.key]).length;
      summary.textContent = showAll
        ? `Showing notes from all ${pages.length} sections that have notes.`
        : `Showing notes from ${completed} of ${pages.length} sections with notes (the ones you've marked complete).`;
      list.replaceChildren();
      if (!shown.length) {
        list.append(h('p', {class: 'd2l-review-empty', text:
          'No completed sections with notes yet. Mark a section as complete at the bottom of its page, ' +
          'or tick "Show all sections" above.'}));
        return;
      }
      for (const p of shown) {
        const block = h('section', {class: 'd2l-review-page'}, [
          h('h2', {}, [h('a', {href: p.href, text: p.title})]),
        ]);
        for (const s of notes[p.key].sections) {
          block.append(
            h('h3', {}, [h('a', {href: p.href + '#' + sectionAnchor(s.heading), text: s.heading.replace(/--/g, '–')})]),
            h('div', {class: 'd2l-review-notes', html: s.html}));
        }
        list.append(block);
      }
      typeset(list);
    }

    const copyBtn = h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Copy as text'});
    copyBtn.addEventListener('click', () => {
      navigator.clipboard.writeText(asText()).then(() => {
        copyBtn.textContent = 'Copied';
        setTimeout(() => { copyBtn.textContent = 'Copy as text'; }, 1500);
      }, () => {});
    });
    const printBtn = h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Print / save as PDF',
      onclick: () => window.print()});

    container.replaceChildren(
      h('div', {class: 'd2l-review-bar'}, [
        h('label', {class: 'd2l-review-toggle', for: 'd2l-review-all'}, [toggle, ' Show all sections']),
        h('span', {class: 'd2l-ex-spacer'}),
        copyBtn, printBtn,
      ]),
      summary,
      list);
    document.body.classList.add('d2l-review-page-body');
    render();
    window.addEventListener('d2l-progress-updated', render);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
