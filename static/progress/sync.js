/* Sync progress across devices (the site has no server of its own).
 *
 * Two ways, both opened from the "Sync" link in the sidebar progress box:
 *   1. Sync link: a URL that carries your progress in its #fragment
 *      (#d2l-sync=...). Opening it on another device merges it in. The
 *      fragment never leaves the browser (it isn't sent to the server).
 *   2. GitHub: with a token that can read/write Gists, progress is stored in
 *      a secret gist ("d2l-progress.json") in your own GitHub account and
 *      synced automatically: pulled when a page loads, pushed after changes.
 *
 * What syncs: completed sections and solved exercises. Merging is a union,
 * so syncing never loses a check mark (unmarking only applies locally).
 *
 * Appended to _static/d2l.js through `include_js`, after progress.js.
 */
// Leading semicolon: this file is appended to other scripts in d2l.js.
;(function () {
  'use strict';

  const SETTINGS_KEY = 'd2l-sync:github';
  const GIST_FILE = 'd2l-progress.json';
  const HASH = 'd2l-sync=';
  const API = 'https://api.github.com';
  const SOLVED = /^d2l-exercise:(.+):solved$/;

  function store(key, value) {
    try {
      if (value === null) window.localStorage.removeItem(key);
      else window.localStorage.setItem(key, value);
    } catch (e) { /* storage unavailable */ }
  }

  function read(key) {
    try { return window.localStorage.getItem(key); } catch (e) { return null; }
  }

  // ------------------------------------------------------------ the data

  function collect() {
    const solved = {};
    try {
      for (let i = 0; i < window.localStorage.length; i++) {
        const m = SOLVED.exec(window.localStorage.key(i));
        if (m && window.localStorage.getItem(window.localStorage.key(i)) === '1') solved[m[1]] = 1;
      }
    } catch (e) { /* ignore */ }
    return {version: 1, progress: window.D2LProgress.load(), solved};
  }

  /* Merge remote data into this browser. Returns true if anything changed. */
  function merge(remote) {
    if (!remote || typeof remote !== 'object') return false;
    let changed = false;
    const progress = window.D2LProgress.load();
    for (const [page, date] of Object.entries(remote.progress || {})) {
      if (typeof date !== 'string') continue;
      if (!progress[page] || date < progress[page]) {  // keep the earliest date
        progress[page] = date;
        changed = true;
      }
    }
    for (const id of Object.keys(remote.solved || {})) {
      if (read(`d2l-exercise:${id}:solved`) !== '1') {
        store(`d2l-exercise:${id}:solved`, '1');
        changed = true;
      }
    }
    if (changed) window.D2LProgress.save(progress, {silent: true});
    return changed;
  }

  function sameData(a, b) {
    const norm = (d) => JSON.stringify({
      p: Object.keys(d.progress || {}).sort().map((k) => [k, d.progress[k]]),
      s: Object.keys(d.solved || {}).sort(),
    });
    return norm(a) === norm(b);
  }

  // ------------------------------------------------------------ sync link

  function toBase64Url(bytes) {
    let s = '';
    bytes.forEach((b) => { s += String.fromCharCode(b); });
    return btoa(s).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
  }

  function fromBase64Url(text) {
    const s = atob(text.replace(/-/g, '+').replace(/_/g, '/'));
    return Uint8Array.from(s, (c) => c.charCodeAt(0));
  }

  async function encodeLink(data) {
    const bytes = new TextEncoder().encode(JSON.stringify(data));
    if (window.CompressionStream) {
      const gz = new Uint8Array(await new Response(
        new Blob([bytes]).stream().pipeThrough(new CompressionStream('gzip'))).arrayBuffer());
      return 'g' + toBase64Url(gz);
    }
    return 'j' + toBase64Url(bytes);
  }

  async function decodeLink(text) {
    let bytes = fromBase64Url(text.slice(1));
    if (text[0] === 'g') {
      bytes = new Uint8Array(await new Response(
        new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    }
    return JSON.parse(new TextDecoder().decode(bytes));
  }

  async function syncLink() {
    const root = new URL(window.D2LProgress.siteRoot(), window.location.href);
    return new URL('index.html#' + HASH + await encodeLink(collect()), root).href;
  }

  /* Opened a sync link? Merge it, tell the reader, and clean the URL. */
  async function importFromHash() {
    const i = window.location.hash.indexOf(HASH);
    if (i < 0) return;
    const payload = window.location.hash.slice(i + HASH.length);
    history.replaceState(null, '', window.location.pathname + window.location.search);
    try {
      const data = await decodeLink(payload);
      const before = Object.keys(window.D2LProgress.load()).length;
      merge(data);
      const after = Object.keys(window.D2LProgress.load()).length;
      notify(`Progress imported from your sync link: ${after - before} new completed section(s).`);
      if (settings()) schedulePush(0);
    } catch (e) {
      notify('That sync link could not be read.');
    }
  }

  // ------------------------------------------------------------ GitHub gist

  function settings() {
    try { return JSON.parse(read(SETTINGS_KEY) || 'null'); } catch (e) { return null; }
  }

  function saveSettings(s) {
    store(SETTINGS_KEY, s ? JSON.stringify(s) : null);
  }

  async function gh(path, token, options = {}) {
    const res = await fetch(API + path, Object.assign({}, options, {
      headers: Object.assign({
        Accept: 'application/vnd.github+json',
        Authorization: 'Bearer ' + token,
        'X-GitHub-Api-Version': '2022-11-28',
      }, options.body ? {'Content-Type': 'application/json'} : {}),
    }));
    if (res.status === 401 || res.status === 403) throw new Error('GitHub rejected the token (check that it can read and write Gists).');
    if (!res.ok) throw new Error(`GitHub error ${res.status}`);
    return res.status === 204 ? null : res.json();
  }

  /* Find the progress gist in the account, or create it. */
  async function connect(token) {
    for (let page = 1; page <= 5; page++) {
      const gists = await gh(`/gists?per_page=100&page=${page}`, token);
      const found = gists.find((g) => g.files && g.files[GIST_FILE]);
      if (found) return found.id;
      if (gists.length < 100) break;
    }
    const created = await gh('/gists', token, {
      method: 'POST',
      body: JSON.stringify({
        description: 'Dive into Deep Learning: my reading progress (synced by the site)',
        public: false,
        files: {[GIST_FILE]: {content: JSON.stringify(collect(), null, 1)}},
      }),
    });
    return created.id;
  }

  async function pull(s) {
    const gist = await gh(`/gists/${s.gistId}`, s.token);
    const file = gist.files && gist.files[GIST_FILE];
    if (!file) return null;
    // Large gists are truncated in the API response; fetch the raw file then.
    const text = file.truncated ? await (await fetch(file.raw_url)).text() : file.content;
    return JSON.parse(text);
  }

  async function push(s) {
    await gh(`/gists/${s.gistId}`, s.token, {
      method: 'PATCH',
      body: JSON.stringify({files: {[GIST_FILE]: {content: JSON.stringify(collect(), null, 1)}}}),
    });
  }

  let syncing = null;
  let pushTimer = null;

  /* Pull, merge, and push if the remote is missing anything. */
  function syncNow() {
    const s = settings();
    if (!s) return Promise.resolve();
    if (syncing) return syncing;
    setState('Syncing…');
    syncing = (async () => {
      try {
        const remote = await pull(s);
        merge(remote);
        if (!remote || !sameData(remote, collect())) await push(s);
        saveSettings(Object.assign(s, {last: new Date().toISOString(), error: null}));
        setState('Synced');
      } catch (e) {
        saveSettings(Object.assign(s, {error: e.message}));
        setState('Sync failed');
      } finally {
        syncing = null;
        refreshDialog();
      }
    })();
    return syncing;
  }

  function schedulePush(delay = 2000) {
    if (!settings()) return;
    clearTimeout(pushTimer);
    pushTimer = setTimeout(syncNow, delay);
  }

  // ------------------------------------------------------------------ UI

  function h(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const c of [].concat(children || [])) if (c) node.append(c);
    return node;
  }

  function setState(text) {
    document.querySelectorAll('.d2l-progress-syncstate').forEach((n) => {
      n.textContent = text ? ` · ${text}` : '';
    });
  }

  function notify(text) {
    const toast = h('div', {class: 'd2l-sync-toast', role: 'status', text});
    document.body.append(toast);
    setTimeout(() => toast.remove(), 6000);
  }

  function ago(iso) {
    if (!iso) return 'never';
    const s = Math.round((Date.now() - new Date(iso)) / 1000);
    if (s < 60) return 'just now';
    if (s < 3600) return `${Math.round(s / 60)} min ago`;
    if (s < 86400) return `${Math.round(s / 3600)} h ago`;
    return new Date(iso).toLocaleDateString();
  }

  let dialog = null;
  let ghBox = null;

  function refreshDialog() {
    if (!ghBox) return;
    ghBox.replaceChildren();
    const s = settings();
    if (s) {
      ghBox.append(
        h('p', {class: 'd2l-sync-status'}, [
          h('strong', {text: 'Connected.'}),
          ` Last synced: ${ago(s.last)}.`,
          s.error ? h('span', {class: 'd2l-sync-error', text: ` ${s.error}`}) : null,
        ]),
        h('div', {class: 'd2l-sync-row'}, [
          h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Sync now', onclick: () => syncNow()}),
          h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Disconnect',
            onclick: () => { saveSettings(null); setState(''); refreshDialog(); }}),
        ]));
      return;
    }
    const input = h('input', {type: 'password', class: 'd2l-sync-input', placeholder: 'GitHub token (github_pat_… or ghp_…)',
      autocomplete: 'off', 'aria-label': 'GitHub token'});
    const status = h('p', {class: 'd2l-sync-status'});
    const btn = h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Connect'});
    btn.addEventListener('click', async () => {
      const token = input.value.trim();
      if (!token) return;
      btn.disabled = true;
      status.textContent = 'Connecting…';
      try {
        const gistId = await connect(token);
        saveSettings({token, gistId, last: null});
        await syncNow();
        refreshDialog();
      } catch (e) {
        status.textContent = e.message;
        btn.disabled = false;
      }
    });
    ghBox.append(
      h('p', {}, [
        'Create a token at ',
        h('a', {href: 'https://github.com/settings/personal-access-tokens/new', target: '_blank', rel: 'noopener',
          text: 'GitHub → Settings → Fine-grained tokens'}),
        ' with only the permission "Gists: Read and write", then paste it on each device. ',
        'The token is stored only in this browser.',
      ]),
      h('div', {class: 'd2l-sync-row'}, [input, btn]),
      status);
  }

  function open() {
    if (!dialog) {
      const linkStatus = h('span', {class: 'd2l-sync-status'});
      const copyBtn = h('button', {class: 'd2l-btn d2l-btn-small', type: 'button', text: 'Copy sync link'});
      copyBtn.addEventListener('click', async () => {
        const link = await syncLink();
        try {
          await navigator.clipboard.writeText(link);
          linkStatus.textContent = ' Copied. Open it on your other device.';
        } catch (e) {
          window.prompt('Copy this link:', link);
        }
      });
      ghBox = h('div');
      dialog = h('dialog', {class: 'd2l-sync-dialog', 'aria-label': 'Sync progress'}, [
        h('div', {class: 'd2l-sync-head'}, [
          h('span', {class: 'd2l-sync-title', text: 'Sync your progress'}),
          h('button', {class: 'd2l-link-btn', type: 'button', text: 'Close', onclick: () => dialog.close()}),
        ]),
        h('section', {class: 'd2l-sync-section'}, [
          h('h3', {text: 'Sync link'}),
          h('p', {text: 'A one-time copy: open this link on another device to add your completed ' +
            'sections and solved exercises there. No account needed.'}),
          h('div', {class: 'd2l-sync-row'}, [copyBtn, linkStatus]),
        ]),
        h('section', {class: 'd2l-sync-section'}, [
          h('h3', {text: 'Automatic sync with GitHub'}),
          h('p', {text: 'Keeps every device in sync automatically by saving your progress to a secret ' +
            'gist in your own GitHub account.'}),
          ghBox,
        ]),
      ]);
      dialog.addEventListener('click', (e) => { if (e.target === dialog) dialog.close(); });
      document.body.append(dialog);
    }
    refreshDialog();
    if (dialog.showModal) dialog.showModal();
    else dialog.setAttribute('open', '');
  }

  window.D2LSync = {open, syncNow, merge, collect, encodeLink, decodeLink};

  function init() {
    importFromHash();
    // A sync link pasted into a tab already on the site only changes the #.
    window.addEventListener('hashchange', importFromHash);
    window.addEventListener('d2l-progress-updated', (e) => {
      if (e.detail && e.detail.local) schedulePush();
    });
    // Solving an exercise is also progress worth syncing.
    window.addEventListener('d2l-exercise-solved', () => schedulePush());
    const s = settings();
    if (s) {
      setState(s.error ? 'Sync failed' : 'Synced');
      syncNow();
    }
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
