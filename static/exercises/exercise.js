/* Interactive coding exercises for the D2L book.
 *
 * Appended to _static/d2l.js through `include_js` in config.ini, so it runs on
 * every page but does nothing unless the page contains an exercise:
 *
 *   ```{=html}
 *   <div class="d2l-exercise" data-exercise="vector-add"></div>
 *   ```
 *
 * Exercise data comes from _static/exercises/<id>.json, generated from
 * exercises/<id>/ by tools/build_exercises.py. Python runs in a Web Worker
 * (worker.js) with Pyodide; the editor is CodeMirror 5. Both load lazily.
 */
// Leading semicolon: this file is appended to other scripts in d2l.js.
;(function () {
  'use strict';

  const PYODIDE_VERSION = '314.0.7';
  const PYODIDE_URL = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;
  const CODEMIRROR_URL = 'https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.18/';
  const LOAD_TIMEOUT_MS = 180000; // downloading Pyodide + packages
  const DEFAULT_TIMEOUT_S = 10; // running the code itself
  const STORAGE_PREFIX = 'd2l-exercise:';

  // ---------------------------------------------------------------- helpers

  function staticRoot() {
    const opts = document.getElementById('documentation_options');
    const root = (opts && opts.getAttribute('data-url_root')) || './';
    return new URL(root + '_static/exercises/', document.baseURI).href;
  }

  function el(tag, attrs, children) {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (k === 'class') node.className = v;
      else if (k === 'text') node.textContent = v;
      else if (k === 'html') node.innerHTML = v;
      else if (k.startsWith('on')) node.addEventListener(k.slice(2), v);
      else if (v !== false && v != null) node.setAttribute(k, v === true ? '' : v);
    }
    for (const child of [].concat(children || [])) {
      if (child) node.append(child);
    }
    return node;
  }

  const storage = {
    get(key) {
      try { return window.localStorage.getItem(STORAGE_PREFIX + key); } catch (e) { return null; }
    },
    set(key, value) {
      try { window.localStorage.setItem(STORAGE_PREFIX + key, value); } catch (e) { /* ignore */ }
    },
    remove(key) {
      try { window.localStorage.removeItem(STORAGE_PREFIX + key); } catch (e) { /* ignore */ }
    },
  };

  const loadedAssets = {};
  function loadAsset(url) {
    if (!loadedAssets[url]) {
      loadedAssets[url] = new Promise((resolve, reject) => {
        const node = url.endsWith('.css')
          ? el('link', {rel: 'stylesheet', href: url})
          : el('script', {src: url});
        node.onload = resolve;
        node.onerror = () => reject(new Error('Failed to load ' + url));
        document.head.append(node);
      });
    }
    return loadedAssets[url];
  }

  function loadCodeMirror() {
    return Promise.all([
      loadAsset(CODEMIRROR_URL + 'codemirror.min.css'),
      loadAsset(CODEMIRROR_URL + 'codemirror.min.js'),
    ]).then(() => loadAsset(CODEMIRROR_URL + 'mode/python/python.min.js'))
      .then(() => window.CodeMirror);
  }

  // ----------------------------------------------------------------- runner

  /* One Pyodide worker shared by all exercises on the page. Only one run at
   * a time; a run that exceeds its timeout (or is stopped) terminates the
   * worker, and the next run starts a fresh one. */
  const runner = {
    worker: null,
    ready: null,
    busy: false,
    current: null, // {id, resolve, timer}
    nextId: 1,
    listeners: new Set(),

    setBusy(busy) {
      this.busy = busy;
      this.listeners.forEach((fn) => fn(busy));
    },

    start(onStatus) {
      if (this.ready) return this.ready;
      const root = staticRoot();
      const worker = new Worker(root + 'worker.js', {type: 'module'});
      this.worker = worker;
      this.ready = new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          reject(new Error('Loading Python took too long. Check your connection and try again.'));
        }, LOAD_TIMEOUT_MS);
        worker.onmessage = (e) => {
          const msg = e.data;
          if (msg.type === 'ready') { clearTimeout(timer); resolve(); }
          else if (msg.type === 'init-error') { clearTimeout(timer); reject(new Error(msg.message)); }
          else this.handle(msg);
        };
        worker.onerror = (e) => {
          clearTimeout(timer);
          reject(new Error(e.message || 'The Python worker failed to start.'));
        };
        worker.postMessage({type: 'init', pyodideUrl: PYODIDE_URL, graderUrl: root + 'grader.py'});
      });
      if (onStatus) onStatus('Loading Python (first run downloads ~10 MB)…');
      const ready = this.ready;
      ready.catch(() => { if (this.ready === ready) this.reset(); });
      return ready;
    },

    reset() {
      if (this.worker) this.worker.terminate();
      this.worker = null;
      this.ready = null;
    },

    handle(msg) {
      const cur = this.current;
      if (!cur || msg.id !== cur.id) return;
      if (msg.type === 'status') cur.onStatus(msg.text);
      else if (msg.type === 'started') {
        cur.onStatus('Running…');
        cur.timer = setTimeout(() => this.finish({
          status: 'timeout', stdout: '',
          error: `Stopped after ${cur.timeout} seconds. Is there an infinite loop?`,
        }, true), cur.timeout * 1000);
      } else if (msg.type === 'result') this.finish(msg.result, false);
    },

    finish(result, kill) {
      const cur = this.current;
      if (!cur) return;
      clearTimeout(cur.timer);
      this.current = null;
      if (kill) this.reset();
      this.setBusy(false);
      cur.resolve(result);
    },

    /* Run code ('run') or grade it ('submit'). Resolves with a result dict;
     * never rejects. */
    run({mode, code, tests, packages, files, timeout, onStatus, session, params, reset}) {
      if (this.busy) return Promise.resolve({status: 'busy'});
      this.setBusy(true);
      return new Promise((resolve) => {
        const id = this.nextId++;
        this.current = {id, resolve, onStatus, timeout: timeout || DEFAULT_TIMEOUT_S, timer: null};
        this.start(onStatus).then(() => {
          if (!this.current || this.current.id !== id) return; // stopped while loading
          onStatus('Loading packages…');
          this.worker.postMessage({type: 'run', id, mode, code, tests, packages, files,
            session, params, reset});
        }, (err) => this.finish({status: 'load-error', stdout: '', error: err.message}, false));
      });
    },

    stop() {
      if (this.current) this.finish({status: 'stopped', stdout: '', error: 'Stopped.'}, true);
    },
  };

  // --------------------------------------------------------------- exercise

  const DIFFICULTY = {easy: 'Easy', medium: 'Medium', hard: 'Hard'};

  function typeset(node) {
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise([node]).catch(() => {});
    }
  }

  function makeEditor(CodeMirror, parent, value, options) {
    if (!CodeMirror) {
      // CDN unavailable: fall back to a plain textarea with the same API bits.
      const ta = el('textarea', {class: 'd2l-ex-textarea', spellcheck: 'false'});
      ta.value = value;
      ta.readOnly = !!options.readOnly;
      parent.append(ta);
      return {
        getValue: () => ta.value,
        setValue: (v) => { ta.value = v; },
        on: (event, fn) => ta.addEventListener(event === 'change' ? 'input' : event, fn),
        refresh: () => {},
        focus: () => ta.focus(),
      };
    }
    return CodeMirror(parent, Object.assign({
      value,
      mode: 'python',
      lineNumbers: true,
      indentUnit: 4,
      tabSize: 4,
      viewportMargin: Infinity,
    }, options, {
      extraKeys: Object.assign({
        Tab: (cm) => (cm.somethingSelected()
          ? cm.indentSelection('add')
          : cm.replaceSelection('    ', 'end')),
        'Shift-Tab': (cm) => cm.indentSelection('subtract'),
      }, options.extraKeys),
    }));
  }

  class Exercise {
    constructor(container, data, CodeMirror) {
      this.data = data;
      this.container = container;
      this.key = data.id + ':code';
      this.build(CodeMirror);
      runner.listeners.add((busy) => this.syncButtons(busy));
      this.syncButtons(runner.busy);
      this.setSolved(storage.get(data.id + ':solved') === '1');
    }

    build(CodeMirror) {
      const d = this.data;
      const saved = storage.get(this.key);
      const hasSaved = saved !== null && saved !== d.starter;

      this.runBtn = el('button', {class: 'd2l-ex-btn d2l-ex-primary', type: 'button',
        title: 'Run your code (Ctrl/Cmd+Enter)', onclick: () => this.onRun()}, '▶ Run');
      this.submitBtn = el('button', {class: 'd2l-ex-btn d2l-ex-submit', type: 'button',
        title: 'Run the tests (Ctrl/Cmd+Shift+Enter)', onclick: () => this.onSubmit()}, 'Submit');
      this.resetBtn = el('button', {class: 'd2l-ex-btn', type: 'button',
        title: 'Restore the starter code', onclick: () => this.onReset()}, 'Reset');
      this.solutionBtn = el('button', {class: 'd2l-ex-btn', type: 'button',
        onclick: () => this.toggleSolution()}, 'Show solution');
      this.status = el('span', {class: 'd2l-ex-status', role: 'status', 'aria-live': 'polite'});
      this.solvedBadge = el('span', {class: 'd2l-ex-solved', hidden: true}, '✓ Solved');

      const editorBox = el('div', {class: 'd2l-ex-editor'});
      this.outputBox = el('div', {class: 'd2l-ex-panel', hidden: true});
      this.solutionBox = el('div', {class: 'd2l-ex-solution', hidden: true});

      const root = el('div', {class: 'd2l-ex'}, [
        el('div', {class: 'd2l-ex-head'}, [
          el('span', {class: 'd2l-ex-kicker'}, 'Exercise'),
          el('span', {class: 'd2l-ex-title'}, d.title),
          el('span', {class: 'd2l-ex-badge d2l-ex-' + d.difficulty}, DIFFICULTY[d.difficulty] || d.difficulty),
          this.solvedBadge,
        ]),
        el('div', {class: 'd2l-ex-desc', html: d.description_html}),
        editorBox,
        el('div', {class: 'd2l-ex-bar'}, [
          this.runBtn, this.submitBtn, this.status,
          el('span', {class: 'd2l-ex-spacer'}),
          this.resetBtn, this.solutionBtn,
        ]),
        this.outputBox,
        this.solutionBox,
        el('div', {class: 'd2l-ex-foot'},
          'Runs in your browser with Pyodide (Python ' + PYODIDE_VERSION.replace(/^(\d)(\d+)\..*/, '$1.$2') +
          ') · NumPy, SciPy, pandas, scikit-learn available · no PyTorch'),
      ]);
      this.container.replaceChildren(root);
      typeset(root);

      this.editor = makeEditor(CodeMirror, editorBox, hasSaved ? saved : d.starter, {
        extraKeys: {
          'Ctrl-Enter': () => this.onRun(),
          'Cmd-Enter': () => this.onRun(),
          'Shift-Ctrl-Enter': () => this.onSubmit(),
          'Shift-Cmd-Enter': () => this.onSubmit(),
        },
      });
      let saveTimer = null;
      this.editor.on('change', () => {
        clearTimeout(saveTimer);
        saveTimer = setTimeout(() => storage.set(this.key, this.editor.getValue()), 300);
      });
      // Start downloading Python as soon as the student starts typing.
      this.editor.on('focus', () => runner.start().catch(() => {}));
      if (hasSaved) this.setStatus('Restored your saved code.');
      this.CodeMirror = CodeMirror;
    }

    setStatus(text, kind) {
      this.status.textContent = text || '';
      this.status.className = 'd2l-ex-status' + (kind ? ' d2l-ex-' + kind : '');
    }

    setSolved(solved) {
      this.solvedBadge.hidden = !solved;
    }

    syncButtons(busy) {
      const mine = this.running;
      this.runBtn.disabled = busy && !mine;
      this.runBtn.textContent = mine ? '■ Stop' : '▶ Run';
      this.runBtn.title = mine ? 'Stop the running code' : 'Run your code (Ctrl/Cmd+Enter)';
      this.submitBtn.disabled = busy;
      this.resetBtn.disabled = busy;
    }

    async execute(mode) {
      if (this.running) { runner.stop(); return; }
      if (runner.busy) return;
      this.running = true;
      this.setStatus('Starting…');
      const result = await runner.run({
        mode,
        code: this.editor.getValue(),
        tests: mode === 'submit' ? this.data.tests : '',
        packages: this.data.packages,
        files: this.data.files,
        timeout: this.data.timeout,
        onStatus: (text) => this.setStatus(text),
      });
      this.running = false;
      this.syncButtons(runner.busy);
      this.showResult(mode, result);
    }

    onRun() { this.execute('run'); }

    onSubmit() {
      if (this.running) return;
      this.execute('submit');
    }

    onReset() {
      if (runner.busy) return;
      const current = this.editor.getValue();
      if (current !== this.data.starter &&
          !window.confirm('Replace your code with the starter code?')) return;
      this.editor.setValue(this.data.starter);
      storage.remove(this.key);
      this.outputBox.hidden = true;
      this.setStatus('Starter code restored.');
    }

    toggleSolution() {
      const box = this.solutionBox;
      if (!box.hidden) {
        box.hidden = true;
        this.solutionBtn.textContent = 'Show solution';
        return;
      }
      if (!box.childElementCount) {
        const codeBox = el('div', {class: 'd2l-ex-editor d2l-ex-readonly'});
        box.append(
          el('div', {class: 'd2l-ex-label'}, 'Reference solution'),
          codeBox,
          this.data.explanation_html
            ? el('div', {class: 'd2l-ex-explanation', html: this.data.explanation_html}) : null);
        box.hidden = false;
        makeEditor(this.CodeMirror, codeBox, this.data.solution, {readOnly: true});
        typeset(box);
      }
      box.hidden = false;
      this.solutionBtn.textContent = 'Hide solution';
    }

    showResult(mode, r) {
      const box = this.outputBox;
      box.replaceChildren();
      box.hidden = false;
      if (r.status === 'busy') { box.hidden = true; return; }

      if (r.status === 'stopped' || r.status === 'timeout' || r.status === 'load-error') {
        this.setStatus(r.status === 'stopped' ? 'Stopped.' : r.status === 'timeout'
          ? 'Timed out.' : 'Could not load Python.', 'bad');
        box.append(el('pre', {class: 'd2l-ex-error'}, r.error));
        return;
      }

      if (mode === 'submit' && r.status === 'ok') {
        const all = r.total > 0 && r.passed === r.total;
        this.setStatus(all ? 'All tests passed!' : `${r.passed} / ${r.total} tests passed`,
          all ? 'good' : 'bad');
        if (all) {
          storage.set(this.data.id + ':solved', '1');
          this.setSolved(true);
        }
        box.append(el('div', {class: 'd2l-ex-summary ' + (all ? 'd2l-ex-good' : 'd2l-ex-bad')},
          all ? `✓ All ${r.total} tests passed` : `${r.passed} of ${r.total} tests passed`));
        const list = el('ul', {class: 'd2l-ex-tests'});
        for (const t of r.tests) {
          const name = t.name.replace(/^test_/, '').replace(/_/g, ' ');
          const item = el('li', {class: t.passed ? 'd2l-ex-pass' : 'd2l-ex-fail'}, [
            el('span', {class: 'd2l-ex-mark'}, t.passed ? '✓' : '✗'),
            el('span', {class: 'd2l-ex-tname'}, name),
            t.description ? el('span', {class: 'd2l-ex-tdesc'}, t.description) : null,
          ]);
          if (!t.passed) {
            const detail = el('div', {class: 'd2l-ex-detail'});
            if (t.message) detail.append(el('div', {}, t.message));
            if ('expected' in t) detail.append(el('div', {class: 'd2l-ex-kv'}, [
              el('span', {}, 'Expected'), el('code', {}, t.expected)]));
            if ('actual' in t) detail.append(el('div', {class: 'd2l-ex-kv'}, [
              el('span', {}, 'Got'), el('code', {}, t.actual)]));
            if (t.error) detail.append(el('pre', {class: 'd2l-ex-error'}, t.error));
            item.append(detail);
          }
          list.append(item);
        }
        box.append(list);
      } else if (r.status === 'ok') {
        this.setStatus('Done.', 'good');
      } else {
        this.setStatus(mode === 'submit' ? 'Your code could not be tested.' : 'Error.', 'bad');
      }

      if (r.stdout) {
        box.append(el('div', {class: 'd2l-ex-label'}, 'Output'),
          el('pre', {class: 'd2l-ex-stdout'}, r.stdout));
      } else if (mode === 'run' && r.status === 'ok') {
        box.append(el('div', {class: 'd2l-ex-muted'},
          'No output. Use print() to show values, or press Submit to run the tests.'));
      }
      if (r.error) box.append(el('pre', {class: 'd2l-ex-error'}, r.error));
    }
  }

  // ------------------------------------------------------------------ mount

  let editorLib = null;

  function mount(container) {
    const id = container.getAttribute('data-exercise');
    if (!editorLib) editorLib = loadCodeMirror().catch(() => null);
    container.textContent = 'Loading exercise…';
    Promise.all([
      fetch(staticRoot() + encodeURIComponent(id) + '.json', {cache: 'no-cache'}).then((res) => {
        if (!res.ok) throw new Error(`exercise "${id}" not found (${res.status})`);
        return res.json();
      }),
      editorLib,
    ]).then(([data, CodeMirror]) => new Exercise(container, data, CodeMirror))
      .catch((err) => {
        container.textContent = 'Could not load this exercise: ' + err.message;
        container.classList.add('d2l-ex-broken');
      });
  }

  // ---------------------------------------------------------------- answers

  /* Hidden answers (and optional coding exercises) for the numbered list in a
   * page's "Exercises" section, from _static/answers/<page>.json, generated
   * from answers/<chapter>/<section>.yml by tools/build_exercises.py. */
  function addAnswers() {
    const list = document.querySelector('#exercises > ol, section#exercises > ol');
    if (!list) return;
    const opts = document.getElementById('documentation_options');
    const rootUrl = new URL((opts && opts.getAttribute('data-url_root')) || './', window.location.href);
    let page = window.location.pathname;
    if (page.endsWith('/')) page += 'index.html';
    if (!page.startsWith(rootUrl.pathname)) return;
    page = decodeURIComponent(page.slice(rootUrl.pathname.length)).replace(/\.html$/, '');
    const url = new URL('_static/answers/' + page + '.json', rootUrl).href;
    fetch(url, {cache: 'no-cache'})
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return;
        const items = Array.from(list.children).filter((li) => li.tagName === 'LI');
        items.forEach((li, i) => {
          const entry = data.answers[i];
          if (!entry) return;
          if (entry.exercise) {
            const box = el('div', {class: 'd2l-exercise', 'data-exercise': entry.exercise});
            li.append(box);
            mount(box);
          }
          if (entry.answer_html) {
            const answer = el('div', {class: 'd2l-answer-body', html: entry.answer_html, hidden: true});
            const button = el('button', {class: 'd2l-answer-btn', type: 'button', 'aria-expanded': 'false'},
              'Show answer');
            button.addEventListener('click', () => {
              answer.hidden = !answer.hidden;
              button.textContent = answer.hidden ? 'Show answer' : 'Hide answer';
              button.setAttribute('aria-expanded', String(!answer.hidden));
              if (!answer.hidden && !answer.dataset.typeset) {
                answer.dataset.typeset = '1';
                typeset(answer);
              }
            });
            li.append(el('div', {class: 'd2l-answer'}, [button, answer]));
          }
        });
      })
      .catch(() => {});
  }

  // ------------------------------------------------------------ book cells

  /* Make the book's own code cells editable and runnable, notebook style.
   * Cells in the "pytorch" tab and cells shared by all frameworks run in
   * one session per page; `torch` there is torchlite, a NumPy-based subset
   * of PyTorch (see torchlite.py), since real PyTorch can't run in a browser.
   */
  const CELL_TIMEOUT_S = 30;

  function showCellResult(box, r) {
    box.replaceChildren();
    box.hidden = false;
    if (r.stdout) box.append(el('pre', {class: 'd2l-cell-stdout'}, r.stdout));
    if (r.result) box.append(el('pre', {class: 'd2l-cell-value'}, r.result));
    for (const img of r.images || []) {
      box.append(el('img', {class: 'd2l-cell-img', alt: 'plot output', src: 'data:image/png;base64,' + img}));
    }
    if (r.error) box.append(el('pre', {class: 'd2l-ex-error'}, r.error));
    if (!box.childElementCount) box.append(el('div', {class: 'd2l-ex-muted'}, 'Done (no output).'));
  }

  function setupBookCells() {
    const blocks = Array.from(document.querySelectorAll('.highlight-python')).filter((b) => {
      if (b.closest('.d2l-ex, .d2l-answer, .d2l-notes, .d2l-playground')) return false;
      const panel = b.closest('.mdl-tabs__panel');
      if (panel && !/^pytorch-/.test(panel.id)) return false;
      return !/interact_select/.test(b.textContent);
    });
    if (!blocks.length) return;
    const session = 'page:' + window.location.pathname;
    const cells = blocks.map((block, index) => ({block, index, original: '', ran: false, editor: null}));

    function codeOf(cell) {
      return cell.editor ? cell.editor.getValue() : cell.original;
    }

    function restart() {
      cells.forEach((c) => {
        c.ran = false;
        c.out.hidden = true;
      });
      restartNext = true;
    }

    let restartNext = false;

    async function runCell(cell) {
      if (runner.busy) return;
      // Like running a notebook from the top: earlier cells that haven't run
      // yet run first, so variables they define exist.
      const queue = cells.slice(0, cell.index).filter((c) => !c.ran).concat([cell]);
      for (const c of queue) {
        c.status.textContent = c === cell ? 'Running…' : '';
        if (c !== cell) cell.status.textContent = `Running earlier cell ${c.index + 1}…`;
        const r = await runner.run({
          mode: 'cell', code: codeOf(c), session, reset: restartNext,
          timeout: CELL_TIMEOUT_S, onStatus: (t) => { cell.status.textContent = t; },
        });
        restartNext = false;
        if (r.status === 'busy') return;
        if (r.status !== 'ok' && r.status !== 'error') {
          // Timed out, stopped, or Python failed to load: the session is gone.
          cells.forEach((x) => { x.ran = false; });
          restartNext = true;
          showCellResult(c.out, {error: r.error});
          cell.status.textContent = '';
          return;
        }
        c.ran = true;
        showCellResult(c.out, r);
        c.status.textContent = '';
        if (r.status === 'error' && c !== cell) {
          cell.status.textContent = `Stopped: cell ${c.index + 1} above failed.`;
          return;
        }
      }
    }

    cells.forEach((cell) => {
      cell.original = cell.block.querySelector('pre').textContent.replace(/\n$/, '');
      const runBtn = el('button', {class: 'd2l-cell-btn d2l-cell-run', type: 'button',
        title: 'Run this cell (runs earlier cells first if needed)'}, '▶ Run');
      const editBtn = el('button', {class: 'd2l-cell-btn', type: 'button', title: 'Edit this code'}, '✎ Edit');
      const resetBtn = el('button', {class: 'd2l-cell-btn', type: 'button', hidden: true,
        title: 'Restore the original code'}, '↺ Original');
      const restartBtn = el('button', {class: 'd2l-cell-btn', type: 'button',
        title: 'Forget all variables on this page and start fresh'}, 'Restart');
      cell.status = el('span', {class: 'd2l-cell-status', role: 'status', 'aria-live': 'polite'});
      cell.out = el('div', {class: 'd2l-cell-out', hidden: true});
      const bar = el('div', {class: 'd2l-cell-bar'}, [runBtn, editBtn, resetBtn, cell.status,
        el('span', {class: 'd2l-ex-spacer'}), restartBtn]);
      cell.block.classList.add('d2l-cell');
      cell.block.after(bar, cell.out);

      runBtn.addEventListener('click', () => runCell(cell));
      restartBtn.addEventListener('click', () => {
        restart();
        cell.status.textContent = 'Session restarted.';
      });
      editBtn.addEventListener('click', () => {
        if (cell.editor) { cell.editor.focus(); return; }
        if (!editorLib) editorLib = loadCodeMirror().catch(() => null);
        editorLib.then((CodeMirror) => {
          const holder = el('div', {class: 'd2l-ex-editor d2l-cell-editor'});
          cell.block.querySelector('.highlight').replaceWith(holder);
          cell.editor = makeEditor(CodeMirror, holder, cell.original, {
            extraKeys: {
              'Ctrl-Enter': () => runCell(cell),
              'Cmd-Enter': () => runCell(cell),
              'Shift-Enter': () => runCell(cell),
            },
          });
          cell.editor.focus();
          resetBtn.hidden = false;
          editBtn.hidden = true;
          runner.start().catch(() => {});
        });
      });
      resetBtn.addEventListener('click', () => {
        if (cell.editor) cell.editor.setValue(cell.original);
      });
    });

    const note = el('div', {class: 'd2l-cell-note'}, [
      el('strong', {}, 'Runnable examples: '),
      'press ▶ Run or ✎ Edit on any PyTorch cell to experiment. They run in your browser on ',
      el('em', {}, 'torchlite'),
      ', a small NumPy-based version of PyTorch that covers the basics (tensors, math, ' +
      'autograd, plotting). Cells share variables like a notebook.',
    ]);
    blocks[0].before(note);
  }

  // ------------------------------------------------------------ playgrounds

  /* Interactive playgrounds: sliders feed variables into a short Python
   * program whose plot and printed output update live. Defined in
   * playgrounds/<id>.yml, embedded with
   *   <div class="d2l-playground" data-playground="<id>"></div>
   */
  function mountPlayground(container) {
    const id = container.getAttribute('data-playground');
    container.textContent = 'Loading playground…';
    const root = new URL('../playgrounds/', staticRoot()).href;
    if (!editorLib) editorLib = loadCodeMirror().catch(() => null);
    Promise.all([
      fetch(root + encodeURIComponent(id) + '.json', {cache: 'no-cache'}).then((res) => {
        if (!res.ok) throw new Error(`playground "${id}" not found (${res.status})`);
        return res.json();
      }),
      editorLib,
    ]).then(([pg, CodeMirror]) => buildPlayground(container, pg, CodeMirror))
      .catch((err) => {
        container.textContent = 'Could not load this playground: ' + err.message;
        container.classList.add('d2l-ex-broken');
      });
  }

  function buildPlayground(container, pg, CodeMirror) {
    const values = {};
    const out = el('div', {class: 'd2l-pg-out'});
    const status = el('span', {class: 'd2l-ex-status', role: 'status', 'aria-live': 'polite'});
    const controls = el('div', {class: 'd2l-pg-controls'});
    let code = pg.code;
    let pending = false;
    let timer = null;

    async function update() {
      if (runner.busy) {  // another cell or exercise is running; try again soon
        clearTimeout(timer);
        timer = setTimeout(update, 300);
        return;
      }
      status.textContent = 'Running…';
      const r = await runner.run({
        mode: 'cell', code, session: 'playground:' + pg.id, reset: true,
        params: Object.assign({}, values), timeout: pg.timeout || 20,
        onStatus: (t) => { status.textContent = t; },
      });
      status.textContent = r.status === 'ok' ? '' : (r.status === 'busy' ? '' : 'Error');
      if (r.status !== 'busy') showCellResult(out, r);
      if (pending) {
        pending = false;
        setTimeout(update, 50);
      }
    }

    function schedule() {
      // Slider moved: rerun after a short pause (or right after the current run).
      if (runner.busy && status.textContent) { pending = true; return; }
      clearTimeout(timer);
      timer = setTimeout(update, 200);
    }

    for (const p of pg.params) {
      values[p.name] = p.value;
      const valueLabel = el('span', {class: 'd2l-pg-value'}, String(p.value));
      const input = el('input', {type: 'range', min: p.min, max: p.max, step: p.step,
        value: p.value, 'aria-label': p.label});
      input.addEventListener('input', () => {
        values[p.name] = Number(input.value);
        valueLabel.textContent = input.value;
        schedule();
      });
      controls.append(el('label', {class: 'd2l-pg-param'}, [
        el('span', {class: 'd2l-pg-label', html: p.label_html || p.label}), input, valueLabel]));
    }

    const codeBox = el('div', {class: 'd2l-ex-editor', hidden: true});
    const codeBtn = el('button', {class: 'd2l-ex-btn', type: 'button'}, 'Show / edit code');
    const runBtn = el('button', {class: 'd2l-ex-btn d2l-ex-primary', type: 'button'}, '▶ Run');
    let editor = null;
    codeBtn.addEventListener('click', () => {
      codeBox.hidden = !codeBox.hidden;
      if (!editor && !codeBox.hidden) {
        editor = makeEditor(CodeMirror, codeBox, code, {
          extraKeys: {'Ctrl-Enter': () => runBtn.click(), 'Cmd-Enter': () => runBtn.click()},
        });
        editor.on('change', () => { code = editor.getValue(); });
      }
    });
    runBtn.addEventListener('click', update);

    const box = el('div', {class: 'd2l-ex d2l-playground-box'}, [
      el('div', {class: 'd2l-ex-head'}, [
        el('span', {class: 'd2l-ex-kicker'}, 'Playground'),
        el('span', {class: 'd2l-ex-title'}, pg.title),
      ]),
      el('div', {class: 'd2l-ex-desc', html: pg.description_html}),
      controls,
      el('div', {class: 'd2l-ex-bar'}, [runBtn, status, el('span', {class: 'd2l-ex-spacer'}), codeBtn]),
      codeBox,
      out,
      el('div', {class: 'd2l-ex-foot'},
        'Move the sliders to experiment. Variables: ' + pg.params.map((p) => p.name).join(', ') +
        ' · runs in your browser'),
    ]);
    container.replaceChildren(box);
    typeset(box);
    // Run once when the playground scrolls into view.
    const io = new IntersectionObserver((entries) => {
      if (entries.some((e) => e.isIntersecting)) {
        io.disconnect();
        update();
      }
    });
    io.observe(box);
  }

  // ------------------------------------------------------------------ notes

  /* "Notes to remember" box before a page's Exercises (or at its end), from
   * _static/notes/<page>.json (generated from notes/<chapter>/<section>.yml). */
  function addNotes() {
    const opts = document.getElementById('documentation_options');
    const rootUrl = new URL((opts && opts.getAttribute('data-url_root')) || './', window.location.href);
    let page = window.location.pathname;
    if (page.endsWith('/')) page += 'index.html';
    if (!page.startsWith(rootUrl.pathname)) return;
    page = decodeURIComponent(page.slice(rootUrl.pathname.length)).replace(/\.html$/, '');
    fetch(new URL('_static/notes/' + page + '.json', rootUrl).href, {cache: 'no-cache'})
      .then((res) => (res.ok ? res.json() : null))
      .then((data) => {
        if (!data) return;
        const copyBtn = el('button', {class: 'd2l-answer-btn', type: 'button'}, 'Copy notes');
        copyBtn.addEventListener('click', () => {
          const done = () => {
            copyBtn.textContent = 'Copied!';
            setTimeout(() => { copyBtn.textContent = 'Copy notes'; }, 1500);
          };
          if (navigator.clipboard) navigator.clipboard.writeText(data.text).then(done, () => {});
        });
        const box = el('div', {class: 'd2l-notes'}, [
          el('div', {class: 'd2l-notes-head'}, [
            el('span', {class: 'd2l-notes-title'}, '📝 Notes to remember'),
            el('span', {class: 'd2l-ex-spacer'}), copyBtn,
          ]),
          el('div', {class: 'd2l-notes-body', html: data.html}),
        ]);
        const exercises = document.querySelector('#exercises');
        const main = document.querySelector('.document section, .document .section');
        if (exercises) exercises.before(box);
        else if (main) main.append(box);
        typeset(box);
      })
      .catch(() => {});
  }

  // ------------------------------------------------------------------- init

  function init() {
    document.querySelectorAll('.d2l-exercise[data-exercise]').forEach(mount);
    document.querySelectorAll('.d2l-playground[data-playground]').forEach(mountPlayground);
    addAnswers();
    addNotes();
    setupBookCells();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
