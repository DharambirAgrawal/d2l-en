/* Web Worker that runs exercise code with Pyodide (Python in the browser).
 *
 * Started by exercise.js as a module worker. Running Python here keeps the page
 * responsive, and lets the page stop runaway code by terminating this worker.
 * (Pyodide 314+ must be loaded with import(); importScripts() fails.)
 *
 * Messages in:
 *   {type: 'init', pyodideUrl, graderUrl}
 *   {type: 'run', id, mode: 'run' | 'submit', code, tests, packages, files}
 *   {type: 'run', id, mode: 'cell', code, session, params, reset}  book cells,
 *       playgrounds (see notebook.py)
 * Messages out:
 *   {type: 'ready'} | {type: 'init-error', message}
 *   {type: 'status', id, text}   progress while loading packages
 *   {type: 'started', id}        user code is about to run (timeout starts)
 *   {type: 'result', id, result} see grader.run_code / grader.run_tests
 */
let pyodideReady = null;
let datasetsBase = '';

// Datasets the book's code can load (torchlite's torchvision shim and
// d2l.download read them from /d2l/datasets). Downloaded the first time a
// cell mentions them.
const DATASETS = [
  {file: 'fashion-mnist.npz', pattern: /FashionMNIST/, label: 'Fashion-MNIST (31 MB, only once)'},
  {file: 'kaggle_house_pred_train.csv', pattern: /KaggleHouse|kaggle_house/, label: 'house prices (0.5 MB)'},
  {file: 'kaggle_house_pred_test.csv', pattern: /KaggleHouse|kaggle_house/, label: 'house prices (0.5 MB)'},
];

async function ensureDatasets(pyodide, code, post) {
  for (const d of DATASETS) {
    const path = `/d2l/datasets/${d.file}`;
    if (!d.pattern.test(code) || pyodide.FS.analyzePath(path).exists) continue;
    post({type: 'status', text: `Downloading ${d.label}…`});
    try {
      const r = await fetch(`${datasetsBase}${d.file}`);
      if (!r.ok) throw new Error(String(r.status));
      pyodide.FS.writeFile(path, new Uint8Array(await r.arrayBuffer()));
    } catch (e) {
      // The Python side raises a clear error when the file is missing.
    }
  }
}

async function init(pyodideUrl, graderUrl) {
  const {loadPyodide} = await import(pyodideUrl + 'pyodide.mjs');
  const pyodide = await loadPyodide({indexURL: pyodideUrl});
  const response = await fetch(graderUrl);
  if (!response.ok) throw new Error(`Could not load ${graderUrl} (${response.status})`);
  pyodide.FS.mkdirTree('/d2l/datasets');
  pyodide.FS.writeFile('/d2l/grader.py', await response.text());
  // Runtime for the book's runnable cells and playgrounds, fetched next to
  // grader.py. Loaded lazily by the first cell run.
  const base = graderUrl.slice(0, graderUrl.lastIndexOf('/') + 1);
  datasetsBase = new URL('../datasets/', base).href;
  for (const name of ['notebook.py', 'torchlite.py', 'd2llite.py', 'd2l_book.py']) {
    const r = await fetch(base + name);
    if (r.ok) pyodide.FS.writeFile('/d2l/' + name, await r.text());
  }
  pyodide.runPython('import sys; sys.path.insert(0, "/d2l"); import grader');
  return pyodide;
}

async function run(msg) {
  const pyodide = await pyodideReady;
  const post = (data) => postMessage(Object.assign({id: msg.id}, data));
  try {
    post({type: 'status', text: 'Loading packages…'});
    try {
      const helpers = Object.entries(msg.files || {})
        .filter(([name]) => name.endsWith('.py')).map(([, text]) => text).join('\n');
      // Book cells run on torchlite (NumPy) and may plot with matplotlib.
      const extra = msg.mode === 'cell'
        ? 'import numpy\n' + (/\bd2l\b|matplotlib|plt\./.test(msg.code) ? 'import matplotlib\n' : '')
        : '';
      // Jupyter magics (%matplotlib inline) aren't Python; drop them so
      // import detection can parse the code.
      const code = msg.code.replace(/^\s*[%!].*$/gm, '');
      await pyodide.loadPackagesFromImports(
        extra + code + '\n' + (msg.tests || '') + '\n' + helpers);
    } catch (e) {
      // Unknown or broken imports are reported by Python when the code runs.
    }
    if (msg.packages && msg.packages.length) await pyodide.loadPackage(msg.packages);
    if (msg.mode === 'cell') await ensureDatasets(pyodide, msg.code, post);
    // Exercise data files (CSV, text, helper modules) go in the working
    // directory, which is also on sys.path.
    for (const [name, text] of Object.entries(msg.files || {})) {
      pyodide.FS.writeFile(name, text);
    }
    pyodide.runPython('import importlib, os, sys; importlib.invalidate_caches();'
      + ' sys.path.insert(0, os.getcwd()) if os.getcwd() not in sys.path else None');
    post({type: 'started'});
    pyodide.globals.set('_d2l_code', msg.code);
    pyodide.globals.set('_d2l_tests', msg.tests || '');
    let json;
    if (msg.mode === 'cell') {
      // A book cell or playground: notebook-style session, torchlite as torch.
      pyodide.globals.set('_d2l_session', msg.session);
      pyodide.globals.set('_d2l_params', JSON.stringify(msg.params || null));
      if (msg.reset) pyodide.runPython('import notebook; notebook.reset(_d2l_session)');
      json = pyodide.runPython(
        'import notebook; notebook.run_cell_json(_d2l_session, _d2l_code, _d2l_params)');
    } else {
      json = pyodide.runPython(msg.mode === 'submit'
        ? 'grader.run_tests_json(_d2l_code, _d2l_tests)'
        : 'grader.run_code_json(_d2l_code)');
    }
    post({type: 'result', result: JSON.parse(json)});
  } catch (e) {
    post({type: 'result', result: {status: 'error', stdout: '', error: String(e)}});
  }
}

onmessage = (event) => {
  const msg = event.data;
  if (msg.type === 'init') {
    pyodideReady = init(msg.pyodideUrl, msg.graderUrl);
    pyodideReady.then(
      () => postMessage({type: 'ready'}),
      (e) => postMessage({type: 'init-error', message: String(e && e.message || e)}));
  } else if (msg.type === 'run') {
    run(msg);
  }
};
