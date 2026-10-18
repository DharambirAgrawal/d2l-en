/* Web Worker that runs exercise code with Pyodide (Python in the browser).
 *
 * Started by exercise.js as a module worker. Running Python here keeps the page
 * responsive, and lets the page stop runaway code by terminating this worker.
 * (Pyodide 314+ must be loaded with import(); importScripts() fails.)
 *
 * Messages in:
 *   {type: 'init', pyodideUrl, graderUrl}
 *   {type: 'run', id, mode: 'run' | 'submit', code, tests, packages}
 * Messages out:
 *   {type: 'ready'} | {type: 'init-error', message}
 *   {type: 'status', id, text}   progress while loading packages
 *   {type: 'started', id}        user code is about to run (timeout starts)
 *   {type: 'result', id, result} see grader.run_code / grader.run_tests
 */
let pyodideReady = null;

async function init(pyodideUrl, graderUrl) {
  const {loadPyodide} = await import(pyodideUrl + 'pyodide.mjs');
  const pyodide = await loadPyodide({indexURL: pyodideUrl});
  const response = await fetch(graderUrl);
  if (!response.ok) throw new Error(`Could not load ${graderUrl} (${response.status})`);
  pyodide.FS.mkdirTree('/d2l');
  pyodide.FS.writeFile('/d2l/grader.py', await response.text());
  pyodide.runPython('import sys; sys.path.insert(0, "/d2l"); import grader');
  return pyodide;
}

async function run(msg) {
  const pyodide = await pyodideReady;
  const post = (data) => postMessage(Object.assign({id: msg.id}, data));
  try {
    post({type: 'status', text: 'Loading packages…'});
    try {
      await pyodide.loadPackagesFromImports(msg.code + '\n' + (msg.tests || ''));
    } catch (e) {
      // Unknown or broken imports are reported by Python when the code runs.
    }
    if (msg.packages && msg.packages.length) await pyodide.loadPackage(msg.packages);
    post({type: 'started'});
    pyodide.globals.set('_d2l_code', msg.code);
    pyodide.globals.set('_d2l_tests', msg.tests || '');
    const json = pyodide.runPython(msg.mode === 'submit'
      ? 'grader.run_tests_json(_d2l_code, _d2l_tests)'
      : 'grader.run_code_json(_d2l_code)');
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
