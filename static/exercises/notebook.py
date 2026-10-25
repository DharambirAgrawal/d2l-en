"""Run the book's code cells in the browser, notebook style.

Each page gets one session (a namespace shared by its cells, like a Jupyter
kernel). `torch` is torchlite (a NumPy-based PyTorch subset) and `d2l` is
d2llite while a cell runs; both are removed again afterwards so graded
exercises never see them. Also used for interactive playgrounds.

Keep compatible with Python 3.10 (the build check runs it there too).
"""
import ast
import base64
import io
import json
import os
import re
import sys

os.environ.setdefault('MPLBACKEND', 'Agg')

import grader  # noqa: E402  (shared output capture and error formatting)

CELL_FILE = 'cell.py'
SESSIONS = {}

_MAGIC = re.compile(r'^\s*[%!].*$', re.M)


def _install_shims(needs_plotting):
    import torchlite
    torchlite.install()
    if needs_plotting:
        import d2llite
        d2llite.install()


def _uninstall_shims():
    for name in list(sys.modules):
        if name in ('torch', 'torchvision', 'd2l') or name.startswith(('torch.', 'torchvision.', 'd2l.')):
            del sys.modules[name]


def _figures():
    if 'matplotlib.pyplot' not in sys.modules:
        return []
    plt = sys.modules['matplotlib.pyplot']
    images = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        images.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    plt.close('all')
    return images


def _execute(code, ns):
    """Exec code; if the last statement is an expression, return its value."""
    tree = ast.parse(code, CELL_FILE)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, CELL_FILE, 'exec'), ns)
    if last is not None:
        return eval(compile(last, CELL_FILE, 'eval'), ns)
    return None


class _Tab:
    """d2lbook's `tab` helper: cells shared by several frameworks branch on
    tab.selected(...). The browser runs the PyTorch version."""

    @staticmethod
    def selected(*names):
        return 'pytorch' in names

    @staticmethod
    def interact_select(*args, **kwargs):
        pass


# Chapters whose training cells are light enough to use all of Fashion-MNIST
# in the browser; later chapters train on a subset (see torchlite).
_FULL_DATA_CHAPTERS = ('chapter_preliminaries', 'chapter_linear-regression', 'chapter_linear-classification')


def _set_train_limit(session_id):
    import torchlite
    limit = os.environ.get('D2L_TRAIN_LIMIT')
    if limit is None and sys.platform == 'emscripten' and 'chapter_' in session_id \
            and not any(c in session_id for c in _FULL_DATA_CHAPTERS):
        limit = 10000
    torchlite.FASHION_MNIST_TRAIN_LIMIT = int(limit) if limit else None


def _pandas_compat():
    """The browser's pandas (3.x) stores text as a new "str" dtype; the book
    (pandas 2) expects "object", e.g. `dtypes[dtypes != 'object']` to pick the
    numeric columns. Keep the old behavior."""
    pd = sys.modules.get('pandas')
    if pd is None:
        return
    try:
        pd.set_option('future.infer_string', False)
    except Exception:
        pass


def run_cell(session_id, code, params=None):
    """Run a cell in a session. Returns a result dict."""
    ns = SESSIONS.setdefault(session_id, {'__name__': '__main__', 'tab': _Tab()})
    if params:
        ns.update(params)
    clean = _MAGIC.sub('', code)
    _set_train_limit(session_id)
    _pandas_compat()
    grader._register_source(CELL_FILE, clean)
    needs_plotting = bool(re.search(r'\bd2l\b|matplotlib|plt\.', clean))
    value = None
    error = None
    with grader._captured() as out:
        try:
            _install_shims(needs_plotting)
            value = _execute(clean, ns)
        except BaseException as e:
            error = grader._format_error(e, (CELL_FILE,))
        finally:
            _uninstall_shims()
    result = None
    if clean.rstrip().endswith(';'):  # like Jupyter, ";" hides the value
        value = None
    if value is not None and error is None:
        try:
            result = repr(value)
        except Exception as e:
            result = f'<unprintable {type(value).__name__}: {e}>'
        if len(result) > grader.MAX_OUTPUT:
            result = result[:grader.MAX_OUTPUT] + ' …'
    return {'status': 'error' if error else 'ok', 'stdout': out.getvalue(),
            'result': result, 'images': _figures(), 'error': error}


def reset(session_id):
    SESSIONS.pop(session_id, None)


def run_cell_json(session_id, code, params_json='null'):
    return json.dumps(run_cell(session_id, code, json.loads(params_json)))
