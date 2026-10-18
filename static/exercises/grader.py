"""Grading harness for the interactive coding exercises.

The same file runs in two places, so both grade identically:

* in the browser, inside the Pyodide Web Worker (``worker.js``), and
* at build time, in ``tools/build_exercises.py``, which checks that every
  reference solution passes its tests and every starter does not.

Keep it compatible with Python 3.10 (Render's build Python).

Exercise tests are plain Python. A ``tests.py`` looks like::

    from submission import vector_add
    from grader import expect_equal, expect_close, expect_raises

    def test_example():
        \"\"\"[1, 2] + [3, 4] = [4, 6]\"\"\"
        expect_equal(vector_add([1, 2], [3, 4]), [4, 6])

``submission`` is a fresh module built from the student's code on every run.
Every function whose name starts with ``test_`` is a test, run in file order;
its docstring's first line is shown as the description.
"""
import ast
import contextlib
import io
import json
import linecache
import math
import sys
import traceback
import types

USER_FILE = 'your_code.py'
TESTS_FILE = 'tests.py'
MAX_OUTPUT = 20000  # characters of stdout/stderr kept per run
MAX_REPR = 400  # characters of expected/actual values shown

BROWSER_PACKAGES = 'the standard library, NumPy, SciPy, pandas, scikit-learn and matplotlib'

# Source of the code being graded, for tests that inspect it (see
# ``uses_any``).
SOURCE = ''

_MISSING = object()


class _Text(str):
    """A description (such as a type name) shown as-is, without quotes."""

    def __repr__(self):
        return str(self)


class ExpectationFailed(AssertionError):
    """A failed check, carrying the expected and actual values to display."""

    def __init__(self, message, expected=_MISSING, actual=_MISSING):
        super().__init__(message)
        self.expected = expected
        self.actual = actual


def _short_repr(value):
    try:
        text = repr(value)
    except Exception as e:  # a broken __repr__ in student code
        text = f'<unprintable {type(value).__name__}: {e}>'
    if len(text) > MAX_REPR:
        text = text[:MAX_REPR] + ' …'
    return text


def expect_equal(actual, expected, message=None):
    """Check ``actual == expected`` exactly (use expect_close for floats)."""
    try:
        equal = actual == expected
        if not isinstance(equal, bool):
            # e.g. NumPy arrays compare element-wise.
            raise TypeError
    except Exception:
        raise ExpectationFailed(
            message or 'Values cannot be compared with == '
            '(for arrays, use expect_close).', expected, actual) from None
    if not equal:
        raise ExpectationFailed(message or 'Values differ.', expected, actual)


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _close(actual, expected, rtol, atol):
    """Return None if close, else a reason string. Handles nested
    lists/tuples and NumPy arrays (if NumPy is installed)."""
    np = sys.modules.get('numpy')
    if np is not None and (isinstance(actual, np.ndarray)
                           or isinstance(expected, np.ndarray)):
        a = np.asarray(actual)
        e = np.asarray(expected)
        if a.shape != e.shape:
            return f'Shape {a.shape} does not match expected shape {e.shape}.'
        try:
            ok = np.allclose(a, e, rtol=rtol, atol=atol, equal_nan=True)
        except TypeError:
            return 'Values are not numeric.'
        return None if ok else 'Values are not close enough.'
    if np is not None and isinstance(actual, np.generic):
        actual = actual.item()
    if np is not None and isinstance(expected, np.generic):
        expected = expected.item()
    if _is_number(expected):
        if not _is_number(actual):
            return f'Expected a number, got {type(actual).__name__}.'
        if math.isnan(expected) and math.isnan(actual):
            return None
        if math.isclose(actual, expected, rel_tol=rtol, abs_tol=atol):
            return None
        return 'Values are not close enough.'
    if isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)):
            return f'Expected a {type(expected).__name__}, got {type(actual).__name__}.'
        if len(actual) != len(expected):
            return f'Length {len(actual)} does not match expected length {len(expected)}.'
        for i, (a, e) in enumerate(zip(actual, expected)):
            reason = _close(a, e, rtol, atol)
            if reason:
                return f'At index {i}: {reason}'
        return None
    return None if actual == expected else 'Values differ.'


def expect_close(actual, expected, rtol=1e-6, atol=1e-8, message=None):
    """Check numbers, (nested) lists or arrays are equal within tolerance:
    ``|actual - expected| <= atol + rtol * |expected|``."""
    reason = _close(actual, expected, rtol, atol)
    if reason:
        raise ExpectationFailed(message or reason, expected, actual)


def expect_true(condition, message):
    if not condition:
        raise ExpectationFailed(message)


def expect_type(value, expected_type, message=None):
    if not isinstance(value, expected_type):
        name = getattr(expected_type, '__name__', str(expected_type))
        raise ExpectationFailed(
            message or f'Expected type {name}, got {type(value).__name__}.',
            _Text(name), _Text(type(value).__name__))


def expect_shape(array, shape, message=None):
    actual = getattr(array, 'shape', _MISSING)
    if actual is _MISSING:
        raise ExpectationFailed(
            message or f'Expected an array, got {type(array).__name__}.',
            tuple(shape), _Text(type(array).__name__))
    if tuple(actual) != tuple(shape):
        raise ExpectationFailed(message or 'Wrong shape.', tuple(shape), tuple(actual))


def expect_raises(exception_type, func, *args, **kwargs):
    """Check that ``func(*args, **kwargs)`` raises ``exception_type``."""
    name = exception_type.__name__
    try:
        result = func(*args, **kwargs)
    except exception_type:
        return
    except Exception as e:
        raise ExpectationFailed(
            f'Expected {name}, but got {type(e).__name__}: {e}', _Text(name),
            _Text(type(e).__name__)) from None
    raise ExpectationFailed(
        f'Expected {name} to be raised, but the call returned normally.',
        _Text(name), _Text(f'returned {_short_repr(result)}'))


def uses_any(*names):
    """Return the names (functions, attributes, or ``'@'``) the graded code
    uses, out of ``names``. For "don't use np.dot"-style rules."""
    try:
        tree = ast.parse(SOURCE)
    except SyntaxError:
        return []
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.MatMult):
            found.add('@')
        elif isinstance(node, ast.Attribute):
            found.add(node.attr)
        elif isinstance(node, ast.Name):
            found.add(node.id)
    return [n for n in names if n in found]


class _CappedWriter(io.TextIOBase):
    """Collects output, dropping anything past MAX_OUTPUT characters so an
    endless print loop can't exhaust memory before the timeout hits."""

    def __init__(self):
        self.parts = []
        self.size = 0
        self.truncated = False

    def writable(self):
        return True

    def write(self, text):
        if self.size < MAX_OUTPUT:
            keep = text[:MAX_OUTPUT - self.size]
            self.parts.append(keep)
            self.size += len(keep)
            if len(keep) < len(text):
                self.truncated = True
        elif text:
            self.truncated = True
        return len(text)

    def getvalue(self):
        out = ''.join(self.parts)
        if self.truncated:
            out += f'\n… output truncated after {MAX_OUTPUT} characters'
        return out


def _register_source(filename, source):
    # Lets tracebacks show the offending source lines.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)


def _format_error(exc, show_files=(USER_FILE,)):
    """Format an exception like Python does, keeping only frames from the
    given files (hiding grader internals)."""
    te = traceback.TracebackException(type(exc), exc, exc.__traceback__)
    frames = [f for f in te.stack if f.filename in show_files]
    lines = []
    if frames:
        lines.append('Traceback (most recent call last):\n')
        lines.extend(traceback.StackSummary.from_list(frames).format())
    lines.extend(te.format_exception_only())
    text = ''.join(lines).rstrip()
    if isinstance(exc, EOFError):
        text += ('\n\ninput() is not supported in exercises. '
                 'Put your values in variables instead.')
    if isinstance(exc, ModuleNotFoundError):
        name = (exc.name or '').split('.')[0]
        if name in ('torch', 'tensorflow', 'jax', 'mxnet'):
            text += (f'\n\n{name} cannot run in the browser. Browser exercises '
                     f'can use {BROWSER_PACKAGES}.')
        else:
            text += (f'\n\nThis package is not available in the browser. '
                     f'Browser exercises can use {BROWSER_PACKAGES}.')
    return text


def _load_module(name, source, filename):
    _register_source(filename, source)
    code = compile(source, filename, 'exec')
    module = types.ModuleType(name)
    module.__file__ = filename
    sys.modules[name] = module
    exec(code, module.__dict__)
    return module


@contextlib.contextmanager
def _captured():
    out = _CappedWriter()
    old_stdin = sys.stdin
    sys.stdin = io.StringIO('')  # input() raises EOFError instead of hanging
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            yield out
    finally:
        sys.stdin = old_stdin


def _cleanup():
    for name in ('submission', 'tests'):
        sys.modules.pop(name, None)


def run_code(code):
    """Run code as a script (the "Run" button). Returns a result dict."""
    global SOURCE
    SOURCE = code
    _cleanup()
    with _captured() as out:
        try:
            _load_module('submission', code, USER_FILE)
            error = None
        except BaseException as e:  # SystemExit, KeyboardInterrupt too
            error = _format_error(e)
    _cleanup()
    return {'status': 'error' if error else 'ok', 'stdout': out.getvalue(),
            'error': error}


def run_tests(code, tests_source):
    """Load code as the ``submission`` module, then run each test_* function
    in ``tests_source`` (the "Submit" button). Returns a result dict."""
    global SOURCE
    SOURCE = code
    _cleanup()
    results = []
    error = None
    with _captured() as out:
        try:
            _load_module('submission', code, USER_FILE)
        except BaseException as e:
            error = _format_error(e)
        if error is None:
            try:
                tests = _load_module('tests', tests_source, TESTS_FILE)
            except ImportError as e:
                if e.name == 'submission':
                    missing = str(e).split("'")[1] if "'" in str(e) else ''
                    error = (f'Your code must define `{missing}`.'
                             if missing else str(e))
                else:
                    error = _format_error(e, (USER_FILE, TESTS_FILE))
            except BaseException as e:
                error = _format_error(e, (USER_FILE, TESTS_FILE))
        if error is None:
            test_funcs = [(n, f) for n, f in vars(tests).items()
                          if n.startswith('test_') and callable(f)]
            for name, func in test_funcs:
                doc = (func.__doc__ or '').strip().splitlines()
                result = {'name': name, 'description': doc[0] if doc else '',
                          'passed': False}
                try:
                    func()
                    result['passed'] = True
                except ExpectationFailed as e:
                    result['message'] = str(e)
                    if e.expected is not _MISSING:
                        result['expected'] = _short_repr(e.expected)
                    if e.actual is not _MISSING:
                        result['actual'] = _short_repr(e.actual)
                except AssertionError as e:
                    result['message'] = str(e) or 'Assertion failed.'
                    result['error'] = _format_error(e, (USER_FILE, TESTS_FILE))
                except BaseException as e:
                    result['message'] = f'Your code raised {type(e).__name__}.'
                    result['error'] = _format_error(e, (USER_FILE, TESTS_FILE))
                results.append(result)
    _cleanup()
    passed = sum(r['passed'] for r in results)
    return {'status': 'error' if error else 'ok', 'stdout': out.getvalue(),
            'error': error, 'tests': results, 'passed': passed,
            'total': len(results)}


def run_code_json(code):
    return json.dumps(run_code(code))


def run_tests_json(code, tests_source):
    return json.dumps(run_tests(code, tests_source))
