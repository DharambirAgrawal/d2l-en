"""Validate the interactive exercises and bundle them for the website.

Each exercise is a folder under exercises/:

    exercises/<id>/
        exercise.yml   title, difficulty, description, explanation, ...
        starter.py     code the student starts from
        solution.py    reference solution
        tests.py       test_* functions (see static/exercises/grader.py)

Checks, with the same grader the browser uses:
  * the reference solution passes every test,
  * the starter code does not pass every test,
  * every exercise embedded in a chapter exists.

Usage:
    python tools/build_exercises.py                   # validate only
    python tools/build_exercises.py --out DIR         # validate, then write
                                                      # DIR/<id>.json + runtime files
"""
import argparse
import glob
import html
import json
import os
import re
import shutil
import subprocess
import sys

import mistune
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXERCISES_DIR = os.path.join(ROOT, 'exercises')
RUNTIME_DIR = os.path.join(ROOT, 'static', 'exercises')
RUNTIME_FILES = ['grader.py', 'worker.js']  # served next to the JSON files

ID_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
DIFFICULTIES = ('easy', 'medium', 'hard')
REQUIRED_FILES = ('exercise.yml', 'starter.py', 'solution.py', 'tests.py')
ALLOWED_KEYS = {'title', 'difficulty', 'description', 'explanation',
                'packages', 'timeout'}
EMBED_PATTERN = re.compile(r'data-exercise="([^"]+)"')

# Runs one exercise's tests in a separate process, so a hanging starter or
# solution can be timed out.
_CHECK_SCRIPT = """
import json, sys
sys.path.insert(0, sys.argv[1])
import grader
code, tests = json.load(sys.stdin)
print(grader.run_tests_json(code, tests))
"""


class ExerciseError(Exception):
    pass


def render_markdown(text):
    """Markdown to HTML. $...$ and $$...$$ become MathJax spans like the ones
    Sphinx emits, so the page's MathJax renders them."""
    math = []

    def stash(match):
        display = match.group(1) is not None
        body = match.group(1) if display else match.group(2)
        tex = (r'\[' + body + r'\]') if display else (r'\(' + body + r'\)')
        tag = 'div' if display else 'span'
        math.append(f'<{tag} class="math notranslate nohighlight">'
                    f'{html.escape(tex, quote=False)}</{tag}>')
        return f'@@MATH{len(math) - 1}@@'

    # Leave code spans/blocks alone: only replace math outside backticks.
    parts = re.split(r'(```.*?```|`[^`\n]*`)', text or '', flags=re.S)
    for i in range(0, len(parts), 2):
        parts[i] = re.sub(r'\$\$(.+?)\$\$|\$([^$\n]+?)\$', stash, parts[i], flags=re.S)
    rendered = mistune.html(''.join(parts))
    return re.sub(r'@@MATH(\d+)@@', lambda m: math[int(m.group(1))], rendered)


def run_tests(code, tests, timeout):
    try:
        proc = subprocess.run(
            [sys.executable, '-I', '-c', _CHECK_SCRIPT, RUNTIME_DIR],
            input=json.dumps([code, tests]), capture_output=True, text=True,
            timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if proc.returncode != 0:
        raise ExerciseError('the grader crashed:\n' + proc.stderr)
    return json.loads(proc.stdout.strip().splitlines()[-1])


def describe_failures(result):
    if result['status'] != 'ok':
        return result['error'] or 'could not be loaded'
    lines = []
    for t in result['tests']:
        if not t['passed']:
            lines.append(f"  - {t['name']}: {t.get('message', '')}")
            for key in ('expected', 'actual'):
                if key in t:
                    lines.append(f'      {key}: {t[key]}')
            if t.get('error'):
                lines.append('      ' + t['error'].replace('\n', '\n      '))
    return '\n'.join(lines)


def load_exercise(folder):
    ex_id = os.path.basename(folder)
    if not ID_PATTERN.match(ex_id):
        raise ExerciseError('folder name must be lowercase words joined by "-"')
    missing = [f for f in REQUIRED_FILES if not os.path.isfile(os.path.join(folder, f))]
    if missing:
        raise ExerciseError('missing ' + ', '.join(missing))

    def read(name):
        with open(os.path.join(folder, name), encoding='utf-8') as f:
            return f.read()

    meta = yaml.safe_load(read('exercise.yml')) or {}
    unknown = set(meta) - ALLOWED_KEYS
    if unknown:
        raise ExerciseError('unknown keys in exercise.yml: ' + ', '.join(sorted(unknown)))
    for key in ('title', 'description'):
        if not meta.get(key):
            raise ExerciseError(f'exercise.yml needs a "{key}"')
    difficulty = meta.get('difficulty', 'easy')
    if difficulty not in DIFFICULTIES:
        raise ExerciseError(f'difficulty must be one of {", ".join(DIFFICULTIES)}')
    timeout = meta.get('timeout', 10)
    if not isinstance(timeout, (int, float)) or not 0 < timeout <= 120:
        raise ExerciseError('timeout must be a number of seconds between 1 and 120')
    packages = meta.get('packages') or []
    if not isinstance(packages, list):
        raise ExerciseError('packages must be a list')

    return {
        'id': ex_id,
        'title': str(meta['title']),
        'difficulty': difficulty,
        'description_html': render_markdown(meta['description']),
        'explanation_html': render_markdown(meta.get('explanation', '')),
        'packages': packages,
        'timeout': timeout,
        'starter': read('starter.py'),
        'solution': read('solution.py'),
        'tests': read('tests.py'),
    }


def validate(ex):
    # The browser's timeout covers running only; allow extra time here for
    # importing packages such as NumPy.
    timeout = ex['timeout'] + 30
    result = run_tests(ex['solution'], ex['tests'], timeout)
    if result is None:
        raise ExerciseError(f'solution.py timed out after {timeout}s')
    if result['status'] != 'ok' or result['total'] == 0 or result['passed'] != result['total']:
        if result['status'] == 'ok' and result['total'] == 0:
            raise ExerciseError('tests.py defines no test_* functions')
        raise ExerciseError('solution.py fails its tests:\n' + describe_failures(result))
    starter = run_tests(ex['starter'], ex['tests'], timeout)
    if starter is not None and starter['status'] == 'ok' and starter['passed'] == starter['total']:
        raise ExerciseError('starter.py already passes every test')
    return result['total']


def embedded_ids():
    """Exercise ids referenced from chapter Markdown files."""
    found = {}
    for path in glob.glob(os.path.join(ROOT, '**', '*.md'), recursive=True):
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(('_build', '.')):
            continue
        with open(path, encoding='utf-8') as f:
            for ex_id in EMBED_PATTERN.findall(f.read()):
                found.setdefault(ex_id, []).append(rel)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', help='directory to write the bundled exercises to')
    args = parser.parse_args()

    folders = sorted(p for p in glob.glob(os.path.join(EXERCISES_DIR, '*')) if os.path.isdir(p))
    exercises, errors = [], []
    for folder in folders:
        name = os.path.basename(folder)
        try:
            ex = load_exercise(folder)
            n = validate(ex)
            exercises.append(ex)
            print(f'  ok  {name} ({n} tests)')
        except ExerciseError as e:
            errors.append(f'{name}: {e}')
            print(f'  FAIL {name}')

    known = {ex['id'] for ex in exercises} | {os.path.basename(f) for f in folders}
    for ex_id, files in embedded_ids().items():
        if ex_id not in known:
            errors.append(f'{", ".join(files)}: embeds unknown exercise "{ex_id}"')

    if errors:
        print('\nExercise check failed:\n\n' + '\n\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    print(f'{len(exercises)} exercises OK')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for ex in exercises:
            with open(os.path.join(args.out, ex['id'] + '.json'), 'w', encoding='utf-8') as f:
                json.dump(ex, f, ensure_ascii=False)
        for name in RUNTIME_FILES:
            shutil.copy(os.path.join(RUNTIME_DIR, name), args.out)
        print(f'Wrote {len(exercises)} exercises to {args.out}')


if __name__ == '__main__':
    main()
