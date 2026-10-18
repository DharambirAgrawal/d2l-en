"""Validate the interactive exercises and answers, and bundle them for the website.

Each exercise is a folder under exercises/:

    exercises/<id>/
        exercise.yml   title, difficulty, description, explanation, ...
        starter.py     code the student starts from
        solution.py    reference solution
        tests.py       test_* functions (see static/exercises/grader.py)
        files/         optional data files (CSV, text, helper .py modules),
                       placed in the working directory before code runs

Answers to the numbered "## Exercises" list at the end of a section live in
answers/<chapter>/<section>.yml, one entry per exercise, in order:

    - answer: |              # Markdown, shown behind a "Show answer" button
        ...
    - exercise: tensor-len   # optional: an interactive exercise shown under
      answer: |              # the question
        ...

Checks, with the same grader the browser uses:
  * the reference solution passes every test,
  * the starter code does not pass every test,
  * every exercise embedded in a chapter or answers file exists,
  * every answers file has exactly one entry per exercise in its section.

Usage:
    python tools/build_exercises.py                   # validate only
    python tools/build_exercises.py --out DIR         # validate, then write
                                                      # DIR/<id>.json + runtime files,
                                                      # and DIR/../answers/<page>.json
"""
import argparse
import configparser
import fnmatch
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXERCISES_DIR = os.path.join(ROOT, 'exercises')
ANSWERS_DIR = os.path.join(ROOT, 'answers')
PLAYGROUNDS_DIR = os.path.join(ROOT, 'playgrounds')
NOTES_DIR = os.path.join(ROOT, 'notes')
RUNTIME_DIR = os.path.join(ROOT, 'static', 'exercises')
RUNTIME_FILES = ['grader.py', 'worker.js', 'notebook.py', 'torchlite.py', 'd2llite.py']

ID_PATTERN = re.compile(r'^[a-z0-9]+(-[a-z0-9]+)*$')
DIFFICULTIES = ('easy', 'medium', 'hard')
REQUIRED_FILES = ('exercise.yml', 'starter.py', 'solution.py', 'tests.py')
ALLOWED_KEYS = {'title', 'difficulty', 'description', 'explanation',
                'packages', 'timeout'}
EMBED_PATTERN = re.compile(r'data-exercise="([^"]+)"')
PLAYGROUND_EMBED = re.compile(r'data-playground="([^"]+)"')
PLAYGROUND_KEYS = {'title', 'description', 'params', 'python', 'timeout'}
PLAYGROUND_JS = os.path.join(RUNTIME_DIR, 'playgrounds.js')

# Runs a playground's code once with its default parameters.
_PLAYGROUND_SCRIPT = """
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
os.chdir(tempfile.mkdtemp())
import notebook
code, params = json.load(sys.stdin)
r = notebook.run_cell('check', code, params)
print(json.dumps({'status': r['status'], 'error': r['error'],
                  'outputs': len(r['images']) + bool(r['stdout']) + bool(r['result'])}))
"""

# Runs one exercise's tests in a separate process, so a hanging starter or
# solution can be timed out.
_CHECK_SCRIPT = """
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[1])
import grader
code, tests, files = json.load(sys.stdin)
workdir = tempfile.mkdtemp()
for name, text in files.items():
    with open(os.path.join(workdir, name), 'w', encoding='utf-8') as f:
        f.write(text)
os.chdir(workdir)
sys.path.insert(0, workdir)
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


def run_tests(code, tests, timeout, files=None):
    try:
        proc = subprocess.run(
            [sys.executable, '-I', '-c', _CHECK_SCRIPT, RUNTIME_DIR],
            input=json.dumps([code, tests, files or {}]), capture_output=True, text=True,
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


MAX_FILES_BYTES = 2_000_000


def read_files(folder):
    """Text files in <folder>/files/, keyed by file name."""
    files, total = {}, 0
    for path in sorted(glob.glob(os.path.join(folder, 'files', '*'))):
        if not os.path.isfile(path):
            continue
        name = os.path.basename(path)
        if not re.match(r'^[A-Za-z0-9_.-]+$', name):
            raise ExerciseError(f'files/{name}: use letters, digits, "_", "-" and "." only')
        with open(path, encoding='utf-8') as f:
            files[name] = f.read()
        total += len(files[name])
    if total > MAX_FILES_BYTES:
        raise ExerciseError(f'files/ is {total} bytes; keep it under {MAX_FILES_BYTES}')
    return files


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

    try:
        meta = yaml.safe_load(read('exercise.yml')) or {}
    except yaml.YAMLError as e:
        raise ExerciseError(f'exercise.yml is not valid YAML (quote titles containing ":"):\n{e}')
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
        'files': read_files(folder),
    }


def validate(ex):
    # The browser's timeout covers running only; allow extra time here for
    # importing packages such as NumPy.
    timeout = ex['timeout'] + 30
    result = run_tests(ex['solution'], ex['tests'], timeout, ex['files'])
    if result is None:
        raise ExerciseError(f'solution.py timed out after {timeout}s')
    if result['status'] != 'ok' or result['total'] == 0 or result['passed'] != result['total']:
        if result['status'] == 'ok' and result['total'] == 0:
            raise ExerciseError('tests.py defines no test_* functions')
        raise ExerciseError('solution.py fails its tests:\n' + describe_failures(result))
    starter = run_tests(ex['starter'], ex['tests'], timeout, ex['files'])
    if starter is not None and starter['status'] == 'ok' and starter['passed'] == starter['total']:
        raise ExerciseError('starter.py already passes every test')
    return result['total']


def book_exclusions():
    """The `exclusions` patterns from config.ini (files not in the book)."""
    config = configparser.ConfigParser(interpolation=None)
    config.read(os.path.join(ROOT, 'config.ini'), encoding='utf-8')
    return config.get('build', 'exclusions', fallback='').split()


def embedded_ids(pattern=EMBED_PATTERN):
    """Exercise (or playground) ids referenced from the book's Markdown files."""
    excluded = book_exclusions()
    found = {}
    for path in glob.glob(os.path.join(ROOT, '**', '*.md'), recursive=True):
        rel = os.path.relpath(path, ROOT)
        if rel.startswith(('_build', '.')) or any(
                fnmatch.fnmatch(rel, pattern) for pattern in excluded):
            continue
        with open(path, encoding='utf-8') as f:
            for ex_id in pattern.findall(f.read()):
                found.setdefault(ex_id, []).append(rel)
    return found


def count_section_exercises(md_path):
    """Number of top-level items in the numbered list under "## Exercises"."""
    count, inside = 0, False
    with open(md_path, encoding='utf-8') as f:
        for line in f:
            if line.startswith('## '):
                inside = line.strip() == '## Exercises'
            elif inside and re.match(r'\d+\. ', line):
                count += 1
    return count


def load_answers(known_ids):
    """Load answers/**/*.yml. Returns ({page: [entry, ...]}, errors)."""
    pages, errors = {}, []
    for path in sorted(glob.glob(os.path.join(ANSWERS_DIR, '**', '*.yml'), recursive=True)):
        rel = os.path.relpath(path, ANSWERS_DIR)
        page = rel[:-len('.yml')].replace(os.sep, '/')
        md = os.path.join(ROOT, page + '.md')
        if not os.path.isfile(md):
            errors.append(f'answers/{rel}: no matching {page}.md')
            continue
        with open(path, encoding='utf-8') as f:
            entries = yaml.safe_load(f) or []
        if not isinstance(entries, list):
            errors.append(f'answers/{rel}: must be a list with one entry per exercise')
            continue
        expected = count_section_exercises(md)
        if len(entries) != expected:
            errors.append(f'answers/{rel}: has {len(entries)} entries, but {page}.md '
                          f'has {expected} exercises')
            continue
        out = []
        for i, entry in enumerate(entries, 1):
            if not isinstance(entry, dict) or not set(entry) <= {'answer', 'exercise'}:
                errors.append(f'answers/{rel}: entry {i} may only have "answer" and "exercise"')
                continue
            ex_id = entry.get('exercise')
            if ex_id and ex_id not in known_ids:
                errors.append(f'answers/{rel}: entry {i} uses unknown exercise "{ex_id}"')
            out.append({'exercise': ex_id,
                        'answer_html': render_markdown(entry.get('answer', ''))})
        pages[page] = out
    return pages, errors


def load_playgrounds():
    """Load and run-check playgrounds/*.yml. Returns ({id: data}, errors)."""
    found, errors = {}, []
    for path in sorted(glob.glob(os.path.join(PLAYGROUNDS_DIR, '*.yml'))):
        pg_id = os.path.basename(path)[:-len('.yml')]
        where = f'playgrounds/{pg_id}.yml'
        with open(path, encoding='utf-8') as f:
            meta = yaml.safe_load(f) or {}
        if not ID_PATTERN.match(pg_id):
            errors.append(f'{where}: name must be lowercase words joined by "-"')
            continue
        if set(meta) - PLAYGROUND_KEYS or not {'title', 'params'} <= set(meta):
            errors.append(f'{where}: needs title and params (allowed: '
                          f'{", ".join(sorted(PLAYGROUND_KEYS))})')
            continue
        with open(PLAYGROUND_JS, encoding='utf-8') as f:
            if f"registry['{pg_id}']" not in f.read():
                errors.append(f"{where}: no drawing code registry['{pg_id}'] in static/exercises/playgrounds.js")
                continue
        params, defaults = [], {}
        for p in meta['params']:
            if not {'name', 'min', 'max', 'step', 'value'} <= set(p):
                errors.append(f'{where}: each param needs name, min, max, step, value')
                break
            if not p['min'] <= p['value'] <= p['max']:
                errors.append(f'{where}: param {p["name"]} default is outside [min, max]')
            label = p.get('label', p['name'])
            params.append({'name': p['name'], 'label': label,
                           'label_html': render_markdown(label).strip()[3:-4],
                           'min': p['min'], 'max': p['max'], 'step': p['step'],
                           'value': p['value']})
            defaults[p['name']] = p['value']
        if meta.get('python'):  # the optional Python version must run
            try:
                proc = subprocess.run(
                    [sys.executable, '-I', '-c', _PLAYGROUND_SCRIPT, RUNTIME_DIR],
                    input=json.dumps([meta['python'], defaults]), capture_output=True,
                    text=True, timeout=120)
                result = json.loads(proc.stdout.strip().splitlines()[-1])
            except Exception as e:
                errors.append(f'{where}: could not run the Python version: {e}')
                continue
            if result['status'] != 'ok':
                errors.append(f'{where}: Python version fails with default values:\n{result["error"]}')
            elif not result['outputs']:
                errors.append(f'{where}: Python version produces no output')
        found[pg_id] = {'id': pg_id, 'title': str(meta['title']),
                        'description_html': render_markdown(meta.get('description', '')),
                        'params': params, 'python': meta.get('python', ''),
                        'timeout': meta.get('timeout', 20)}
    return found, errors


def md_headings(md_path):
    """Normalized text of every '##'+ heading in a Markdown file."""
    with open(md_path, encoding='utf-8') as f:
        return {' '.join(m.split()) for m in re.findall(r'^#{2,}\s+(.+?)\s*$', f.read(), re.M)}


def load_notes():
    """notes/<chapter>/<section>.yml: [{after: <heading>, notes: [Markdown, ...]}].
    Each entry is shown at the end of the section with that heading."""
    pages, errors = {}, []
    for path in sorted(glob.glob(os.path.join(NOTES_DIR, '**', '*.yml'), recursive=True)):
        rel = os.path.relpath(path, NOTES_DIR)
        page = rel[:-len('.yml')].replace(os.sep, '/')
        md = os.path.join(ROOT, page + '.md')
        if not os.path.isfile(md):
            errors.append(f'notes/{rel}: no matching {page}.md')
            continue
        with open(path, encoding='utf-8') as f:
            entries = yaml.safe_load(f) or []
        headings = md_headings(md)
        sections = []
        for i, entry in enumerate(entries if isinstance(entries, list) else [None], 1):
            if (not isinstance(entry, dict) or set(entry) != {'after', 'notes'}
                    or not isinstance(entry['notes'], list)):
                errors.append(f'notes/{rel}: entry {i} must have "after" (a heading) and "notes" (a list)')
                continue
            heading = ' '.join(str(entry['after']).split())
            if heading not in headings:
                errors.append(f'notes/{rel}: entry {i}: no section "{heading}" in {page}.md')
                continue
            items = [str(n) for n in entry['notes']]
            sections.append({
                'heading': heading,
                'html': '<ul>' + ''.join(f'<li>{render_markdown(n).strip()}</li>' for n in items) + '</ul>',
                'text': '\n'.join('- ' + ' '.join(n.split()) for n in items),
            })
        pages[page] = {'sections': sections}
    return pages, errors


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

    answers, answer_errors = load_answers(known)
    errors += answer_errors
    playgrounds, pg_errors = load_playgrounds()
    errors += pg_errors
    for pg_id, files in embedded_ids(PLAYGROUND_EMBED).items():
        if pg_id not in playgrounds:
            errors.append(f'{", ".join(files)}: embeds unknown playground "{pg_id}"')
    notes, note_errors = load_notes()
    errors += note_errors

    if errors:
        print('\nExercise check failed:\n\n' + '\n\n'.join(errors), file=sys.stderr)
        sys.exit(1)
    print(f'{len(exercises)} exercises, answers for {len(answers)} sections, '
          f'{len(playgrounds)} playgrounds, notes for {len(notes)} sections: OK')

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        for ex in exercises:
            with open(os.path.join(args.out, ex['id'] + '.json'), 'w', encoding='utf-8') as f:
                json.dump(ex, f, ensure_ascii=False)
        for name in RUNTIME_FILES:
            shutil.copy(os.path.join(RUNTIME_DIR, name), args.out)
        # The book's d2l library (its #@save code) for runnable cells.
        import build_d2l_book
        build_d2l_book.generate(os.path.join(args.out, 'd2l_book.py'))
        answers_out = os.path.join(os.path.dirname(os.path.abspath(args.out)), 'answers')
        for page, entries in answers.items():
            path = os.path.join(answers_out, page + '.json')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump({'answers': entries}, f, ensure_ascii=False)
        static_out = os.path.dirname(os.path.abspath(args.out))
        for pg_id, data in playgrounds.items():
            os.makedirs(os.path.join(static_out, 'playgrounds'), exist_ok=True)
            with open(os.path.join(static_out, 'playgrounds', pg_id + '.json'), 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        for page, data in notes.items():
            path = os.path.join(static_out, 'notes', page + '.json')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False)
        # Index for the "My Notes" review page: which pages have notes.
        os.makedirs(os.path.join(static_out, 'notes'), exist_ok=True)
        with open(os.path.join(static_out, 'notes', 'index.json'), 'w', encoding='utf-8') as f:
            json.dump({'pages': sorted(notes)}, f)
        print(f'Wrote {len(exercises)} exercises, answers for {len(answers)} sections, '
              f'{len(playgrounds)} playgrounds, notes for {len(notes)} sections')


if __name__ == '__main__':
    main()
