"""Run every PyTorch/shared code cell of book pages through the browser
runtime (static/exercises/notebook.py + torchlite + the generated d2l_book),
in plain CPython, and report failures.

Usage: python run_book_cells.py D2L_BOOK_DIR page.md [page.md ...]
  D2L_BOOK_DIR: a folder containing d2l_book.py (tools/build_d2l_book.py output)
"""
import os
import re
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, 'static', 'exercises'))
sys.path.insert(0, os.path.abspath(sys.argv[1]))
import notebook  # noqa: E402


def cells(md):
    out = []
    with open(md, encoding='utf-8') as f:
        text = f.read()
    for c in re.findall(r'```\{\.python \.input[^}]*\}\n(.*?)```', text, re.S):
        lines = c.strip('\n').split('\n')
        head = lines[0].strip()
        if 'interact_select' in c:
            continue
        if head.startswith(('%%tab', '#@tab')):
            tabs = head.split(None, 1)[1] if ' ' in head else ''
            if 'pytorch' not in tabs and 'all' not in tabs:
                continue
            body = '\n'.join(lines[1:])
        else:
            body = '\n'.join(lines)
        out.append(body)
    return out


def main():
    failed = 0
    total = 0
    pages = sys.argv[2:]
    work = tempfile.mkdtemp()
    os.makedirs(os.path.join(work, 'sub'))
    for page in pages:
        os.chdir(os.path.join(work, 'sub'))  # book cells write to ../data
        cs = cells(os.path.join(REPO, page))
        print(f'== {page}: {len(cs)} cells', flush=True)
        for i, code in enumerate(cs):
            total += 1
            r = notebook.run_cell(page, code)
            if r['status'] != 'ok':
                failed += 1
                print(f'--- cell {i} FAILED:\n{code}\n>>> {r["error"]}\n', flush=True)
    print(f'{total - failed}/{total} book cells passed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
