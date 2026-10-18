#!/usr/bin/env bash
set -euo pipefail

# Use a modern pip so legacy packages build in an isolated environment
# (setuptools<81 still ships pkg_resources, which notedown imports)
python -m pip install --upgrade pip "setuptools<81" wheel

# pandoc-attributes' setup.py calls pypandoc.convert(), which no longer
# exists in current pypandoc. Install it before pypandoc so the build can't
# import pypandoc, and fall back to an empty long_description.
python -m pip install --use-pep517 pandoc-attributes

# Install the D2L book builder
# (pybtex 0.25+ removed _FakeEntryPoint, which sphinxcontrib-bibtex 2.4 imports)
python -m pip install \
  "git+https://github.com/d2l-ai/d2l-book.git" \
  pypandoc_binary \
  "pybtex==0.24.0" \
  numpy

# Check the interactive exercises first (fails fast if one is broken)
python tools/build_exercises.py

# Make Pandoc available to the book builder
PANDOC_PATH="$(python -c 'import pypandoc; print(pypandoc.get_pandoc_path())')"
export PATH="$(dirname "$PANDOC_PATH"):$PATH"

# Fill in the navbar link placeholders in config.ini
sed -i.bak \
  -e "s@###_ALTERNATE_VERSION_###@Stable Version@g" \
  -e "s@###_ALTERNATE_VERSION_BASE_LINK_###@https://d2l.ai@g" \
  -e "s@###_CURRENT_VERSION_BASE_LINK_###@https://d2l.ai@g" \
  config.ini
rm -f config.ini.bak

# Start from a clean output folder (Render may cache _build between deploys)
rm -rf _build

# Convert each framework's Markdown to notebooks; "rst --tab all" only
# merges these, it does not create them. eval_notebook = False in config.ini,
# so no code is run.
d2lbook build eval --tab pytorch
d2lbook build eval --tab jax
d2lbook build eval --tab tensorflow

# Convert the book source
d2lbook build rst --tab all

# Include the book's homepage
cp static/frontpage/frontpage.html _build/rst_all/

# Generate the HTML website
d2lbook build html --tab all

# Include homepage images
mkdir -p _build/html/_images
cp -r static/frontpage/_images/. _build/html/_images/

# Add the interactive exercises (JSON + Python runner files)
python tools/build_exercises.py --out _build/html/_static/exercises
