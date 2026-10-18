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
python -m pip install \
  "git+https://github.com/d2l-ai/d2l-book.git" \
  pypandoc_binary

# Make Pandoc available to the book builder
export PATH="$(dirname "$(python -c \
  'import pypandoc; print(pypandoc.get_pandoc_path())')"):$PATH"

# Convert the book source
d2lbook build rst --tab all

# Include the book's homepage
cp static/frontpage/frontpage.html _build/rst_all/

# Generate the HTML website
d2lbook build html --tab all

# Include homepage images
mkdir -p _build/html/_images
cp -r static/frontpage/_images/. _build/html/_images/
