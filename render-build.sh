#!/usr/bin/env bash
set -euo pipefail

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
