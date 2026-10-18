"""Remove Google Analytics from the built site's _static/d2l.js.

d2lbook's default config defines `google_analytics_tracking_id` (empty), so it
always adds its Google Analytics loader. The framework tabs and Colab buttons
call two of the loader's helpers (`tagClick`, `captureOutboundLink`), so we
replace the block with no-op versions instead of deleting it.

Usage: python tools/strip_analytics.py _build/html/_static/d2l.js
"""
import re
import sys

STUB = '''// Analytics removed (tools/strip_analytics.py): no requests to Google.
var ga = function () {};
var captureOutboundLink = function (url) { document.location = url; };
var tagClick = function (tab) {};
'''

path = sys.argv[1]
with open(path, encoding='utf-8') as f:
    js = f.read()
pattern = re.compile(
    r"\(function\(i,s,o,g,r,a,m\)\{i\['GoogleAnalyticsObject'\].*?"
    r"var tagClick = function\(tab\) \{.*?\n\}\n", re.S)
js, n = pattern.subn(STUB, js)
if 'google-analytics' in js or 'googletagmanager' in js:
    sys.exit(f'{path}: analytics code still present after stripping')
with open(path, 'w', encoding='utf-8') as f:
    f.write(js)
print(f'Removed analytics from {path}' if n else f'No analytics found in {path}')
