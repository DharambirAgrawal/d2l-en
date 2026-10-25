"""Generate static/projects/battles.csv for the Chapter 2 project (Podium).

Head-to-head votes between 12 fictional AI chat models in September 2026:
a user sends one prompt, sees two anonymous answers (A on the left, B on
the right) and votes A, B, tie, or both_bad. Hidden truth: Bradley-Terry
strengths (logit scale), slightly different per category, plus a bias for
the answer shown first (A). Popular models are matched more often and
strong models are matched against each other more, so raw win rates are
biased. One model (nova-2) launches on Sept 22 and has few votes."""
import csv, os
import numpy as np

rng = np.random.default_rng(7)
MODELS = ['atlas-4', 'atlas-4-mini', 'nova-2', 'nova-1.5', 'sage-70b', 'sage-8b',
          'orbit-pro', 'orbit-flash', 'lumen-3', 'kite-7b', 'quartz-xl', 'ember-13b']
STRENGTH = np.array([1.20, 0.35, 1.25, 0.80, 0.55, -0.55, 1.00, 0.10, 0.75, -0.90, 0.30, -0.35])
CATS = ['coding', 'math', 'writing', 'chat', 'other']
CAT_P = [0.30, 0.12, 0.18, 0.30, 0.10]
CAT_ADJ = np.zeros((12, 5))
CAT_ADJ[MODELS.index('quartz-xl'), 0] = 0.6     # a coding specialist
CAT_ADJ[MODELS.index('quartz-xl'), 2] = -0.4
CAT_ADJ[MODELS.index('lumen-3'), 2] = 0.5       # a great writer
CAT_ADJ[MODELS.index('sage-70b'), 1] = 0.5      # good at math
POPULAR = np.array([3, 2, 3, 2, 1.5, 1, 2.5, 1.5, 1.5, 1, 1, 1], float)
POS_BIAS = 0.18                                  # extra logit for the left answer
LANGS = ['en', 'en', 'en', 'en', 'en', 'es', 'hi', 'zh', 'de', 'fr', 'pt', 'ja']

rows = []
bid = 0
for day in range(1, 31):
    n_today = rng.poisson(230)
    live = np.ones(12, bool)
    if day < 22:
        live[MODELS.index('nova-2')] = False
    for _ in range(n_today):
        p = POPULAR * live
        a = rng.choice(12, p=p / p.sum())
        # matchmaking prefers opponents of similar strength
        w = POPULAR * live * np.exp(-1.2 * np.abs(STRENGTH - STRENGTH[a]))
        w[a] = 0
        b = rng.choice(12, p=w / w.sum())
        c = rng.choice(5, p=CAT_P)
        d = STRENGTH[a] + CAT_ADJ[a, c] - STRENGTH[b] - CAT_ADJ[b, c] + POS_BIAS
        pa = 1 / (1 + np.exp(-d))
        u = rng.random()
        both = 0.12 if (STRENGTH[a] < 0 and STRENGTH[b] < 0) else 0.04
        if u < both:
            winner = 'both_bad'
        elif u < both + 0.10:
            winner = 'tie'
        else:
            winner = 'model_a' if rng.random() < pa else 'model_b'
        bid += 1
        tokens = int(np.clip(rng.lognormal(4.3, 1.0), 3, 6000))
        ma, mb = MODELS[a], MODELS[b]
        r = rng.random()
        if r < 0.02:  # logging inconsistencies
            ma = ma.upper()
        elif r < 0.035:
            mb = ' ' + mb + ' '
        if rng.random() < 0.025:
            winner = ''  # user closed the tab before voting
        hour = int(rng.integers(0, 24))
        ts = f'2026-09-{day:02d} {hour:02d}:{int(rng.integers(0, 60)):02d}'
        lang = LANGS[rng.integers(len(LANGS))]
        row = [f'b{bid:05d}', ts, ma, mb, CATS[c], lang if rng.random() > 0.03 else '', tokens, winner]
        rows.append(row)
        if rng.random() < 0.01:   # double submission
            rows.append(list(row))
rows.sort(key=lambda r: r[1])
with open(os.path.join(os.path.dirname(__file__), '../../static/projects/battles.csv'), 'w', newline='') as f:
    wr = csv.writer(f)
    wr.writerow(['battle_id', 'time', 'model_a', 'model_b', 'category', 'language', 'prompt_tokens', 'winner'])
    wr.writerows(rows)
print(len(rows), 'rows')
