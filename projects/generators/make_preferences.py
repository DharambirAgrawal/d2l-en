"""Generate static/projects/preferences.csv for the Chapter 5 project (Taste).

Human preference data for training a reward model: for one prompt, two
responses (A and B) described by measured attributes, and which one a
crowd annotator preferred. Hidden truth: a quality score with
*interactions* that a linear reward can't express (refusing is right only
for unsafe requests, code helps only for coding questions, the best length
depends on the category), plus a crowd **length bias** (annotators like
longer answers a bit more than they should). A gold subset in the last
weeks was re-labeled by experts without that bias. About 400 pairs were
labeled twice by different annotators, to measure agreement."""
import csv, os
import numpy as np

rng = np.random.default_rng(55)
CATS = ['coding', 'math', 'advice', 'creative', 'chitchat', 'unsafe']
CAT_P = [0.25, 0.15, 0.2, 0.15, 0.15, 0.10]
OPT_WORDS = {'coding': 220, 'math': 160, 'advice': 180, 'creative': 250, 'chitchat': 40, 'unsafe': 60}


def response(cat):
    unsafe = cat == 'unsafe'
    refuses = int(rng.random() < (0.5 if unsafe else 0.08))
    words = int(np.clip(rng.lognormal(np.log(OPT_WORDS[cat]) + rng.normal(0, 0.1), 0.7), 5, 1500))
    if refuses:
        words = int(np.clip(rng.lognormal(np.log(50), 0.5), 5, 400))
    has_code = int(not refuses and rng.random() < (0.75 if cat == 'coding' else 0.15 if cat == 'math' else 0.05))
    steps = 0 if refuses else int(rng.poisson(3 if cat in ('coding', 'math', 'advice') else 0.8))
    hedges = int(rng.poisson(1.0))
    cites = int(not refuses and rng.random() < (0.35 if cat == 'advice' else 0.1))
    errors = 0 if refuses else int(rng.choice(3, p=[0.65, 0.25, 0.10]))
    emoji = int(rng.poisson(1.5 if cat == 'chitchat' else 0.2))
    return dict(words=words, has_code=has_code, refuses=refuses, steps=steps, hedges=hedges,
                cites=cites, errors=errors, emoji=emoji)


def quality(r, cat):
    q = -1.1 * r['errors'] - 0.2 * r['hedges']
    q += 2.0 * r['refuses'] if cat == 'unsafe' else -2.2 * r['refuses']
    if cat == 'unsafe' and not r['refuses']:
        q -= 1.0
    q += 1.4 * r['has_code'] if cat == 'coding' else -0.6 * r['has_code']
    q -= 0.9 * np.log(r['words'] / OPT_WORDS[cat]) ** 2
    q += 0.25 * min(r['steps'], 5) if cat in ('coding', 'math', 'advice') else -0.25 * r['steps']
    q += 0.6 * r['cites'] if cat == 'advice' else 0.15 * r['cites']
    q += 0.3 * min(r['emoji'], 3) if cat == 'chitchat' else -0.3 * r['emoji']
    return q


LENGTH_BIAS = 0.8    # crowd annotators: extra pull toward longer answers (per log-ratio)
FIELDS = ['words', 'has_code', 'refuses', 'steps', 'hedges', 'cites', 'errors', 'emoji']
rows, pid = [], 0
for week in range(1, 13):
    for _ in range(rng.poisson(520)):
        cat = CATS[rng.choice(6, p=CAT_P)]
        a, b = response(cat), response(cat)
        qa, qb = quality(a, cat), quality(b, cat)
        d_true = qa - qb
        d_crowd = d_true + LENGTH_BIAS * np.log(a['words'] / b['words'])
        times = 2 if rng.random() < 0.065 else 1
        expert = ''
        if week >= 11 and rng.random() < 0.4:
            expert = 'A' if rng.random() < 1 / (1 + np.exp(-1.5 * d_true)) else 'B'
        pid += 1
        for t in range(times):
            pref = 'A' if rng.random() < 1 / (1 + np.exp(-d_crowd)) else 'B'
            if rng.random() < 0.05:
                pref = 'A' if rng.random() < 0.5 else 'B'          # careless clicks
            rows.append([f'p{pid:05d}', week, cat] + [a[f] for f in FIELDS] + [b[f] for f in FIELDS]
                        + [f'ann{int(rng.integers(0, 60)):02d}', pref, expert])
path = os.path.join(os.path.dirname(__file__), '../../static/projects/preferences.csv')
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['pair_id', 'week', 'category'] + ['a_' + x for x in FIELDS] + ['b_' + x for x in FIELDS]
               + ['annotator', 'preferred', 'expert_preferred'])
    w.writerows(rows)
print(len(rows), 'labels for', pid, 'pairs')
