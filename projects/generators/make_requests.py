"""Generate static/projects/llm_requests.csv for the Chapter 3 project (Pace).

Requests to an LLM serving cluster over 30 days. Target: time to first
token (TTFT) in milliseconds. Hidden truth:

  TTFT = overhead + prefill + queueing (+ rare cold starts)
  prefill  = new tokens / 1000 * rate[model, gpu] * (1 + prompt/40000)
  queueing = queue_depth * slot_ms[model] * gpu_factor

where new tokens = prompt_tokens - cached_tokens (a prefix cache skips
tokens it has already processed). Slopes differ ~40x between models, so a
linear model on the raw columns is poor until you engineer interaction
features. About 2% cold starts add 1.5-6 s (a heavy right tail)."""
import csv, os
import numpy as np

rng = np.random.default_rng(31)
MODELS = ['lumen-8b', 'lumen-70b', 'lumen-400b']
GPUS = ['A100', 'H100']
RATE = {('lumen-8b', 'A100'): 14, ('lumen-8b', 'H100'): 6,
        ('lumen-70b', 'A100'): 95, ('lumen-70b', 'H100'): 42,
        ('lumen-400b', 'A100'): 430, ('lumen-400b', 'H100'): 190}
SLOT = {'lumen-8b': 12, 'lumen-70b': 45, 'lumen-400b': 160}
GPU_FACTOR = {'A100': 1.8, 'H100': 1.0}
HOUR_P = np.r_[np.full(8, .02), np.full(10, .07), np.full(6, .14 / 6)]
HOUR_P /= HOUR_P.sum()
N_CLIENTS = 150
CLIENT_P = 1 / np.arange(1, N_CLIENTS + 1) ** 0.8
CLIENT_P /= CLIENT_P.sum()
CLIENT_MS = rng.gamma(2, 45, N_CLIENTS) - 90   # network distance, SDK overhead
rows = []
rid = 0
for day in range(30):
    n = rng.poisson(200 * (1 + day / 60))      # traffic grows
    hours = np.sort(rng.choice(24, n, p=HOUR_P))
    for hour in hours:
        client = rng.choice(N_CLIENTS, p=CLIENT_P)
        model = MODELS[rng.choice(3, p=[.5, .35, .15])]
        gpu = GPUS[rng.choice(2, p=[.4, .6])]
        prompt = int(np.clip(rng.lognormal(7.0, 1.0), 20, 32000))
        cached = int(prompt * rng.uniform(.5, .95)) if rng.random() < .4 else 0
        peak = 8 <= hour < 18
        queue = int(rng.poisson(1.5 + 5 * peak))
        new = prompt - cached
        prefill = new / 1000 * RATE[model, gpu] * (1 + prompt / 40000)
        queueing = queue * SLOT[model] * GPU_FACTOR[gpu]
        ttft = (35 + CLIENT_MS[client] + 90 + prefill + queueing) * rng.lognormal(0, .12) + rng.normal(0, 8)
        if rng.random() < .02:
            ttft += rng.uniform(1500, 6000)      # cold start: weights loading
        rid += 1
        q = '' if rng.random() < .03 else queue  # monitoring gaps
        rows.append([f'r{rid:05d}', day, int(hour), f'app{client:03d}', model, gpu, prompt, cached, q, round(max(ttft, 20), 1)])
path = os.path.join(os.path.dirname(__file__), '../../static/projects/llm_requests.csv')
with open(path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['request_id', 'day', 'hour', 'client', 'model', 'gpu', 'prompt_tokens', 'cached_tokens', 'queue_depth', 'ttft_ms'])
    w.writerows(rows)
print(len(rows), 'requests')
