# Project: Pace
:label:`sec_project_pace`

*Predict how long an AI model takes to start answering, and promise
it honestly.*

**Uses:** everything up to this point, especially linear regression and
minibatch SGD (:numref:`sec_linear_regression`), the `Module` / `DataModule` /
`Trainer` design (:numref:`sec_oo-design`), the squared-loss vs. robust-loss
trade-off, generalization and validation (:numref:`sec_generalization_basics`),
and weight decay (:numref:`sec_weight_decay`); plus pandas preprocessing and
probability from Chapter 2. Nothing beyond Chapter 3.

**Time:** two or three evenings. **Build it in your own GitHub repository**
named `pace`.

## The Problem

When you send a message to a chat assistant, the first word appears after
a pause. That pause is the **time to first token (TTFT)**. It is the
latency users notice most, and AI providers promise it to business
customers in contracts ("90% of requests start within 1 second"). TTFT is
made of three parts:

- **Prefill:** the model reads your whole prompt before it can write
  anything. Longer prompts take longer, bigger models take much longer,
  and newer GPUs are faster.
- **Queueing:** waiting behind other requests on a busy server.
- **Overhead:** network distance and software around the model.

A **prefix cache** helps: if the start of your prompt (a long system
prompt, earlier chat turns) was processed recently, those tokens are
skipped.

Inference teams at OpenAI, Anthropic, Google, and Meta model this to
choose which GPU serves which model, to route requests, to plan capacity,
and to set promises they can keep. A well-engineered linear model is
surprisingly good at it, because prefill time really is close to linear in
the number of new tokens.

You'll build **Pace**, a TTFT predictor that explains where the time goes
("the 400B model on an A100 spends about 0.5 s per 1,000 new tokens") and
gives a promise that holds for 90% of requests.

## Data

[Download llm_requests.csv](../_static/projects/llm_requests.csv) (7,522
simulated requests to a serving cluster over 30 days, in time order). One
row per request:

| Column | Meaning |
|:--|:--|
| `request_id` | unique id |
| `day` | day index 0--29 |
| `hour` | hour of the day (busy 8:00--18:00) |
| `client` | which customer application sent it (150 apps, `app000`--`app149`) |
| `model` | `lumen-8b`, `lumen-70b`, or `lumen-400b` (sizes in parameters) |
| `gpu` | `A100` (older) or `H100` (newer) |
| `prompt_tokens` | length of the prompt in tokens |
| `cached_tokens` | tokens served from the prefix cache (0 if none) |
| `queue_depth` | requests waiting on that server when this one arrived |
| `ttft_ms` | **target**: time to first token, in milliseconds |

Like real logs, it's messy. About 3% of `queue_depth` values are missing
(monitoring gaps). About 2% of requests hit a **cold start**: the server
was still loading the model's weights, adding 1.5--6 seconds. Clients
differ in ways no column captures directly. A few big clients send most
requests; most clients send only a handful.

## Steps

Pace is built **step by step while you read the chapter**: each step appears
at the end of its section, right after you learn what it needs, and
**revisits** earlier topics so nothing fades. All steps are listed here too.

```{=html}
<div class="d2l-project-steps"></div>
```

## Acceptance Criteria

- Time-based split (train: days 0--23, test: days 24--29). Imputation and
  standardization are fitted on the training data only.
- Test MAE (approximate; your numbers will differ slightly):

  | Model | Test MAE (ms) |
  |:--|--:|
  | Average (baseline) | about 514 |
  | Raw numeric columns | about 472 |
  | + one-hot model and GPU | about 311 |
  | + per-model-and-GPU slopes (interactions) | about 196 |
  | + Huber loss | about 172 |
  | + one-hot client | about 155 |

- A table of prefill cost in **ms per 1,000 new tokens** for each model
  and GPU, with one sentence on what it means for buying GPUs. With Huber
  loss it's about 500 vs. 220 for the 400B model on A100 vs. H100.
- A promise that covers about 90% of test requests: "your first token
  within $1.19\times$ our prediction".
- Loss curves for at least one model, and the learning curve from Step 6.

## Stretch Goals

- Attention makes prefill slightly worse than linear for long prompts. Add
  a (new tokens × prompt tokens) feature per model. Does it help, and for
  which prompts?
- Predict $\log(\textrm{TTFT})$ instead and compare MAE. Which errors does
  it reduce?
- Flag likely cold starts: what fraction of requests over 1.5 s can you
  explain from the columns, and what fraction is just bad luck?

## Optional Side Projects

Short, fun builds for more practice with this chapter's ideas. Each is a
weekend at most; pick whichever excites you. Same rule: only what you've
learned so far.

### Scale: Predict a Giant Model's Loss Before Training It

*Fit a scaling law and extrapolate, the way AI labs plan billion-dollar
training runs.*

- **Why it's real:** before training a frontier model, OpenAI, Anthropic,
  Google DeepMind, and Meta train many small models and fit **scaling
  laws** to predict how good the big one will be. The
  "Chinchilla" result (DeepMind, 2022), which changed how every lab sizes
  its models, is a regression on such runs.
- **Uses:** linear regression on transformed features (take logs), the
  analytic solution and SGD (:numref:`sec_linear_regression`),
  validation by holding out the largest runs
  (:numref:`sec_generalization_basics`).
- **Build:** simulate 40 small training runs with compute $C$ from $10^{15}$
  to $10^{19}$ FLOPs and final loss $L = 1.7 + 30\, C^{-0.12}$ plus 1%
  noise. Fit $\log(L - 1.7) = \log a - b \log C$ with linear regression on
  the 30 smallest runs, and predict the 10 largest. Then pretend you don't
  know the 1.7 (the "irreducible loss"): try values from 1.0 to 2.5 and
  pick the one whose fit has the smallest validation error.
- **Done when:** a log-log plot shows the fit and the extrapolation, your
  predicted loss for a $10^{21}$ FLOP run is within 2% of the truth, and
  you can explain why a small error in $b$ becomes a large error far away.

### Scout: Find Good Hyperparameters Automatically

*A small hyperparameter search tool, like a mini Google Vizier.*

- **Why it's real:** choosing learning rates, batch sizes, and weight decay
  is a big part of every ML team's compute bill; Google, Meta, and Anthropic
  all run internal tuning services.
- **Uses:** validation sets and $K$-fold cross-validation
  (:numref:`sec_generalization_basics`), weight decay (:numref:`sec_weight_decay`),
  your SGD `Trainer` (:numref:`sec_linear_scratch`).
- **Build:** a function `search(train_fn, space, budget)` that runs random
  search over log-scaled ranges (learning rate, batch size, $\lambda$), uses
  $K$-fold cross-validation, and returns the best settings with a results
  table.
- **Done when:** on Pace's data, Scout finds settings at least as good as
  your hand-tuned ones, and you show that random search beats a grid with
  the same budget.

### Drift: A Model That Keeps Up With the World

*Online learning that adapts when the data changes.*

- **Why it's real:** traffic, hardware, and user behavior shift over time
  (concept drift). Serving, pricing, and ad models update continuously
  instead of retraining from scratch.
- **Uses:** SGD on one example at a time, learning rates, and why the IID
  assumption matters (:numref:`sec_generalization_basics`).
- **Build:** stream Pace's requests in time order. Halfway through,
  simulate a change: a new inference engine makes H100 prefill 30% faster.
  Compare a model trained once on the first half against one that keeps
  doing SGD updates on each new request.
- **Done when:** a plot of rolling error over time shows the frozen model
  degrading and the online model recovering, and you explain how the
  learning rate trades stability for speed of adaptation.

## For Your README and Resume

Suggested layout: `pace/data.py` (loading, splitting, features),
`pace/model.py` (Module, losses), `pace/train.py` (Trainer), a notebook
with the results table and plots, and `tests/`.

> *Built **Pace**, a time-to-first-token predictor for LLM serving:
> linear regression trained with my own minibatch-SGD framework, features
> for prefix caching and per-model/GPU prefill cost, Huber loss for
> cold-start outliers, and a latency promise that held for 90% of unseen
> requests; cut error by 70% (MAE 514 → 155 ms) versus the average
> baseline.*

Use your own measured numbers in the bullet.
