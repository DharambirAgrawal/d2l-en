# Project: Relay
:label:`sec_project_relay`

*Route support tickets to the right team, and notice when the world changes.*

**Uses:** everything up to this point, especially softmax regression and
cross-entropy (:numref:`sec_softmax`), the `Classifier` and accuracy
(:numref:`sec_classification`), training from scratch and concisely
(:numref:`sec_softmax_scratch`, :numref:`sec_softmax_concise`), test-set
error bars (:numref:`chap_classification_generalization`), and distribution
shift (:numref:`sec_environment-and-distribution-shift`); plus pandas,
probability, SGD, and weight decay from Chapters 2 and 3. Nothing beyond
Chapter 4.

**Time:** two or three evenings. **Build it in your own GitHub repository**
named `relay`.

## The Problem

Every large company receives thousands of support messages a day: "I was
charged twice", "my package never arrived", "the app crashes at checkout".
Each must reach the right team (billing, shipping, technical, ...), and a
misrouted ticket waits in the wrong queue while the customer gets angrier.
Zendesk, Intercom, Salesforce, and in-house tools at Amazon, Shopify, and
Google all route tickets with classifiers, and a well-built linear model on
word counts is the strong, cheap, explainable baseline they start from.

You'll build **Relay**, a ticket router that predicts the team, explains
its decision with the words that drove it, accounts for the cost of
mistakes, and **notices** when a product launch changes what customers
write about.

## Data

[Download tickets.csv](../_static/projects/tickets.csv) (8,000 simulated
tickets from January to June 2026, in time order). One row per ticket:

| Column | Meaning |
|:--|:--|
| `ticket_id` | unique id |
| `created` | date (YYYY-MM-DD) |
| `channel` | email, chat, web, or phone |
| `tier` | customer plan: free, plus, business |
| `subject` | subject line (missing for about 3% of tickets) |
| `body` | the message |
| `team` | **label**: account, billing, returns, shipping, or technical |

Like real data it's imperfect: some tickets mention two problems, about 4%
of labels are wrong (a tired human routed them), and in **June** a new "pay
in installments" feature launched, so the mix of teams and the words
customers use change. Treat June as *production*: don't look at it until
the last step.

## Steps

Relay is built **step by step while you read the chapter**: each step
appears at the end of its section, right after you learn what it needs, and
**revisits** earlier topics so nothing fades. All steps are listed here too.

```{=html}
<div class="d2l-project-steps"></div>
```

## Acceptance Criteria

- Time-based split; vocabulary built from training months only.
- Results (approximate; your numbers will differ slightly):

  | Model | Accuracy |
  |:--|--:|
  | Most common team (baseline) | about 25% (May) |
  | Softmax regression, bag of words | about 96--97% (May) |
  | Same model on June (after the launch) | about 88% |
  | + 50 labeled June tickets, retrained | about 93% (rest of June) |

- May accuracy reported with an error bar, and a sentence on whether two of
  your variants really differ.
- The top words per team, and three confident mistakes explained (model
  error or label noise?).
- The June diagnosis: domain-classifier accuracy, unknown-word rate, BBSE
  estimate vs. the truth, and why label-shift correction alone fails here.

## Stretch Goals

- Use word *pairs* (bigrams) as extra features. Does May accuracy improve?
  What happens to the vocabulary size?
- Add `channel` and `tier` as one-hot features. Do they help?
- Let Relay say "I'm not sure": send tickets whose top probability is below
  a threshold to a human, and plot accuracy vs. the fraction sent to humans.

## Optional Side Projects

Short, fun builds for more practice with this chapter's ideas. Each is a
weekend at most; pick whichever excites you. Same rule: only what you've
learned so far.

### Sieve: A Spam Filter You Can Explain

*Catch spam text messages without blocking real ones.*

- **Why it's real:** Gmail's first spam filters were linear models on word
  counts, and they still sit in today's pipelines as fast, explainable
  first stages. Wrongly blocking a real message costs far more than letting
  one spam through.
- **Uses:** softmax (two-class) regression on word counts, cross-entropy,
  unequal costs and decision rules (:numref:`sec_classification`), error
  bars (:numref:`chap_classification_generalization`).
- **Build:** use the public SMS Spam Collection (5,574 messages, UCI
  Machine Learning Repository). Train on counts of words, then choose the
  decision threshold that minimizes cost when a blocked real message costs
  10× a missed spam.
- **Done when:** you report precision and recall for spam at your threshold
  with error bars, and list the 10 most "spammy" and 10 most "safe" words.

### Gauge: Is the Model as Sure as It Says?

*Measure and fix the confidence of a classifier.*

- **Why it's real:** a doctor, a self-driving car, or an AI assistant must
  know when *not* to trust its own answer. Google and Anthropic study the
  calibration of their models; a model saying "90% sure" should be right
  about 90% of the time.
- **Uses:** softmax and temperature (:numref:`sec_softmax`), Fashion-MNIST
  and your softmax regression (:numref:`sec_softmax_concise`), validation
  vs. test sets.
- **Build:** group the test predictions of your Fashion-MNIST model by
  confidence (0.0--0.1, ..., 0.9--1.0) and plot accuracy vs. confidence (a
  *reliability diagram*); compute the expected calibration error (ECE).
  Then fit a single **temperature** $T$ on the validation set (divide the
  logits by $T$, choose $T$ to minimize validation cross-entropy).
- **Done when:** your reliability diagram before and after shows the
  improvement, and ECE drops while accuracy stays the same (explain why it
  can't change).

### Sentinel: An Alarm for Data Drift

*Watch a production model's inputs and raise an alarm when they shift.*

- **Why it's real:** models fail silently when the world changes. Every ML
  platform (Google's Vertex AI, AWS SageMaker, Uber's Michelangelo) has a
  drift monitor.
- **Uses:** domain classifiers for covariate shift and the confusion-matrix
  method for label shift (:numref:`sec_environment-and-distribution-shift`),
  Hoeffding's inequality (:numref:`chap_classification_generalization`).
- **Build:** feed Relay's tickets day by day. Each day, compare the day's
  tickets with the training data: domain-classifier accuracy, unknown-word
  rate, and the BBSE label-distribution estimate. Choose alarm thresholds
  from the day-to-day variation in January--May so that false alarms are
  rare.
- **Done when:** Sentinel stays quiet through May and fires within the first
  days of June, and a plot shows the three signals over time.

## For Your README and Resume

Suggested layout: `relay/data.py` (loading, splitting, vocabulary),
`relay/model.py` (softmax regression, losses, `decide`), `relay/train.py`,
`relay/drift.py` (Step 7), a notebook with the results table and plots, and
`tests/`.

> *Built **Relay**, a support-ticket router: softmax regression on bag-of-words
> features trained with my own SGD framework (96.7% accuracy vs. 25% baseline,
> with error bars), cost-aware routing decisions, and a drift diagnosis that
> caught a product launch (domain classifier, label-shift estimation) and
> recovered accuracy from 88% to 93% with 50 new labels.*

Use your own measured numbers in the bullet.
