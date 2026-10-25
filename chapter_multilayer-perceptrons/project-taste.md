# Project: Taste
:label:`sec_project_taste`

*Train a reward model: a network that learns which AI answers people
prefer. Then catch it learning the wrong lesson.*

**Uses:** everything up to this point, especially MLPs and activation
functions (:numref:`sec_mlp`), their implementation
(:numref:`sec_mlp-implementation`), backpropagation (:numref:`sec_backprop`),
initialization (:numref:`sec_numerical_stability`), generalization and
early stopping (Section 5.5), dropout
(:numref:`sec_dropout`), and $K$-fold cross-validation
(:numref:`sec_kaggle_house`); plus the Bradley--Terry model from Podium
(:numref:`sec_project_podium`) and classification tools from Chapter 4.
Nothing beyond Chapter 5.

**Time:** two or three evenings. **Build it in your own GitHub repository**
named `taste`.

## The Problem

Chat assistants such as ChatGPT, Claude, and Gemini are trained in part with
**reinforcement learning from human feedback (RLHF)**. People compare two
answers to the same prompt and pick the better one. A **reward model** is
trained on these comparisons to give every answer a score. The assistant
is then tuned to produce answers that score high. The reward model is
trained with exactly Podium's Bradley--Terry loss, with a neural network
in place of a table of ratings:

$$P(A \succ B) = \sigma\big(r_\theta(A) - r_\theta(B)\big).$$

Two things make this hard, and both are central to this chapter:

- **Good answers depend on context.** Refusing is right for a dangerous
  request and wrong for a homework question. Code helps a programming
  question and annoys someone asking for a birthday poem. A linear score
  can't express "it depends". A hidden layer can.
- **Human labels are noisy and biased.** Annotators disagree with each
  other, and they tend to prefer **longer** answers even when shorter
  ones are better. A reward model learns that bias too. A model tuned
  against it then becomes verbose: **reward hacking**, a real problem at
  every AI lab.

You'll build **Taste**, a reward model that beats a linear baseline by a
wide margin, and an audit that measures how much of the annotators' length
bias it learned.

## Data

[Download preferences.csv](../_static/projects/preferences.csv) (6,770
simulated labels on 6,346 pairs of answers, collected over 12 weeks). Real
reward models read the text itself; to keep this project inside Chapter 5,
each answer is described by **eight measured attributes**, the kind of
signals labs actually log about responses. One row per label:

| Column | Meaning |
|:--|:--|
| `pair_id`, `week` | the pair, and when it was labeled (1--12) |
| `category` | prompt type: coding, math, advice, creative, chitchat, or unsafe (a request the assistant should decline) |
| `a_words`, `b_words` | length of answer A and B in words |
| `a_has_code`, ... | contains code (0/1) |
| `a_refuses`, ... | declines to help (0/1) |
| `a_steps`, ... | number of numbered steps |
| `a_hedges`, ... | hedging phrases ("might", "possibly") |
| `a_cites`, ... | cites a source (0/1) |
| `a_errors`, ... | factual errors found by a checker (0--2) |
| `a_emoji`, ... | emoji count |
| `annotator` | who labeled it |
| `preferred` | **label**: `A` or `B` |
| `expert_preferred` | for 441 pairs in weeks 11--12: a careful expert's choice (otherwise empty) |

424 pairs were labeled **twice** by different annotators, so you can
measure how often people agree. About 5% of clicks are careless.

## Steps

Taste is built **step by step while you read the chapter**: each step
appears at the end of its section, right after you learn what it needs, and
**revisits** earlier topics so nothing fades. All steps are listed here too.

```{=html}
<div class="d2l-project-steps"></div>
```

## Acceptance Criteria

- Split by week: train 1--8, validation 9--10, test 11--12. Standardize
  with training statistics (use $\log$ of the word counts).
- Results (approximate; your numbers will differ slightly):

  | Model | Validation accuracy | Test accuracy |
  |:--|--:|--:|
  | "The longer answer wins" | about 63% | |
  | Linear reward (8 attributes) | about 68% | about 67% |
  | MLP reward, one hidden layer (64 units) | about 74--75% | about 74% |
  | Agreement between two annotators | about 66% | |

- An explanation of why the model can beat the annotator agreement,
  and the ceiling you estimate from it (about 78%).
- The deep-network experiment from Step 4. Six sigmoid layers with tiny
  initial weights stay at about 50%; ReLU with He initialization reaches
  about 73%.
- The audit from Step 7: agreement with experts (about 77% for the MLP,
  70% for the linear model). Show how often the model picks the longer
  answer compared with experts (about 66% vs. 58%), and the reward vs.
  length curve for chitchat. It peaks near 80 words, while the
  simulation's true best length is 40.

## Stretch Goals

- **Remove the length bias.** During training, add a separate term
  $\beta(\log w_A - \log w_B)$ to the score difference, where $w$ is the
  word count, and give the MLP every attribute *except* length. At
  inference, drop the length term. Does expert agreement improve?
- **Best-of-$N$.** For 200 test prompts, create $N$ candidate answers by
  randomly changing attributes. Pick the highest-reward one, and plot its
  average length and its expert-quality win rate as $N$ grows. This is
  reward hacking in miniature.
- Weight each label by how reliable its annotator is: estimate
  reliability from the duplicated pairs.

## Optional Side Projects

Short, fun builds for more practice with this chapter's ideas. Each is a
weekend at most; pick whichever excites you. Same rule: only what you've
learned so far.

### Grok: Watch a Network Suddenly Understand

*Train an MLP on modular arithmetic and see "grokking": memorization
first, understanding much later.*

- **Why it's real:** grokking (Power et al., OpenAI, 2022) is one of the
  most surprising findings in deep learning, and a favorite example of
  interpretability researchers at Anthropic and Google DeepMind. A network
  first memorizes its training data and then, long after training
  accuracy hits 100%, suddenly generalizes.
- **Uses:** MLPs (:numref:`sec_mlp-implementation`), generalization in deep
  networks (Section 5.5), weight decay
  (:numref:`sec_weight_decay`).
- **Build:** all pairs $(a, b)$ with $a, b < 97$; label $(a + b) \bmod
  97$. Input: two one-hot vectors concatenated; a two-layer MLP with 256
  hidden units; train on a random 40% of the pairs with full-batch Adam
  (`torch.optim.AdamW`, learning rate $10^{-3}$, weight decay 1) for tens of
  thousands of steps. Log training and validation accuracy on a log-scale
  x-axis.
- **Done when:** your plot shows training accuracy at 100% long before
  validation accuracy jumps, and you show that without weight decay the
  jump doesn't happen (or comes much later).

### Reel: Rank a Video Feed

*The model behind "For You" feeds: predict whether someone will finish a
video.*

- **Why it's real:** Meta, TikTok, and YouTube rank billions of items a day
  with MLPs on user and item features (Meta's open-source DLRM is one).
  What matters is the *interaction*: a cooking fan and a cooking video.
- **Uses:** MLPs, dropout, cross-entropy (Chapters 4--5).
- **Build:** simulate 2,000 users with interest scores in 8 topics and
  5,000 videos with a topic, a length, and a quality score. A view is
  "watched to the end" with a probability that depends on the user's
  interest *in that video's topic*. Compare logistic regression on the
  concatenated features with an MLP. Then add a learned embedding per user
  (a one-hot row times a matrix).
- **Done when:** the MLP clearly beats the linear model on held-out views,
  and you can show one user's top-10 feed changing with their interests.

### Probe: What Does a Language Model Know?

*Train small classifiers on a real model's hidden states.*

- **Why it's real:** *probing* is a basic tool of interpretability research
  (Anthropic, OpenAI, Google DeepMind). If a linear classifier can read a
  property off a layer's activations, the model represents it there.
- **Uses:** softmax regression (Chapter 4), MLPs, and generalization
  (Chapter 5).
- **Build:** run a small open model (GPT-2 small through Hugging Face
  `transformers`, with `output_hidden_states=True`) on 2,000 short
  sentences labeled with a property, such as past vs. present tense or
  positive vs. negative. Average each layer's hidden states over tokens,
  then train a linear probe and an MLP probe per layer.
- **Done when:** a plot shows probe accuracy by layer, and you discuss why
  an MLP probe that beats a linear one is weaker evidence (it may learn the
  task itself).

## For Your README and Resume

Suggested layout: `taste/data.py` (loading, pairing, standardizing),
`taste/model.py` (reward MLP, pairwise loss), `taste/train.py`,
`taste/audit.py` (expert agreement, length bias), a notebook with the
results table and plots, and `tests/`.

> *Built **Taste**, a reward model of the kind used in RLHF: an MLP trained
> with a pairwise Bradley--Terry loss on 6,700 human preference labels
> (74% accuracy vs. 67% for a linear reward; human agreement 66%). Debugged
> initialization and overfitting in deeper variants, and audited the model
> against expert labels, finding a learned length bias (it prefers the
> longer answer 66% of the time vs. 58% for experts).*

Use your own measured numbers in the bullet.
