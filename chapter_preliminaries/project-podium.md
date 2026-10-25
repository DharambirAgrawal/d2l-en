# Project: Podium
:label:`sec_project_podium`

*Rank AI chat models from head-to-head votes, and say how sure you are.*

**Uses:** everything up to this point: tensors and broadcasting
(:numref:`sec_ndarray`), data cleaning with pandas (:numref:`sec_pandas`),
matrices and norms (:numref:`sec_linear-algebra`), derivatives and
gradients (:numref:`sec_calculus`), automatic differentiation
(:numref:`sec_autograd`), and probability (:numref:`sec_prob`). Nothing
beyond Chapter 2.

**Time:** one or two evenings. **Build it in your own GitHub repository**
named `podium`.

## The Problem

Which AI model is best? Benchmarks with fixed answers saturate quickly, so
the most-watched leaderboards in 2026 rank chat models with **votes**: a
user types a prompt, sees two anonymous answers side by side, and picks the
better one. Turning tens of thousands of such votes into a ranking is
harder than it looks:

- Raw win rates mislead. A model that is mostly matched against strong
  opponents looks worse than it is.
- People have biases. For example, they slightly prefer the answer shown
  first.
- A brand-new model has few votes. Is it really #1, or just lucky?

The standard answer is the **Bradley--Terry model**: every model $i$ gets
a rating $r_i$, and

$$P(i \textrm{ beats } j) = \sigma(r_i - r_j) = \frac{1}{1 + e^{-(r_i - r_j)}}.$$

This is the same formula behind chess Elo ratings. It is also the loss
used to train the **reward models** in RLHF: at OpenAI, Anthropic, and
Google, human preference pairs are turned into a score with exactly this
equation. You'll build **Podium**, a leaderboard that fits these ratings
from scratch, corrects for the position bias, and puts honest error bars
on every rank.

## Data

[Download battles.csv](../_static/projects/battles.csv) (6,831 simulated
votes between 12 fictional models, September 2026, in time order). One row
per vote:

| Column | Meaning |
|:--|:--|
| `battle_id` | id of the vote |
| `time` | when it happened |
| `model_a` | the model whose answer was shown **first** (left) |
| `model_b` | the model shown second (right) |
| `category` | coding, math, writing, chat, or other |
| `language` | language of the prompt (missing for about 3%) |
| `prompt_tokens` | length of the prompt |
| `winner` | `model_a`, `model_b`, `tie`, `both_bad`, or empty |

Like real logs it is messy. Some model names are upper-cased or padded
with spaces. About 1% of votes were submitted twice. About 2% have no
winner because the user closed the tab. One model, `nova-2`, launched on
September 22, so it has far fewer votes than the rest.

## Steps

Podium is built **step by step while you read the chapter**: each step
appears at the end of its section, right after you learn what it needs, and
**revisits** earlier topics so nothing fades. All steps are listed here too.

```{=html}
<div class="d2l-project-steps"></div>
```

## Acceptance Criteria

- After cleaning: 66 duplicate rows and 150 votes without a winner removed,
  leaving 6,615 votes. The 5,665 votes with a decisive winner are used for
  fitting.
- The answer shown first wins about 52% of decisive votes. Your fitted
  position bias is about **+0.14** on the logit scale.
- Ratings on an Elo-like scale ($1000 + \frac{400}{\ln 10} r_i$, mean
  ratings at 1000), approximately:

  | Model | Rating | Raw win rate |
  |:--|--:|--:|
  | nova-2 | 1151 | 59.0% |
  | atlas-4 | 1150 | 61.5% |
  | orbit-pro | 1111 | 56.1% |
  | ... | | |
  | kite-7b | 799 | 34.7% |

- `quartz-xl` is 4th by raw win rate but only 6th by rating. Explain
  why, using its opponents.
- Bootstrap 95% intervals for every rating (about ±20 for the
  well-measured models and about ±30 for `nova-2`). Report
  $P(\textrm{nova-2} > \textrm{atlas-4}) \approx 0.6$ and say what you'd
  tell a team that wants to announce "nova-2 is #1".
- A leaderboard plot: ratings with error bars, sorted.

## Stretch Goals

- Fit separate ratings per `category`. Which model is the best at coding
  but not overall?
- Use ties properly: count a tie as half a win for each side, or use the
  Davidson extension of Bradley--Terry.
- Simulate votes from your fitted ratings and refit. Do you recover them?
  This is how leaderboard teams test their own pipelines.

## Optional Side Projects

Short, fun builds for when you want more practice with this chapter's
ideas. Each is a weekend at most; pick whichever excites you. Same rule:
only what you've learned so far.

### Lens: Search This Book with Vectors

*Build the retrieval half of a RAG system from scratch.*

- **Why it's real:** every AI assistant that "searches your documents"
  (retrieval-augmented generation, RAG) turns text into vectors and
  returns the nearest ones. Vector databases (Pinecone, pgvector, FAISS)
  do exactly this at scale.
- **Uses:** tensors and broadcasting (:numref:`sec_ndarray`), pandas
  (:numref:`sec_pandas`), dot products, norms, and cosine similarity
  (:numref:`sec_linear-algebra`).
- **Build:** split this book's Markdown files into paragraphs. Turn each
  paragraph into a vector of word counts over a fixed vocabulary, and
  down-weight common words. A good weight is
  $\log(N / \textrm{number of paragraphs containing the word})$, where $N$
  is the total number of paragraphs. For a query, compute the cosine
  similarity with every paragraph in **one** matrix--vector product and
  return the top 5.
- **Done when:** "how does broadcasting work" returns paragraphs from
  :numref:`sec_ndarray`. Write 10 test queries with their correct
  sections and report how often the right one is in your top 5
  (recall@5). Explain why normalizing the vectors matters.

### Ripple: Google's First Algorithm

*Rank web pages the way Google started, with one matrix--vector product,
repeated.*

- **Why it's real:** PageRank made Google. The same idea ranks people in
  social networks, papers by citations, and packages in dependency graphs.
- **Uses:** matrix--vector products, norms (:numref:`sec_linear-algebra`),
  probability (a random surfer, :numref:`sec_prob`).
- **Build:** take a small link graph (for example, the links between this
  book's pages, or 20 Wikipedia articles). Build the column-stochastic
  transition matrix $\mathbf{M}$, then iterate
  $\mathbf{r} \leftarrow 0.85\,\mathbf{M}\mathbf{r} + 0.15/n$ until the
  $\ell_1$ change is below $10^{-8}$.
- **Done when:** ranks sum to 1, the top pages make sense, and you can
  explain why the 0.15 "teleport" term guarantees convergence (and what
  happens to dead-end pages without it).

### Spark: An Autograd Engine in 100 Lines

*Build the core of PyTorch yourself.*

- **Why it's real:** every deep learning framework (PyTorch, JAX,
  TensorFlow) is built around reverse-mode automatic differentiation. Being
  able to build one is a classic signal in ML engineering interviews at
  places like Google and Anthropic.
- **Uses:** the chain rule (:numref:`sec_calculus`) and how autograd records
  a graph (:numref:`sec_autograd`).
- **Build:** a `Value` class that stores a number, its gradient, and its
  parents; support `+`, `*`, `**`, `exp`, `log`, `tanh`; implement
  `backward()` with a topological sort.
- **Done when:** its gradients match PyTorch's (or a numerical gradient) on
  random expressions to $10^{-6}$, and you can fit $y = 3x + 1$ with it by
  gradient descent.

## For Your README and Resume

Suggested repository layout: `podium/data.py` (loading and cleaning),
`podium/model.py` (Bradley--Terry fit), `podium/bootstrap.py`,
`notebooks/leaderboard.ipynb` (plots and results), `tests/`, and a README
with the leaderboard, the error bars, and what you learned about raw win
rates.

> *Built **Podium**, a model leaderboard from 6,600 pairwise human votes:
> cleaned the vote logs with pandas, fit Bradley--Terry ratings (the loss
> behind RLHF reward models) by gradient descent with PyTorch autograd,
> corrected a 52/48 position bias, and showed with bootstrap intervals that
> a new model's apparent #1 rank was a statistical tie.*

Use your own measured numbers in the bullet.
