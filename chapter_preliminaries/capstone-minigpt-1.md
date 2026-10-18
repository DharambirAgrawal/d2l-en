# Capstone: Quill, Part 1
:label:`sec_capstone_minigpt_1`

*Letters from counts: the first page of your own GPT.*

**The capstone:** across the whole book, you'll build **Quill**, your own
ChatGPT-like model, one part per chapter, in a single GitHub repository
(`quill`). Each part uses only what you have learned so far and must beat
the previous part's results. By the end, you'll have a small Transformer
language model that you understand line by line.

| Part | After chapter | You add |
|:--|:--|:--|
| **1** | **2. Preliminaries** | **Tokenizer, bigram model from counts, sampling, perplexity, softmax gradient** |
| 2 | 3. Linear regression | Train the bigram model as a neural network with minibatch SGD |
| 3 | 4--5. Classification, MLPs | Longer context: an MLP over the previous few characters |
| ... | 9--10. RNNs | A recurrent language model |
| ... | 11. Attention and Transformers | Self-attention, then a full GPT-style Transformer |

**Uses (Part 1):** tensors and indexing (:numref:`sec_ndarray`), reductions
and broadcasting (:numref:`sec_linear-algebra`), the chain rule
(:numref:`sec_calculus`), and sampling, conditional probability, and
expectations (:numref:`sec_prob`).

## Data

Use H. G. Wells' *The Time Machine* (public domain), the same text the book
uses for language models later:
[timemachine.txt](http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt)
(about 180 KB). Use the first 90% for training and the last 10% as
held-out text.

## Milestones

1. **Tokenizer.** Build a character vocabulary (`itos`: sorted distinct
   characters; `stoi`: the reverse map) with `encode` and `decode`.
   `decode(encode(text)) == text` must hold.
2. **Bigram counts.** Build a $V \times V$ count matrix $\mathbf{N}$ where
   $N_{ij}$ counts how often character $j$ follows character $i$. Do it
   without a Python loop (hint: `np.add.at`). Check that the counts sum to
   the number of adjacent pairs.
3. **Probabilities.** Turn counts into next-character distributions with
   add-$\alpha$ smoothing:
   $P_{ij} = (N_{ij} + \alpha) / \sum_k (N_{ik} + \alpha)$ (rows sum to 1;
   use `keepdims`). Explain why $\alpha = 0$ is dangerous on new text.
4. **Generate text.** Sample one character at a time from the current
   row of $\mathbf{P}$ with a seeded random generator. Show a few
   200-character samples. Why does always taking the most likely character
   produce worse text than sampling?
5. **Evaluate.** Compute the average negative log-likelihood
   $\ell = -\frac{1}{T-1}\sum_t \log P_{x_t, x_{t+1}}$ and perplexity
   $e^{\ell}$ on held-out text. Compare with uniform guessing (perplexity
   $V$). Plot held-out perplexity against $\alpha$ and choose the best
   $\alpha$.
6. **The gradient that trains GPT.** Implement a numerically stable
   softmax, the cross-entropy loss on logits, and its gradient
   $(\textrm{softmax}(\mathbf{O}) - \textrm{onehot}(\mathbf{y})) / n$.
   Derive the gradient with the chain rule, then verify it with a
   central-difference numerical gradient check. Part 2 trains with exactly
   this gradient.

## Acceptance Criteria

- Vocabulary size around 70 characters; the encode/decode round trip passes.
- Held-out perplexity of about **11--12** with $\alpha \approx 1$, versus about
  70 for uniform guessing. **This is your baseline: every later part must
  beat it.**
- Your gradient matches the numerical gradient to about $10^{-6}$ relative
  error.
- A README section "Part 1 results" with the perplexity table, a sample of
  generated text, and the perplexity-vs-$\alpha$ plot.

## Stretch Goals

- Add a *temperature* parameter to sampling (divide log-probabilities by
  $T$ before normalizing) and show samples at $T = 0.5, 1, 2$.
- Train a *trigram* model (two characters of context). Does held-out
  perplexity improve? What happens to the table size, and why is that a
  problem as context grows? (That problem is why neural models exist.)

## For Your README and Resume

Structure the repo so it can grow: `quill/tokenizer.py`, `quill/bigram.py`,
`quill/train.py` (later parts), `notebooks/`, `tests/`, and a README with
a results table that gains one row per part:

| Part | Model | Held-out perplexity |
|:--|:--|:--|
| 1 | Character bigram (counts) | about 11.4 |

> *Building **Quill**, a GPT-style language model from scratch in NumPy/PyTorch,
> part by part: character tokenizer, n-gram baseline, neural training with
> hand-derived gradients, and (in progress) self-attention and Transformer
> layers; tracking held-out perplexity at each stage.*
