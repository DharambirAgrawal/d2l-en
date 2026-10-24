# Capstone: Quill, Part 3
:label:`sec_capstone_quill_3`

*More context: predicting the next character from the last several.*

Quill so far looks at **one** character to guess the next. That's why its
text is word-shaped soup: after "th" it can't know that "e" is likely,
because it only sees "h". Next-character prediction is exactly a
**classification** problem with $V \approx 70$ classes, so this chapter's
tools apply directly: in this part Quill becomes a softmax regression over
the previous $k$ characters, trained with cross-entropy, measured with
accuracy and perplexity, given an error bar, and tested on a *different*
book to see distribution shift in language.

**Uses:** softmax regression and cross-entropy (:numref:`sec_softmax`),
`Classifier` and accuracy (:numref:`sec_classification`), the concise
implementation with `F.cross_entropy` on logits
(:numref:`sec_softmax_concise`), test-set error bars
(:numref:`chap_classification_generalization`), distribution shift
(:numref:`sec_environment-and-distribution-shift`), plus your Part 1
tokenizer and Part 2 training code. Keep working in your `quill`
repository.

## The Model

Take the previous $k$ characters $c_{t-k}, \ldots, c_{t-1}$, one-hot encode
each, and **concatenate** them into one input vector of length $kV$. A
single linear layer maps it to $V$ logits, and softmax gives the
distribution of the next character:

$$P(c_t \mid c_{t-k}, \ldots, c_{t-1}) = \mathrm{softmax}\Big(\mathbf{b} +
\sum_{i=1}^{k} \mathbf{W}^{(i)}_{c_{t-i},:}\Big).$$

That's softmax regression, exactly as in this chapter. Multiplying a
one-hot vector by a matrix just selects a row, so you never need to build
the $kV$-long vectors: add up $k$ rows, one per position. With $k = 1$ it
is Part 2's bigram model.

## Milestones

1. **Context windows.** Build a `DataModule` that yields
   $(\textrm{previous } k \textrm{ ids}, \textrm{next id})$ pairs from the same
   90/10 split as before (start the validation windows $k$ characters early
   so every validation character is predicted).
2. **Model.** A `Classifier`-style module whose `forward` returns logits
   (the row sum above, or `nn.Linear` on concatenated one-hot vectors) and
   whose `loss` is `F.cross_entropy` on logits. Initialize to zero so the
   first perplexity is exactly $V$.
3. **Train and choose $k$.** Train for $k = 1, 2, 3, 5, 8$ with minibatch
   SGD (batch 256, learning rate around 3, 20 epochs). Plot validation
   perplexity against $k$. Also report next-character **accuracy** (how
   often the most likely character is right).
4. **Compare with counting.** Compute held-out perplexity of count-based
   trigram and 4-gram models (Part 1's method with two and three
   characters of context, with add-$\alpha$ smoothing). The linear model
   uses more context than a trigram, yet loses to it. Explain why: the
   logits are a **sum** of separate contributions from each position, so the
   model can't express "e is likely after t *followed by* h" any more
   strongly than "after t" plus "after h". Counting can, but needs a table
   with $V^k$ rows. (The next chapter's MLP fixes exactly this.)
5. **An error bar.** The validation perplexity is $\exp$ of an average of
   about 18,000 per-character losses. Compute the standard error of that
   average and turn it into an interval for the perplexity. Is the
   difference between $k = 5$ and $k = 8$ real?
6. **Distribution shift.** Evaluate your best model on another H. G. Wells
   novel, *The War of the Worlds* (Project Gutenberg, ebook 36; strip the
   license header and footer). Drop characters that aren't in your
   vocabulary and count them (curly quotes are the main ones). Report the
   perplexity and explain the increase with this chapter's vocabulary:
   which kind of shift is it?
7. **Sample with temperature.** Generate text starting from "The T" at
   temperatures 0.5 and 1.0 (divide the logits by $T$ before the softmax).

## Acceptance Criteria

- Initial perplexity equals the vocabulary size (about 70).
- Held-out results (approximate; your numbers will differ slightly):

  | Model | Held-out perplexity |
  |:--|--:|
  | Bigram (Parts 1--2) | about 11.4 |
  | Linear, $k = 3$ | about 9.1 |
  | Linear, $k = 5$ | about 8.4--8.5 |
  | Linear, $k = 8$ | about 8.4 |
  | Count trigram (for comparison) | about 7.3 |
  | Count 4-gram (for comparison) | about 5.8 |

- Next-character accuracy of about 39% for $k = 5$.
- An interval for the validation perplexity (the standard error of the
  average loss is about 0.011 nats).
- *The War of the Worlds* perplexity (about 9.2 for $k = 5$) with your
  explanation of the shift.
- Your README's results table gains a row:

  | Part | Model | Held-out perplexity |
  |:--|:--|:--|
  | 3 | Softmax regression over 5 previous characters | about 8.4 |

## Stretch Goals

- Add weight decay. Does it help at $k = 8$, where the model has
  $8 \times 70 \times 70 \approx 39{,}000$ weights?
- Give each position its own learning rate or scale, or add a count-based
  trigram feature as an extra input. Can a linear model now beat the
  trigram?

## How Far Quill Has Come

| Part | After chapter | Model | Held-out perplexity | Status |
|:--|:--|:--|--:|:--|
| 0 | | random guessing (uniform over 70 characters) | 70.0 | starting point |
| 1 | 2 | bigram from counts | **11.4** | done |
| 2 | 3 | bigram trained with minibatch SGD | **11.7** (→ 11.4) | done |
| 3 | 4 | softmax regression over 5 previous characters | **8.4** | done after this part |
| 4 | 5 | MLP over the previous several characters | goal: below the 4-gram's 5.8 | next |
| 5 | 9--10 | recurrent network (RNN/GRU) | goal: about 4--5 | later |
| 6 | 11 | GPT-style Transformer | goal: below 4 | later |

Samples from each version, starting with "T" (Part 3 starts with "The T"
and uses temperature 0.5; newlines shown as spaces):

| Version | Sample |
|:--|:--|
| Random guessing | `T(AmY)OR8h,LUO[hwDbeE yEFsYQk!fK)cv;aEhg?ocdnOjr,pLR1eEqDXL]:9iiXu;p8x_]xll"J(:;q-ESpxf;Wf"dJ[ccUW:S-RVkL U;hL"` |
| Part 1: counts | `Tg thKPhas heckitarld walupis s9)_ughequgorkndes ted s t?)Gmaderes t whewis f afr d tur?'U(d orine l. mom masa` |
| Part 2: SGD | `Thashand s ged steng 'uldinde omMhrSoghilorkndes the rdimed addely t whewis f adie. turang pemione ieelom lasa` |
| Part 3: 5 characters, $T = 0.5$ | `The Time Trevery terend of the wored boundes of the s of the mine far the bound lownith thed ande tromithe the mo that per ma` |

Real words appear now ("The Time", "of the", "the bound", "that"), and the
spacing looks like English. At $T = 1$ the text is still much noisier: the
model's probabilities are spread too thin, the price of a linear model.

| Part | Skills you gained (and can explain in an interview) |
|:--|:--|
| 1 | Tokenization; probability tables from data; smoothing; sampling; perplexity as an evaluation metric; deriving the softmax/cross-entropy gradient and checking it numerically |
| 2 | Training a model by minibatch SGD; designing Module/DataModule/Trainer code; tuning the learning rate; train vs. validation curves; weight decay as regularization (and when it hurts) |
| 3 | Framing language modeling as classification; one-hot features and why they become embedding lookups; numerically stable cross-entropy; error bars on evaluation metrics; the limits of linear models (no interactions); measuring distribution shift between texts |

## Why This Matters

Every large language model is, at its core, this part's model: a
classifier over the vocabulary, trained with cross-entropy to predict the
next token from the previous ones. What changes in the rest of the book is
*how the context is turned into logits*. Here it's a sum of per-position
contributions; next comes a hidden layer that can combine them
(Chapter 5), then recurrence (Chapters 9--10), then attention
(Chapter 11).
