# Capstone: Quill, Part 2
:label:`sec_capstone_quill_2`

*Learning by gradient descent: the same model, now trained like a neural
network.*

In Part 1 (:numref:`sec_capstone_minigpt_1`) you built Quill's first model by
**counting** which character follows which. Large language models can't be
built by counting: they have billions of parameters, learned by **gradient
descent**. In this part, you'll train the bigram model as a one-layer neural
network with minibatch SGD, using the training framework from this chapter,
and check that learning recovers what counting found.

**Uses:** minibatch SGD and the `Module` / `DataModule` / `Trainer` design
(:numref:`sec_linear_scratch`, :numref:`sec_oo-design`), training vs.
validation loss (:numref:`sec_generalization_basics`), weight decay
(:numref:`sec_weight_decay`), plus your Part 1 tokenizer and the softmax
gradient you derived there. Keep working in your `quill` repository.

## The Model

For a current character $c$ (a token id), the model has a row of **logits**
$\mathbf{W}_{c,:} \in \mathbb{R}^V$ (one score per possible next character)
and predicts

$$P(\textrm{next} = j \mid c) = \textrm{softmax}(\mathbf{W}_{c,:})_j.$$

That's a linear layer applied to a one-hot input: $\textrm{onehot}(c)^\top\mathbf{W}$
just selects row $c$. The parameters are the $V \times V$ matrix $\mathbf{W}$
(about 4,900 numbers for this book's 70 characters), and the loss is the
average cross-entropy (negative log-likelihood) of the true next characters,
the same quantity you used for perplexity in Part 1.

## Milestones

1. **Data module.** Build a `DataModule` that turns the text into
   (current, next) id pairs, with the same 90/10 train/validation split as in
   Part 1, and yields shuffled minibatches.
2. **Model.** A `Module` holding $\mathbf{W}$ (initialized to zeros, so it
   starts as the uniform model: perplexity = $V$), with `forward` returning
   logits for a batch of current ids and `loss` returning mean cross-entropy.
3. **Gradients by hand.** Implement the backward pass yourself using your
   Part 1 result: for a minibatch, the gradient for row $c$ accumulates
   $(\textrm{softmax}(\mathbf{W}_{c,:}) - \textrm{onehot}(y)) / |\mathcal{B}|$ over
   examples whose current character is $c$. Check it against a numerical
   gradient on a tiny batch.
4. **Train.** Use your `Trainer` with minibatch SGD. The per-row gradients
   are small, so you'll need a **large learning rate** (try 1, 3, 10, 30).
   Plot training and validation perplexity per epoch.
5. **Compare with counting.** After a few epochs, validation perplexity should
   approach the count-based model's. Explain why: what is the minimizer of the
   cross-entropy over a row of $\mathbf{W}$, and how does it relate to the
   normalized counts?
6. **Weight decay vs. smoothing.** Add weight decay $\lambda$. On the full
   text, show that it slightly *hurts* validation perplexity. Then train on
   only the first 5% of the text and show that a well-chosen $\lambda$
   *helps*. Connect this to add-$\alpha$ smoothing from Part 1: both pull the
   model toward "every character is somewhat possible".
7. **Sample.** Generate text from the trained model (softmax of a row, then
   sample), as in Part 1.

## Acceptance Criteria

- Initial validation perplexity equals the vocabulary size (about 70).
- Gradient check passes (relative error about $10^{-6}$).
- With batch size 256 and learning rate 10, validation perplexity is about
  **11.7 after 3 epochs** and keeps approaching the count baseline of **about
  11.4** (within 2% after about 10 epochs).
- A plot of validation perplexity vs. $\lambda$ for the full text and for the
  5% subset, with one sentence explaining the difference.
- Your README's results table gains a row:

  | Part | Model | Held-out perplexity |
  |:--|:--|:--|
  | 1 | Character bigram (counts) | about 11.4 |
  | 2 | Character bigram (trained with SGD) | about 11.4--11.7 |

## Stretch Goals

- Use PyTorch (or this site's runnable cells) with `nn.Embedding`-style
  indexing and autograd instead of your hand-written gradient, and confirm
  both give the same parameters after one step.
- Add a learning-rate schedule (start high, decay each epoch). Does it reach
  the count baseline faster?

## How Far Quill Has Come

| Part | After chapter | Model | Held-out perplexity | Status |
|:--|:--|:--|--:|:--|
| 0 | | random guessing (uniform over 70 characters) | 70.0 | starting point |
| 1 | 2 | bigram from counts | **11.4** | done |
| 2 | 3 | bigram trained with minibatch SGD | **11.7** (→ 11.4) | done after this part |
| 3 | 4--5 | MLP over the previous few characters | goal: about 6--8 | next |
| 4 | 9--10 | recurrent network (RNN/GRU) | goal: about 4--6 | later |
| 5 | 11 | GPT-style Transformer | goal: below 4 | later |

Goals for future parts are rough targets; we'll replace them with the real
numbers you measure. Lower perplexity means the model is less "surprised"
by real text: 70 is pure guessing; about 1 would be perfect prediction.

Samples from each version, all starting with "T" and the same random seed
(newlines shown as spaces):

| Version | Sample |
|:--|:--|
| Random guessing | `T(AmY)OR8h,LUO[hwDbeE yEFsYQk!fK)cv;aEhg?ocdnOjr,pLR1eEqDXL]:9iiXu;p8x_]xll"J(:;q-ESpxf;Wf"dJ[ccUW:S-RVkL U;hL"` |
| Part 1: counts | `Tg thKPhas heckitarld walupis s9)_ughequgorkndes ted s t?)Gmaderes t whewis f afr d tur?'U(d orine l. mom masa` |
| Part 2: SGD | `Thashand s ged steng 'uldinde omMhrSoghilorkndes the rdimed addely t whewis f adie. turang pemione ieelom lasa` |

Random guessing produces symbol soup. Both bigram versions have learned
which letters follow which: spaces, common pairs like "th", "he", "ed", and
short word-like chunks. Parts 1 and 2 *look* alike because they are the same
model; Part 2's achievement is *how* it learned. Real words and grammar need
more context, which Part 3 adds.

| Part | Skills you gained (and can explain in an interview) |
|:--|:--|
| 1 | Tokenization; probability tables from data; smoothing; sampling; perplexity as an evaluation metric; deriving the softmax/cross-entropy gradient and checking it numerically |
| 2 | Training a model by minibatch SGD; designing Module/DataModule/Trainer code; tuning the learning rate; train vs. validation curves; weight decay as regularization (and when it hurts) |

## Why This Matters

Part 2 is the turning point of the capstone: the model is now *learned* by
gradient descent rather than *calculated*. That's what lets Part 3 add more
context (an MLP over the previous several characters), where counting
becomes impossible because the table would need $V^k$ rows.
