# Capstone: Quill, Part 4
:label:`sec_capstone_quill_4`

*A hidden layer that combines characters: Quill finally beats counting.*

In Part 3 (:numref:`sec_capstone_quill_3`) Quill became a softmax
regression over the previous $k$ characters, and it hit a wall at
perplexity 8.4. Counting 4-grams did better (5.8), because a linear model
can only **add up** separate votes from each position. It can't express "e
is likely after t *followed by* h". This chapter's tool is exactly what's
missing: a hidden layer computes features of *combinations* of inputs. In
this part, Quill becomes the model of Bengio et al. (2003), the first
neural language model. Each character gets a learned **embedding**, the
embeddings of the last $k$ characters are concatenated, and an MLP
predicts the next character.

**Uses:** MLPs and ReLU (:numref:`sec_mlp`), the concise implementation
(:numref:`sec_mlp-implementation`), backpropagation
(:numref:`sec_backprop`), initialization (:numref:`sec_numerical_stability`),
early stopping and dropout (Section 5.5, :numref:`sec_dropout`), plus your
Part 3 data module, cross-entropy, and error bars. Keep working in your
`quill` repository.

## The Model

Give every character $c$ a learned vector $\mathbf{e}_c \in \mathbb{R}^d$
(an *embedding*). For the previous $k$ characters, concatenate their
embeddings into one vector of length $kd$, and pass it through an MLP:

$$\mathbf{h} = \mathrm{ReLU}\big(\mathbf{W}_1 [\mathbf{e}_{c_{t-k}}; \ldots;
\mathbf{e}_{c_{t-1}}] + \mathbf{b}_1\big), \quad
P(c_t \mid \textrm{context}) = \mathrm{softmax}(\mathbf{W}_2\mathbf{h} + \mathbf{b}_2).$$

An embedding is just Part 3's trick made learnable: multiplying a one-hot
vector by a $V \times d$ matrix selects a row. In PyTorch, `nn.Embedding(V,
d)` does that lookup. Unlike Part 3, the model can now learn that "t then
h" is a pattern of its own.

## Milestones

1. **Data.** Reuse Part 3's $(\textrm{previous } k \textrm{ ids},
   \textrm{next id})$ windows with the same 90/10 split.
2. **Model.** `nn.Embedding(V, 16)`, flatten, `nn.Linear(k * 16, 256)`,
   ReLU, `nn.Linear(256, V)`. Train with `F.cross_entropy`, minibatch SGD
   (batch 128, learning rate 0.3), and $k = 8$. After 30 epochs, validation
   perplexity is about **5.6**, already below the count 4-gram.
3. **Context length.** Train $k = 3, 5, 8, 12$. Unlike Part 3, more context
   helps at first (5.8 → 5.3 for $k$ = 3 → 5). Then the model starts to
   overfit: with $k = 12$, validation perplexity rises after about epoch 7 while
   training perplexity keeps falling. Plot both curves and mark where
   early stopping would stop.
4. **Bigger, then regularized.** With 512 hidden units and no
   regularization, the model memorizes the book (training perplexity about
   3, validation worse than with 256 units). Add **dropout** 0.3 after the
   hidden layer (about 5.1). Then add a second hidden layer of 512 with
   dropout: about **4.5**.
5. **Initialization check.** Print the standard deviation of the
   activations of each layer at initialization and after one epoch. Then
   initialize the embeddings with standard deviation 10 instead of 1. What
   happens to the first epochs, and why?
6. **Error bar and accuracy.** Report next-character accuracy (about 56%,
   up from 39% in Part 3) and the standard error of the average loss
   (about 0.012), as in Part 3. Is the improvement over the 4-gram real?
7. **Sample.** Generate from "The T" at temperatures 0.5 and 1.0.
8. **Look inside.** Find the 5 nearest neighbors (cosine similarity) of a
   few characters' embeddings: "a", "e", "t", ".", "1". Do vowels cluster?
   Digits? Punctuation?

## Acceptance Criteria

- Held-out results (approximate; your numbers will differ slightly):

  | Model | Held-out perplexity |
  |:--|--:|
  | Count 4-gram (Part 3's comparison) | about 5.8 |
  | MLP, $k = 3$, 256 units | about 5.75 |
  | MLP, $k = 5$, 256 units | about 5.3 |
  | MLP, $k = 8$, 256 units | about 5.6 |
  | MLP, $k = 8$, 512 units, dropout 0.3 | about 5.1 |
  | MLP, $k = 8$, two layers of 512, dropout 0.3 | about **4.5** |

- Curves showing overfitting without dropout and its absence with it.
- Next-character accuracy of about 56% and an interval for the perplexity.
- Your README's results table gains a row:

  | Part | Model | Held-out perplexity |
  |:--|:--|:--|
  | 4 | MLP over 8 characters with embeddings and dropout | about 4.5 |

## Stretch Goals

- Replace ReLU with GELU, or use the Adam optimizer (Chapter 12 explains
  it). How low can you get with a fixed budget of 30 epochs?
- Count the parameters (about 366,000 for the best model). How does
  perplexity change as you scale the hidden size: 64, 128, 256, 512, 1024?
  Plot it on a log scale. That is a small **scaling law** (see Pace's
  side project *Scale*).
- Train on words instead of characters for the 2,000 most common words
  plus an "unknown" token. Is a word-level perplexity comparable to a
  character-level one? (Hint: no. Why?)

## How Far Quill Has Come

| Part | After chapter | Model | Held-out perplexity | Status |
|:--|:--|:--|--:|:--|
| 0 | | random guessing (uniform over 70 characters) | 70.0 | starting point |
| 1 | 2 | bigram from counts | **11.4** | done |
| 2 | 3 | bigram trained with minibatch SGD | **11.7** (→ 11.4) | done |
| 3 | 4 | softmax regression over 5 previous characters | **8.4** | done |
| 4 | 5 | MLP over 8 previous characters, with embeddings and dropout | **4.5** | done after this part |
| 5 | 9--10 | recurrent network (RNN/GRU) | goal: below 4.5 | next |
| 6 | 11 | GPT-style Transformer | goal: below 4 | later |

Samples from each version, starting with "T" (Parts 3 and 4 start with
"The T" and use temperature 0.5; newlines shown as spaces):

| Version | Sample |
|:--|:--|
| Random guessing | `T(AmY)OR8h,LUO[hwDbeE yEFsYQk!fK)cv;aEhg?ocdnOjr,pLR1eEqDXL]:9iiXu;p8x_]xll"J(:;q-ESpxf;Wf"dJ[ccUW:S-RVkL U;hL"` |
| Part 1: counts | `Tg thKPhas heckitarld walupis s9)_ughequgorkndes ted s t?)Gmaderes t whewis f afr d tur?'U(d orine l. mom masa` |
| Part 2: SGD | `Thashand s ged steng 'uldinde omMhrSoghilorkndes the rdimed addely t whewis f adie. turang pemione ieelom lasa` |
| Part 3: 5 characters, $T = 0.5$ | `The Time Trevery terend of the wored boundes of the s of the mine far the bound lownith thed ande tromithe the mo that per ma` |
| Part 4: MLP, $T = 0.5$ | `The Time Machine was a hour of the way of their over the truth a fire. I had seen eques from myself. My ginner of the little` |

Quill now writes "The Time Machine", full sentences with periods and
capitals, and phrases like "I had seen" and "of the little". It still has
no memory beyond 8 characters, so the sentences don't hang together. That
is what recurrent networks (Part 5) add.

| Part | Skills you gained (and can explain in an interview) |
|:--|:--|
| 1 | Tokenization; probability tables from data; smoothing; sampling; perplexity as an evaluation metric; deriving the softmax/cross-entropy gradient and checking it numerically |
| 2 | Training a model by minibatch SGD; designing Module/DataModule/Trainer code; tuning the learning rate; train vs. validation curves; weight decay as regularization (and when it hurts) |
| 3 | Framing language modeling as classification; one-hot features and why they become embedding lookups; numerically stable cross-entropy; error bars on evaluation metrics; the limits of linear models (no interactions); measuring distribution shift between texts |
| 4 | Learned embeddings; MLPs that combine inputs; the first neural language model (Bengio et al., 2003); diagnosing overfitting from curves; dropout and early stopping; scaling width and depth |

## Why This Matters

Embeddings plus a stack of nonlinear layers trained with cross-entropy
is still the core of every large language model. A GPT's first layer is an
`nn.Embedding`, and half of every Transformer block is an MLP exactly like
this one (with GELU). What the MLP can't do is look further back than its
fixed window, or decide *which* earlier characters matter. Recurrent
networks (Part 5) add memory, and attention (Part 6) adds the ability to
look anywhere in the context.
