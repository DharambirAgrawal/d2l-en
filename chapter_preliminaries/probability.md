```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Probability and Statistics
:label:`sec_prob`

Machine learning is, one way or another, all about uncertainty. In supervised
learning, we predict something unknown (the *target*) from something known
(the *features*). We might predict the most likely target value, or the value
with the smallest expected distance from the target, or we might want to
*quantify our uncertainty*: given a patient's features, *how likely* are they
to have a heart attack next year? In unsupervised learning, deciding whether
measurements are anomalous requires knowing how likely such values are in the
population of interest. In reinforcement learning, agents must reason about
how the environment might change and what rewards to expect for each action.

*Probability* is the mathematics of reasoning under uncertainty: given a
probabilistic model of a process, we can reason about how likely various
events are. Using probabilities for the frequencies of repeatable events (like
coin tosses) is uncontroversial; *frequentist* scholars apply probability
*only* to such events. *Bayesian* scholars use probability more broadly to
formalize reasoning under uncertainty, with two distinctive features:
(i) assigning degrees of belief to non-repeatable events, e.g., what is the
*probability* that a dam will collapse?; and (ii) subjectivity: Bayesian
probability gives unambiguous rules for updating beliefs given new evidence,
but different people may start with different *prior* beliefs. *Statistics*
reasons backwards: from collecting and organizing data to inferences about
the process that generated it. Whenever we hunt for patterns in a dataset
that might describe a broader population, we're thinking statistically.
Whole careers and institutions are devoted to probability and statistics; this
section just lays the foundation you need to start building models.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from mxnet import np, npx
from mxnet.numpy.random import multinomial
import random
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
import random
import torch
from torch.distributions.multinomial import Multinomial
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
import random
import tensorflow as tf
from tensorflow_probability import distributions as tfd
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
import random
import jax
from jax import numpy as jnp
import numpy as np
```

## A Simple Example: Tossing Coins

Say we toss a coin and want to quantify how likely heads (vs. tails) is. If
the coin is *fair*, both outcomes are equally likely, and over $n$ tosses we
*expect* the same fraction of heads as tails. One intuition is symmetry: for
every outcome with $n_\textrm{h}$ heads and $n_\textrm{t} = (n - n_\textrm{h})$
tails, there's an equally likely outcome with $n_\textrm{t}$ heads and
$n_\textrm{h}$ tails, which is only possible if on average $1/2$ of tosses come
up heads and $1/2$ tails. Still, if you repeat the experiment many times with
$n=1000000$ tosses each, you might never see $n_\textrm{h} = n_\textrm{t}$
exactly.


Formally, $1/2$ is a *probability*: it captures how certain we are that any
given toss comes up heads. Probabilities assign scores between $0$ and $1$ to
outcomes of interest, called *events*; here the event is $\textrm{heads}$, with
probability $P(\textrm{heads})$. A probability of $1$ means certainty (a trick
coin with heads on both sides) and $0$ means impossibility (tails on both
sides). The frequencies $n_\textrm{h}/n$ and $n_\textrm{t}/n$ are not
probabilities but *statistics*. Probabilities are *theoretical* quantities of
the data-generating process (here, $1/2$ is a property of the coin), while
statistics are *empirical* quantities computed from observed data. The two
are intertwined: we design statistics called *estimators* that produce
*estimates* of model parameters such as probabilities from a dataset. When an
estimator is *consistent*, its estimates converge to the true probability,
and the inferred probabilities in turn tell us about future data from the
same population.

Suppose we find a real coin with unknown $P(\textrm{heads})$. To study it
statistically, we (i) collect data and (ii) design an estimator. Collecting
data is easy: toss the coin many times and record the outcomes. Drawing
realizations from an underlying random process is called *sampling*. A
natural estimator is the number of observed *heads* divided by the total
number of tosses.

Now suppose the coin is actually fair, $P(\textrm{heads}) = 0.5$. Any random
number generator can simulate it. Python's `random.random` returns numbers in
$[0,1]$, where the probability of landing in any sub-interval
$[a, b] \subset [0,1]$ is $b-a$. So testing whether the result exceeds `0.5`
gives `0` or `1` with probability `0.5` each:

```{.python .input}
%%tab all
num_tosses = 100
heads = sum([random.random() > 0.5 for _ in range(num_tosses)])
tails = num_tosses - heads
print("heads, tails: ", [heads, tails])
```

More generally, we can simulate draws of any variable with finitely many
outcomes (like a coin or a die) with the multinomial function: the first
argument is the number of draws, the second the list of outcome
probabilities. For ten tosses of a fair coin we use `[0.5, 0.5]`, with index 0
as heads and index 1 as tails. It returns a vector with one entry per outcome
(here, 2): the number of heads and the number of tails.

```{.python .input}
%%tab mxnet
fair_probs = [0.5, 0.5]
multinomial(100, fair_probs)
```

```{.python .input}
%%tab pytorch
fair_probs = torch.tensor([0.5, 0.5])
Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab tensorflow
fair_probs = tf.ones(2) / 2
tfd.Multinomial(100, fair_probs).sample()
```

```{.python .input}
%%tab jax
fair_probs = [0.5, 0.5]
# jax.random does not have multinomial distribution implemented
np.random.multinomial(100, fair_probs)
```

Each run gives a new random result. Dividing by the number of tosses gives
the *frequency* of each outcome; like the probabilities they estimate, the
frequencies sum to $1$.

```{.python .input}
%%tab mxnet
multinomial(100, fair_probs) / 100
```

```{.python .input}
%%tab pytorch
Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab tensorflow
tfd.Multinomial(100, fair_probs).sample() / 100
```

```{.python .input}
%%tab jax
np.random.multinomial(100, fair_probs) / 100
```

Even though our simulated coin is fair (we set `[0.5, 0.5]` ourselves), the
counts of heads and tails may differ, because we drew relatively few samples.
If we only saw the outcome, how could we tell a slightly unfair coin from a
deviation caused by the small sample? Let's simulate 10,000 tosses.

```{.python .input}
%%tab mxnet
counts = multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

```{.python .input}
%%tab pytorch
counts = Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(10000, fair_probs).sample()
counts / 10000
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(10000, fair_probs).astype(np.float32)
counts / 10000
```

In general, for averages of repeated events (like coin tosses), the estimates
are guaranteed to converge to the true probabilities as the number of
repetitions grows. This is the *law of large numbers*, and the
*central limit theorem* says that in many situations the error shrinks at a
rate of $(1/\sqrt{n})$ as the sample size $n$ grows. To build intuition, let's
watch our estimate evolve as the number of tosses grows from 1 to 10,000.

```{.python .input}
%%tab pytorch
counts = Multinomial(1, fair_probs).sample((10000,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)
estimates = estimates.numpy()

d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

```{.python .input}
%%tab mxnet
counts = multinomial(1, fair_probs, size=10000)
cum_counts = counts.astype(np.float32).cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab tensorflow
counts = tfd.Multinomial(1, fair_probs).sample(10000)
cum_counts = tf.cumsum(counts, axis=0)
estimates = cum_counts / tf.reduce_sum(cum_counts, axis=1, keepdims=True)
estimates = estimates.numpy()
```

```{.python .input}
%%tab jax
counts = np.random.multinomial(1, fair_probs, size=10000).astype(np.float32)
cum_counts = counts.cumsum(axis=0)
estimates = cum_counts / cum_counts.sum(axis=1, keepdims=True)
```

```{.python .input}
%%tab mxnet, tensorflow, jax
d2l.set_figsize((4.5, 3.5))
d2l.plt.plot(estimates[:, 0], label=("P(coin=heads)"))
d2l.plt.plot(estimates[:, 1], label=("P(coin=tails)"))
d2l.plt.axhline(y=0.5, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Samples')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend();
```

Each solid curve is one of the coin's two values, showing our estimated
probability of that value after each group of experiments; the dashed black
line is the true probability. With more data, the curves converge to the
truth. You can already glimpse the advanced questions statisticians care
about: How fast does convergence happen? If we had already tested many coins
from the same plant, how could we use that information?

Experiment with the convergence yourself:

```{=html}
<div class="d2l-playground" data-playground="coin-convergence"></div>
```

##  A More Formal Treatment

We've already posed a probabilistic model, generated synthetic data, run a
statistical estimator, assessed convergence empirically, and reported error
metrics (checking the deviation). To go further, we need more precision.


For a random process, the set of possible outcomes $\mathcal{S}$ is the
*sample space* (or *outcome space*), and each element is a distinct
*outcome*. For one coin toss, $\mathcal{S} = \{\textrm{heads}, \textrm{tails}\}$;
for one die, $\mathcal{S} = \{1, 2, 3, 4, 5, 6\}$; for two coins, the outcomes
are
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails}), (\textrm{tails}, \textrm{heads}),  (\textrm{tails}, \textrm{tails})\}$.
*Events* are subsets of the sample space: "the first toss comes up heads" is
$\{(\textrm{heads}, \textrm{heads}), (\textrm{heads}, \textrm{tails})\}$. When
an experiment's outcome $z$ satisfies $z \in \mathcal{A}$, event $\mathcal{A}$
has occurred. For one die roll, define "seeing a $5$" ($\mathcal{A} = \{5\}$)
and "seeing an odd number" ($\mathcal{B} = \{1, 3, 5\}$). If the die shows
$5$, both $\mathcal{A}$ and $\mathcal{B}$ occurred; if $z = 3$, only
$\mathcal{B}$ did.


A *probability* function maps events to real values
${P: \mathcal{A} \subseteq \mathcal{S} \rightarrow [0,1]}$. The probability
$P(\mathcal{A})$ of an event $\mathcal{A}$ in sample space $\mathcal{S}$ has
these properties:

* The probability of any event $\mathcal{A}$ is a nonnegative real number, i.e., $P(\mathcal{A}) \geq 0$;
* The probability of the entire sample space is $1$, i.e., $P(\mathcal{S}) = 1$;
* For any countable sequence of events $\mathcal{A}_1, \mathcal{A}_2, \ldots$ that are *mutually exclusive* (i.e., $\mathcal{A}_i \cap \mathcal{A}_j = \emptyset$ for all $i \neq j$), the probability that any of them happens is equal to the sum of their individual probabilities, i.e., $P(\bigcup_{i=1}^{\infty} \mathcal{A}_i) = \sum_{i=1}^{\infty} P(\mathcal{A}_i)$.

These axioms, proposed by :citet:`Kolmogorov.1933`, quickly yield important
consequences. The probability that $\mathcal{A}$ *or* its complement
$\mathcal{A}'$ occurs is 1 (since $\mathcal{A} \cup \mathcal{A}' = \mathcal{S}$).
Also $P(\emptyset) = 0$, because
$1 = P(\mathcal{S} \cup \mathcal{S}') = P(\mathcal{S} \cup \emptyset) = P(\mathcal{S}) + P(\emptyset) = 1 + P(\emptyset)$.
So the probability that $\mathcal{A}$ *and* $\mathcal{A}'$ occur together is
$P(\mathcal{A} \cap \mathcal{A}') = 0$: impossible events have zero
probability.



## Random Variables

Events like "the die comes up odd" or "the first toss is heads" invoke
*random variables*: mappings from an underlying sample space to a set of
(possibly many) values. How is that different from the sample space itself?
Random variables can be much coarser: a binary variable like "greater than
0.5" can be defined even when the sample space is infinite (e.g., points
between $0$ and $1$). And several random variables can share one sample
space: "my home alarm goes off" and "my house was burgled" are both binary
random variables on the same sample space, so knowing one tells us something
about the other. If the alarm went off, we might suspect a burglary.


Each value of a random variable corresponds to a subset of the sample space,
so "$X$ takes value $v$", written $X=v$, is an *event* with probability
$P(X=v)$. When the context is clear, we abuse notation. $P(X)$ can mean the
*distribution* of $X$, the function giving the probability of each value.
$P(X,Y) = P(X) P(Y)$ is shorthand for a statement true for all values of $X$
and $Y$: for all $i,j$, $P(X=i \textrm{ and } Y=j) = P(X=i)P(Y=j)$. And we may
write $P(v)$ when the random variable is obvious. Since events are sets of
outcomes, we can also specify ranges: $P(1 \leq X \leq 3)$ is the probability
of the event $\{1 \leq X \leq 3\}$.


There's a subtle difference between *discrete* random variables, like coin
flips or die rolls, and *continuous* ones, like the weight and height of a
randomly chosen person. We rarely care about someone's exact height: measured
precisely enough, no two people have the same height, and you aren't even the
same height when you wake up and go to sleep. There's little point asking for
the probability that someone is exactly 1.801392782910287192 meters tall;
we'd rather ask whether their height is in an interval, say 1.79 to 1.81
meters. For this we use probability *densities*: a height of exactly 1.80
meters has zero probability but nonzero density, and the probability of an
interval is the *integral* of the density over it.

## Multiple Random Variables

We couldn't even get through the last section without statements about
several random variables at once (recall $P(X,Y) = P(X) P(Y)$). Most of
machine learning is about such relationships. The sample space is the
population of interest, say a business's customers, photos on the Internet,
or proteins known to biologists, and each random variable is the (unknown)
value of a different attribute. Sampling an individual gives a realization of
every random variable. Because the values of random variables correspond to
subsets of the sample space that may overlap partially, fully, or not at all,
knowing one variable's value can change our beliefs about another's. If a
patient has trouble breathing and has lost their sense of smell, we think
COVID-19 is more likely than if they breathed normally and could smell fine.


With multiple random variables, we can form an event for every combination of
values they can jointly take. The function giving the probability of each
combination (e.g., $A=a$ and $B=b$) is the *joint probability* function: it
returns the probability of the intersection of the corresponding subsets of
the sample space. The *joint probability* that $A$ and $B$ take values $a$ and
$b$ is written $P(A = a, B = b)$, where the comma means "and". For any $a$ and
$b$,

$$P(A=a, B=b) \leq P(A=a) \textrm{ and } P(A=a, B=b) \leq P(B = b),$$

since for $A=a$ and $B=b$ to happen, $A=a$ *and* $B=b$ must each happen. The
joint probability tells us everything we can know about these random
variables probabilistically, and from it we can derive many other useful
quantities, including the individual distributions $P(A)$ and $P(B)$: to
recover $P(A=a)$, sum $P(A=a, B=v)$ over all values $v$ of $B$:
$P(A=a) = \sum_v P(A=a, B=v)$.


The ratio $\frac{P(A=a, B=b)}{P(A=a)} \leq 1$ turns out to be extremely
important. It is the *conditional probability*, written with "$\mid$":

$$P(B=b \mid A=a) = P(A=a,B=b)/P(A=a).$$

It is the new probability of $B=b$ once we know $A=a$ happened: restrict
attention to the part of the sample space where $A=a$ and renormalize so the
probabilities sum to 1. Conditional probabilities are ordinary probabilities
and obey all the axioms, as long as every term is conditioned on the same
event. For instance, for disjoint events $\mathcal{B}$ and $\mathcal{B}'$,
$P(\mathcal{B} \cup \mathcal{B}' \mid A = a) = P(\mathcal{B} \mid A = a) + P(\mathcal{B}' \mid A = a)$.


From the definition of conditional probability we get the famous
*Bayes' theorem*. By construction, $P(A, B) = P(B\mid A) P(A)$ and
$P(A, B) = P(A\mid B) P(B)$. Combining them gives
$P(B\mid A) P(A) = P(A\mid B) P(B)$ and hence

$$P(A \mid B) = \frac{P(B\mid A) P(A)}{P(B)}.$$






This simple equation has profound implications because it lets us reverse the
order of conditioning: if we can estimate $P(B\mid A)$, $P(A)$, and $P(B)$, we
can estimate $P(A\mid B)$. Often one term is easy to estimate directly and the
other isn't, and Bayes' theorem comes to the rescue. For instance, knowing how
common symptoms are for a disease, plus the overall prevalence of the disease
and of the symptoms, we can work out how likely someone is to have the disease
given their symptoms. If we don't know $P(B)$ (e.g., the prevalence of the
symptoms), a simplified version helps:

$$P(A \mid B) \propto P(B \mid A) P(A).$$

Since $P(A \mid B)$ must sum to $1$, i.e., $\sum_a P(A=a \mid B) = 1$, we can
compute

$$P(A \mid B) = \frac{P(B \mid A) P(A)}{\sum_a P(B \mid A=a) P(A = a)}.$$

In Bayesian statistics, an observer has (subjective) prior beliefs about the
plausibility of the hypotheses, encoded in the *prior* $P(H)$, and a
*likelihood function* $P(E \mid H)$ saying how likely each value of the
collected evidence is under each hypothesis. Bayes' theorem says how to update
the *prior* $P(H)$ in light of evidence $E$ to get the *posterior*
$P(H \mid E) = \frac{P(E \mid H) P(H)}{P(E)}$. Informally: "posterior equals
prior times likelihood, divided by the evidence". Since the evidence $P(E)$ is
the same for every hypothesis, we can just normalize over the hypotheses.

$\sum_a P(A=a \mid B) = 1$ also lets us *marginalize* over random variables,
i.e., drop variables from a joint distribution such as $P(A, B)$:

$$\sum_a P(B \mid A=a) P(A=a) = \sum_a P(B, A=a) = P(B).$$

*Independence* is another fundamental concept, the backbone of many ideas in
statistics. Two variables are *independent* if conditioning on the value of
$A$ doesn't change the distribution of $B$, and vice versa. Formally,
$A \perp B$ requires $P(A \mid B) = P(A)$, and therefore
$P(A,B) = P(A \mid B) P(B) = P(A) P(B)$. Independence is often a reasonable
assumption: if $A$ is the toss of one fair coin and $B$ the toss of another,
knowing whether $A$ came up heads shouldn't change the probability that $B$
does.


Independence is especially useful between successive draws of our data from
a distribution (enabling strong statistical conclusions), or between
variables in our data (letting us use simpler models that encode it). On the
other hand, estimating dependencies is often the very goal of learning: we
estimate the probability of disease given symptoms precisely because we
believe diseases and symptoms are *not* independent.


Since conditional probabilities are proper probabilities, independence
applies to them too. $A$ and $B$ are *conditionally independent* given $C$ if
and only if $P(A, B \mid C) = P(A \mid C)P(B \mid C)$. Interestingly, two
variables can be independent in general yet dependent given a third, often
when $A$ and $B$ are both causes of $C$. Broken bones and lung cancer may be
independent in the general population, but among hospital patients broken
bones are negatively correlated with lung cancer: the broken bone
*explains away* why someone is in the hospital, lowering the probability that
lung cancer is the reason.


Conversely, two dependent variables can become independent given a third,
often when two otherwise unrelated things have a common cause. Shoe size and
reading level are highly correlated among elementary school students, but the
correlation disappears once we condition on age.



## An Example
:label:`subsec_probability_hiv_app`

Let's test our skills. A doctor gives a patient an HIV test. The test is
fairly accurate: it fails only 1% of the time when a healthy patient is
reported as diseased (healthy patients test positive in 1% of cases), and it
never misses HIV when the patient has it. Let $D_1 \in \{0, 1\}$ be the
diagnosis ($0$ negative, $1$ positive) and $H \in \{0, 1\}$ the HIV status.

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_1 = 1 \mid H)$        |     1 |  0.01 |
| $P(D_1 = 0 \mid H)$        |     0 |  0.99 |

The columns sum to 1 (the rows don't), since these are conditional
probabilities. What is the probability the patient has HIV if the test is
positive, $P(H = 1 \mid D_1 = 1)$? Intuitively it depends on how common the
disease is, since that drives the number of false alarms. Suppose the disease
is rare, e.g., $P(H=1) = 0.0015$. To apply Bayes' theorem, we first
marginalize:

$$\begin{aligned}
P(D_1 = 1)
=& P(D_1=1, H=0) + P(D_1=1, H=1)  \\
=& P(D_1=1 \mid H=0) P(H=0) + P(D_1=1 \mid H=1) P(H=1) \\
=& 0.011485.
\end{aligned}
$$

This gives

$$P(H = 1 \mid D_1 = 1) = \frac{P(D_1=1 \mid H=1) P(H=1)}{P(D_1=1)} = 0.1306.$$

So there's only a 13.06% chance the patient actually has HIV, despite the
quite accurate test. Probability can be counterintuitive! Faced with such
terrifying news, the patient would likely ask for another test. The second
test has different characteristics and isn't as good as the first:

| Conditional probability | $H=1$ | $H=0$ |
|:------------------------|------:|------:|
| $P(D_2 = 1 \mid H)$          |  0.98 |  0.03 |
| $P(D_2 = 0 \mid H)$          |  0.02 |  0.97 |

Unfortunately, it's positive too. Assuming the tests are conditionally
independent, we compute the probabilities needed for Bayes' theorem:

$$\begin{aligned}
P(D_1 = 1, D_2 = 1 \mid H = 0)
& = P(D_1 = 1 \mid H = 0) P(D_2 = 1 \mid H = 0)
=& 0.0003, \\
P(D_1 = 1, D_2 = 1 \mid H = 1)
& = P(D_1 = 1 \mid H = 1) P(D_2 = 1 \mid H = 1)
=& 0.98.
\end{aligned}
$$

Marginalizing gives the probability that both tests are positive:

$$\begin{aligned}
&P(D_1 = 1, D_2 = 1)\\
&= P(D_1 = 1, D_2 = 1, H = 0) + P(D_1 = 1, D_2 = 1, H = 1)  \\
&= P(D_1 = 1, D_2 = 1 \mid H = 0)P(H=0) + P(D_1 = 1, D_2 = 1 \mid H = 1)P(H=1)\\
&= 0.00176955.
\end{aligned}
$$

Finally, the probability that the patient has HIV given two positive tests is

$$P(H = 1 \mid D_1 = 1, D_2 = 1)
= \frac{P(D_1 = 1, D_2 = 1 \mid H=1) P(H=1)}{P(D_1 = 1, D_2 = 1)}
= 0.8307.$$

The second test gives much higher confidence that something is wrong. Even
though it is much less accurate than the first, it greatly improved our
estimate. The assumption that the tests are conditionally independent was
crucial: if we ran the *same* test twice, we'd expect the same outcome both
times and learn nothing new. Notice that the diagnosis behaves like a
classifier hiding in plain sight: our ability to tell whether a patient is
healthy grows as we get more features (test outcomes).


Explore how prevalence and test accuracy change the answer:

```{=html}
<div class="d2l-playground" data-playground="bayes-test"></div>
```

## Expectations

Decisions often need more than the probabilities of individual events: we
combine them into useful aggregates. For random variables with continuous
scalar values, we often want the value to expect *on average*, formally the
*expectation*. For an investment, the first question is the expected return,
averaging over all outcomes weighted by their probabilities. Say an
investment fails completely with 50% probability, returns 2$\times$ with 40%
probability, and returns 10$\times$ with 10% probability. The expected return
sums each return times its probability:
$0.5 \cdot 0 + 0.4 \cdot 2 + 0.1 \cdot 10 = 1.8$, i.e., 1.8$\times$.


In general, the *expectation* (or average) of a random variable $X$ is

$$E[X] = E_{x \sim P}[x] = \sum_{x} x P(X = x).$$

For densities, $E[X] = \int x \;dp(x)$. The expected value of a function of
$x$ is

$$E_{x \sim P}[f(x)] = \sum_x f(x) P(x) \textrm{ and } E_{x \sim P}[f(x)] = \int f(x) p(x) \;dx$$

for discrete probabilities and densities, respectively. In the investment
example, $f$ might be the *utility* (happiness) of the return. Behavioral
economists have long noted that losing money hurts more than gaining the same
amount helps, and that money's value is sub-linear: having 100k dollars
versus zero can be the difference between paying rent, eating well, and good
healthcare versus homelessness, while 200k versus 100k matters much less.
Hence the cliché that "the utility of money is logarithmic".


If a total loss had utility $-1$ and returns of $1$, $2$, and $10$ had
utilities $1$, $2$, and $4$, the expected happiness of investing would be
$0.5 \cdot (-1) + 0.4 \cdot 2 + 0.1 \cdot 4 = 0.7$ (an expected 30% loss of
utility). With that utility function, you'd be better off keeping your money
in the bank.

For financial decisions, we also want to know how *risky* an investment is:
not just the expected value, but how much actual values *vary* around it. We
can't just take the expected difference between actual and expected values,
since the expectation of a difference is the difference of expectations:
$E[X - E[X]] = E[X] - E[E[X]] = 0$. But we can take the expectation of any
non-negative function of the difference. The *variance* is the expected
*squared* difference:

$$\textrm{Var}[X] = E\left[(X - E[X])^2\right] = E[X^2] - E[X]^2.$$

The equality follows by expanding $(X - E[X])^2 = X^2 - 2 X E[X] + E[X]^2$
and taking expectations term by term. The square root of the variance is the
*standard deviation*. Both carry the same information (each can be computed
from the other), but the standard deviation is in the same units as the
original quantity.

The variance of a function of a random variable is defined analogously:

$$\textrm{Var}_{x \sim P}[f(x)] = E_{x \sim P}[f^2(x)] - E_{x \sim P}[f(x)]^2.$$

For our investment, the variance is
$0.5 \cdot 0 + 0.4 \cdot 2^2 + 0.1 \cdot 10^2 - 1.8^2 = 8.36$: a risky
investment by any standard. By convention, the mean and variance are often
written $\mu$ and $\sigma^2$, especially when they parametrize a Gaussian
distribution.

Expectations and variances extend to vector-valued random variables.
Expectations apply elementwise:
$\boldsymbol{\mu} \stackrel{\textrm{def}}{=} E_{\mathbf{x} \sim P}[\mathbf{x}]$
has coordinates $\mu_i = E_{\mathbf{x} \sim P}[x_i]$. *Covariances* are more
complicated: they are the expectation of the *outer product* of the
difference between the random variable and its mean:

$$\boldsymbol{\Sigma} \stackrel{\textrm{def}}{=} \textrm{Cov}_{\mathbf{x} \sim P}[\mathbf{x}] = E_{\mathbf{x} \sim P}\left[(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top\right].$$

$\boldsymbol{\Sigma}$ is the covariance matrix. To see what it does, take a
vector $\mathbf{v}$ of the same size as $\mathbf{x}$:

$$\mathbf{v}^\top \boldsymbol{\Sigma} \mathbf{v} = E_{\mathbf{x} \sim P}\left[\mathbf{v}^\top(\mathbf{x} - \boldsymbol{\mu}) (\mathbf{x} - \boldsymbol{\mu})^\top \mathbf{v}\right] = \textrm{Var}_{x \sim P}[\mathbf{v}^\top \mathbf{x}].$$

So $\boldsymbol{\Sigma}$ gives the variance of any linear function of
$\mathbf{x}$ through a simple matrix multiplication. The off-diagonal
elements say how correlated the coordinates are: 0 means no correlation, and
larger positive values mean stronger correlation.



## Discussion

Machine learning has plenty to be uncertain about: the label for an input,
the estimated value of a parameter, or even whether deployment data comes
from the same distribution as the training data.

*Aleatoric uncertainty* is intrinsic to the problem: genuine randomness not
accounted for by the observed variables. *Epistemic uncertainty* is about a
model's parameters, the kind we can hope to reduce with more data. We may be
epistemically uncertain about a coin's probability of heads, but even once
we know it, aleatoric uncertainty about the next toss remains: however long
we watch someone toss a fair coin, we'll never be more or less than 50% sure
the next toss is heads. The terms come from mechanical modeling (see, e.g.,
:citet:`Der-Kiureghian.Ditlevsen.2009` for a review of
[uncertainty quantification](https://en.wikipedia.org/wiki/Uncertainty_quantification)).
They slightly abuse language, though: *epistemic* refers to anything about
*knowledge*, so philosophically all uncertainty is epistemic.


Sampling from an unknown distribution lets us estimate its parameters, but
the rate can be slow. For coin tossing (and many other cases), the best
estimators converge at a rate of $1/\sqrt{n}$, where $n$ is the sample size
(e.g., the number of tosses). Going from 10 to 1000 observations (usually
easy) cuts uncertainty tenfold, but the next 1000 observations help much
less, only a 1.41 times reduction. This is a persistent feature of machine
learning: easy early gains, then very large amounts of data (and often
computation) for further gains. For an empirical review of this for
large-scale language models, see :citet:`Revels.Lubin.Papamarkou.2016`.

We also sharpened our language and tools for statistical modeling, including
conditional probabilities and one of the most important equations in
statistics, Bayes' theorem. It separates the information in the data, via a
likelihood term $P(B \mid A)$ measuring how well observations $B$ match
parameters $A$, from a prior $P(A)$ saying how plausible $A$ was to begin
with. We applied it to assign probabilities to diagnoses based on both the
test's efficacy *and* the disease's prevalence (our prior).

Finally, we met expectations and variances, our first nontrivial summaries
of a probability distribution. There are many more than linear and quadratic
expectations, but these two already say a lot about a distribution's
behavior. For instance,
[Chebyshev's inequality](https://en.wikipedia.org/wiki/Chebyshev%27s_inequality)
states $P(|X - \mu| \geq k \sigma) \leq 1/k^2$, where $\mu$ is the
expectation, $\sigma^2$ the variance, and $k > 1$ a confidence parameter we
choose. It says draws lie within $[-\sqrt{2} \sigma, \sqrt{2} \sigma]$ of the
expectation with at least 50% probability.




## Exercises

1. Give an example where observing more data can reduce the amount of uncertainty about the outcome to an arbitrarily low level.
1. Give an example where observing more data will only reduce the amount of uncertainty up to a point and then no further. Explain why this is the case and where you expect this point to occur.
1. We empirically demonstrated convergence to the mean for the toss of a coin. Calculate the variance of the estimate of the probability that we see a head after drawing $n$ samples.
    1. How does the variance scale with the number of observations?
    1. Use Chebyshev's inequality to bound the deviation from the expectation.
    1. How does it relate to the central limit theorem?
1. Assume that we draw $m$ samples $x_i$ from a probability distribution with zero mean and unit variance. Compute the averages $z_m \stackrel{\textrm{def}}{=} m^{-1} \sum_{i=1}^m x_i$. Can we apply Chebyshev's inequality for every $z_m$ independently? Why not?
1. Given two events with probability $P(\mathcal{A})$ and $P(\mathcal{B})$, compute upper and lower bounds on $P(\mathcal{A} \cup \mathcal{B})$ and $P(\mathcal{A} \cap \mathcal{B})$. Hint: graph the situation using a [Venn diagram](https://en.wikipedia.org/wiki/Venn_diagram).
1. Assume that we have a sequence of random variables, say $A$, $B$, and $C$, where $B$ only depends on $A$, and $C$ only depends on $B$, can you simplify the joint probability $P(A, B, C)$? Hint: this is a [Markov chain](https://en.wikipedia.org/wiki/Markov_chain).
1. In :numref:`subsec_probability_hiv_app`, assume that the outcomes of the two tests are not independent. In particular assume that either test on its own has a false positive rate of 10% and a false negative rate of 1%. That is, assume that $P(D =1 \mid H=0) = 0.1$ and that $P(D = 0 \mid H=1) = 0.01$. Moreover, assume that for $H = 1$ (infected) the test outcomes are conditionally independent, i.e., that $P(D_1, D_2 \mid H=1) = P(D_1 \mid H=1) P(D_2 \mid H=1)$ but that for healthy patients the outcomes are coupled via $P(D_1 = D_2 = 1 \mid H=0) = 0.02$.
    1. Work out the joint probability table for $D_1$ and $D_2$, given $H=0$ based on the information you have so far.
    1. Derive the probability that the patient is diseased ($H=1$) after one test returns positive. You can assume the same baseline probability $P(H=1) = 0.0015$ as before.
    1. Derive the probability that the patient is diseased ($H=1$) after both tests return positive.
1. Assume that you are an asset manager for an investment bank and you have a choice of stocks $s_i$ to invest in. Your portfolio needs to add up to $1$ with weights $\alpha_i$ for each stock. The stocks have an average return $\boldsymbol{\mu} = E_{\mathbf{s} \sim P}[\mathbf{s}]$ and covariance $\boldsymbol{\Sigma} = \textrm{Cov}_{\mathbf{s} \sim P}[\mathbf{s}]$.
    1. Compute the expected return for a given portfolio $\boldsymbol{\alpha}$.
    1. If you wanted to maximize the return of the portfolio, how should you choose your investment?
    1. Compute the *variance* of the portfolio.
    1. Formulate an optimization problem of maximizing the return while keeping the variance constrained to an upper bound. This is the Nobel-Prize winning [Markovitz portfolio](https://en.wikipedia.org/wiki/Markowitz_model) :cite:`Mangram.2013`. To solve it you will need a quadratic programming solver, something way beyond the scope of this book.
