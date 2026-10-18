```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Automatic Differentiation
:label:`sec_autograd`

Recall from :numref:`sec_calculus` that computing derivatives is the crucial
step in every optimization algorithm we'll use to train deep networks. The
calculations are straightforward, but doing them by hand is tedious and
error-prone, and gets worse as models grow more complex.

Fortunately, all modern deep learning frameworks do this for us with
*automatic differentiation* (*autograd*). As data passes through each
function, the framework builds a *computational graph* tracking how each value
depends on others. To compute derivatives, autograd works backwards through
this graph applying the chain rule, an algorithm called *backpropagation*.

Autograd libraries have been a hot topic for the past decade, but they have a
long history: the earliest references date back over half a century
:cite:`Wengert.1964`. The core ideas of modern backpropagation come from a
1980 PhD thesis :cite:`Speelpenning.1980` and were developed further in the
late 1980s :cite:`Griewank.1989`. Backpropagation is now the default way to
compute gradients, but not the only one: the Julia language, for instance,
uses forward propagation :cite:`Revels.Lubin.Papamarkou.2016`. First, let's
master the autograd package.

```{.python .input}
%%tab mxnet
from mxnet import autograd, np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
import torch
```

```{.python .input}
%%tab tensorflow
import tensorflow as tf
```

```{.python .input}
%%tab jax
from jax import numpy as jnp
```

## A Simple Function

Say we want to differentiate $y = 2\mathbf{x}^{\top}\mathbf{x}$ with respect
to the column vector $\mathbf{x}$. First, we give `x` an initial value.

```{.python .input  n=1}
%%tab mxnet
x = np.arange(4.0)
x
```

```{.python .input  n=7}
%%tab pytorch
x = torch.arange(4.0)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(4, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(4.0)
x
```

:begin_tab:`pytorch, tensorflow`
Before computing the gradient of $y$ with respect to $\mathbf{x}$, we need
somewhere to store it. We avoid allocating new memory for every derivative,
since deep learning computes derivatives with respect to the same parameters
a great many times and could run out of memory. The gradient of a
scalar-valued function with respect to a vector $\mathbf{x}$ is a vector with
the same shape as $\mathbf{x}$.
:end_tab:

```{.python .input  n=8}
%%tab mxnet
# We allocate memory for a tensor's gradient by invoking `attach_grad`
x.attach_grad()
# After we calculate a gradient taken with respect to `x`, we will be able to
# access it via the `grad` attribute, whose values are initialized with 0s
x.grad
```

```{.python .input  n=9}
%%tab pytorch
# Can also create x = torch.arange(4.0, requires_grad=True)
x.requires_grad_(True)
x.grad  # The gradient is None by default
```

```{.python .input}
%%tab tensorflow
x = tf.Variable(x)
```

Now we compute our function of `x` and assign the result to `y`.

```{.python .input  n=10}
%%tab mxnet
# Our code is inside an `autograd.record` scope to build the computational
# graph
with autograd.record():
    y = 2 * np.dot(x, x)
y
```

```{.python .input  n=11}
%%tab pytorch
y = 2 * torch.dot(x, x)
y
```

```{.python .input}
%%tab tensorflow
# Record all computations onto a tape
with tf.GradientTape() as t:
    y = 2 * tf.tensordot(x, x, axes=1)
y
```

```{.python .input}
%%tab jax
y = lambda x: 2 * jnp.dot(x, x)
y(x)
```

:begin_tab:`pytorch`
Now we take the gradient of `y` with respect to `x` by calling its
`backward` method, and read it from `x`'s `grad` attribute.
:end_tab:

:begin_tab:`tensorflow`
Now we compute the gradient of `y` with respect to `x` by calling the
`gradient` method.
:end_tab:

:begin_tab:`jax`
Now we take the gradient of `y` with respect to `x` by passing it through
the `grad` transform.
:end_tab:

```{.python .input}
%%tab mxnet
y.backward()
x.grad
```

```{.python .input  n=12}
%%tab pytorch
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
x_grad = t.gradient(y, x)
x_grad
```

```{.python .input}
%%tab jax
from jax import grad
# The `grad` transform returns a Python function that
# computes the gradient of the original function
x_grad = grad(y)(x)
x_grad
```

We know the gradient of $y = 2\mathbf{x}^{\top}\mathbf{x}$ with respect to
$\mathbf{x}$ should be $4\mathbf{x}$. Let's verify that autograd agrees.

```{.python .input  n=13}
%%tab mxnet
x.grad == 4 * x
```

```{.python .input  n=14}
%%tab pytorch
x.grad == 4 * x
```

```{.python .input}
%%tab tensorflow
x_grad == 4 * x
```

```{.python .input}
%%tab jax
x_grad == 4 * x
```

:begin_tab:`pytorch`
Now let's compute another function of `x` and take its gradient. PyTorch
does not reset the gradient buffer when recording a new gradient; it adds the
new gradient to the stored one. That's handy for optimizing a sum of several
objective functions. To reset the buffer, call `x.grad.zero_()`:
:end_tab:

:begin_tab:`tensorflow`
Now let's compute another function of `x` and take its gradient. TensorFlow
resets the gradient buffer whenever we record a new gradient.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x.sum()
y.backward()
x.grad  # Overwritten by the newly calculated gradient
```

```{.python .input  n=20}
%%tab pytorch
x.grad.zero_()  # Reset the gradient
y = x.sum()
y.backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = tf.reduce_sum(x)
t.gradient(y, x)  # Overwritten by the newly calculated gradient
```

```{.python .input}
%%tab jax
y = lambda x: x.sum()
grad(y)(x)
```

## Backward for Non-Scalar Variables

When `y` is a vector, the natural derivative of `y` with respect to a vector
`x` is a matrix, the *Jacobian*, holding the partial derivative of each
component of `y` with respect to each component of `x`. For higher-order `y`
and `x`, the result can be an even higher-order tensor.

Jacobians appear in some advanced techniques, but more often we want to sum
the gradients of each component of `y` with respect to all of `x`, giving a
vector shaped like `x`. For example, we often have a vector of loss values,
one per example in a *batch* of training examples, and just want to sum up
the per-example gradients.

:begin_tab:`pytorch`
Frameworks differ in how they interpret gradients of non-scalar tensors, so
PyTorch avoids confusion: calling `backward` on a non-scalar raises an error
unless we say how to reduce it to a scalar. Formally, we provide a vector
$\mathbf{v}$ so that `backward` computes
$\mathbf{v}^\top \partial_{\mathbf{x}} \mathbf{y}$ instead of
$\partial_{\mathbf{x}} \mathbf{y}$. Confusingly (for reasons that become clear
later), this argument representing $\mathbf{v}$ is named `gradient`. For
details, see Yang Zhang's
[Medium post](https://zhang-yang.medium.com/the-gradient-argument-in-pytorchs-backward-function-explained-by-examples-68f266950c29).
:end_tab:

:begin_tab:`tensorflow`
By default, TensorFlow returns the gradient of the sum,
$\partial_{\mathbf{x}} \sum_i y_i$, instead of the Jacobian
$\partial_{\mathbf{x}} \mathbf{y}$.
:end_tab:

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x  
y.backward()
x.grad  # Equals the gradient of y = sum(x * x)
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y = x * x
y.backward(gradient=torch.ones(len(y)))  # Faster: y.sum().backward()
x.grad
```

```{.python .input}
%%tab tensorflow
with tf.GradientTape() as t:
    y = x * x
t.gradient(y, x)  # Same as y = tf.reduce_sum(x * x)
```

```{.python .input}
%%tab jax
y = lambda x: x * x
# grad is only defined for scalar output functions
grad(lambda x: y(x).sum())(x)
```

## Detaching Computation

Sometimes we want to move some calculations outside the recorded graph, e.g.,
auxiliary intermediate terms we don't want gradients for. Then we *detach*
them from the final result. Toy example: `z = x * y` and `y = x * x`, but we
want only the *direct* influence of `x` on `z`, not the influence through
`y`. We create `u` with the same value as `y` but with its *provenance* (how
it was created) wiped out, so `u` has no ancestors in the graph and gradients
don't flow through it to `x`. The gradient of `z = x * u` is then `u`, not
`3 * x * x` as you might expect from `z = x * x * x`.

```{.python .input}
%%tab mxnet
with autograd.record():
    y = x * x
    u = y.detach()
    z = u * x
z.backward()
x.grad == u
```

```{.python .input  n=21}
%%tab pytorch
x.grad.zero_()
y = x * x
u = y.detach()
z = u * x

z.sum().backward()
x.grad == u
```

```{.python .input}
%%tab tensorflow
# Set persistent=True to preserve the compute graph. 
# This lets us run t.gradient more than once
with tf.GradientTape(persistent=True) as t:
    y = x * x
    u = tf.stop_gradient(y)
    z = u * x

x_grad = t.gradient(z, x)
x_grad == u
```

```{.python .input}
%%tab jax
import jax

y = lambda x: x * x
# jax.lax primitives are Python wrappers around XLA operations
u = jax.lax.stop_gradient(y(x))
z = lambda x: u * x

grad(lambda x: z(x).sum())(x) == y(x)
```

This detaches `y`'s ancestors from the graph leading to `z`, but the graph
leading to `y` remains, so we can still compute the gradient of `y` with
respect to `x`.

```{.python .input}
%%tab mxnet
y.backward()
x.grad == 2 * x
```

```{.python .input}
%%tab pytorch
x.grad.zero_()
y.sum().backward()
x.grad == 2 * x
```

```{.python .input}
%%tab tensorflow
t.gradient(y, x) == 2 * x
```

```{.python .input}
%%tab jax
grad(lambda x: y(x).sum())(x) == 2 * x
```

## Gradients and Python Control Flow

So far, the path from input to output was a fixed function such as
`z = x * x * x`. But code can compute results much more freely: depending on
auxiliary variables, or branching on intermediate results. A big benefit of
autograd is that even if the computation passes through a maze of Python
control flow (conditionals, loops, arbitrary function calls), we can still
compute the gradient of the result. In the snippet below, both the number of
`while` iterations and the `if` branch depend on the input `a`.

```{.python .input}
%%tab mxnet
def f(a):
    b = a * 2
    while np.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab pytorch
def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab tensorflow
def f(a):
    b = a * 2
    while tf.norm(b) < 1000:
        b = b * 2
    if tf.reduce_sum(b) > 0:
        c = b
    else:
        c = 100 * b
    return c
```

```{.python .input}
%%tab jax
def f(a):
    b = a * 2
    while jnp.linalg.norm(b) < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c
```

We call the function with a random input. Since the input is random, we
don't know what the computational graph will look like, but each call of
`f(a)` on a specific input builds a specific graph, on which we can then run
`backward`.

```{.python .input}
%%tab mxnet
a = np.random.normal()
a.attach_grad()
with autograd.record():
    d = f(a)
d.backward()
```

```{.python .input}
%%tab pytorch
a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
```

```{.python .input}
%%tab tensorflow
a = tf.Variable(tf.random.normal(shape=()))
with tf.GradientTape() as t:
    d = f(a)
d_grad = t.gradient(d, a)
d_grad
```

```{.python .input}
%%tab jax
from jax import random
a = random.normal(random.PRNGKey(1), ())
d = f(a)
d_grad = grad(f)(a)
```

The function `f` is contrived, but its dependence on the input is simple: it
is *linear* in `a` with a piecewise-defined scale. So `f(a) / a` is a vector
of constant entries and must match the gradient of `f(a)` with respect to `a`.

```{.python .input}
%%tab mxnet
a.grad == d / a
```

```{.python .input}
%%tab pytorch
a.grad == d / a
```

```{.python .input}
%%tab tensorflow
d_grad == d / a
```

```{.python .input}
%%tab jax
d_grad == d / a
```

Dynamic control flow is very common in deep learning; when processing text,
for example, the graph depends on the input's length. Then automatic
differentiation is vital, since the gradient can't be computed *a priori*.

## Discussion

You've now seen the power of automatic differentiation. Libraries that
compute derivatives automatically and efficiently have been a massive
productivity boost for deep learning practitioners, freeing them from menial
work, and they make it possible to design huge models whose gradients would
take prohibitively long to derive by hand. Interestingly, while we use
autograd to *optimize* models (statistically), *optimizing* autograd
libraries themselves (computationally) is a rich subject for framework
designers, drawing on compilers and graph manipulation to compute results as
quickly and memory-efficiently as possible.

For now, remember the basics: (i) attach gradients to the variables we want
derivatives with respect to; (ii) record the computation of the target value;
(iii) run the backpropagation function; and (iv) read the resulting gradient.


## Exercises

1. Why is the second derivative much more expensive to compute than the first derivative?
1. After running the function for backpropagation, immediately run it again and see what happens. Investigate.
1. In the control flow example where we calculate the derivative of `d` with respect to `a`, what would happen if we changed the variable `a` to a random vector or a matrix? At this point, the result of the calculation `f(a)` is no longer a scalar. What happens to the result? How do we analyze this?
1. Let $f(x) = \sin(x)$. Plot the graph of $f$ and of its derivative $f'$. Do not exploit the fact that $f'(x) = \cos(x)$ but rather use automatic differentiation to get the result. 
1. Let $f(x) = ((\log x^2) \cdot \sin x) + x^{-1}$. Write out a dependency graph tracing results from $x$ to $f(x)$. 
1. Use the chain rule to compute the derivative $\frac{df}{dx}$ of the aforementioned function, placing each term on the dependency graph that you constructed previously. 
1. Given the graph and the intermediate derivative results, you have a number of options when computing the gradient. Evaluate the result once starting from $x$ to $f$ and once from $f$ tracing back to $x$. The path from $x$ to $f$ is commonly known as *forward differentiation*, whereas the path from $f$ to $x$ is known as backward differentiation. 
1. When might you want to use forward, and when backward, differentiation? Hint: consider the amount of intermediate data needed, the ability to parallelize steps, and the size of matrices and vectors involved.
