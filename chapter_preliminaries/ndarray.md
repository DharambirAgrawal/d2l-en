```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Data Manipulation
:label:`sec_ndarray`

To get anything done, we need to (i) acquire data and (ii) process it once
it's in the computer, and there's no point acquiring data without a way to
store it. So let's get our hands dirty with $n$-dimensional arrays, also
called *tensors*. If you know NumPy, this will be a breeze. In every modern
deep learning framework, the *tensor class* (`ndarray` in MXNet, `Tensor` in
PyTorch and TensorFlow) resembles NumPy's `ndarray`, plus two killer
features: it supports automatic differentiation, and it uses GPUs to
accelerate numerical computation (NumPy runs only on CPUs). These make neural
networks easy to code and fast to run.



## Getting Started

:begin_tab:`pytorch`
To start, we import the PyTorch library. Note that the package name is `torch`.
:end_tab:

:begin_tab:`tensorflow`
To start, we import `tensorflow`, usually with the alias `tf`.
:end_tab:

```{.python .input}
%%tab mxnet
from mxnet import np, npx
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
import jax
from jax import numpy as jnp
```

A tensor represents a (possibly multidimensional) array of numerical values.
With one axis, it's a *vector*; with two, a *matrix*; with $k > 2$ axes, we
just call it a $k^\textrm{th}$-*order tensor*.

:begin_tab:`pytorch`
PyTorch has many functions for creating tensors prepopulated with values.
`arange(n)` creates a vector of evenly spaced values from 0 (included) to `n`
(not included), with step $1$ by default. Unless specified otherwise, new
tensors are stored in main memory and computed on the CPU.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow has many functions for creating tensors prepopulated with values.
`range(n)` creates a vector of evenly spaced values from 0 (included) to `n`
(not included), with step $1$ by default. Unless specified otherwise, new
tensors are stored in main memory and computed on the CPU.
:end_tab:

```{.python .input}
%%tab mxnet
x = np.arange(12)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(12, dtype=torch.float32)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(12, dtype=tf.float32)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(12)
x
```

:begin_tab:`pytorch`
Each value is an *element* of the tensor; `x` has 12. The `numel` method gives
the total number of elements.
:end_tab:

:begin_tab:`tensorflow`
Each value is an *element* of the tensor; `x` has 12. The `size` function gives
the total number of elements.
:end_tab:

```{.python .input}
%%tab mxnet, jax
x.size
```

```{.python .input}
%%tab pytorch
x.numel()
```

```{.python .input}
%%tab tensorflow
tf.size(x)
```

The `shape` attribute gives a tensor's *shape*, its length along each axis.
For a vector, the shape has a single element, equal to the size.

```{.python .input}
%%tab all
x.shape
```

`reshape` changes a tensor's shape without changing its size or values. Here
we turn the vector `x` of shape (12,) into a matrix `X` of shape (3, 4). The
elements are laid out one row at a time, so `x[3] == X[0, 3]`.

```{.python .input}
%%tab mxnet, pytorch, jax
X = x.reshape(3, 4)
X
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(x, (3, 4))
X
```

Specifying every shape component is redundant: given the size $n$ and target
shape ($h$, $w$), we know $w = n/h$. Put `-1` for the component that should be
inferred: instead of `x.reshape(3, 4)`, we could write `x.reshape(-1, 4)` or
`x.reshape(3, -1)`.

We often need tensors of all 0s or 1s. `zeros` creates a tensor of shape
(2, 3, 4) with every element set to 0.

```{.python .input}
%%tab mxnet
np.zeros((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.zeros((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.zeros((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.zeros((2, 3, 4))
```

Similarly, `ones` creates a tensor of all 1s.

```{.python .input}
%%tab mxnet
np.ones((2, 3, 4))
```

```{.python .input}
%%tab pytorch
torch.ones((2, 3, 4))
```

```{.python .input}
%%tab tensorflow
tf.ones((2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.ones((2, 3, 4))
```

We often want to sample each element randomly (and independently) from a
given distribution, e.g., to initialize neural network parameters. This
creates a tensor with elements drawn from a standard Gaussian (normal)
distribution with mean 0 and standard deviation 1.

```{.python .input}
%%tab mxnet
np.random.normal(0, 1, size=(3, 4))
```

```{.python .input}
%%tab pytorch
torch.randn(3, 4)
```

```{.python .input}
%%tab tensorflow
tf.random.normal(shape=[3, 4])
```

```{.python .input}
%%tab jax
# Any call of a random function in JAX requires a key to be
# specified, feeding the same key to a random function will
# always result in the same sample being generated
jax.random.normal(jax.random.PRNGKey(0), (3, 4))
```

Finally, we can give the exact value of each element with (possibly nested)
Python lists of numbers. Here, the outer list corresponds to axis 0 and the
inner lists to axis 1.

```{.python .input}
%%tab mxnet
np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab pytorch
torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab tensorflow
tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

```{.python .input}
%%tab jax
jnp.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
```

## Indexing and Slicing

As with Python lists, we can access elements by index (starting at 0), use
negative indices to count from the end, and select ranges with slices
(`X[start:stop]`), which include `start` *but not* `stop`. When only one index
(or slice) is given for a $k^\textrm{th}$-order tensor, it applies along axis
0. So `[-1]` selects the last row and `[1:3]` the second and third rows.

```{.python .input}
%%tab all
X[-1], X[1:3]
```

:begin_tab:`pytorch`
We can also *write* elements of a matrix by specifying indices.
:end_tab:

:begin_tab:`tensorflow`
TensorFlow `Tensors` are immutable and can't be assigned to. `Variables` are
mutable containers of state that support assignment (note that gradients don't
flow backwards through `Variable` assignments). Beyond assigning the whole
`Variable`, we can write elements by specifying indices.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
X[1, 2] = 17
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[1, 2].assign(9)
X_var
```

```{.python .input}
%%tab jax
# JAX arrays are immutable. jax.numpy.ndarray.at index
# update operators create a new array with the corresponding
# modifications made
X_new_1 = X.at[1, 2].set(17)
X_new_1
```

To assign the same value to multiple elements, index on the left-hand side of
the assignment. For instance, `[:2, :]` selects the first and second rows,
where `:` takes all elements along axis 1 (columns). This works for vectors
and higher-order tensors too.

```{.python .input}
%%tab mxnet, pytorch
X[:2, :] = 12
X
```

```{.python .input}
%%tab tensorflow
X_var = tf.Variable(X)
X_var[:2, :].assign(tf.ones(X_var[:2,:].shape, dtype=tf.float32) * 12)
X_var
```

```{.python .input}
%%tab jax
X_new_2 = X_new_1.at[:2, :].set(12)
X_new_2
```

## Operations

Now we can manipulate tensors with mathematical operations. Among the most
useful are *elementwise* operations, which apply a scalar operation to each
element, or, for two tensors, a binary operation to each pair of
corresponding elements. Any function from a scalar to a scalar can be made
elementwise.

We write such *unary* scalar operators (one input) as
$f: \mathbb{R} \rightarrow \mathbb{R}$: they map any real number to another.
Most standard operators, including unary ones like $e^x$, can be applied
elementwise.

```{.python .input}
%%tab mxnet
np.exp(x)
```

```{.python .input}
%%tab pytorch
torch.exp(x)
```

```{.python .input}
%%tab tensorflow
tf.exp(x)
```

```{.python .input}
%%tab jax
jnp.exp(x)
```

Likewise, *binary* scalar operators map pairs of real numbers to one real
number: $f: \mathbb{R}, \mathbb{R} \rightarrow \mathbb{R}$. Given vectors
$\mathbf{u}$ and $\mathbf{v}$ *of the same shape* and a binary operator $f$,
we produce $\mathbf{c} = F(\mathbf{u},\mathbf{v})$ by setting
$c_i \gets f(u_i, v_i)$ for all $i$, where $c_i, u_i$, and $v_i$ are the
$i^\textrm{th}$ elements of $\mathbf{c}, \mathbf{u}$, and $\mathbf{v}$. We
have *lifted* the scalar function to an elementwise vector operation
$F: \mathbb{R}^d, \mathbb{R}^d \rightarrow \mathbb{R}^d$. The standard
arithmetic operators, addition (`+`), subtraction (`-`), multiplication
(`*`), division (`/`), and exponentiation (`**`), are all lifted to
elementwise operations on identically-shaped tensors of any shape.

```{.python .input}
%%tab mxnet
x = np.array([1, 2, 4, 8])
y = np.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab tensorflow
x = tf.constant([1.0, 2, 4, 8])
y = tf.constant([2.0, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

```{.python .input}
%%tab jax
x = jnp.array([1.0, 2, 4, 8])
y = jnp.array([2, 2, 2, 2])
x + y, x - y, x * y, x / y, x ** y
```

Besides elementwise computations, we can do linear algebra operations such as
dot products and matrix multiplications, covered in
:numref:`sec_linear-algebra`.

We can also *concatenate* tensors, stacking them end-to-end along a chosen
axis. Concatenating two matrices along rows (axis 0) gives an axis-0 length
of $6$ ($3 + 3$); along columns (axis 1), an axis-1 length of $8$ ($4 + 4$).

```{.python .input}
%%tab mxnet
X = np.arange(12).reshape(3, 4)
Y = np.array([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
np.concatenate([X, Y], axis=0), np.concatenate([X, Y], axis=1)
```

```{.python .input}
%%tab pytorch
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)
```

```{.python .input}
%%tab tensorflow
X = tf.reshape(tf.range(12, dtype=tf.float32), (3, 4))
Y = tf.constant([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
tf.concat([X, Y], axis=0), tf.concat([X, Y], axis=1)
```

```{.python .input}
%%tab jax
X = jnp.arange(12, dtype=jnp.float32).reshape((3, 4))
Y = jnp.array([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
jnp.concatenate((X, Y), axis=0), jnp.concatenate((X, Y), axis=1)
```

We can build a binary tensor from a *logical statement*. For `X == Y`, each
entry `i, j` is `1` if `X[i, j]` equals `Y[i, j]` and `0` otherwise.

```{.python .input}
%%tab all
X == Y
```

Summing all elements gives a tensor with just one element.

```{.python .input}
%%tab mxnet, pytorch, jax
X.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(X)
```

## Broadcasting
:label:`subsec_broadcasting`

Elementwise binary operations need tensors of the same shape, but under
certain conditions the *broadcasting mechanism* lets shapes differ. It works
in two steps: (i) expand one or both arrays by copying elements along axes of
length 1 until the shapes match; (ii) apply the elementwise operation.

```{.python .input}
%%tab mxnet
a = np.arange(3).reshape(3, 1)
b = np.arange(2).reshape(1, 2)
a, b
```

```{.python .input}
%%tab pytorch
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
a, b
```

```{.python .input}
%%tab tensorflow
a = tf.reshape(tf.range(3), (3, 1))
b = tf.reshape(tf.range(2), (1, 2))
a, b
```

```{.python .input}
%%tab jax
a = jnp.arange(3).reshape((3, 1))
b = jnp.arange(2).reshape((1, 2))
a, b
```

`a` is $3\times1$ and `b` is $1\times2$, so their shapes don't match.
Broadcasting produces a $3\times2$ matrix by replicating `a` along the columns
and `b` along the rows before adding elementwise.

```{.python .input}
%%tab all
a + b
```

## Saving Memory

Operations can allocate new memory for their results. `Y = Y + X` makes `Y`
point to newly allocated memory instead of its old tensor. Python's `id()`,
which gives an object's memory address, shows this: Python first evaluates
`Y + X` into new memory, then points `Y` there.

```{.python .input}
%%tab all
before = id(Y)
Y = Y + X
id(Y) == before
```

This is undesirable for two reasons. First, we don't want to allocate memory
unnecessarily: models can have hundreds of megabytes of parameters updated
many times per second, so we want to update them *in place*. Second, several
variables may point to the same parameters; without in-place updates we must
update every reference, or risk memory leaks and stale parameters.

:begin_tab:`pytorch`
In-place operations are easy: assign the result to a previously allocated
array with slice notation, `Y[:] = <expression>`. Here we create `Z` with
`zeros_like` (same shape as `Y`) and then overwrite its values.
:end_tab:

:begin_tab:`tensorflow`
`Variables` are TensorFlow's mutable containers of state, used to store model
parameters. We assign the result of an operation to a `Variable` with
`assign`. Here we create `Z` with `zeros_like` (same shape as `Y`) and then
overwrite its values.
:end_tab:

```{.python .input}
%%tab mxnet
Z = np.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab pytorch
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))
```

```{.python .input}
%%tab tensorflow
Z = tf.Variable(tf.zeros_like(Y))
print('id(Z):', id(Z))
Z.assign(X + Y)
print('id(Z):', id(Z))
```

```{.python .input}
%%tab jax
# JAX arrays do not allow in-place operations
```

:begin_tab:`pytorch`
If `X` isn't needed afterwards, `X[:] = X + Y` or `X += Y` also reduces the
memory overhead.
:end_tab:

:begin_tab:`tensorflow`
Even with state stored in `Variables`, you may want to avoid excess
allocations for tensors that aren't model parameters. Since `Tensors` are
immutable and gradients don't flow through `Variable` assignments, TensorFlow
has no explicit way to run a single operation in place. Instead, the
`tf.function` decorator compiles the computation into an optimized TensorFlow
graph that prunes unused values and reuses allocations no longer needed,
minimizing memory overhead.
:end_tab:

```{.python .input}
%%tab mxnet, pytorch
before = id(X)
X += Y
id(X) == before
```

```{.python .input}
%%tab tensorflow
@tf.function
def computation(X, Y):
    Z = tf.zeros_like(Y)  # This unused value will be pruned out
    A = X + Y  # Allocations will be reused when no longer needed
    B = A + Y
    C = B + Y
    return C + Y

computation(X, Y)
```

## Conversion to Other Python Objects

:begin_tab:`tensorflow`
Converting to a NumPy tensor (`ndarray`), or back, is easy. The result doesn't
share memory, which matters: computation on the CPU or GPU shouldn't have to
wait for NumPy, which might be using the same memory for something else.
:end_tab:

:begin_tab:`pytorch`
Converting to a NumPy tensor (`ndarray`), or back, is easy. The torch tensor
and NumPy array share their underlying memory, so changing one in place
changes the other.
:end_tab:

```{.python .input}
%%tab mxnet
A = X.asnumpy()
B = np.array(A)
type(A), type(B)
```

```{.python .input}
%%tab pytorch
A = X.numpy()
B = torch.from_numpy(A)
type(A), type(B)
```

```{.python .input}
%%tab tensorflow
A = X.numpy()
B = tf.constant(A)
type(A), type(B)
```

```{.python .input}
%%tab jax
A = jax.device_get(X)
B = jax.device_put(A)
type(A), type(B)
```

To convert a size-1 tensor to a Python scalar, use `item` or Python's built-in
functions.

```{.python .input}
%%tab mxnet
a = np.array([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab pytorch
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab tensorflow
a = tf.constant([3.5]).numpy()
a, a.item(), float(a), int(a)
```

```{.python .input}
%%tab jax
a = jnp.array([3.5])
a, a.item(), float(a), int(a)
```

## Summary

The tensor class is the main interface for storing and manipulating data in
deep learning libraries. It provides construction routines, indexing and
slicing, basic math operations, broadcasting, memory-efficient assignment,
and conversion to and from other Python objects.


## Exercises

1. Run the code in this section. Change the conditional statement `X == Y` to `X < Y` or `X > Y`, and then see what kind of tensor you can get.
1. Replace the two tensors that operate by element in the broadcasting mechanism with other shapes, e.g., 3-dimensional tensors. Is the result the same as expected?
