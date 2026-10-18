```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Linear Algebra
:label:`sec_linear-algebra`

We can now load data into tensors and manipulate them with basic operations.
To build sophisticated models we also need some linear algebra. This section
is a gentle introduction to the essentials, from scalar arithmetic up to
matrix multiplication.

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
from jax import numpy as jnp
```

## Scalars


Most everyday math manipulates numbers one at a time; formally, these are
*scalars*. To convert Palo Alto's balmy $72$ degrees Fahrenheit to Celsius,
evaluate $c = \frac{5}{9}(f - 32)$ with $f = 72$. Here $5$, $9$, and $32$ are
constant scalars, and the variables $c$ and $f$ represent unknown scalars.

We write scalars as ordinary lowercase letters (e.g., $x$, $y$, and $z$) and
the space of all (continuous) *real-valued* scalars as $\mathbb{R}$. Skipping
rigorous definitions of *spaces*, $x \in \mathbb{R}$ formally says that $x$ is
a real-valued scalar. The symbol $\in$ ("in") denotes set membership: e.g.,
$x, y \in \{0, 1\}$ means $x$ and $y$ can only be $0$ or $1$.

Scalars are implemented as tensors with one element. Below, we do addition,
multiplication, division, and exponentiation with two scalars.

```{.python .input}
%%tab mxnet
x = np.array(3.0)
y = np.array(2.0)

x + y, x * y, x / y, x ** y
```

```{.python .input}
%%tab pytorch
x = torch.tensor(3.0)
y = torch.tensor(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab tensorflow
x = tf.constant(3.0)
y = tf.constant(2.0)

x + y, x * y, x / y, x**y
```

```{.python .input}
%%tab jax
x = jnp.array(3.0)
y = jnp.array(2.0)

x + y, x * y, x / y, x**y
```

## Vectors

For now, think of a vector as a fixed-length array of scalars, called its
*elements* (or *entries*, *components*). When vectors represent real-world
examples, their values mean something: a loan applicant's vector might hold
income, length of employment, and number of previous defaults; a heart attack
study's patient vector might hold recent vital signs, cholesterol levels, and
minutes of exercise per day. We write vectors in bold lowercase (e.g.,
$\mathbf{x}$, $\mathbf{y}$, and $\mathbf{z}$).

Vectors are implemented as $1^{\textrm{st}}$-order tensors, which can have any
length memory allows. Caution: Python, like most programming languages, uses
*zero-based indexing* (indices start at $0$), while linear algebra subscripts
start at $1$.

```{.python .input}
%%tab mxnet
x = np.arange(3)
x
```

```{.python .input}
%%tab pytorch
x = torch.arange(3)
x
```

```{.python .input}
%%tab tensorflow
x = tf.range(3)
x
```

```{.python .input}
%%tab jax
x = jnp.arange(3)
x
```

We refer to an element with a subscript: $x_2$ is the second element of
$\mathbf{x}$ (not bold, since it's a scalar). By default, we draw vectors
with their elements stacked vertically:

$$\mathbf{x} =\begin{bmatrix}x_{1}  \\ \vdots  \\x_{n}\end{bmatrix}.$$
:eqlabel:`eq_vec_def`

Here $x_1, \ldots, x_n$ are the elements. Later we'll distinguish these
*column vectors* from *row vectors*, whose elements are stacked horizontally.
In code, we access elements by indexing.

```{.python .input}
%%tab all
x[2]
```

$\mathbf{x} \in \mathbb{R}^n$ says that $\mathbf{x}$ has $n$ elements; $n$ is
its *dimensionality*. In code, this is the tensor's length, given by Python's
`len`.

```{.python .input}
%%tab all
len(x)
```

The `shape` attribute also gives the length: it's a tuple of the tensor's
length along each axis, so a one-axis tensor has a one-element shape.

```{.python .input}
%%tab all
x.shape
```

"Dimension" is often overloaded to mean both the number of axes and the
length along an axis. To avoid confusion, we use *order* for the number of
axes and *dimensionality* only for the number of components.


## Matrices

Scalars are $0^{\textrm{th}}$-order tensors, vectors $1^{\textrm{st}}$-order,
and matrices $2^{\textrm{nd}}$-order. We write matrices in bold capitals
(e.g., $\mathbf{X}$, $\mathbf{Y}$, and $\mathbf{Z}$) and represent them in
code as tensors with two axes. $\mathbf{A} \in \mathbb{R}^{m \times n}$ means
$\mathbf{A}$ holds $m \times n$ real-valued scalars in $m$ rows and $n$
columns; when $m = n$, it is *square*. Think of a matrix as a table. An
element is indexed by row and column: $a_{ij}$ is the entry in row $i$ and
column $j$ of $\mathbf{A}$:

$$\mathbf{A}=\begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \\ \end{bmatrix}.$$
:eqlabel:`eq_matrix_def`


In code, $\mathbf{A} \in \mathbb{R}^{m \times n}$ is a $2^{\textrm{nd}}$-order
tensor of shape ($m$, $n$). Any tensor with $m \times n$ elements can be
turned into an $m \times n$ matrix by passing that shape to `reshape`:

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab pytorch
A = torch.arange(6).reshape(3, 2)
A
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6), (3, 2))
A
```

```{.python .input}
%%tab jax
A = jnp.arange(6).reshape(3, 2)
A
```

Exchanging a matrix's rows and columns gives its *transpose*, written
$\mathbf{A}^\top$: if $\mathbf{B} = \mathbf{A}^\top$, then $b_{ij} = a_{ji}$
for all $i$ and $j$. So the transpose of an $m \times n$ matrix is
$n \times m$:

$$
\mathbf{A}^\top =
\begin{bmatrix}
    a_{11} & a_{21} & \dots  & a_{m1} \\
    a_{12} & a_{22} & \dots  & a_{m2} \\
    \vdots & \vdots & \ddots  & \vdots \\
    a_{1n} & a_{2n} & \dots  & a_{mn}
\end{bmatrix}.
$$

In code:

```{.python .input}
%%tab mxnet, pytorch, jax
A.T
```

```{.python .input}
%%tab tensorflow
tf.transpose(A)
```

*Symmetric* matrices are square matrices equal to their own transpose:
$\mathbf{A} = \mathbf{A}^\top$. This matrix is symmetric:

```{.python .input}
%%tab mxnet
A = np.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab pytorch
A = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

```{.python .input}
%%tab tensorflow
A = tf.constant([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == tf.transpose(A)
```

```{.python .input}
%%tab jax
A = jnp.array([[1, 2, 3], [2, 0, 4], [3, 4, 5]])
A == A.T
```

Matrices are handy for datasets: typically, rows are records and columns are
attributes.



## Tensors

You can go far with scalars, vectors, and matrices, but eventually you'll
need higher-order tensors, the generic extension to $n^{\textrm{th}}$-order
arrays. Objects of the *tensor class* get their name precisely because they
too can have any number of axes; whether "tensor" means the math object or
the code object should be clear from context. We write general tensors as
capitals in a special font (e.g., $\mathsf{X}$, $\mathsf{Y}$, and
$\mathsf{Z}$), indexed like matrices (e.g., $x_{ijk}$ and
$[\mathsf{X}]_{1, 2i-1, 3}$).

Tensors matter more once we work with images. Each image is a
$3^{\textrm{rd}}$-order tensor with axes for height, width, and *channel*:
at each location, the red, green, and blue intensities are stacked along the
channel axis. A collection of images is a $4^{\textrm{th}}$-order tensor, with
images indexed along the first axis. Higher-order tensors are built like
vectors and matrices, by adding shape components.

```{.python .input}
%%tab mxnet
np.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab pytorch
torch.arange(24).reshape(2, 3, 4)
```

```{.python .input}
%%tab tensorflow
tf.reshape(tf.range(24), (2, 3, 4))
```

```{.python .input}
%%tab jax
jnp.arange(24).reshape(2, 3, 4)
```

## Basic Properties of Tensor Arithmetic

Scalars, vectors, matrices, and higher-order tensors share some handy
properties. For example, elementwise operations produce outputs with the same
shape as their operands.

```{.python .input}
%%tab mxnet
A = np.arange(6).reshape(2, 3)
B = A.copy()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab pytorch
A = torch.arange(6, dtype=torch.float32).reshape(2, 3)
B = A.clone()  # Assign a copy of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab tensorflow
A = tf.reshape(tf.range(6, dtype=tf.float32), (2, 3))
B = A  # No cloning of A to B by allocating new memory
A, A + B
```

```{.python .input}
%%tab jax
A = jnp.arange(6, dtype=jnp.float32).reshape(2, 3)
B = A
A, A + B
```

The elementwise product of two matrices is their *Hadamard product*
(denoted $\odot$). For $\mathbf{A}, \mathbf{B} \in \mathbb{R}^{m \times n}$:



$$
\mathbf{A} \odot \mathbf{B} =
\begin{bmatrix}
    a_{11}  b_{11} & a_{12}  b_{12} & \dots  & a_{1n}  b_{1n} \\
    a_{21}  b_{21} & a_{22}  b_{22} & \dots  & a_{2n}  b_{2n} \\
    \vdots & \vdots & \ddots & \vdots \\
    a_{m1}  b_{m1} & a_{m2}  b_{m2} & \dots  & a_{mn}  b_{mn}
\end{bmatrix}.
$$

```{.python .input}
%%tab all
A * B
```

Adding or multiplying a scalar and a tensor gives a result with the tensor's
shape: the scalar is added to (or multiplies) each element.

```{.python .input}
%%tab mxnet
a = 2
X = np.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab pytorch
a = 2
X = torch.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

```{.python .input}
%%tab tensorflow
a = 2
X = tf.reshape(tf.range(24), (2, 3, 4))
a + X, (a * X).shape
```

```{.python .input}
%%tab jax
a = 2
X = jnp.arange(24).reshape(2, 3, 4)
a + X, (a * X).shape
```

**Try it yourself.** Elementwise operations are the bread and butter of
tensor code. Implement the simplest one, vector addition, in plain Python:

```{=html}
<div class="d2l-exercise" data-exercise="vector-add"></div>
```

## Reduction
:label:`subsec_lin-alg-reduction`

We often want the sum of a tensor's elements. For a vector $\mathbf{x}$ of
length $n$, we write $\sum_{i=1}^n x_i$, and there's a simple function for it:

```{.python .input}
%%tab mxnet
x = np.arange(3)
x, x.sum()
```

```{.python .input}
%%tab pytorch
x = torch.arange(3, dtype=torch.float32)
x, x.sum()
```

```{.python .input}
%%tab tensorflow
x = tf.range(3, dtype=tf.float32)
x, tf.reduce_sum(x)
```

```{.python .input}
%%tab jax
x = jnp.arange(3, dtype=jnp.float32)
x, x.sum()
```

For tensors of any shape, sum over all axes. For an $m \times n$ matrix
$\mathbf{A}$, the sum is $\sum_{i=1}^{m} \sum_{j=1}^{n} a_{ij}$.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum()
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A)
```

By default, `sum` *reduces* a tensor along all axes to a scalar. We can
instead specify the axes to reduce. `axis=0` sums along the rows (axis 0);
since that axis is reduced away, it's missing from the output's shape.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=0).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=0).shape
```

`axis=1` reduces the column dimension (axis 1), summing across the columns.

```{.python .input}
%%tab mxnet, pytorch, jax
A.shape, A.sum(axis=1).shape
```

```{.python .input}
%%tab tensorflow
A.shape, tf.reduce_sum(A, axis=1).shape
```

Reducing along both rows and columns is the same as summing all elements.

```{.python .input}
%%tab mxnet, pytorch, jax
A.sum(axis=[0, 1]) == A.sum()  # Same as A.sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(A, axis=[0, 1]), tf.reduce_sum(A)  # Same as tf.reduce_sum(A)
```

A related quantity is the *mean* (or *average*): the sum divided by the
number of elements. It's common enough to have its own function, which works
like `sum`.

```{.python .input}
%%tab mxnet, jax
A.mean(), A.sum() / A.size
```

```{.python .input}
%%tab pytorch
A.mean(), A.sum() / A.numel()
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A), tf.reduce_sum(A) / tf.size(A).numpy()
```

The mean can also reduce along specific axes.

```{.python .input}
%%tab mxnet, pytorch, jax
A.mean(axis=0), A.sum(axis=0) / A.shape[0]
```

```{.python .input}
%%tab tensorflow
tf.reduce_mean(A, axis=0), tf.reduce_sum(A, axis=0) / A.shape[0]
```

## Non-Reduction Sum
:label:`subsec_lin-alg-non-reduction`

Sometimes we want to keep the number of axes unchanged when computing a sum
or mean, so that we can use broadcasting.

```{.python .input}
%%tab mxnet, pytorch, jax
sum_A = A.sum(axis=1, keepdims=True)
sum_A, sum_A.shape
```

```{.python .input}
%%tab tensorflow
sum_A = tf.reduce_sum(A, axis=1, keepdims=True)
sum_A, sum_A.shape
```

Since `sum_A` keeps both axes after summing each row, we can divide `A` by
`sum_A` with broadcasting to make each row sum to $1$.

```{.python .input}
%%tab all
A / sum_A
```

`cumsum` computes the cumulative sum along an axis, say `axis=0` (row by
row). By design, it doesn't reduce the input along any axis.

```{.python .input}
%%tab mxnet, pytorch, jax
A.cumsum(axis=0)
```

```{.python .input}
%%tab tensorflow
tf.cumsum(A, axis=0)
```

## Dot Products

So far we've only done elementwise operations, sums, and averages, which
wouldn't justify a section on linear algebra. Things get more interesting
with one of the most fundamental operations: the dot product. For
$\mathbf{x}, \mathbf{y} \in \mathbb{R}^d$, the *dot product*
$\mathbf{x}^\top \mathbf{y}$ (or *inner product*,
$\langle \mathbf{x}, \mathbf{y}  \rangle$) sums the products of elements at
the same position: $\mathbf{x}^\top \mathbf{y} = \sum_{i=1}^{d} x_i y_i$.

```{.python .input}
%%tab mxnet
y = np.ones(3)
x, y, np.dot(x, y)
```

```{.python .input}
%%tab pytorch
y = torch.ones(3, dtype = torch.float32)
x, y, torch.dot(x, y)
```

```{.python .input}
%%tab tensorflow
y = tf.ones(3, dtype=tf.float32)
x, y, tf.tensordot(x, y, axes=1)
```

```{.python .input}
%%tab jax
y = jnp.ones(3, dtype = jnp.float32)
x, y, jnp.dot(x, y)
```

Equivalently, the dot product is an elementwise multiplication followed by a
sum:

```{.python .input}
%%tab mxnet
np.sum(x * y)
```

```{.python .input}
%%tab pytorch
torch.sum(x * y)
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(x * y)
```

```{.python .input}
%%tab jax
jnp.sum(x * y)
```

Dot products are useful everywhere. Given values $\mathbf{x}  \in \mathbb{R}^n$
and weights $\mathbf{w} \in \mathbb{R}^n$, the weighted sum of the values is
the dot product $\mathbf{x}^\top \mathbf{w}$. When the weights are
nonnegative and sum to $1$, i.e., $\left(\sum_{i=1}^{n} {w_i} = 1\right)$,
it's a *weighted average*. For two vectors normalized to unit length, the dot
product is the cosine of the angle between them (we define *length* formally
later in this section).


## Matrix--Vector Products

With dot products, we can understand the *product* of an $m \times n$ matrix
$\mathbf{A}$ and an $n$-dimensional vector $\mathbf{x}$. Write the matrix in
terms of its rows

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix},$$

where each $\mathbf{a}^\top_{i} \in \mathbb{R}^n$ is the $i^\textrm{th}$ row of
$\mathbf{A}$. The matrix--vector product $\mathbf{A}\mathbf{x}$ is the column
vector of length $m$ whose $i^\textrm{th}$ element is the dot product
$\mathbf{a}^\top_i \mathbf{x}$:

$$
\mathbf{A}\mathbf{x}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_m \\
\end{bmatrix}\mathbf{x}
= \begin{bmatrix}
 \mathbf{a}^\top_{1} \mathbf{x}  \\
 \mathbf{a}^\top_{2} \mathbf{x} \\
\vdots\\
 \mathbf{a}^\top_{m} \mathbf{x}\\
\end{bmatrix}.
$$

Multiplying by $\mathbf{A}\in \mathbb{R}^{m \times n}$ is a transformation
that projects vectors from $\mathbb{R}^{n}$ to $\mathbb{R}^{m}$. These are
remarkably useful: rotations, for example, are multiplications by certain
square matrices. Matrix--vector products also describe the key computation
of each neural network layer, from the previous layer's outputs.

:begin_tab:`pytorch`
In code, use the `mv` function. The column dimension of `A` (its length along
axis 1) must equal the dimension of `x` (its length). Python's `@` operator
does both matrix--vector and matrix--matrix products (depending on its
arguments), so we can also write `A@x`.
:end_tab:

:begin_tab:`tensorflow`
In code, use the `matvec` function. The column dimension of `A` (its length
along axis 1) must equal the dimension of `x` (its length).
:end_tab:

```{.python .input}
%%tab mxnet
A.shape, x.shape, np.dot(A, x)
```

```{.python .input}
%%tab pytorch
A.shape, x.shape, torch.mv(A, x), A@x
```

```{.python .input}
%%tab tensorflow
A.shape, x.shape, tf.linalg.matvec(A, x)
```

```{.python .input}
%%tab jax
A.shape, x.shape, jnp.matmul(A, x)
```

**Try it yourself.** Implement the matrix--vector product with NumPy,
using only elementwise operations and sums:

```{=html}
<div class="d2l-exercise" data-exercise="matvec-numpy"></div>
```

To build intuition for matrices as transformations, play with a $2 \times 2$ matrix:

```{=html}
<div class="d2l-playground" data-playground="matrix-transform"></div>
```

## Matrix--Matrix Multiplication

Once you have the hang of dot products and matrix--vector products,
*matrix--matrix multiplication* is straightforward. Take two matrices
$\mathbf{A} \in \mathbb{R}^{n \times k}$ and
$\mathbf{B} \in \mathbb{R}^{k \times m}$:

$$\mathbf{A}=\begin{bmatrix}
 a_{11} & a_{12} & \cdots & a_{1k} \\
 a_{21} & a_{22} & \cdots & a_{2k} \\
\vdots & \vdots & \ddots & \vdots \\
 a_{n1} & a_{n2} & \cdots & a_{nk} \\
\end{bmatrix},\quad
\mathbf{B}=\begin{bmatrix}
 b_{11} & b_{12} & \cdots & b_{1m} \\
 b_{21} & b_{22} & \cdots & b_{2m} \\
\vdots & \vdots & \ddots & \vdots \\
 b_{k1} & b_{k2} & \cdots & b_{km} \\
\end{bmatrix}.$$


Let $\mathbf{a}^\top_{i} \in \mathbb{R}^k$ be the $i^\textrm{th}$ row of
$\mathbf{A}$ and $\mathbf{b}_{j} \in \mathbb{R}^k$ the $j^\textrm{th}$ column
of $\mathbf{B}$:

$$\mathbf{A}=
\begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix},
\quad \mathbf{B}=\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}.
$$


The product $\mathbf{C} \in \mathbb{R}^{n \times m}$ has elements $c_{ij}$
equal to the dot product of row $i$ of $\mathbf{A}$ and column $j$ of
$\mathbf{B}$, i.e., $\mathbf{a}^\top_i \mathbf{b}_j$:

$$\mathbf{C} = \mathbf{AB} = \begin{bmatrix}
\mathbf{a}^\top_{1} \\
\mathbf{a}^\top_{2} \\
\vdots \\
\mathbf{a}^\top_n \\
\end{bmatrix}
\begin{bmatrix}
 \mathbf{b}_{1} & \mathbf{b}_{2} & \cdots & \mathbf{b}_{m} \\
\end{bmatrix}
= \begin{bmatrix}
\mathbf{a}^\top_{1} \mathbf{b}_1 & \mathbf{a}^\top_{1}\mathbf{b}_2& \cdots & \mathbf{a}^\top_{1} \mathbf{b}_m \\
 \mathbf{a}^\top_{2}\mathbf{b}_1 & \mathbf{a}^\top_{2} \mathbf{b}_2 & \cdots & \mathbf{a}^\top_{2} \mathbf{b}_m \\
 \vdots & \vdots & \ddots &\vdots\\
\mathbf{a}^\top_{n} \mathbf{b}_1 & \mathbf{a}^\top_{n}\mathbf{b}_2& \cdots& \mathbf{a}^\top_{n} \mathbf{b}_m
\end{bmatrix}.
$$

So $\mathbf{AB}$ amounts to $m$ matrix--vector products, or $m \times n$ dot
products, stitched into an $n \times m$ matrix. Below, `A` has two rows and
three columns and `B` has three rows and four columns, so the product has two
rows and four columns.

```{.python .input}
%%tab mxnet
B = np.ones(shape=(3, 4))
np.dot(A, B)
```

```{.python .input}
%%tab pytorch
B = torch.ones(3, 4)
torch.mm(A, B), A@B
```

```{.python .input}
%%tab tensorflow
B = tf.ones((3, 4), tf.float32)
tf.matmul(A, B)
```

```{.python .input}
%%tab jax
B = jnp.ones((3, 4))
jnp.matmul(A, B)
```

*Matrix--matrix multiplication* is often shortened to
*matrix multiplication*; don't confuse it with the Hadamard product.


## Norms
:label:`subsec_lin-algebra-norms`

Some of the most useful operators in linear algebra are *norms*. Informally,
a vector's norm tells us how *big* it is; the $\ell_2$ norm, for instance, is
its (Euclidean) length. This notion of *size* concerns the magnitude of the
components, not the dimensionality.

A norm is a function $\| \cdot \|$ mapping a vector to a scalar with three
properties:

1. Given any vector $\mathbf{x}$, if we scale (all elements of) the vector
   by a scalar $\alpha \in \mathbb{R}$, its norm scales accordingly:
   $$\|\alpha \mathbf{x}\| = |\alpha| \|\mathbf{x}\|.$$
2. For any vectors $\mathbf{x}$ and $\mathbf{y}$:
   norms satisfy the triangle inequality:
   $$\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|.$$
3. The norm of a vector is nonnegative and it only vanishes if the vector is zero:
   $$\|\mathbf{x}\| > 0 \textrm{ for all } \mathbf{x} \neq 0.$$

Many functions are valid norms, each encoding a different notion of size. The
Euclidean norm from school geometry (the hypotenuse of a right triangle) is
the square root of the sum of squared elements, formally the $\ell_2$ *norm*:

$$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}.$$

The method `norm` calculates the $\ell_2$ norm.

```{.python .input}
%%tab mxnet
u = np.array([3, -4])
np.linalg.norm(u)
```

```{.python .input}
%%tab pytorch
u = torch.tensor([3.0, -4.0])
torch.norm(u)
```

```{.python .input}
%%tab tensorflow
u = tf.constant([3.0, -4.0])
tf.norm(u)
```

```{.python .input}
%%tab jax
u = jnp.array([3.0, -4.0])
jnp.linalg.norm(u)
```

The $\ell_1$ norm is also common; its distance is the Manhattan distance. It
sums the absolute values of the elements:

$$\|\mathbf{x}\|_1 = \sum_{i=1}^n \left|x_i \right|.$$

It is less sensitive to outliers than the $\ell_2$ norm. To compute it, take
absolute values and sum.

```{.python .input}
%%tab mxnet
np.abs(u).sum()
```

```{.python .input}
%%tab pytorch
torch.abs(u).sum()
```

```{.python .input}
%%tab tensorflow
tf.reduce_sum(tf.abs(u))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(u, ord=1) # same as jnp.abs(u).sum()
```

Both are special cases of the general $\ell_p$ *norms*:

$$\|\mathbf{x}\|_p = \left(\sum_{i=1}^n \left|x_i \right|^p \right)^{1/p}.$$

Matrices are more complicated: they are collections of entries *and* objects
that transform vectors into other vectors. Asking how much longer
$\mathbf{X} \mathbf{v}$ can be than $\mathbf{v}$ leads to the *spectral* norm.
For now, we use the much easier *Frobenius norm*, the square root of the sum
of squared entries:

$$\|\mathbf{X}\|_\textrm{F} = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}.$$

It behaves like the $\ell_2$ norm of a matrix-shaped vector. This computes the
Frobenius norm of a matrix:

```{.python .input}
%%tab mxnet
np.linalg.norm(np.ones((4, 9)))
```

```{.python .input}
%%tab pytorch
torch.norm(torch.ones((4, 9)))
```

```{.python .input}
%%tab tensorflow
tf.norm(tf.ones((4, 9)))
```

```{.python .input}
%%tab jax
jnp.linalg.norm(jnp.ones((4, 9)))
```

Different norms measure size differently. Compare their "unit balls":

```{=html}
<div class="d2l-playground" data-playground="lp-norm-balls"></div>
```

Here's a hint of why this matters. Deep learning often solves optimization
problems: *maximize* the probability of observed data; *maximize* a
recommender's revenue; *minimize* the distance between predictions and
ground truth; *minimize* the distance between photos of the same person while
*maximizing* it between photos of different people. These distances, the
objectives of deep learning algorithms, are often expressed as norms.


## Discussion

This section covered all the linear algebra you need for a large part of
modern deep learning. There is much more, and much of it is useful for
machine learning: matrices can be factored into decompositions that reveal
low-dimensional structure in real data, and entire subfields use matrix and
tensor decompositions to find structure and make predictions. But this book
focuses on deep learning, and we think you'll be keener to learn more math
after applying machine learning to real datasets. So we'll introduce more
mathematics later as needed.

To learn more linear algebra, there are many excellent books and online
resources. For a more advanced crash course, see :citet:`Strang.1993`,
:citet:`Kolter.2008`, and :citet:`Petersen.Pedersen.ea.2008`.

To recap:

* Scalars, vectors, matrices, and tensors are the basic objects of linear
  algebra, with zero, one, two, and any number of axes, respectively.
* Tensors can be sliced via indexing, or reduced along given axes with
  operations such as `sum` and `mean`.
* Elementwise products are Hadamard products. Dot products, matrix--vector
  products, and matrix--matrix products are not elementwise and generally
  return shapes different from their operands.
* Matrix--matrix products take much longer to compute than Hadamard products
  (cubic rather than quadratic time).
* Norms capture notions of the magnitude of a vector (or matrix), and are
  often applied to the difference of two vectors to measure their distance.
* Common vector norms are the $\ell_1$ and $\ell_2$ norms; common matrix norms
  are the *spectral* and *Frobenius* norms.


## Exercises

1. Prove that the transpose of the transpose of a matrix is the matrix itself: $(\mathbf{A}^\top)^\top = \mathbf{A}$.
1. Given two matrices $\mathbf{A}$ and $\mathbf{B}$, show that sum and transposition commute: $\mathbf{A}^\top + \mathbf{B}^\top = (\mathbf{A} + \mathbf{B})^\top$.
1. Given any square matrix $\mathbf{A}$, is $\mathbf{A} + \mathbf{A}^\top$ always symmetric? Can you prove the result by using only the results of the previous two exercises?
1. We defined the tensor `X` of shape (2, 3, 4) in this section. What is the output of `len(X)`? Write your answer without implementing any code, then check your answer using code.
1. For a tensor `X` of arbitrary shape, does `len(X)` always correspond to the length of a certain axis of `X`? What is that axis?
1. Run `A / A.sum(axis=1)` and see what happens. Can you analyze the results?
1. When traveling between two points in downtown Manhattan, what is the distance that you need to cover in terms of the coordinates, i.e., in terms of avenues and streets? Can you travel diagonally?
1. Consider a tensor of shape (2, 3, 4). What are the shapes of the summation outputs along axes 0, 1, and 2?
1. Feed a tensor with three or more axes to the `linalg.norm` function and observe its output. What does this function compute for tensors of arbitrary shape?
1. Consider three large matrices, say $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ and $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{14}}$, initialized with Gaussian random variables. You want to compute the product $\mathbf{A} \mathbf{B} \mathbf{C}$. Is there any difference in memory footprint and speed, depending on whether you compute $(\mathbf{A} \mathbf{B}) \mathbf{C}$ or $\mathbf{A} (\mathbf{B} \mathbf{C})$? Why?
1. Consider three large matrices, say $\mathbf{A} \in \mathbb{R}^{2^{10} \times 2^{16}}$, $\mathbf{B} \in \mathbb{R}^{2^{16} \times 2^{5}}$ and $\mathbf{C} \in \mathbb{R}^{2^{5} \times 2^{16}}$. Is there any difference in speed depending on whether you compute $\mathbf{A} \mathbf{B}$ or $\mathbf{A} \mathbf{C}^\top$? Why? What changes if you initialize $\mathbf{C} = \mathbf{B}^\top$ without cloning memory? Why?
1. Consider three matrices, say $\mathbf{A}, \mathbf{B}, \mathbf{C} \in \mathbb{R}^{100 \times 200}$. Construct a tensor with three axes by stacking $[\mathbf{A}, \mathbf{B}, \mathbf{C}]$. What is the dimensionality? Slice out the second coordinate of the third axis to recover $\mathbf{B}$. Check that your answer is correct.
