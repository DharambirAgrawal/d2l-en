```{.python .input}
%load_ext d2lbook.tab
tab.interact_select(['mxnet', 'pytorch', 'tensorflow', 'jax'])
```

# Calculus
:label:`sec_calculus`

For a long time, the area of a circle was a mystery. Then, in Ancient
Greece, Archimedes had the clever idea of inscribing polygons with more and
more vertices inside a circle (:numref:`fig_circle_area`). A polygon with $n$
vertices splits into $n$ triangles. As the partition gets finer, each
triangle's height approaches the radius $r$ and its base approaches
$2 \pi r/n$ (the ratio of arc to secant approaches 1). So the polygon's area
approaches $n \cdot r \cdot \frac{1}{2} (2 \pi r/n) = \pi r^2$.

![Finding the area of a circle as a limit procedure.](../img/polygon-circle.svg)
:label:`fig_circle_area`

This limiting procedure is the root of both *differential calculus* and
*integral calculus*. Differential calculus tells us how to increase or
decrease a function's value by changing its arguments, which is exactly what
deep learning's *optimization problems* need: we repeatedly update the
parameters to decrease the loss. Optimization is about fitting models to
training data, and calculus is its key prerequisite. But remember, the
ultimate goal is to do well on *previously unseen* data. That problem,
*generalization*, is a focus of other chapters.

```{.python .input}
%%tab mxnet
%matplotlib inline
from d2l import mxnet as d2l
from matplotlib_inline import backend_inline
from mxnet import np, npx
npx.set_np()
```

```{.python .input}
%%tab pytorch
%matplotlib inline
from d2l import torch as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab tensorflow
%matplotlib inline
from d2l import tensorflow as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

```{.python .input}
%%tab jax
%matplotlib inline
from d2l import jax as d2l
from matplotlib_inline import backend_inline
import numpy as np
```

## Derivatives and Differentiation

A *derivative* is the rate of change of a function with respect to its
arguments. It tells us how fast a loss would increase or decrease if we
*increased* or *decreased* each parameter by an infinitesimally small amount.
Formally, for $f: \mathbb{R} \rightarrow \mathbb{R}$, mapping scalars to
scalars, the *derivative* of $f$ at a point $x$ is

$$f'(x) = \lim_{h \rightarrow 0} \frac{f(x+h) - f(x)}{h}.$$
:eqlabel:`eq_derivative`

The right-hand side is a *limit*: what the ratio of the change in the
function value, $f(x + h) - f(x)$, to the perturbation $h$ converges to as
$h$ shrinks to zero.

When $f'(x)$ exists, $f$ is *differentiable* at $x$; when it exists for all
$x$ in a set, e.g., the interval $[a,b]$, $f$ is differentiable on that set.
Not all functions are differentiable, including many we'd like to optimize,
such as accuracy and the area under the receiving operating characteristic
(AUC). But computing the derivative of the loss is a crucial step in nearly
every algorithm for training deep networks, so we often optimize a
differentiable *surrogate* instead.


The derivative $f'(x)$ is the *instantaneous* rate of change of $f(x)$ with
respect to $x$. For intuition, let's define $u = f(x) = 3x^2-4x$.

```{.python .input}
%%tab mxnet
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab pytorch
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab tensorflow
def f(x):
    return 3 * x ** 2 - 4 * x
```

```{.python .input}
%%tab jax
def f(x):
    return 3 * x ** 2 - 4 * x
```

At $x=1$, $\frac{f(x+h) - f(x)}{h}$ approaches $2$ as $h$ approaches $0$.
This isn't a proof, but it quickly suggests $f'(1) = 2$.

```{.python .input}
%%tab all
for h in 10.0**np.arange(-1, -6, -1):
    print(f'h={h:.5f}, numerical limit={(f(1+h)-f(1))/h:.5f}')
```

Several equivalent notations exist for derivatives. Given $y = f(x)$, these
all mean the same thing:

$$f'(x) = y' = \frac{dy}{dx} = \frac{df}{dx} = \frac{d}{dx} f(x) = Df(x) = D_x f(x),$$

where $\frac{d}{dx}$ and $D$ are *differentiation operators*. Derivatives of
some common functions:

$$\begin{aligned} \frac{d}{dx} C & = 0 && \textrm{for any constant $C$} \\ \frac{d}{dx} x^n & = n x^{n-1} && \textrm{for } n \neq 0 \\ \frac{d}{dx} e^x & = e^x \\ \frac{d}{dx} \ln x & = x^{-1}. \end{aligned}$$

Functions composed of differentiable functions are often differentiable
themselves. For differentiable $f$ and $g$ and a constant $C$, these rules
come in handy:

$$\begin{aligned} \frac{d}{dx} [C f(x)] & = C \frac{d}{dx} f(x) && \textrm{Constant multiple rule} \\ \frac{d}{dx} [f(x) + g(x)] & = \frac{d}{dx} f(x) + \frac{d}{dx} g(x) && \textrm{Sum rule} \\ \frac{d}{dx} [f(x) g(x)] & = f(x) \frac{d}{dx} g(x) + g(x) \frac{d}{dx} f(x) && \textrm{Product rule} \\ \frac{d}{dx} \frac{f(x)}{g(x)} & = \frac{g(x) \frac{d}{dx} f(x) - f(x) \frac{d}{dx} g(x)}{g^2(x)} && \textrm{Quotient rule} \end{aligned}$$

Applying them to $3 x^2 - 4x$:

$$\frac{d}{dx} [3 x^2 - 4x] = 3 \frac{d}{dx} x^2 - 4 \frac{d}{dx} x = 6x - 4.$$

At $x = 1$, the derivative is indeed $2$. Derivatives give the *slope* of a
function at a particular point.

## Visualization Utilities

We can visualize slopes with `matplotlib`, using a few helper functions.
`use_svg_display` makes `matplotlib` output crisper SVG graphics. The comment
`#@save` saves a function, class, or code block to the `d2l` package so we
can reuse it later, e.g., as `d2l.use_svg_display()`.

```{.python .input}
%%tab all
def use_svg_display():  #@save
    """Use the svg format to display a plot in Jupyter."""
    backend_inline.set_matplotlib_formats('svg')
```

`set_figsize` sets the figure size. Since
`from matplotlib import pyplot as plt` was saved with `#@save`, we can call
`d2l.plt`.

```{.python .input}
%%tab all
def set_figsize(figsize=(3.5, 2.5)):  #@save
    """Set the figure size for matplotlib."""
    use_svg_display()
    d2l.plt.rcParams['figure.figsize'] = figsize
```

`set_axes` sets the axes' properties, such as labels, ranges, and scales.

```{.python .input}
%%tab all
#@save
def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """Set the axes for matplotlib."""
    axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
    axes.set_xscale(xscale), axes.set_yscale(yscale)
    axes.set_xlim(xlim),     axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()
```

With these three functions, we define `plot` to overlay multiple curves.
Most of its code just makes sure the input sizes and shapes match.

```{.python .input}
%%tab all
#@save
def plot(X, Y=None, xlabel=None, ylabel=None, legend=[], xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """Plot data points."""

    def has_one_axis(X):  # True if X (tensor or list) has 1 axis
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))
    
    if has_one_axis(X): X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
        
    set_figsize(figsize)
    if axes is None:
        axes = d2l.plt.gca()
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        axes.plot(x,y,fmt) if len(x) else axes.plot(y,fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
```

Now we plot $u = f(x)$ and its tangent line $y = 2x - 3$ at $x=1$, where the
coefficient $2$ is the tangent's slope.

```{.python .input}
%%tab all
x = np.arange(0, 3, 0.1)
plot(x, [f(x), 2 * x - 3], 'x', 'f(x)', legend=['f(x)', 'Tangent line (x=1)'])
```

See the limit in the derivative's definition happen:

```{=html}
<div class="d2l-playground" data-playground="derivative-limit"></div>
```

## Partial Derivatives and Gradients
:label:`subsec_calculus-grad`

So far we've differentiated functions of one variable. Deep learning also
needs functions of *many* variables, so here are the derivative notions for
such *multivariate* functions.


Let $y = f(x_1, x_2, \ldots, x_n)$ be a function of $n$ variables. The
*partial derivative* of $y$ with respect to its $i^\textrm{th}$ parameter
$x_i$ is

$$ \frac{\partial y}{\partial x_i} = \lim_{h \rightarrow 0} \frac{f(x_1, \ldots, x_{i-1}, x_i+h, x_{i+1}, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}.$$


To compute it, treat $x_1, \ldots, x_{i-1}, x_{i+1}, \ldots, x_n$ as
constants and differentiate $y$ with respect to $x_i$. These notations are
all common and equivalent:

$$\frac{\partial y}{\partial x_i} = \frac{\partial f}{\partial x_i} = \partial_{x_i} f = \partial_i f = f_{x_i} = f_i = D_i f = D_{x_i} f.$$

Stacking the partial derivatives with respect to all variables gives the
*gradient*. For $f: \mathbb{R}^n \rightarrow \mathbb{R}$ with input
$\mathbf{x} = [x_1, x_2, \ldots, x_n]^\top$ and scalar output, the gradient
with respect to $\mathbf{x}$ is the vector of $n$ partial derivatives:

$$\nabla_{\mathbf{x}} f(\mathbf{x}) = \left[\partial_{x_1} f(\mathbf{x}), \partial_{x_2} f(\mathbf{x}), \ldots
\partial_{x_n} f(\mathbf{x})\right]^\top.$$ 

When unambiguous, we write $\nabla f(\mathbf{x})$ for
$\nabla_{\mathbf{x}} f(\mathbf{x})$. Handy rules for multivariate functions:

* For all $\mathbf{A} \in \mathbb{R}^{m \times n}$ we have $\nabla_{\mathbf{x}} \mathbf{A} \mathbf{x} = \mathbf{A}^\top$ and $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A}  = \mathbf{A}$.
* For square matrices $\mathbf{A} \in \mathbb{R}^{n \times n}$ we have that $\nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{A} \mathbf{x}  = (\mathbf{A} + \mathbf{A}^\top)\mathbf{x}$ and in particular
$\nabla_{\mathbf{x}} \|\mathbf{x} \|^2 = \nabla_{\mathbf{x}} \mathbf{x}^\top \mathbf{x} = 2\mathbf{x}$.

Similarly, for any matrix $\mathbf{X}$,
$\nabla_{\mathbf{X}} \|\mathbf{X} \|_\textrm{F}^2 = 2\mathbf{X}$.



## Chain Rule

Deep learning gradients are often hard to compute because we work with deeply
nested functions (of functions (of functions...)). The *chain rule* takes
care of this. For a single variable, if $y = f(g(x))$ and both $y=f(u)$ and
$u=g(x)$ are differentiable, then


$$\frac{dy}{dx} = \frac{dy}{du} \frac{du}{dx}.$$



For multivariate functions, suppose $y = f(\mathbf{u})$ has variables
$u_1, u_2, \ldots, u_m$, and each $u_i = g_i(\mathbf{x})$ has variables
$x_1, x_2, \ldots, x_n$, i.e., $\mathbf{u} = g(\mathbf{x})$. Then

$$\frac{\partial y}{\partial x_{i}} = \frac{\partial y}{\partial u_{1}} \frac{\partial u_{1}}{\partial x_{i}} + \frac{\partial y}{\partial u_{2}} \frac{\partial u_{2}}{\partial x_{i}} + \ldots + \frac{\partial y}{\partial u_{m}} \frac{\partial u_{m}}{\partial x_{i}} \ \textrm{ and so } \ \nabla_{\mathbf{x}} y =  \mathbf{A} \nabla_{\mathbf{u}} y,$$

where the *matrix* $\mathbf{A} \in \mathbb{R}^{n \times m}$ holds the
derivative of $\mathbf{u}$ with respect to $\mathbf{x}$. So evaluating the
gradient means computing a vector--matrix product, one of the key reasons
linear algebra is so central to deep learning systems.



## Discussion

We've only scratched the surface, but some ideas already stand out. First,
the composition rules for differentiation can be applied mechanically, so
gradients can be computed *automatically*; this takes no creativity, freeing
us to think about other things. Second, differentiating vector-valued
functions means multiplying matrices as we trace the dependency graph of
variables from output to input: the graph is traversed *forward* to evaluate a
function and *backwards* to compute gradients. Later chapters formally
introduce backpropagation, a procedure for applying the chain rule.

For optimization, gradients tell us how to move a model's parameters to lower
the loss, and every step of the optimization algorithms in this book computes
a gradient.

## Exercises

1. So far we took the rules for derivatives for granted. 
   Using the definition and limits prove the properties 
   for (i) $f(x) = c$, (ii) $f(x) = x^n$, (iii) $f(x) = e^x$ and (iv) $f(x) = \log x$.
1. In the same vein, prove the product, sum, and quotient rule from first principles. 
1. Prove that the constant multiple rule follows as a special case of the product rule. 
1. Calculate the derivative of $f(x) = x^x$. 
1. What does it mean that $f'(x) = 0$ for some $x$? 
   Give an example of a function $f$ 
   and a location $x$ for which this might hold. 
1. Plot the function $y = f(x) = x^3 - \frac{1}{x}$ 
   and plot its tangent line at $x = 1$.
1. Find the gradient of the function 
   $f(\mathbf{x}) = 3x_1^2 + 5e^{x_2}$.
1. What is the gradient of the function 
   $f(\mathbf{x}) = \|\mathbf{x}\|_2$? What happens for $\mathbf{x} = \mathbf{0}$?
1. Can you write out the chain rule for the case 
   where $u = f(x, y, z)$ and $x = x(a, b)$, $y = y(a, b)$, and $z = z(a, b)$?
1. Given a function $f(x)$ that is invertible, 
   compute the derivative of its inverse $f^{-1}(x)$. 
   Here we have that $f^{-1}(f(x)) = x$ and conversely $f(f^{-1}(y)) = y$. 
   Hint: use these properties in your derivation.
