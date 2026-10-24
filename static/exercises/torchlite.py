"""torchlite: a tiny NumPy-backed subset of PyTorch for running the book's
examples in the browser (Pyodide can't run real PyTorch).

It covers what the early chapters use: tensor creation, shapes, indexing,
elementwise math, reductions, linear algebra, a small reverse-mode autograd,
`torch.distributions.multinomial.Multinomial`, and (Chapter 3) `nn.Module`,
`nn.Linear`/`LazyLinear`, `nn.MSELoss`, `nn.init`, `optim.SGD` and
`utils.data.TensorDataset`/`DataLoader`, and (Chapter 4) softmax,
cross-entropy, `nn.Flatten` and a `torchvision` subset with the real
Fashion-MNIST dataset. It is for learning and
experimenting, not a full PyTorch: anything missing raises a clear error.

Loaded as the module `torch` only for the runnable book cells
(see static/exercises/notebook.py). Keep it compatible with Python 3.10.
"""
import builtins
import math
import os
import sys
import types

import numpy as np

__version__ = '0.1 (torchlite: NumPy-based PyTorch subset)'


class _DType:
    def __init__(self, name, np_dtype):
        self.name = name
        self.np = np.dtype(np_dtype)
        self.is_floating_point = self.np.kind == 'f'

    def __repr__(self):
        return f'torch.{self.name}'


float32 = float = _DType('float32', np.float32)
float64 = double = _DType('float64', np.float64)
float16 = half = _DType('float16', np.float16)
int64 = long = _DType('int64', np.int64)
int32 = int = _DType('int32', np.int32)
uint8 = _DType('uint8', np.uint8)
bool = _DType('bool', np.bool_)  # noqa: A001 (mirrors torch.bool)
_DTYPES = [float32, float64, float16, int64, int32, uint8, bool]


def _dtype_of(arr):
    for d in _DTYPES:
        if d.np == arr.dtype:
            return d
    return _DType(str(arr.dtype), arr.dtype)


def _np_dtype(dtype):
    if dtype is None:
        return None
    if isinstance(dtype, _DType):
        return dtype.np
    if dtype is builtins.float:
        return np.float32
    if dtype is builtins.int:
        return np.int64
    return np.dtype(dtype)


def _default_array(data, dtype=None):
    """Convert like torch.tensor: Python floats -> float32, ints -> int64."""
    if isinstance(data, Tensor):
        arr = data._data.copy()
    else:
        arr = np.array(_unwrap(data))
    if dtype is not None:
        return arr.astype(_np_dtype(dtype))
    if arr.dtype == np.float64 and not (isinstance(data, np.ndarray) and data.dtype == np.float64):
        arr = arr.astype(np.float32)
    elif arr.dtype.kind in 'iu' and arr.dtype != np.uint8:
        arr = arr.astype(np.int64)
    return arr


def _unwrap(x):
    if isinstance(x, Tensor):
        return x._data
    if isinstance(x, (list, tuple)):
        return type(x)(_unwrap(v) for v in x)
    return x


class Size(tuple):
    def __repr__(self):
        return f'torch.Size({list(self)})'

    def numel(self):
        return builtins.int(np.prod(self)) if self else 1


def _promote(result, *operands):
    """Mimic torch type promotion: float32 stays float32 with ints/scalars;
    int / int is float32."""
    if result.dtype == np.float64:
        kinds = [o.dtype for o in operands if isinstance(o, np.ndarray)]
        if kinds and not any(k == np.float64 for k in kinds):
            return result.astype(np.float32)
    return result


# ------------------------------------------------------------------ autograd

class _Node:
    """How a tensor was computed: parent tensors and a function mapping the
    output gradient to each parent's gradient."""

    def __init__(self, parents, backward):
        self.parents = parents
        self.backward = backward


def _unbroadcast(grad, shape):
    """Sum a broadcast gradient back down to `shape`."""
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    for i, n in enumerate(shape):
        if n == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


_grad_enabled = True


class no_grad:
    def __enter__(self):
        global _grad_enabled
        self._prev, _grad_enabled = _grad_enabled, False

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev

    def __call__(self, fn):
        def wrapper(*a, **k):
            with no_grad():
                return fn(*a, **k)
        return wrapper


def _make(data, parents=(), backward=None):
    out = Tensor._wrap(data)
    if _grad_enabled and parents and any(p.requires_grad for p in parents):
        out.requires_grad = True
        out._node = _Node(parents, backward)
    return out


def _as_tensor(x):
    return x if isinstance(x, Tensor) else Tensor._wrap(np.asarray(x))


class Tensor:
    __module__ = 'torch'

    def __init__(self, *args):
        self._data = np.array(args[0], dtype=np.float32) if args else np.zeros(0, np.float32)
        self.requires_grad = False
        self.grad = None
        self._node = None

    @classmethod
    def _wrap(cls, arr):
        t = cls.__new__(cls)
        t._data = arr if isinstance(arr, np.ndarray) else np.asarray(arr)
        t.requires_grad = False
        t.grad = None
        t._node = None
        return t

    # ---- basic properties
    @property
    def shape(self):
        return Size(self._data.shape)

    def size(self, dim=None):
        return self.shape if dim is None else self._data.shape[dim]

    @property
    def dtype(self):
        return _dtype_of(self._data)

    @property
    def ndim(self):
        return self._data.ndim

    def dim(self):
        return self._data.ndim

    @property
    def T(self):
        return _make(self._data.T, (self,), lambda g: (g.T,))

    def t(self):
        return self.T

    def numel(self):
        return builtins.int(self._data.size)

    def __len__(self):
        if self._data.ndim == 0:
            raise TypeError('len() of a 0-d tensor')
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        if self._data.ndim == 0:
            body = np.array2string(self._data, separator=', ')
        else:
            body = np.array2string(self._data, separator=', ', prefix='tensor(')
        extra = ''
        if self._data.dtype not in (np.float32, np.int64, np.bool_):
            extra = f', dtype={self.dtype!r}'
        if self.requires_grad:
            extra += ', requires_grad=True' if self._node is None else ', grad_fn=<Backward>'
        return f'tensor({body}{extra})'

    __str__ = __repr__

    # ---- conversion
    def numpy(self):
        if self.requires_grad:
            raise RuntimeError("Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.")
        return self._data

    def item(self):
        if self._data.size != 1:
            raise ValueError('only one element tensors can be converted to Python scalars')
        return self._data.reshape(()).item()

    def tolist(self):
        return self._data.tolist()

    def __float__(self):
        return builtins.float(self.item())

    def __int__(self):
        return builtins.int(self.item())

    def __bool__(self):
        if self._data.size != 1:
            raise RuntimeError('Boolean value of Tensor with more than one value is ambiguous')
        return builtins.bool(self._data.reshape(()).item())

    def __index__(self):
        return builtins.int(self.item())

    def __array__(self, dtype=None, copy=None):
        return self._data if dtype is None else self._data.astype(dtype)

    def type(self, dtype=None):
        return self.to(dtype) if dtype is not None else f'torch.{self.dtype.name}Tensor'

    def to(self, *args, **kwargs):
        dtype = kwargs.get('dtype')
        for a in args:
            if isinstance(a, _DType):
                dtype = a
        return Tensor._wrap(self._data.astype(_np_dtype(dtype))) if dtype else self

    def float(self):
        return Tensor._wrap(self._data.astype(np.float32))

    def double(self):
        return Tensor._wrap(self._data.astype(np.float64))

    def long(self):
        return Tensor._wrap(self._data.astype(np.int64))

    def int(self):
        return Tensor._wrap(self._data.astype(np.int32))

    def bool(self):
        return Tensor._wrap(self._data.astype(np.bool_))

    def clone(self):
        return _make(self._data.copy(), (self,), lambda g: (g,))

    def detach(self):
        return Tensor._wrap(self._data)

    def cpu(self):
        return self

    def requires_grad_(self, flag=True):
        if flag and self._data.dtype.kind != 'f':
            raise RuntimeError('only Tensors of floating point dtype can require gradients')
        self.requires_grad = flag
        return self

    # ---- autograd
    def backward(self, gradient=None, retain_graph=None, create_graph=False):
        if not self.requires_grad:
            raise RuntimeError('element 0 of tensors does not require grad and does not have a grad_fn')
        if gradient is None:
            if self._data.size != 1:
                raise RuntimeError('grad can be implicitly created only for scalar outputs')
            seed = np.ones_like(self._data, dtype=np.float32)
        else:
            seed = np.asarray(_unwrap(gradient), dtype=self._data.dtype).reshape(self._data.shape)
        # Topological order of the graph.
        order, seen = [], set()

        def visit(t):
            if id(t) in seen:
                return
            seen.add(id(t))
            if t._node is not None:
                for p in t._node.parents:
                    visit(p)
            order.append(t)
        visit(self)
        grads = {id(self): seed}
        for t in reversed(order):
            g = grads.pop(id(t), None)
            if g is None:
                continue
            if t._node is None:
                if t.requires_grad:  # leaf: accumulate, like PyTorch
                    g = g.astype(t._data.dtype)
                    t.grad = Tensor._wrap(g.copy()) if t.grad is None else Tensor._wrap(t.grad._data + g)
                continue
            for p, pg in zip(t._node.parents, t._node.backward(g)):
                if pg is None or not p.requires_grad:
                    continue
                pg = _unbroadcast(np.asarray(pg), p._data.shape)
                grads[id(p)] = grads[id(p)] + pg if id(p) in grads else pg

    def zero_(self):
        self._data[...] = 0
        return self

    # In-place initializers (used through `param.data.normal_(...)` etc.).
    def normal_(self, mean=0.0, std=1.0):
        self._data[...] = _rng.normal(mean, std, self._data.shape)
        return self

    def uniform_(self, a=0.0, b=1.0):
        self._data[...] = _rng.uniform(a, b, self._data.shape)
        return self

    def fill_(self, value):
        self._data[...] = _unwrap(value)
        return self

    def copy_(self, other):
        self._data[...] = _unwrap(other)
        return self

    @property
    def data(self):
        """The same values without gradient tracking (shares memory)."""
        return Tensor._wrap(self._data)

    @data.setter
    def data(self, value):
        self._data = np.array(_unwrap(value), dtype=self._data.dtype)

    @property
    def device(self):
        return device('cpu')

    @property
    def is_leaf(self):
        return self._node is None

    # ---- shape
    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], (tuple, list, Size)) else shape
        old = self._data.shape
        return _make(self._data.reshape(shape), (self,), lambda g: (g.reshape(old),))

    view = reshape

    def flatten(self):
        return self.reshape(-1)

    def unsqueeze(self, dim):
        return Tensor._wrap(np.expand_dims(self._data, dim))

    def squeeze(self, dim=None):
        return Tensor._wrap(np.squeeze(self._data) if dim is None else np.squeeze(self._data, dim))

    def __getitem__(self, idx):
        idx = _unwrap(idx)
        shape = self._data.shape

        def back(g):
            full = np.zeros(shape, dtype=g.dtype)
            np.add.at(full, idx, g)
            return (full,)
        return _make(self._data[idx], (self,), back)

    def __setitem__(self, idx, value):
        self._data[_unwrap(idx)] = _unwrap(value)

    # ---- arithmetic
    def _binary(self, other, op, back, reverse=False):
        if isinstance(other, (builtins.int, builtins.float, builtins.bool)) and not isinstance(other, Tensor):
            # Python numbers adopt the tensor's type, as in PyTorch:
            # float32 * 0.01 stays float32; int64 * 0.5 becomes float32.
            dt = self._data.dtype
            if dt.kind != 'f':
                dt = np.float32 if isinstance(other, builtins.float) else dt
            other = np.asarray(other, dtype=dt)
        o = _as_tensor(other)
        a, b = (o, self) if reverse else (self, o)
        out = _promote(op(a._data, b._data), a._data, b._data)
        return _make(out, (a, b), lambda g: back(g, a._data, b._data))

    def __add__(self, o):
        return self._binary(o, np.add, lambda g, a, b: (g, g))

    def __radd__(self, o):
        return self._binary(o, np.add, lambda g, a, b: (g, g), True)

    def __sub__(self, o):
        return self._binary(o, np.subtract, lambda g, a, b: (g, -g))

    def __rsub__(self, o):
        return self._binary(o, np.subtract, lambda g, a, b: (g, -g), True)

    def __mul__(self, o):
        return self._binary(o, np.multiply, lambda g, a, b: (g * b, g * a))

    def __rmul__(self, o):
        return self._binary(o, np.multiply, lambda g, a, b: (g * b, g * a), True)

    def __truediv__(self, o):
        div = lambda a, b: np.true_divide(a, b).astype(np.float32) \
            if a.dtype.kind in 'iub' and np.asarray(b).dtype.kind in 'iub' else np.true_divide(a, b)
        return self._binary(o, div, lambda g, a, b: (g / b, -g * a / b ** 2))

    def __rtruediv__(self, o):
        return self._binary(o, np.true_divide, lambda g, a, b: (g / b, -g * a / b ** 2), True)

    def __pow__(self, o):
        return self._binary(o, np.power, lambda g, a, b: (
            g * b * np.power(a, b - 1),
            g * np.power(a, b) * np.log(np.where(a > 0, a, 1))))

    def __rpow__(self, o):
        return self._binary(o, np.power, lambda g, a, b: (
            g * b * np.power(a, b - 1), g * np.power(a, b) * np.log(np.where(a > 0, a, 1))), True)

    def __matmul__(self, o):
        return matmul(self, o)

    def __rmatmul__(self, o):
        return matmul(o, self)

    def __neg__(self):
        return _make(-self._data, (self,), lambda g: (-g,))

    def __abs__(self):
        return abs(self)

    # In-place ops modify the data and keep the same object (same id()).
    def __iadd__(self, o):
        self._data += _unwrap(o)
        return self

    def __isub__(self, o):
        self._data -= _unwrap(o)
        return self

    def __imul__(self, o):
        self._data *= _unwrap(o)
        return self

    def __itruediv__(self, o):
        self._data /= _unwrap(o)
        return self

    # ---- comparisons (elementwise, no gradient)
    def _cmp(self, o, op):
        return Tensor._wrap(op(self._data, _unwrap(o)))

    def __eq__(self, o):
        return self._cmp(o, np.equal)

    def __ne__(self, o):
        return self._cmp(o, np.not_equal)

    def __lt__(self, o):
        return self._cmp(o, np.less)

    def __le__(self, o):
        return self._cmp(o, np.less_equal)

    def __gt__(self, o):
        return self._cmp(o, np.greater)

    def __ge__(self, o):
        return self._cmp(o, np.greater_equal)

    __hash__ = object.__hash__

    def all(self):
        return Tensor._wrap(np.all(self._data))

    def any(self):
        return Tensor._wrap(np.any(self._data))

    # ---- reductions and math
    def sum(self, dim=None, keepdim=False, axis=None, keepdims=None, dtype=None):
        return sum(self, dim if dim is not None else axis,
                   keepdims if keepdims is not None else keepdim)

    def mean(self, dim=None, keepdim=False, axis=None, keepdims=None):
        return mean(self, dim if dim is not None else axis,
                    keepdims if keepdims is not None else keepdim)

    def cumsum(self, dim=None, axis=None):
        d = dim if dim is not None else axis
        return Tensor._wrap(np.cumsum(self._data, axis=d))

    def norm(self, p=2):
        return norm(self, p)

    def exp(self):
        return exp(self)

    def log(self):
        return log(self)

    def abs(self):
        return abs(self)

    def sqrt(self):
        return sqrt(self)

    def max(self, dim=None):
        return Tensor._wrap(self._data.max() if dim is None else self._data.max(axis=dim))

    def min(self, dim=None):
        return Tensor._wrap(self._data.min() if dim is None else self._data.min(axis=dim))

    def argmax(self, dim=None, keepdim=False, axis=None):
        d = dim if dim is not None else axis
        out = np.argmax(self._data, axis=d)
        return Tensor._wrap(np.expand_dims(out, d) if keepdim and d is not None else out)

    def softmax(self, dim):
        return softmax(self, dim)

    def log_softmax(self, dim):
        return log_softmax(self, dim)

    def dot(self, other):
        return dot(self, other)

    def mv(self, v):
        return mv(self, v)

    def mm(self, m):
        return mm(self, m)

    def zeros_like(self):
        return zeros_like(self)


def _axes(dim):
    if isinstance(dim, (list, tuple)):
        return tuple(dim)
    return dim


def sum(x, dim=None, keepdim=False, axis=None, keepdims=None):
    x = _as_tensor(x)
    d = _axes(dim if dim is not None else axis)
    k = keepdims if keepdims is not None else keepdim
    shape = x._data.shape

    def back(g):
        if d is not None and not k:
            g = np.expand_dims(g, d)
        return (np.broadcast_to(g, shape),)
    return _make(np.asarray(x._data.sum(axis=d, keepdims=k)), (x,), back)


def mean(x, dim=None, keepdim=False, axis=None, keepdims=None):
    x = _as_tensor(x)
    if x._data.dtype.kind != 'f':
        raise RuntimeError('mean(): could not infer output dtype. Input dtype must be '
                           'either a floating point or complex dtype.')
    d = _axes(dim if dim is not None else axis)
    k = keepdims if keepdims is not None else keepdim
    n = x._data.size if d is None else np.prod([x._data.shape[i] for i in np.atleast_1d(d)])
    s = sum(x, d, k)
    return s / builtins.float(n)


def _unary(x, fn, dfn):
    x = _as_tensor(x)
    data = fn(x._data)
    if data.dtype == np.float64 and x._data.dtype != np.float64:
        data = data.astype(np.float32)
    return _make(data, (x,), lambda g: (g * dfn(x._data, data),))


def exp(x):
    return _unary(x, np.exp, lambda a, out: out)


def log(x):
    return _unary(x, np.log, lambda a, out: 1 / a)


def sqrt(x):
    return _unary(x, np.sqrt, lambda a, out: 0.5 / out)


def sin(x):
    return _unary(x, np.sin, lambda a, out: np.cos(a))


def cos(x):
    return _unary(x, np.cos, lambda a, out: -np.sin(a))


def tanh(x):
    return _unary(x, np.tanh, lambda a, out: 1 - out ** 2)


def sigmoid(x):
    return _unary(x, lambda a: 1 / (1 + np.exp(-a)), lambda a, out: out * (1 - out))


def relu(x):
    return _unary(x, lambda a: np.maximum(a, 0), lambda a, out: (a > 0).astype(a.dtype))


def log_softmax(x, dim=-1):
    """Stable log-softmax: x - max - log(sum(exp(x - max)))."""
    x = _as_tensor(x)
    z = x._data - x._data.max(axis=dim, keepdims=True)
    out = z - np.log(np.exp(z).sum(axis=dim, keepdims=True))
    p = np.exp(out)
    return _make(out, (x,), lambda g: (g - p * g.sum(axis=dim, keepdims=True),))


def softmax(x, dim=-1):
    x = _as_tensor(x)
    z = np.exp(x._data - x._data.max(axis=dim, keepdims=True))
    out = z / z.sum(axis=dim, keepdims=True)
    return _make(out, (x,), lambda g: (out * (g - (g * out).sum(axis=dim, keepdims=True)),))


def cross_entropy(input, target, reduction='mean', label_smoothing=0.0):
    """torch.nn.functional.cross_entropy for (n, q) logits and class indices
    (or (n, q) class probabilities), computed stably from the logits."""
    x = _as_tensor(input)
    if x._data.ndim != 2:
        raise NotImplementedError('cross_entropy in the browser supports (batch, classes) inputs only')
    logits = x._data
    n, q = logits.shape
    t = _unwrap(target)
    t = np.asarray(t)
    if t.dtype.kind in 'iu':
        if t.shape != (n,):
            raise RuntimeError(f'expected target of shape ({n},), got {tuple(t.shape)}')
        Y = np.zeros_like(logits)
        Y[np.arange(n), t] = 1
    else:
        Y = t.astype(logits.dtype)
    if label_smoothing:
        Y = Y * (1 - label_smoothing) + label_smoothing / q
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    per = -(Y * logp).sum(axis=1)
    P = np.exp(logp)
    if reduction == 'mean':
        out, scale = np.asarray(per.mean(), dtype=logits.dtype), np.full(n, 1.0 / n, logits.dtype)
    elif reduction == 'sum':
        out, scale = np.asarray(per.sum(), dtype=logits.dtype), np.ones(n, logits.dtype)
    else:
        out, scale = per.astype(logits.dtype), None

    def back(g):
        w = g * scale if scale is not None else g
        return ((P - Y) * np.reshape(w, (n, 1)),)
    return _make(out, (x,), back)


def abs(x):  # noqa: A001 (mirrors torch.abs)
    return _unary(x, np.abs, lambda a, out: np.sign(a))


def norm(x, p=2, dim=None):
    x = _as_tensor(x)
    if x._data.dtype.kind != 'f':
        raise RuntimeError('linalg.vector_norm: Expected a floating point or complex tensor as input.')
    if p == 2 and dim is None:
        return sqrt(sum(x * x))
    if p == 1 and dim is None:
        return sum(abs(x))
    return Tensor._wrap(np.linalg.norm(x._data, ord=p, axis=dim))


def matmul(a, b):
    a, b = _as_tensor(a), _as_tensor(b)
    A, B = a._data, b._data
    if A.dtype != B.dtype:
        raise RuntimeError(f'expected m1 and m2 to have the same dtype, but got: '
                           f'{_dtype_of(A).name} != {_dtype_of(B).name}')

    def back(g):
        if A.ndim == 1 and B.ndim == 1:
            return g * B, g * A
        if B.ndim == 1:
            return np.outer(g, B), A.T @ g
        if A.ndim == 1:
            return B @ g, np.outer(A, g)
        return g @ B.T, A.T @ g
    return _make(np.asarray(A @ B), (a, b), back)


def dot(a, b):
    if _as_tensor(a).ndim != 1 or _as_tensor(b).ndim != 1:
        raise RuntimeError('1D tensors expected, but got other shapes')
    return matmul(a, b)


def mv(m, v):
    return matmul(m, v)


def mm(a, b):
    return matmul(a, b)


# ---------------------------------------------------------------- creation

def tensor(data, dtype=None, requires_grad=False, device=None):
    t = Tensor._wrap(_default_array(data, dtype))
    return t.requires_grad_(True) if requires_grad else t


def as_tensor(data, dtype=None):
    return tensor(data, dtype)


def from_numpy(arr):
    return Tensor._wrap(arr)  # shares memory, like PyTorch


def _size_args(size):
    if len(size) == 1 and isinstance(size[0], (tuple, list, Size)):
        return tuple(size[0])
    return tuple(size)


def zeros(*size, dtype=None, requires_grad=False):
    t = Tensor._wrap(np.zeros(_size_args(size), dtype=_np_dtype(dtype) or np.float32))
    return t.requires_grad_(True) if requires_grad else t


def ones(*size, dtype=None, requires_grad=False):
    t = Tensor._wrap(np.ones(_size_args(size), dtype=_np_dtype(dtype) or np.float32))
    return t.requires_grad_(True) if requires_grad else t


def full(size, value, dtype=None):
    return Tensor._wrap(np.full(tuple(size), value, dtype=_np_dtype(dtype) or np.float32))


def zeros_like(x, dtype=None):
    x = _as_tensor(x)
    return Tensor._wrap(np.zeros_like(x._data, dtype=_np_dtype(dtype)))


def ones_like(x, dtype=None):
    x = _as_tensor(x)
    return Tensor._wrap(np.ones_like(x._data, dtype=_np_dtype(dtype)))


def arange(start, end=None, step=1, dtype=None, requires_grad=False):
    if end is None:
        start, end = 0, start
    arr = np.arange(start, end, step)
    if dtype is not None:
        arr = arr.astype(_np_dtype(dtype))
    elif arr.dtype.kind == 'f':
        arr = arr.astype(np.float32)
    else:
        arr = arr.astype(np.int64)
    t = Tensor._wrap(arr)
    return t.requires_grad_(True) if requires_grad else t


def linspace(start, end, steps, dtype=None, requires_grad=False):
    t = Tensor._wrap(np.linspace(start, end, steps).astype(_np_dtype(dtype) or np.float32))
    return t.requires_grad_(True) if requires_grad else t


_rng = np.random.default_rng()


def manual_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)


def randn(*size, dtype=None, requires_grad=False, **kwargs):
    if 'size' in kwargs:
        size = (kwargs.pop('size'),)
    t = Tensor._wrap(_rng.standard_normal(_size_args(size)).astype(_np_dtype(dtype) or np.float32))
    return t.requires_grad_(True) if requires_grad else t


def rand(*size, dtype=None):
    return Tensor._wrap(_rng.random(_size_args(size)).astype(_np_dtype(dtype) or np.float32))


def normal(mean, std, size, requires_grad=False):
    t = Tensor._wrap(_rng.normal(mean, std, size=tuple(size)).astype(np.float32))
    return t.requires_grad_(True) if requires_grad else t


def randperm(n, dtype=None):
    return Tensor._wrap(_rng.permutation(n).astype(np.int64))


def eye(n, m=None, dtype=None):
    return Tensor._wrap(np.eye(n, m, dtype=_np_dtype(dtype) or np.float32))


def randint(low, high=None, size=(), dtype=None):
    if high is None:
        low, high = 0, low
    return Tensor._wrap(_rng.integers(low, high, size=tuple(size)).astype(np.int64))


def cat(tensors, dim=0):
    return Tensor._wrap(np.concatenate([_unwrap(t) for t in tensors], axis=dim))


concat = cat


def stack(tensors, dim=0):
    return Tensor._wrap(np.stack([_unwrap(t) for t in tensors], axis=dim))


def is_tensor(x):
    return isinstance(x, Tensor)


def numel(x):
    return x.numel()


def cumsum(x, dim):
    return _as_tensor(x).cumsum(dim)


def manual_device(*a, **k):
    return 'cpu'


class device(str):
    """torch.device('cpu'); only the CPU exists in the browser."""

    def __new__(cls, name='cpu'):
        return str.__new__(cls, name)

    @property
    def type(self):
        return str(self).split(':')[0]

    def __repr__(self):
        return f"device(type='{self.type}')"


class _Cuda:
    @staticmethod
    def is_available():
        return False

    @staticmethod
    def device_count():
        return 0


cuda = _Cuda()


# ----------------------------------------------------------- distributions

class Multinomial:
    """torch.distributions.multinomial.Multinomial (sampling only)."""

    def __init__(self, total_count=1, probs=None):
        self.total_count = builtins.int(total_count)
        p = np.asarray(_unwrap(probs), dtype=np.float64)
        self.probs = p / p.sum()

    def sample(self, sample_shape=()):
        shape = tuple(sample_shape) if not isinstance(sample_shape, builtins.int) else (sample_shape,)
        counts = _rng.multinomial(self.total_count, self.probs, size=shape or None)
        return Tensor._wrap(np.asarray(counts, dtype=np.float32))


distributions = types.ModuleType('torch.distributions')
distributions.__doc__ = 'Probability distributions (torchlite: only Multinomial sampling).'
distributions.multinomial = types.ModuleType('torch.distributions.multinomial')
distributions.multinomial.Multinomial = Multinomial
distributions.Multinomial = Multinomial


class _Unsupported(types.ModuleType):
    """Placeholder for parts of PyTorch that torchlite doesn't implement."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f'{self.__name__}.{name} is not available in the browser. This page runs a small '
            'NumPy-based subset of PyTorch (torchlite); run this code in Colab or locally '
            'for full PyTorch.')


# ---------------------------------------------------------------- torch.nn

class Parameter(Tensor):
    """A tensor that is a model parameter (requires grad, registered by Module)."""
    __module__ = 'torch.nn'

    def __new__(cls, data=None, requires_grad=True):
        t = Tensor._wrap(np.array(_unwrap(data), dtype=np.float32) if data is not None
                         else np.zeros(0, np.float32))
        t.__class__ = cls
        t.requires_grad = requires_grad
        return t

    def __init__(self, *args, **kwargs):
        pass

    def __repr__(self):
        return 'Parameter containing:\n' + Tensor.__repr__(self)


class Module:
    """torch.nn.Module: parameters and submodules are registered when assigned
    as attributes; calling the module runs forward()."""
    __module__ = 'torch.nn'

    def __init__(self, *args, **kwargs):
        object.__setattr__(self, '_parameters', {})
        object.__setattr__(self, '_modules', {})
        object.__setattr__(self, 'training', True)

    def _ensure(self):
        if '_parameters' not in self.__dict__:
            Module.__init__(self)

    def __setattr__(self, name, value):
        self._ensure()
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
        object.__setattr__(self, name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def named_parameters(self, prefix=''):
        self._ensure()
        seen = set()
        for name, p in self._parameters.items():
            if id(p) not in seen:
                seen.add(id(p))
                yield prefix + name, p
        for mname, m in self._modules.items():
            for name, p in m.named_parameters(prefix + mname + '.'):
                if id(p) not in seen:
                    seen.add(id(p))
                    yield name, p

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def children(self):
        self._ensure()
        return iter(self._modules.values())

    def named_children(self):
        self._ensure()
        return iter(self._modules.items())

    def modules(self):
        yield self
        for m in self.children():
            yield from m.modules()

    def apply(self, fn):
        for m in self.children():
            m.apply(fn)
        fn(self)
        return self

    def train(self, mode=True):
        self._ensure()
        object.__setattr__(self, 'training', mode)
        for m in self.children():
            m.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def to(self, *args, **kwargs):
        return self

    def state_dict(self):
        return {name: Tensor._wrap(p._data.copy()) for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        for name, p in self.named_parameters():
            p._data[...] = _unwrap(state[name])

    def extra_repr(self):
        return ''

    def __repr__(self):
        self._ensure()
        lines = [f'{type(self).__name__}({self.extra_repr()}']
        for name, m in self._modules.items():
            lines.append(f'  ({name}): ' + repr(m).replace('\n', '\n  '))
        return lines[0] + (')' if len(lines) == 1 else '\n' + '\n'.join(lines[1:]) + '\n)')


class Linear(Module):
    """y = x W^T + b, with PyTorch's default uniform initialization."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        k = 1 / math.sqrt(in_features)
        self.weight = Parameter(_rng.uniform(-k, k, (out_features, in_features)))
        self.bias = Parameter(_rng.uniform(-k, k, out_features)) if bias else None

    def forward(self, X):
        out = matmul(X, self.weight.T)
        return out + self.bias if self.bias is not None else out

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class LazyLinear(Linear):
    """Linear layer whose input size is inferred on the first call. Like
    PyTorch, the weight is an uninitialized parameter until then; in-place
    initializers called earlier are replayed once the shape is known."""

    def __init__(self, out_features, bias=True):
        Module.__init__(self)
        self.in_features, self.out_features = None, out_features
        self.weight = _LazyParameter()
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, X):
        w = self.weight
        if isinstance(w, _LazyParameter):
            # Materialize in place, like PyTorch, so an optimizer created
            # before the first call still updates this same parameter.
            n = _as_tensor(X).shape[-1]
            self.in_features = n
            k = 1 / math.sqrt(n)
            pending = w.pending
            w._data = _rng.uniform(-k, k, (self.out_features, n)).astype(np.float32)
            w.__class__ = Parameter
            for op, args in pending:
                getattr(w.data, op)(*args)
        return Linear.forward(self, X)


class _LazyParameter(Parameter):
    """Placeholder weight of a LazyLinear before its first call."""

    def __new__(cls):
        t = Parameter.__new__(cls, np.zeros(0))
        t.pending = []
        return t

    @property
    def data(self):
        lazy = self

        class _Recorder:
            def __getattr__(self, op):
                def record(*args):
                    lazy.pending.append((op, args))
                return record
        return _Recorder()


class MSELoss(Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, y_hat, y):
        l = (y_hat - y) ** 2
        if self.reduction == 'mean':
            return mean(l)
        if self.reduction == 'sum':
            return sum(l)
        return l


class CrossEntropyLoss(Module):
    def __init__(self, reduction='mean', label_smoothing=0.0):
        super().__init__()
        self.reduction, self.label_smoothing = reduction, label_smoothing

    def forward(self, y_hat, y):
        return cross_entropy(y_hat, y, self.reduction, self.label_smoothing)


class Flatten(Module):
    def __init__(self, start_dim=1, end_dim=-1):
        super().__init__()
        self.start_dim, self.end_dim = start_dim, end_dim

    def forward(self, X):
        shape = X.shape
        end = self.end_dim % len(shape)
        return X.reshape(*shape[:self.start_dim], -1, *shape[end + 1:])


class ReLU(Module):
    def forward(self, X):
        return relu(X)


class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = -1 if dim is None else dim

    def forward(self, X):
        return softmax(X, self.dim)


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, str(i), layer)

    def forward(self, X):
        for m in self._modules.values():
            X = m(X)
        return X

    def __getitem__(self, i):
        return list(self._modules.values())[i]

    def __len__(self):
        return len(self._modules)


class _Init(types.ModuleType):
    @staticmethod
    def normal_(t, mean=0.0, std=1.0):
        t.data.normal_(mean, std)
        return t

    @staticmethod
    def uniform_(t, a=0.0, b=1.0):
        t.data.uniform_(a, b)
        return t

    @staticmethod
    def zeros_(t):
        t.data.fill_(0)
        return t

    @staticmethod
    def ones_(t):
        t.data.fill_(1)
        return t

    @staticmethod
    def constant_(t, value):
        t.data.fill_(value)
        return t

    @staticmethod
    def xavier_uniform_(t, gain=1.0):
        fan_out, fan_in = t.shape[0], t.shape[1]
        a = gain * math.sqrt(6 / (fan_in + fan_out))
        t.data.uniform_(-a, a)
        return t


class _NN(types.ModuleType):
    """torch.nn: the supported parts, and a clear error for the rest."""

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        raise NotImplementedError(
            f'torch.nn.{name} is not available in the browser yet. This page runs torchlite, '
            'a small NumPy-based subset of PyTorch; run this code in Colab or locally for full PyTorch.')


nn = _NN('torch.nn')
nn.Module = Module
nn.Parameter = Parameter
nn.Linear = Linear
nn.LazyLinear = LazyLinear
nn.MSELoss = MSELoss
nn.Sequential = Sequential
nn.CrossEntropyLoss = CrossEntropyLoss
nn.Flatten = Flatten
nn.ReLU = ReLU
nn.Softmax = Softmax
nn.init = _Init('torch.nn.init')
nn.functional = types.ModuleType('torch.nn.functional')
nn.functional.relu = relu
nn.functional.softmax = softmax
nn.functional.log_softmax = log_softmax
nn.functional.cross_entropy = cross_entropy
nn.functional.mse_loss = lambda y_hat, y, reduction='mean': MSELoss(reduction)(y_hat, y)


# -------------------------------------------------------------- torch.optim

class SGD:
    """torch.optim.SGD with weight decay and parameter groups."""

    def __init__(self, params, lr=0.01, weight_decay=0.0, momentum=0.0):
        params = list(params)
        groups = params if params and isinstance(params[0], dict) else [{'params': params}]
        self.param_groups = []
        for g in groups:
            g = dict(g)
            g['params'] = list(g['params']) if not isinstance(g['params'], Tensor) else [g['params']]
            g.setdefault('lr', lr)
            g.setdefault('weight_decay', weight_decay)
            self.param_groups.append(g)

    def zero_grad(self, set_to_none=True):
        for g in self.param_groups:
            for p in g['params']:
                p.grad = None

    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                grad = p.grad._data
                if g['weight_decay']:
                    grad = grad + g['weight_decay'] * p._data
                p._data -= g['lr'] * grad


optim = types.ModuleType('torch.optim')
optim.SGD = SGD


# --------------------------------------------------------- torch.utils.data

class TensorDataset:
    def __init__(self, *tensors):
        self.tensors = tensors

    def __len__(self):
        return len(self.tensors[0])

    def __getitem__(self, i):
        return tuple(t[i] for t in self.tensors)


class DataLoader:
    """Minibatches from a dataset (optionally shuffled each epoch): fast paths
    for TensorDataset and datasets with a vectorized `_batch(indices)`."""

    def __init__(self, dataset, batch_size=1, shuffle=False, **kwargs):
        self.dataset, self.batch_size, self.shuffle = dataset, batch_size, shuffle

    def __len__(self):
        return math.ceil(len(self.dataset) / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)
        order = _rng.permutation(n) if self.shuffle else np.arange(n)
        for i in range(0, n, self.batch_size):
            idx = order[i:i + self.batch_size]
            if hasattr(self.dataset, 'tensors'):
                yield tuple(Tensor._wrap(t._data[idx]) for t in self.dataset.tensors)
            elif hasattr(self.dataset, '_batch'):
                yield self.dataset._batch(idx)
            else:
                items = [self.dataset[int(j)] for j in idx]
                yield tuple(Tensor._wrap(np.stack([np.asarray(_unwrap(v)) for v in col]))
                            for col in zip(*items))


utils = types.ModuleType('torch.utils')
utils.data = types.ModuleType('torch.utils.data')
utils.data.TensorDataset = TensorDataset
utils.data.DataLoader = DataLoader
linalg = types.ModuleType('torch.linalg')
linalg.norm = norm
linalg.matmul = matmul


# ------------------------------------------------------- torchvision (subset)

# Datasets are .npz files fetched by the page (worker.js) into this folder.
DATASETS_DIR = os.environ.get('D2L_DATASETS', '/d2l/datasets')
_DATASET_CACHE = {}


def _load_dataset(name):
    if name not in _DATASET_CACHE:
        path = os.path.join(DATASETS_DIR, name + '.npz')
        if not os.path.exists(path):
            raise RuntimeError(f'The {name} dataset could not be downloaded to the browser. '
                               'Check your connection and run the cell again.')
        with np.load(path) as f:
            _DATASET_CACHE[name] = {k: f[k] for k in f.files}
    return _DATASET_CACHE[name]


def _resize_matrix(n_in, n_out):
    """Bilinear interpolation weights (half-pixel centers, like PIL/torchvision)."""
    src = np.clip((np.arange(n_out) + 0.5) * (n_in / n_out) - 0.5, 0, n_in - 1)
    lo = np.floor(src).astype(np.int64)
    hi = np.minimum(lo + 1, n_in - 1)
    w = (src - lo).astype(np.float32)
    M = np.zeros((n_out, n_in), np.float32)
    np.add.at(M, (np.arange(n_out), lo), 1 - w)
    np.add.at(M, (np.arange(n_out), hi), w)
    return M


class _Transform:
    def __call__(self, img):
        out = self._batch(np.asarray(_unwrap(img))[None])[0]
        return Tensor._wrap(out) if out.dtype.kind == 'f' else out

    def __repr__(self):
        return type(self).__name__ + '()'


class Compose(_Transform):
    def __init__(self, transforms):
        self.transforms = list(transforms)

    def _batch(self, a):
        for t in self.transforms:
            a = t._batch(a)
        return a


class Resize(_Transform):
    def __init__(self, size, antialias=True):
        self.size = (size, size) if isinstance(size, builtins.int) else tuple(size)

    def _batch(self, a):
        h, w = a.shape[-2:]
        if (h, w) == self.size:
            return a
        out = _resize_matrix(h, self.size[0]) @ a.astype(np.float32) @ _resize_matrix(w, self.size[1]).T
        if a.dtype == np.uint8:  # like resizing a PIL image
            out = np.clip(np.rint(out), 0, 255).astype(np.uint8)
        return out


class ToTensor(_Transform):
    """uint8 images (H, W) -> float32 tensors (1, H, W) in [0, 1]."""

    def _batch(self, a):
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255
        return a[:, None] if a.ndim == 3 else a

    def __call__(self, img):
        return Tensor._wrap(self._batch(np.asarray(_unwrap(img))[None])[0])


class Normalize(_Transform):
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32).reshape(-1, 1, 1)
        self.std = np.asarray(std, np.float32).reshape(-1, 1, 1)

    def _batch(self, a):
        return (a - self.mean) / self.std


class FashionMNIST:
    """torchvision.datasets.FashionMNIST: the full dataset (60,000 training
    and 10,000 test images), stored as uint8 and transformed per minibatch."""
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __init__(self, root=None, train=True, transform=None, target_transform=None, download=False):
        d = _load_dataset('fashion-mnist')
        self.train, self.transform = train, transform
        self._x = d['train_x' if train else 'test_x']
        self._y = d['train_y' if train else 'test_y'].astype(np.int64)

    @property
    def data(self):
        return Tensor._wrap(self._x)

    @property
    def targets(self):
        return Tensor._wrap(self._y)

    def __len__(self):
        return len(self._y)

    def _batch(self, idx):
        X = self._x[idx]
        if self.transform is not None:
            X = self.transform._batch(X)
        return Tensor._wrap(X), Tensor._wrap(self._y[idx])

    def __getitem__(self, i):
        X, y = self._batch(np.array([i]))
        return Tensor._wrap(X._data[0]), builtins.int(y._data[0])

    def __repr__(self):
        return (f'Dataset FashionMNIST\n    Number of datapoints: {len(self)}\n'
                f'    Split: {"Train" if self.train else "Test"}')


torchvision = types.ModuleType('torchvision')
torchvision.__version__ = __version__
torchvision.transforms = types.ModuleType('torchvision.transforms')
for _t in (Compose, Resize, ToTensor, Normalize):
    setattr(torchvision.transforms, _t.__name__, _t)
torchvision.datasets = types.ModuleType('torchvision.datasets')
torchvision.datasets.FashionMNIST = FashionMNIST


def install():
    """Register torchlite as the modules `torch`, `torch.distributions`, ..."""
    me = sys.modules[__name__]
    sys.modules['torch'] = me
    sys.modules['torch.distributions'] = distributions
    sys.modules['torch.distributions.multinomial'] = distributions.multinomial
    sys.modules['torch.nn'] = nn
    sys.modules['torch.nn.functional'] = nn.functional
    sys.modules['torch.nn.init'] = nn.init
    sys.modules['torch.optim'] = optim
    sys.modules['torch.utils'] = utils
    sys.modules['torch.utils.data'] = utils.data
    sys.modules['torch.linalg'] = linalg
    sys.modules['torchvision'] = torchvision
    sys.modules['torchvision.transforms'] = torchvision.transforms
    sys.modules['torchvision.datasets'] = torchvision.datasets


def uninstall():
    for name in list(sys.modules):
        if name in ('torch', 'torchvision') or name.startswith(('torch.', 'torchvision.')):
            del sys.modules[name]
