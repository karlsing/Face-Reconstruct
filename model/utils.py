from typing import TypeVar, Callable, Optional, Union
from torch import Tensor, Size
from inspect import isfunction

T = TypeVar('T')

def extract(a: Tensor, t: Tensor, x_shape: Size):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def exists(x: Optional[T]):
    return x is not None


def default(val: T, d: Union[T, Callable[[], T]]):
    if exists(val):
        return val
    return d() if isfunction(d) else d
