import socket
from functools import wraps
from itertools import islice
from typing import TypeVar

T = TypeVar("T")


def mkdir(func: T) -> T:
    @wraps(func)
    def wrapper(*args, **kwargs):
        path = func(*args, **kwargs)
        path.mkdir(parents=True, exist_ok=True)
        return path

    return wrapper


def batched(iterable, n):
    if n < 1:
        raise ValueError("n must be at least one")
    iterator = iter(iterable)
    while batch := list(islice(iterator, n)):
        yield batch


def batchedlist(iterable, n):
    return list(batched(iterable, n))


def is_online() -> bool:
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=3)
    except OSError:
        return False
    else:
        return True
