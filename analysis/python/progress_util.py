"""
tqdm helpers for ProcessPoolExecutor: progress advances as tasks *complete* (as_completed),
while results stay in the original payload order.

If tqdm is not installed, iterables are passed through unchanged.
"""

from __future__ import annotations

import sys
from concurrent.futures import Executor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def tqdm_maybe(iterable=None, *, disable: bool = False, **kwargs):
    """tqdm(iterable) or tqdm(**kwargs) for manual updates; no-op iterator if missing/disabled."""
    if disable:
        if iterable is not None:
            return iterable
        return _DummyPbar()

    try:
        from tqdm import tqdm
    except ImportError:
        if iterable is not None:
            return iterable
        return _DummyPbar()

    kwargs.setdefault("file", sys.stderr)
    if iterable is not None:
        return tqdm(iterable, **kwargs)
    return tqdm(**kwargs)


class _DummyPbar:
    def update(self, n: int = 1) -> None:
        pass

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


def executor_map_as_completed(
    ex: Executor,
    fn: Callable[[T], R],
    payloads: list[T],
    desc: str,
    *,
    disable: bool = False,
) -> list[R]:
    """submit(fn, p_i) for each payload; tqdm over as_completed; return [r_0, r_1, ...] in order."""
    n = len(payloads)
    if n == 0:
        return []
    fut_to_i = {ex.submit(fn, p): i for i, p in enumerate(payloads)}
    out: list[R | None] = [None] * n
    it = as_completed(fut_to_i)
    it = tqdm_maybe(it, total=n, desc=desc, disable=disable)
    for fut in it:
        idx = fut_to_i[fut]
        out[idx] = fut.result()
    return out  # type: ignore[return-value]
