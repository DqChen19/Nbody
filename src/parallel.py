"""
Run many independent integrations concurrently on the CPU.

One integration is a long chain of small, dependent scalar operations, so XLA
runs it on a single core. Independent systems (different initial conditions,
e.g. MCMC samples or a parameter grid) can therefore run side by side on
separate threads: JAX releases the GIL while a compiled computation executes,
and all threads share one compiled executable.

jax.vmap is not a good fit on CPU here: under vmap every lax.cond evaluates
both branches and every lax.while_loop (Kepler solves, transit Newton
refinement) runs until the slowest batch member converges, so a batch is
slower per system than separate calls.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional

import jax


def default_workers() -> int:
    """One worker per physical core (hyperthreads add little for this workload)."""
    logical = os.cpu_count() or 1
    return max(1, logical // 2)


def map_parallel(fn: Callable[..., Any], args: Iterable[Any], max_workers: Optional[int] = None,) -> list:
    """
    Evaluate fn(arg) for every arg on a thread pool; results keep the input order.

    Example
    -------
        intr = Integrator_no_grad.default(h=0.1, t0=0.0, tmax=1000.0)

        def run(ic):
            s = State.from_ic(ic)
            tt = create_transit_timing(tmax=1000.0, ic=ic)
            return intr(s, tt)

        results = map_parallel(run, list_of_ics)

    Every system should share the same hierarchy and number of bodies (and
    output sizes), so all calls reuse one compiled executable. Call fn once
    first if you want to keep compilation out of a timing.
    """
    def call(arg):
        return jax.block_until_ready(fn(arg))

    with ThreadPoolExecutor(max_workers or default_workers()) as pool:
        return list(pool.map(call, args))
