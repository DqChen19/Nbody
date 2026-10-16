# Nbody (JAX port of NbodyGradient)

This package is a JAX/Python re-implementation of the Julia package
[`NbodyGradient`](https://github.com/ericagol/NbodyGradient). It reproduces the AHL21 symplectic integrator, transit-timing
detection, and analytic-derivative machinery, replacing Julia's mutable in-place structs
with JAX pytrees and Julia's native loops with `jax.lax.scan` / `jax.lax.while_loop` /
`jax.lax.fori_loop` so the whole step can be `jax.jit`-compiled.

This document focuses on two things: how each Python module maps back to its Julia
counterpart, and how the package is organized internally.

### Small tutorial/benchmark tests
Nbody_notebook.ipynb shows a couple of examples that used for Trappist-1 systems.

### Comparison

tmax = 1000.0
h = 0.1

| Package     | NbodyGradient.jl | Nbody.py | Nbody.py after compiled|
| ----------- | ----------- | ----------- |----------- |
| No Grad     |    0.6s     |      4.64s  | 0.24s |
| With Grad   |    1.5s     |      10.37s |  1.66s|