# Nbody (JAX port of NbodyGradient)

This package is a JAX/Python re-implementation of the Julia package
[`NbodyGradient`](https://github.com/ericagol/NbodyGradient) (local reference copy:
`~/Desktop/NbodyGradient`). It reproduces the AHL21 symplectic integrator, transit-timing
detection, and analytic-derivative machinery, replacing Julia's mutable in-place structs
with JAX pytrees and Julia's native loops with `jax.lax.scan` / `jax.lax.while_loop` /
`jax.lax.fori_loop` so the whole step can be `jax.jit`-compiled.

This document focuses on two things: how each Python module maps back to its Julia
counterpart, and how the package is organized internally.

## 1. Module structure

```
Nbody/
├── __init__.py                  # Global constants + public API surface
├── pre_alloc_arrays.py          # Derivatives / dTime / Jacobian scratch pytrees
├── utils.py                     # comp_sum, G3/H1.../H8 series, cubic1, low-level numerics
│
├── ics/                         # Initial conditions: orbital elements -> Cartesian state
│   ├── initial_conditions.py    #   Elements / ElementsIC / CartesianIC (+ A-matrix helpers)
│   ├── setup_hierarchy.py       #   Build the hierarchy (epsilon) matrix from a bin vector
│   ├── kepler.py                #   Kepler's equation solver (mean -> eccentric -> true anomaly)
│   ├── kepler_init.py           #   One Kepler orbit -> Cartesian state + Jacobian
│   ├── init_nbody.py            #   Full hierarchy of orbits -> (x, v, jac_init) for all bodies
│   └── defaults.py              #   Built-in example systems (trappist-1, kepler-36)
│
├── integrator/                  # State, step schemes, and time-stepping dispatch
│   ├── State.py                 #   State pytree (x, v, t, m, jac_step, pair, ...)
│   ├── Transits.py              #   TransitTiming / TransitParameters / TransitSnapshot pytrees
│   ├── snapshot.py               #   g_func / gd_func / calc_bv (+ Jacobian variants) — the transit condition
│   ├── timing.py                 #   Newton refinement of transit time (find_transit_grad/no_grad)
│   ├── convert.py                #   Cartesian -> orbital elements (for diagnostics/output)
│   ├── kernels.py                #   fori_loop step driver + transit-detection loop (occultor scan)
│   ├── Integrator_no_grad.py     #   Integrator_no_grad class: __call__ dispatch, no Jacobian tracking
│   ├── Integrator_with_grad.py   #   Integrator class: __call__ dispatch, with Jacobian tracking
│   └── ahl21/
│       ├── ahl21_no_grad.py      #     One AHL21 step, positions/velocities only
│       └── ahl21.py              #     One AHL21 step, with Jacobian + dq/dt propagation
│
└── outputs/                     # Post-processing / recording
    ├── elements.py               #   Cartesian state -> orbital elements (batched, vmap)
    └── Outputs.py                #   CartesianOutput: record the trajectory at every step
```

### Dependency flow

`ics/` has no dependency on `integrator/`; it only turns an `InitialConditions` object into
`(x, v, jac_init)` used by `State.from_ic`. `integrator/ahl21/` depends on `State.py` and
`utils.py` only, and is where the actual physics (kick/drift/Kepler-solve sub-steps) lives.
`integrator/kernels.py` and `integrator/timing.py` sit one layer above the AHL21 step
functions: they call a step function generically (`scheme_no_grad(s, h)` /
`scheme_grad(s, d, h)`) without depending on the AHL21 module directly, so any future
integrator with the same call signature could be substituted. `Integrator_no_grad.py` /
`Integrator_with_grad.py` are the user-facing entry points that wire a concrete scheme
(`ahl21_no_grad` / `ahl21`) into `kernels.py`'s loop drivers and expose the
`intr(s, arg)`-style dispatch that mirrors Julia's callable `Integrator` struct.
`outputs/` is purely a consumer of `State`/`ic` and does not feed back into the integrator.

### Static vs. dynamic data (JAX pytrees)

Every stateful object (`State`, `Derivatives`, `TransitTiming`, `TransitParameters`,
`TransitSnapshot`) is a frozen `dataclass` registered as a JAX pytree via
`tree_flatten`/`tree_unflatten`. Fields are split into two groups:

- **children** — JAX arrays that change every step (`x`, `v`, `t`, `jac_step`, `tt`,
  `gsave`, ...). These are traced and can be differentiated through.
- **aux_data** — plain Python values that are fixed for the lifetime of a run (`n`,
  `pair`, `ntt`, `ti`, `occs`). These are baked into the compiled program as constants,
  the same way a Julia struct's type parameters are fixed at compile time.

This split is what lets `ahl21_no_grad.py` and `ahl21.py` replace Julia's `for i in 1:n,
j in i+1:n` pairwise loops with a single `lax.scan` over a precomputed `(i, j)` index
array: the loop *shape* (how many pairs, which pairs, in what order) is static
`aux_data`-like information known at trace time, while only the physical quantities
being accumulated (positions, velocities, Jacobian rows) are traced `children`.
