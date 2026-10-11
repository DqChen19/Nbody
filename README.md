# Nbody (JAX port of NbodyGradient)

This package is a JAX/Python re-implementation of the Julia package
[`NbodyGradient`](https://github.com/ericagol/NbodyGradient) (local reference copy:
`~/Desktop/NbodyGradient`). It reproduces the AHL21 symplectic integrator, transit-timing
detection, and analytic-derivative machinery, replacing Julia's mutable in-place structs
with JAX pytrees and Julia's native loops with `jax.lax.scan` / `jax.lax.while_loop` /
`jax.lax.fori_loop` so the whole step can be `jax.jit`-compiled.

This document focuses on two things: how each Python module maps back to its Julia
counterpart, and how the package is organized internally.

## 1. Julia ↔ Python correspondence

| Concept | Julia (`NbodyGradient/src/`) | Python (`Nbody/`) | Notes |
|---|---|---|---|
| Package entry point / exports | `NbodyGradient.jl` | `__init__.py` | Defines global constants (`GNEWT`, `YEAR`, ...) and re-exports the public API. |
| Simulation state | `struct State` in `integrator/Integrator.jl` | `integrator/State.py` | Julia's `State` is a mutable struct updated in place (`set_state!`); Python's `State` is an immutable, registered JAX pytree (`dataclasses.replace` instead of mutation). |
| Derivative scratch buffers | `PreAllocArrays.jl` (`Derivatives`, `dTime`, `Jacobian` structs) | `pre_alloc_arrays.py` | Julia pre-allocates and mutates these arrays to avoid GC pressure; Python keeps the same field layout but they are functionally updated (new pytree each call) since JAX arrays are immutable. |
| Integrator dispatcher (`intr(s, ...)` callable) | `integrator/Integrator.jl` (`function (intr::Integrator)(...)`, dispatches on `time`, `N`, or a transit-output type, with a `grad::Bool` keyword) | `integrator/Integrator_with_grad.py` (`Integrator`) and `integrator/Integrator_no_grad.py` (`Integrator_no_grad`) | Julia uses one struct with a `grad=true/false` keyword; Python splits this into two separate classes with the same `__call__` dispatch pattern (`None` / step-count / target-time / `TransitTiming` / `TransitParameters` / `TransitSnapshot`), because JAX needs the gradient/no-gradient code paths to be traced separately. |
| AHL21 step, gradient-enabled | `integrator/ahl21/ahl21.jl` | `integrator/ahl21/ahl21.py` | Same kick/drift/Kepler sub-steps and Jacobian chain rule; Julia's nested `for i in 1:n, j in i+1:n` pair loops become `jax.lax.scan` over a precomputed static `(i, j)` index array so the loop body is traced once, not unrolled per pair. |
| AHL21 step, no-gradient | `integrator/ahl21/ahl21_no_grad.jl` | `integrator/ahl21/ahl21_no_grad.py` | Same relationship as above, without Jacobian propagation. This is the fast path used for pure orbit/transit-time integration. |
| Old/retired integrator variants | `integrator/ahl21/ahl21_old.jl`, `integrator/dh17/dh17_no_grad.jl`, everything under `OLD/` | *(not ported)* | Only the current AHL21 scheme has a Python port; DH17 and the deprecated Kepler solvers were not carried over. |
| Cartesian ↔ orbital-element conversion | `integrator/convert.jl` | `integrator/convert.py` | Direct function-for-function port (`calc_a`, `calc_e`, `calc_I`, `calc_Omega`, `calc_varpi`, `calc_t0`, `calc_P`, `get_orbital_elements`, ...). |
| Sky-plane geometry / transit condition `g` | `transits/snapshot.jl` | `integrator/snapshot.py` | Moved from a `transits/` folder into `integrator/` in Python since it's used by both the step kernels and the transit-detection kernels. Same `calc_bv`, `g_func`, `gd_func`, plus `calc_dbvdq`/`calc_dbvdelements` for snapshot derivatives. |
| Transit-time refinement (Newton iteration) | Inline in `transits/timing.jl`'s `(intr::Integrator)(s, tt::TransitTiming)` method | `integrator/timing.py` (`find_transit_grad`, `find_transit_no_grad`, `dtbvdq`, `calc_dtdelements`, `calc_dtbvdelements`) | Julia interleaves "step, check `g`, Newton-refine" directly inside the callable; Python factors the Newton refinement (`lax.while_loop`) and the `dtbvdq`/element-derivative conversions into standalone functions. |
| Transit *detection* loop (the `for i in tt.occs ... if gi>0 && gsave[i]<0 ...` block) | `transits/timing.jl` | `integrator/kernels.py` (`detect_transit_no_grad`, `detect_transit_grad`, `record_transit_no_grad`, `record_transit_grad`, `integrate_transit_output_no_grad`, `integrate_transit_output_grad`) | This has no single-file Julia counterpart: `kernels.py` is the Python module that turns Julia's imperative "step N times, and on each step loop over occultors checking for a sign change" logic into a `lax.fori_loop` (steps) with a Python-unrolled loop over the *static* tuple of occultors, each guarded by `lax.cond`. |
| Transit output containers | `struct TransitTiming`, `struct TransitParameters`, `struct TransitSnapshot` (`transits/Transits.jl`) | `integrator/Transits.py` | Same three container types and `create_transit_*`/`zero_transit_*` constructors; Python additionally registers each as a JAX pytree (`tree_flatten`/`tree_unflatten`), splitting fields into dynamic `children` (arrays) vs static `aux_data` (`ntt`, `ti`, `occs`). |
| Elements / initial-condition types | `ics/InitialConditions.jl` (`Elements`, `ElementsIC`, `CartesianIC`) | `ics/initial_conditions.py` | Same three types and constructors (`from_elements`, `from_kwargs`, `from_matrix`, `from_file`), including the `amatrix`/`sigma_m` hierarchy-mass helpers. |
| Elements → Cartesian conversion | `ics/init_nbody.jl` | `ics/init_nbody.py` | `kepcalc`, `d_dm`, `init_nbody` ported function-for-function, including the `jac_init` Jacobian-of-initial-conditions construction. |
| Kepler-orbit → Cartesian state (single orbit) | `ics/kepler_init.jl` | `ics/kepler_init.py` | `kepler_cartesian`/`kepler_init`; Python computes the Jacobian via `jax.jacfwd` instead of Julia's hand-differentiated formulas. |
| Kepler equation solver | `ics/kepler.jl` | `ics/kepler.py` | `ekepler`/`kepler`, same Newton iteration, `lax.while_loop` instead of Julia's `while`. |
| Hierarchy matrix construction | `ics/setup_hierarchy.jl` | `ics/setup_hierarchy.py` | Same recursive `_bottom_level`/`_nlevel`/`_symmetric`/`_symmetric_nest` structure; Python builds the matrix with plain NumPy (it is static, precomputed once before tracing) and converts to a JAX array at the end. |
| Built-in example systems (TRAPPIST-1, Kepler-36, ...) | `ics/defaults.jl` | `ics/defaults.py` | `get_default_ICs`, `available_systems`. |
| Batched orbital-element outputs | `outputs/elements.jl` | `outputs/elements.py` | `convert_to_elements`, `convert_relative_states_to_elements`, `get_orbital_elements`; Python uses `jax.vmap` where Julia loops over bodies. |
| Cartesian trajectory recorder | `outputs/Outputs.jl` | `outputs/Outputs.py` | `CartesianOutput`, `run_cartesian_output`. |
| Misc numerics (Kahan summation, `G3`/`H1`...`H8` series, cubic solver) | `utils.jl` | `utils.py` | Direct port; `while` loops become `jax.lax.while_loop`. |

## 2. Module structure

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
