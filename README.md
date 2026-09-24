# Nbody (JAX port of NbodyGradient)

This package is a JAX/Python re-implementation of the Julia package
[`NbodyGradient`](https://github.com/ericagol/NbodyGradient) (local reference copy:
`~/Desktop/NbodyGradient`). It reproduces the AHL21 symplectic integrator, transit-timing
detection, and analytic-derivative machinery, replacing Julia's mutable in-place structs
with JAX pytrees and Julia's native loops with `jax.lax.scan` / `jax.lax.while_loop` /
`jax.lax.fori_loop` so the whole step can be `jax.jit`-compiled.

### Module structure

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
