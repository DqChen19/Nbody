from __future__ import annotations

from dataclasses import replace
from functools import partial
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from jax import lax


from ...utils import comp_sum, G3, H1, H2
ITMAX_GAMMA  = 20
YEAR    = 365.242
GNEWT   = 39.4845 / (YEAR * YEAR)
Array = jax.Array

# ════════════════════════════════════
# Main function
# ════════════════════════════════════
def ahl21_no_grad(state,h,):
    """
    One derivative-free AHL21 integration step.

    Required Integrator interface
    -----------------------------
    state_new = ahl21_no_grad(state, h)

    Notes
    -----
    This function does not update state.t. The enclosing Integrator
    kernel owns the absolute simulation clock.
    """
    h = jnp.asarray(h,dtype=state.x.dtype,)

    h2 = 0.5 * h
    h6 = h / 6.0

    x = state.x
    v = state.v

    xerror = state.xerror
    verror = state.verror

    m = state.m
    pair = state.pair
    n = state.n

    # drift_kepler/kepler_drift each run a lax.while_loop Kepler root-solve
    # per pair; unrolling that in Python (like the plain-arithmetic pair
    # functions below) multiplies real compiled code per pair and blows up
    # compile time for larger n (e.g. TRAPPIST-1's 28 pairs). They stay
    # scan-based, so `pair` needs to be an array there (dynamically
    # indexable by the scan's traced loop indices) rather than the static
    # tuple used everywhere else.
    pair_arr = jnp.asarray(pair, dtype=jnp.bool_)

    # 1. Fast kick.
    v, verror = kickfast(x,v,verror,h6,m,pair,n,)
    # 2. Half drift.
    x, xerror = drift(x,v,xerror,h2,)
    # 3. Drift + Kepler.
    x, v, xerror, verror = drift_kepler(x,v,xerror,verror,h2,m,pair_arr,n,)
    # 4. Kick-pair correction.
    v, verror = phic(x,v,verror,h,m,pair,n,)
    # 5. Fourth-order Kepler-pair correction.
    v, verror = phisalpha(x,v,verror,h,jnp.asarray(2.0, dtype=x.dtype),m,pair,n,)
    # 6. Kepler + drift in reverse pair order.
    x, v, xerror, verror = kepler_drift(x, v, xerror, verror, h2, m, pair_arr, n,)
    # 7. Half drift.
    x, xerror = drift(x, v, xerror, h2,)
    # 8. Final fast kick.
    v, verror = kickfast(x, v, verror,h6, m, pair, n,)

    # Compute dqdt (phase-space time derivatives) needed by the transit-timing
    # Newton refinement (gd_func). dqdt[7*i + 0:3] = velocity of body i,
    # dqdt[7*i + 3:6] = gravitational acceleration of body i from all others.
    state = replace(state, x=x, v=v, xerror=xerror, verror=verror )
    dqdt = compute_dqdt_no_grad(state)

    return replace(state, dqdt=dqdt)


# ════════════════════════════════
# Helpful functions
# ════════════════════════════════
def compute_dqdt_no_grad(state):
    """
    Sum pairwise Newtonian accelerations over all body pairs (independent
    of the `pair` hierarchy flag -- this is a plain N-body force sum used
    only for the transit-timing Newton refinement, not part of the AHL21
    step itself).

    n is a Python int (static), so the pair loop is unrolled in Python at
    trace time: with i, j as plain ints, `x[:, i]`/`.at[:, i].add(...)`
    compile to static slices that XLA can fuse, instead of the
    dynamic-index gather/scatter a `lax.scan` over traced indices would
    require. For the small n (a handful of bodies) this code targets,
    unrolling produces a far smaller, flatter compiled graph than
    scan-based control flow.
    """
    x = state.x
    v = state.v
    m = state.m
    n = state.n

    dtype = x.dtype
    eps = jnp.finfo(dtype).eps

    a = jnp.zeros_like(x)
    for i in range(n - 1):
        for j in range(i + 1, n):
            rij = x[:, j] - x[:, i]
            r2 = jnp.maximum(jnp.dot(rij, rij), eps,)
            r2inv = 1.0 / r2
            r3inv = r2inv * jnp.sqrt(r2inv)
            fac = GNEWT * rij * r3inv
            a = a.at[:, i].add(m[j] * fac).at[:, j].add(-m[i] * fac)

    # dqdt[7*i:7*i+3] = v[:, i]; dqdt[7*i+3:7*i+6] = a[:, i]; slot 7*i+6 stays 0.
    dqdt = jnp.zeros((n, 7), dtype=dtype)
    dqdt = dqdt.at[:, 0:3].set(v.T)
    dqdt = dqdt.at[:, 3:6].set(a.T)

    return dqdt.reshape(7 * n)

def _safe_signed_denominator(value: Array, eps: Array,):
    """
    Replace a denominator close to zero while preserving its sign.
    """
    replacement = jnp.where(value >= 0.0, eps, -eps)
    return jnp.where(jnp.abs(value) > eps, value, replacement,)

def _cbrt_jax(x):
    """Real cube root supporting negative inputs."""
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)

def _cubic1_jax(a, b, c):
    '''Return one real solution of x³ + a x² + b x + c = 0'''
    dtype = jnp.result_type(a, b, c)

    one_third = jnp.asarray(1.0 / 3.0, dtype=dtype)
    one_half = jnp.asarray(0.5, dtype=dtype)

    a3 = a * one_third
    q = a3 * a3 - b * one_third
    r = a3**3 + one_half * (-a3 * b + c)

    r2 = r * r
    q3 = q * q * q

    eps = jnp.finfo(dtype).eps
    safe_b = _safe_signed_denominator(b, eps)

    def three_real(_: Any) -> Array:
        return -c / safe_b

    def one_real(_: Any) -> Array:
        discriminant = jnp.maximum(r2 - q3, 0.0)
        cubic_term = (jnp.abs(r) + jnp.sqrt(discriminant))
        root_a = (-jnp.sign(r) * _cbrt_jax(cubic_term))
        root_b = jnp.where(root_a == 0.0, 0.0, q / root_a,)
        return root_a + root_b - a3

    return lax.cond(r2 < q3, three_real, one_real, operand=None,)

# ══════════════════════════════
# Kepler solver
# ══════════════════════════════

def _jac_delxv_gamma_no_grad(x0: Array, v0: Array, k: Array, h: Array, *, drift_first: bool,):
    """
    JAX-compatible translation of Julia jac_delxv_gamma!(x0, v0, k, h, drift_first).

    Parameters
    ----------
    x0, v0: Relative position and velocity, shape (3,).
    k: G * (m_i + m_j).
    h: Integration step.

    drift_first: Static Python bool.

    Returns
    -------
    delxv: Shape (6,), containing delta position and delta velocity.
    """
    dtype = x0.dtype
    zero = jnp.asarray(0.0, dtype=dtype)

    if drift_first:
        x0_drifted = x0 - h * v0
        r0 = jnp.linalg.norm(x0_drifted)
        eta = jnp.dot(x0_drifted, v0)
    else:
        r0 = jnp.linalg.norm(x0)
        eta = jnp.dot(x0, v0)

    r0inv = 1.0 / r0

    beta0 = (2.0 * k * r0inv - jnp.dot(v0, v0))

    beta0inv = 1.0 / beta0
    signb = jnp.sign(beta0)
    sqb = jnp.sqrt(signb * beta0)

    zeta = k - r0 * beta0

    gamma_fallback = h * r0inv * sqb

    def cubic_guess(_):
        return _cubic1_jax(3.0 * eta * sqb / zeta,
                        6.0 * r0 * signb * beta0 / zeta,
                        -6.0 * h * signb * beta0 * sqb / zeta,)

    def noncubic_guess(_):
        def quadratic_guess(_):
            reta = r0 / eta
            disc = reta**2 + 2.0 * h / eta

            return lax.cond(disc > zero,
                lambda _: sqb * (-reta + jnp.sqrt(disc)),
                lambda _: gamma_fallback,
                operand=None,)

        return lax.cond(eta != zero, quadratic_guess, lambda _: gamma_fallback,operand=None,)

    gamma_guess = lax.cond(zeta != zero, cubic_guess, noncubic_guess, operand=None,)

    c2 = -2.0 * zeta
    c3 = 2.0 * eta * signb * sqb
    c4 = -sqb * h * beta0
    c5 = 2.0 * eta * signb * sqb

    def sincos_half_gamma(gamma):
        xx = 0.5 * gamma

        return lax.cond(beta0 > zero,
            lambda z: (jnp.sin(z), jnp.cos(z),),
            lambda z: (jnp.sinh(z), jnp.exp(-z) + jnp.sinh(z),),
            xx,)

    initial_carry = (gamma_guess, 2.0 * gamma_guess, 3.0 * gamma_guess, jnp.asarray(0, dtype=jnp.int32),jnp.asarray(True),)

    def condition(carry):
        _, _, _, iteration, keep_going = carry

        return (keep_going & (iteration < 20))

    def body(carry):
        gamma, gamma1, _, iteration, _ = carry

        new_gamma2 = gamma1
        new_gamma1 = gamma

        sx, cx = sincos_half_gamma(gamma)
        
        numerator = (k * gamma + c2 * sx * cx + c3 * sx**2 + c4)
        denominator = (2.0 * signb * zeta * sx**2 + c5 * sx * cx  + r0 * beta0)

        new_gamma = gamma - numerator / denominator
        new_iteration = iteration + 1
        keep_going = ((new_iteration < ITMAX_GAMMA) & (new_gamma != new_gamma2) & (new_gamma != new_gamma1))

        return (new_gamma, new_gamma1, new_gamma2, new_iteration, keep_going,)

    gamma, _, _, _, _ = lax.while_loop(condition, body, initial_carry,)
    sx, cx = sincos_half_gamma(gamma)

    g1bs = 2.0 * sx * cx / sqb
    g2bs = 2.0 * signb * sx**2 * beta0inv
    g0bs = 1.0 - beta0 * g2bs
    g3bs = G3(gamma, beta0, sqb)

    r = (r0 * g0bs + eta * g1bs + k * g2bs)
    rinv = 1.0 / r
    dfdt = ( -k * g1bs * rinv * r0inv)

    if drift_first:
        fm1 = -k * r0inv * g2bs
        gmh = (k * r0inv * (h * g2bs - r0 * g3bs))
        dgdtm1 = (k * r0inv * rinv * (h * g1bs - r0 * g2bs))

    else: #kepler first
        h1 = H1(gamma, beta0)
        h2 = H2(gamma, beta0, sqb)
        fm1 = (k * rinv * (g2bs - k * r0inv * h1))
        gmh = (k * rinv * (r0 * h2 + eta * h1))
        dgdtm1 = (-k * rinv * g2bs)

    delx = fm1 * x0 + gmh * v0
    delv = dfdt * x0 + dgdtm1 * v0

    return jnp.concatenate((delx, delv), axis=0,)

def drift(x: Array,v: Array,xerror: Array, h: Array,) -> tuple[Array, Array]:
    """
    Drift all bodies using compensated summation.
    Shapes
    ------
    x, v, xerror (3, n)
    """
    return comp_sum(x, xerror, h * v,)


def kickfast(x, v, verror, h, m, pair, n: int,):
    """
    Apply fast kicks to all pair[i, j] == True pairs.

    n is a Python int and `pair` is now static metadata (see State.pair),
    so this loop is unrolled in Python at trace time: `pair[i][j]` is a
    plain Python bool, so an inactive pair contributes no ops to the
    compiled graph at all (rather than a `lax.cond` that still has to be
    dispatched every step), and i, j are static ints so `x[:, i]` compiles
    to a static slice instead of a dynamic gather.
    """

    for i in range(n - 1):
        for j in range(i + 1, n):
            if not pair[i][j]:
                continue

            rij = x[:, i] - x[:, j]
            r2 = jnp.dot(rij, rij)
            r2inv = 1.0 / r2
            r3inv = r2inv * jnp.sqrt(r2inv)
            fac = h * GNEWT * r3inv * rij

            vi, vi_error = comp_sum(v[:, i], verror[:, i], -m[j] * fac,)
            vj, vj_error = comp_sum(v[:, j], verror[:, j], m[i] * fac,)

            v = v.at[:, i].set(vi).at[:, j].set(vj)
            verror = verror.at[:, i].set(vi_error).at[:, j].set(vj_error)

    return v, verror


def phic(x, v, verror, h, m, pair, n: int,):
    dtype = x.dtype

    a = jnp.zeros_like(x)
    # First pass: 2h/3 kick and acceleration.
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not pair[i][j]:
                continue

            rij = x[:, i] - x[:, j]
            r2 = jnp.dot(rij, rij)

            r2inv = 1.0 / r2
            r3inv = r2inv * jnp.sqrt(r2inv)

            fac = GNEWT * rij * r3inv
            facv = fac * (2.0 * h / 3.0)

            vi, vi_error = comp_sum(v[:, i], verror[:, i], -m[j] * facv,)
            vj, vj_error = comp_sum(v[:, j], verror[:, j], m[i] * facv,)

            v = v.at[:, i].set(vi).at[:, j].set(vj)
            verror = verror.at[:, i].set(vi_error).at[:, j].set(vj_error)

            a = a.at[:, i].add(-m[j] * fac).at[:, j].add(m[i] * fac)

    coeff = h**3 * GNEWT / 36.0

    # Second pass: high-order correction.
    for i in range(n - 1):
        for j in range(i + 1, n):
            if not pair[i][j]:
                continue

            aij = a[:, i] - a[:, j]
            rij = x[:, i] - x[:, j]

            r2 = jnp.dot(rij, rij)
            r1 = jnp.sqrt(r2)
            ardot = jnp.dot(aij, rij)

            fac1 = coeff / (r2 * r2 * r1)
            fac = fac1 * (rij * (3.0 * ardot) - r2 * aij)

            vi, vi_error = comp_sum(v[:, i], verror[:, i], m[j] * fac,)
            vj, vj_error = comp_sum(v[:, j], verror[:, j], -m[i] * fac,)

            v = v.at[:, i].set(vi).at[:, j].set(vj)
            verror = verror.at[:, i].set(vi_error).at[:, j].set(vj_error)

    return v, verror

def phisalpha(x, v, verror, h, alpha, m, pair, n: int,):
    dtype = x.dtype

    coeff = (alpha * h**3 * GNEWT / 48.0)
    a = jnp.zeros_like(x)

    # Kepler-pair acceleration (all pairs NOT flagged as fast-kick pairs).
    for i in range(n - 1):
        for j in range(i + 1, n):
            if pair[i][j]:
                continue

            rij = x[:, i] - x[:, j]
            r2 = jnp.dot(rij, rij)

            r3 = r2 * jnp.sqrt(r2)
            fac = GNEWT * rij / r3

            a = a.at[:, i].add(-m[j] * fac).at[:, j].add(m[i] * fac)

    # Fourth-order correction.
    for i in range(n - 1):
        for j in range(i + 1, n):
            if pair[i][j]:
                continue

            aij = a[:, i] - a[:, j]
            rij = x[:, i] - x[:, j]

            r2 = jnp.dot(rij, rij)

            r1 = jnp.sqrt(r2)
            ardot = jnp.dot(aij, rij)

            fac1 = coeff / (r2 * r2 * r1)
            fac2 = (2.0 * GNEWT * (m[i] + m[j]) / r1 + 3.0 * ardot)
            fac = fac1 * (rij * fac2 - r2 * aij)

            vi, vi_error = comp_sum(v[:, i], verror[:, i], m[j] * fac,)
            vj, vj_error = comp_sum(v[:, j], verror[:, j], -m[i] * fac,)

            v = v.at[:, i].set(vi).at[:, j].set(vj)
            verror = verror.at[:, i].set(vi_error).at[:, j].set(vj_error)

    return v, verror


def _kepler_driftij_gamma(x, v, xerror, verror, m, i: int, j: int, h, *, drift_first: bool,):
    x0 = x[:, i] - x[:, j]
    v0 = v[:, i] - v[:, j]

    total_mass = m[i] + m[j]
    gm = GNEWT * total_mass
    eps = jnp.finfo(x.dtype).eps

    def apply_kepler(carry):
        x_current, v_current, xerror_current, verror_current = carry
        delxv = _jac_delxv_gamma_no_grad(x0, v0, gm, h, drift_first=drift_first,)

        mass_inverse = 1.0 / total_mass

        mi = m[i] * mass_inverse
        mj = m[j] * mass_inverse

        xi, xi_error = comp_sum(x_current[:, i], xerror_current[:, i], mj * delxv[:3],)
        xj, xj_error = comp_sum(x_current[:, j], xerror_current[:, j], -mi * delxv[:3],)

        vi, vi_error = comp_sum(v_current[:, i], verror_current[:, i], mj * delxv[3:],)
        vj, vj_error = comp_sum(v_current[:, j],verror_current[:, j], -mi * delxv[3:],)

        x_current = (x_current.at[:, i].set(xi).at[:, j].set(xj))
        xerror_current = (xerror_current.at[:, i].set(xi_error).at[:, j].set(xj_error))

        v_current = (v_current.at[:, i].set(vi).at[:, j].set(vj))
        verror_current = (verror_current.at[:, i].set(vi_error).at[:, j].set(vj_error))

        return (x_current, v_current, xerror_current, verror_current,)

    return lax.cond(jnp.abs(gm) > eps, apply_kepler, lambda carry: carry, (x, v, xerror, verror), )


def _pair_indices(n: int):
    """
    Static (i, j) index pairs with i < j, in the same order as
    `for i in range(n - 1): for j in range(i + 1, n):`.

    n is a Python int (static), so this is computed once at trace time.
    Consumed by `lax.scan` so that the Kepler-solve pairs (which each run
    a `lax.while_loop` root-find) compile to a single shared loop body
    regardless of how many bodies/pairs there are, instead of one
    physically unrolled copy of the solver per pair -- unrolling here is
    what made compilation blow up for systems like TRAPPIST-1 (28 pairs).
    """
    ii, jj = np.triu_indices(n, k=1)
    return jnp.asarray(ii, dtype=jnp.int32), jnp.asarray(jj, dtype=jnp.int32)


def _pair_indices_reverse(n: int):
    """
    Static (i, j) index pairs, in the same order as
    `for i in range(n - 2, -1, -1): for j in range(n - 1, i, -1):`
    (used by `kepler_drift`, which walks pairs in reverse order).
    """
    pairs = [(i, j) for i in range(n - 2, -1, -1) for j in range(n - 1, i, -1)]

    if not pairs:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty

    ii, jj = zip(*pairs)
    return jnp.asarray(ii, dtype=jnp.int32), jnp.asarray(jj, dtype=jnp.int32)


def drift_kepler(x, v, xerror, verror, h, m, pair, n: int,):
    """`pair` must be a JAX bool array here (dynamically indexed by scan)."""
    ii, jj = _pair_indices(n)

    def pair_step(carry, idx):
        i, j = idx
        new_carry = lax.cond(
            jnp.logical_not(pair[i, j]),
            lambda current: _kepler_driftij_gamma(*current, m, i, j, h, drift_first=True,),
            lambda current: current,
            carry,)
        return new_carry, None

    carry, _ = lax.scan(pair_step, (x, v, xerror, verror), (ii, jj),)

    return carry


def kepler_drift(x, v, xerror, verror, h, m, pair, n: int,):
    """`pair` must be a JAX bool array here (dynamically indexed by scan)."""
    ii, jj = _pair_indices_reverse(n)

    def pair_step(carry, idx):
        i, j = idx
        new_carry = lax.cond(
            jnp.logical_not(pair[i, j]),
            lambda current: _kepler_driftij_gamma(*current, m, i, j, h, drift_first=False,),
            lambda current: current,
            carry,
        )
        return new_carry, None

    carry, _ = lax.scan(pair_step, (x, v, xerror, verror), (ii, jj),)

    return carry
