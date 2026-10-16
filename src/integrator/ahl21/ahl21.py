from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

from ...utils import comp_sum, comp_sum_matrix
from ...pre_alloc_arrays import Derivatives
from ..State import State
from .ahl21_no_grad import _jac_delxv_gamma_no_grad, phic, phisalpha
Array = jax.Array
YEAR  = 365.242
GNEWT = 39.4845 / (YEAR * YEAR)


def ahl21(state: State, d: Derivatives, h: Array):
    """
    One AHL21 step with full Jacobian and dq/dt tracking.

    Returns
    -------
    (state, d)
    """
    h    = jnp.asarray(h, dtype=state.x.dtype)
    n    = state.n
    sevn = 7 * n
    dtype = state.x.dtype
    h2   = h * 0.5
    h6   = h / 6.0
    two  = 2.0

    d = d.zero_out()
    s = replace(state, dqdt=jnp.zeros(sevn, dtype=dtype))

    # # 1. Fast kick(h/6)
    s, d = _kickfast_grad(s, d, h6)
    d = replace(d, dqdt_kick=d.dqdt_kick / jnp.asarray(6.0, dtype=dtype))
    s = _apply_kick_jacobian(s, d)

    # 2. drift (h/2)
    # dqdt update for drift (first occurrence, set)
    dqdt = s.dqdt
    x, xerror = comp_sum(s.x, s.xerror, h2 * s.v)
    jac_step  = s.jac_step
    jac_error = s.jac_error

    pos_idx = jnp.array([7*i+k for i in range(s.n) for k in range(3)])
    vel_idx = pos_idx + 3

    new_sum, new_err = comp_sum_matrix(jac_step[pos_idx, :], jac_error[pos_idx, :], h2 * jac_step[vel_idx, :])
    jac_step  = jac_step.at[pos_idx, :].set(new_sum)
    jac_error = jac_error.at[pos_idx, :].set(new_err)
   
    dqdt = dqdt.at[pos_idx].set(0.5 * s.v.T.reshape(-1) + h2 * dqdt[vel_idx])
    s = replace(s,x=x, xerror=xerror, jac_step=jac_step, jac_error=jac_error, dqdt=dqdt)

    # 3. forward Kepler loop
    ii, jj = _solve_pairs_forward(s.pair, n)

    s = _kepler_pass(s, ii, jj, h2, drift_first=True)

    # 4. Kick-pair correction
    q0 = _pack_q(s.x, s.v, s.m, n)   # starting state at this step

    verror0 = jnp.zeros_like(s.v)
    v1, verror1 = phic(s.x, s.v, verror0, h, s.m, s.pair, n)
    v2, verror2 = phisalpha(s.x, v1, verror1, h, two, s.m, s.pair, n)
    s = replace(s, v=v2, verror=verror2)
    
    # phic + phisalpha only add a kick dv(x, m, h) to v, so jac_phi is
    # nonzero only in the velocity rows / position+mass columns.
    jac_phi, dqdt_phi = _phi_jacobian(q0, h, two, s.pair, n)

    jac_copy = jac_phi @ s.jac_step
    tmp7n    = jac_phi @ s.dqdt
    s = replace(s, dqdt=s.dqdt + dqdt_phi + tmp7n)
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)
    d = replace(d, jac_phi=jac_phi, dqdt_phi=dqdt_phi)

    #5. backward Kepler
    ii_b, jj_b = _solve_pairs_backward(s.pair, n)

    s = _kepler_pass(s, ii_b, jj_b, h2, drift_first=False)

    #6. drift_grad(h/2)
    x, xerror = comp_sum(s.x, s.xerror, h2 * s.v)
    jac_step  = s.jac_step
    jac_error = s.jac_error

    pos_idx = jnp.array([7*i+k for i in range(s.n) for k in range(3)])
    vel_idx = pos_idx + 3

    new_sum, new_err = comp_sum_matrix(jac_step[pos_idx, :], jac_error[pos_idx, :], h2 * jac_step[vel_idx, :])
    jac_step  = jac_step.at[pos_idx, :].set(new_sum)
    jac_error = jac_error.at[pos_idx, :].set(new_err)
    
    dqdt = s.dqdt
    dqdt = dqdt.at[pos_idx].add(0.5 * s.v.T.reshape(-1) + h2 * dqdt[vel_idx])
    s = replace(s, x=x, xerror=xerror, jac_step=jac_step, jac_error=jac_error, dqdt=dqdt)

    #7. final kickfast(h/6)
    d = replace(d, dqdt_kick=jnp.zeros(sevn, dtype=dtype))
    s, d = _kickfast_grad(s, d, h6)
    d = replace(d, dqdt_kick=d.dqdt_kick / jnp.asarray(6.0, dtype=dtype))
    s = _apply_kick_jacobian(s, d)

    return s, d

def _apply_kick_jacobian(s: State, d: Derivatives) -> State:
    """
    Fold the fast-kick Jacobian into dqdt and jac_step.

    pair is static: with no fast-kick pairs jac_kick is all zeros, so the two
    (7n x 7n) matmuls are skipped. The compensated sum still runs (with a
    zero addend) so jac_step/jac_error are bitwise unchanged from before.
    """
    if any(any(row) for row in s.pair):
        tmp7n = d.jac_kick @ s.dqdt
        jac_copy = d.jac_kick @ s.jac_step
    else:
        tmp7n = jnp.zeros_like(s.dqdt)
        jac_copy = jnp.zeros_like(s.jac_step)
    s = replace(s, dqdt=s.dqdt + d.dqdt_kick + tmp7n)
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    return replace(s, jac_step=js, jac_error=je)

def _solve_pairs_forward(pair, n: int):
    pairs = [(i, j) for i in range(n - 1) for j in range(i + 1, n) if not pair[i][j]]
    if not pairs:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    ii, jj = zip(*pairs)
    return jnp.asarray(ii, dtype=jnp.int32), jnp.asarray(jj, dtype=jnp.int32)


def _solve_pairs_backward(pair, n: int):
    pairs = [(i, j) for i in range(n - 2, -1, -1) for j in range(n - 1, i, -1) if not pair[i][j]]
    if not pairs:
        empty = jnp.zeros((0,), dtype=jnp.int32)
        return empty, empty
    ii, jj = zip(*pairs)
    return jnp.asarray(ii, dtype=jnp.int32), jnp.asarray(jj, dtype=jnp.int32)

# Fast kick with Jacobian
def _kickfast_grad(s: State, d: Derivatives, h: Array):
    n     = s.n
    dtype = s.x.dtype
    eps   = jnp.finfo(dtype).eps
    eye3  = jnp.eye(3, dtype=dtype)

    v         = s.v
    verror    = s.verror
    jac_kick  = d.jac_kick
    dqdt_kick = d.dqdt_kick

    for i in range(n - 1):
        indi = i * 7
        for j in range(i + 1, n):
            indj = j * 7

            def _do_kick(carry, _i=i, _j=j, _indi=indi, _indj=indj):
                v_c, ve_c, jk_c, dqk_c = carry
                rij   = s.x[:, _i] - s.x[:, _j]
                r2    = jnp.maximum(jnp.dot(rij, rij), eps)
                r2inv = 1.0 / r2
                r3inv = r2inv * jnp.sqrt(r2inv)
                fac2  = h * GNEWT * r3inv
                fv    = fac2 * rij

                vi,  vei  = comp_sum(v_c[:, _i], ve_c[:, _i], -s.m[_j] * fv)
                vj_, vej  = comp_sum(v_c[:, _j], ve_c[:, _j],  s.m[_i] * fv)
                v_c  = v_c .at[:, _i].set(vi ).at[:, _j].set(vj_)
                ve_c = ve_c.at[:, _i].set(vei).at[:, _j].set(vej)

                dqk_c = (dqk_c
                    .at[_indi + 3:_indi + 6].add(-s.m[_j] * fv / h)
                    .at[_indj + 3:_indj + 6].add( s.m[_i] * fv / h))

                jk_c = (jk_c
                    .at[_indi + 3:_indi + 6, _indj + 6].add(-fv)
                    .at[_indj + 3:_indj + 6, _indi + 6].add( fv))

                fv3 = fv * 3.0 * r2inv
                outer_ij = jnp.outer(fv3, rij)
                jk_c = (jk_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( s.m[_j] * outer_ij)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-s.m[_j] * outer_ij)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( s.m[_i] * outer_ij)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-s.m[_i] * outer_ij))

                jk_c = (jk_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add(-fac2 * s.m[_j] * eye3)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add( fac2 * s.m[_j] * eye3)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add(-fac2 * s.m[_i] * eye3)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add( fac2 * s.m[_i] * eye3))

                return v_c, ve_c, jk_c, dqk_c

            if s.pair[i][j]:
                v, verror, jac_kick, dqdt_kick = _do_kick((v, verror, jac_kick, dqdt_kick))

    s = replace(s, v=v, verror=verror)
    d = replace(d, jac_kick=jac_kick, dqdt_kick=dqdt_kick)
    return s, d

# ====================================
# jac_phi/dqdt_phi
# ====================================
def _pack_q(x: Array, v: Array, m: Array, n: int) -> Array:
    """Pack (x, v, m) to 7n vector, as jac_step"""
    blocks = []
    for i in range(n):
        blocks.append(x[:, i])
        blocks.append(v[:, i])
        blocks.append(m[i:i + 1])
    return jnp.concatenate(blocks)

def _unpack_q(q: Array, n: int):
    dtype = q.dtype
    x = jnp.zeros((3, n), dtype=dtype)
    v = jnp.zeros((3, n), dtype=dtype)
    m = jnp.zeros((n,), dtype=dtype)
    for i in range(n):
        indi = i * 7
        x = x.at[:, i].set(q[indi:indi + 3])
        v = v.at[:, i].set(q[indi + 3:indi + 6])
        m = m.at[i].set(q[indi + 6])
    return x, v, m

def _phi_kick(x: Array, m: Array, h: Array, alpha, pair: tuple, n: int) -> Array:
    """
    Velocity kick dv of phic() followed by phisalpha(), vectorized over all
    pairs and without compensated summation. Used only for the derivatives
    in _phi_jacobian (the derivative of a compensated sum is the plain sum);
    the state itself is still advanced by phic/phisalpha.

    phisalpha reads only x, and phic's second pass only the first pass's
    accelerations, so the kick does not depend on v.
    """
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    fast = np.array(pair, dtype=bool) & upper
    fast = fast | fast.T                     # pair[i][j] == True: phic pairs
    kepler = (upper & ~np.array(pair, dtype=bool))
    kepler = kepler | kepler.T               # pair[i][j] == False: phisalpha pairs

    rij = x[:, :, None] - x[:, None, :]      # rij[:, i, j] = x_i - x_j
    any_pair = jnp.asarray(fast | kepler)
    r2 = jnp.where(any_pair, jnp.sum(rij * rij, axis=0), 1.0)
    r1 = jnp.sqrt(r2)
    r3inv = 1.0 / (r2 * r1)

    def accel(mask):
        # a_i = -sum_j G m_j r_ij / r_ij^3 over the masked pairs.
        w = jnp.where(jnp.asarray(mask), m[None, :] * r3inv, 0.0)
        return -GNEWT * jnp.einsum("kij,ij->ki", rij, w)

    def pair_sum(mask, fac):
        # dv_i = sum_j m_j fac_ij over the masked pairs (fac_ij is antisymmetric).
        w = jnp.where(jnp.asarray(mask), m[None, :], 0.0)
        return jnp.einsum("kij,ij->ki", fac, w)

    def correction(a, coeff, extra):
        aij = a[:, :, None] - a[:, None, :]
        ardot = jnp.sum(aij * rij, axis=0)
        fac1 = coeff / (r2 * r2 * r1)
        return fac1 * (rij * (extra + 3.0 * ardot) - r2 * aij)

    dv = jnp.zeros_like(x)
    if fast.any():
        a_fast = accel(fast)
        dv = dv + (2.0 * h / 3.0) * a_fast
        dv = dv + pair_sum(fast, correction(a_fast, h**3 * GNEWT / 36.0, 0.0))
    if kepler.any():
        a_kep = accel(kepler)
        extra = 2.0 * GNEWT * (m[:, None] + m[None, :]) / r1
        dv = dv + pair_sum(kepler, correction(a_kep, alpha * h**3 * GNEWT / 48.0, extra))
    return dv

def _phi_jacobian(q: Array, h: Array, alpha, pair: tuple, n: int):
    """
    jac_phi = d(phisalpha o phic)(q)/dq - I and dqdt_phi = d/dh, in the
    (7n,) [x, v, m]-per-body layout. Only the velocity rows of
    jac_phi are nonzero, in the position and mass columns.
    """
    x, _, m = _unpack_q(q, n)
    dtype = q.dtype
    jx, jm, jh = jax.jacfwd(lambda xx, mm, hh: _phi_kick(xx, mm, hh, alpha, pair, n), argnums=(0, 1, 2))(x, m, h)
    # jx: (3, n, 3, n) = d dv[k, i] / d x[l, j];  jm: (3, n, n);  jh: (3, n)
    jac = jnp.zeros((n, 7, n, 7), dtype=dtype)
    jac = jac.at[:, 3:6, :, 0:3].set(jnp.transpose(jx, (1, 0, 3, 2)))
    jac = jac.at[:, 3:6, :, 6].set(jnp.transpose(jm, (1, 0, 2)))
    dqdt = jnp.zeros((n, 7), dtype=dtype).at[:, 3:6].set(jh.T)
    return jac.reshape(7 * n, 7 * n), dqdt.reshape(7 * n)

# ====================================
# Kepler step with Jacobian for one pair (i, j) 
# ====================================
def _pair_update_physics(q: Array, h: Array, drift_first: bool):
    """
    return q
    q = (xi, vi, mi, xj, vj, mj), dim = 14
    
    Design for jax.jacforward
    """
    xi, vi, mi, xj, vj, mj = q[0:3], q[3:6], q[6], q[7:10], q[10:13], q[13]
    x0 = xi - xj
    v0 = vi - vj
    gm = GNEWT * (mi + mj)
    delxv = _jac_delxv_gamma_no_grad(x0, v0, gm, h, drift_first=drift_first)

    mass_inverse = 1.0 / (mi + mj)
    mi_frac = mi * mass_inverse
    mj_frac = mj * mass_inverse

    xi_new = xi + mj_frac * delxv[:3]
    xj_new = xj - mi_frac * delxv[:3]
    vi_new = vi + mj_frac * delxv[3:]
    vj_new = vj - mi_frac * delxv[3:]

    # delxv is returned as jacfwd's aux output, so the caller reuses this
    # primal Kepler solve instead of running it a second time.
    return jnp.concatenate([xi_new, vi_new, mi[None], xj_new, vj_new, mj[None]]), delxv

def _kepler_delxv_jacobian(x0: Array, v0: Array, k: Array, h: Array, drift_first: bool):
    """
    delxv = _jac_delxv_gamma_no_grad(x0, v0, k, h) and its (6, 8) Jacobian
    w.r.t. p = (x0, v0, k, h).

    The Kepler solve depends on the pair only through these 8 numbers, so
    forward mode carries 8 tangents through it instead of the 15 of
    (xi, vi, mi, xj, vj, mj, h); _pair_jacobian applies the chain rule.
    """
    def f(p):
        out = _jac_delxv_gamma_no_grad(p[0:3], p[3:6], p[6], p[7], drift_first=drift_first)
        return out, out

    p = jnp.concatenate([x0, v0, k[None], h[None]])
    jac, delxv = jax.jacfwd(f, has_aux=True)(p)
    return delxv, jac

def _pair_jacobian(delxv: Array, jac_delxv: Array, mi: Array, mj: Array, mijinv: Array):
    """
    Chain rule from d(delxv)/d(x0, v0, k, h) to the pair update of
    q = (xi, vi, mi, xj, vj, mj), where
        xi' = xi + mj/M delx,  xj' = xj - mi/M delx   (same for v),
        x0 = xi - xj,  v0 = vi - vj,  k = G (mi + mj),  M = mi + mj.

    mi, mj are the mass fractions m_i/M, m_j/M. Returns (jac_ij, dqdt_ij):
    jac_ij = d(q')/dq - I (14, 14) and dqdt_ij = d(q')/dh (14,), matching
    jax.jacfwd of _pair_update_physics.
    """
    dtype = delxv.dtype
    jxv, jk, jh = jac_delxv[:, 0:6], jac_delxv[:, 6], jac_delxv[:, 7]

    # d(delxv)/dq, (6, 14)
    jkm = GNEWT * jk[:, None]
    ddel = jnp.concatenate([jxv, jkm, -jxv, jkm], axis=1)

    # d(mj/M)/dq and d(mi/M)/dq: nonzero only in the mass columns 6 and 13
    zero6 = jnp.zeros(6, dtype=dtype)
    dmj = jnp.concatenate([zero6, (-mj * mijinv)[None], zero6, (mi * mijinv)[None]])
    dmi = -dmj

    drow_i =  mj * ddel + jnp.outer(delxv, dmj)
    drow_j = -mi * ddel - jnp.outer(delxv, dmi)
    zrow = jnp.zeros((1, 14), dtype=dtype)
    jac_ij = jnp.concatenate([drow_i, zrow, drow_j, zrow], axis=0)

    zero1 = jnp.zeros(1, dtype=dtype)
    dqdt_ij = jnp.concatenate([mj * jh, zero1, -mi * jh, zero1])
    return jac_ij, dqdt_ij

def _kepler_step(carry, m: Array, i, j, h: Array, drift_first: bool):
    """
    Kepler drift for one pair (i, j), with Jacobian and dq/dt updates.

    carry = (x, v, xerror, verror, jac_step, jac_error, dqdt): only the
    arrays a pair update touches, so lax.scan does not thread the whole
    State/Derivatives through every pair.
    """
    x, v, xerror, verror, jac_step, jac_error, dqdt = carry
    sevn = jac_step.shape[0]
    indi, indj = 7 * i, 7 * j
    dtype = x.dtype

    # ---- Kepler solve and its Jacobian w.r.t. its 8 actual inputs ----
    delxv, jac_delxv = _kepler_delxv_jacobian(x[:, i] - x[:, j], v[:, i] - v[:, j],
                                              GNEWT * (m[i] + m[j]), h, drift_first)

    mijinv = 1.0 / (m[i] + m[j])
    mi = m[i] * mijinv
    mj = m[j] * mijinv
    jac_ij, dqdt_ij = _pair_jacobian(delxv, jac_delxv, mi, mj, mijinv)

    xi,  xei  = comp_sum(x[:, i], xerror[:, i],  mj * delxv[:3])
    xj_, xej  = comp_sum(x[:, j], xerror[:, j], -mi * delxv[:3])
    vi,  vei  = comp_sum(v[:, i], verror[:, i],  mj * delxv[3:])
    vj_, vej  = comp_sum(v[:, j], verror[:, j], -mi * delxv[3:])
    x      = x.at[:, i].set(xi ).at[:, j].set(xj_)
    xerror = xerror.at[:, i].set(xei).at[:, j].set(xej)
    v      = v.at[:, i].set(vi ).at[:, j].set(vj_)
    verror = verror.at[:, i].set(vei).at[:, j].set(vej)

    # dynamic_slice_in_dim(x, start_index=, slice_size=, axis=)
    jac_tmp1 = jnp.concatenate([lax.dynamic_slice_in_dim(jac_step,  indi, 7, axis=0),
                                lax.dynamic_slice_in_dim(jac_step,  indj, 7, axis=0)])
    jac_err1 = jnp.concatenate([lax.dynamic_slice_in_dim(jac_error, indi, 7, axis=0),
                                lax.dynamic_slice_in_dim(jac_error, indj, 7, axis=0)])
    dqdt_tmp1 = jnp.concatenate([lax.dynamic_slice_in_dim(dqdt, indi, 7, axis=0),
                                 lax.dynamic_slice_in_dim(dqdt, indj, 7, axis=0)])

    jac_tmp2 = jac_ij @ jac_tmp1
    jac_tmp1, jac_err1 = comp_sum_matrix(jac_tmp1, jac_err1, jac_tmp2)
    dqdt_ij_new = dqdt_ij * 0.5 + dqdt_tmp1 + jac_ij @ dqdt_tmp1

    jac_step  = lax.dynamic_update_slice_in_dim(jac_step,  jac_tmp1[0:7,  :sevn], indi, axis=0)
    jac_step  = lax.dynamic_update_slice_in_dim(jac_step,  jac_tmp1[7:14, :sevn], indj, axis=0)
    jac_error = lax.dynamic_update_slice_in_dim(jac_error, jac_err1[0:7,  :sevn], indi, axis=0)
    jac_error = lax.dynamic_update_slice_in_dim(jac_error, jac_err1[7:14, :sevn], indj, axis=0)
    dqdt      = lax.dynamic_update_slice_in_dim(dqdt,      dqdt_ij_new[0:7],      indi, axis=0)
    dqdt      = lax.dynamic_update_slice_in_dim(dqdt,      dqdt_ij_new[7:14],     indj, axis=0)

    return (x, v, xerror, verror, jac_step, jac_error, dqdt)

def _kepler_pass(s: State, ii: Array, jj: Array, h: Array, *, drift_first: bool) -> State:
    """Apply _kepler_step to the pairs (ii[k], jj[k]) in order."""
    m = s.m

    def body(carry, idx):
        return _kepler_step(carry, m, idx[0], idx[1], h, drift_first), None

    init = (s.x, s.v, s.xerror, s.verror, s.jac_step, s.jac_error, s.dqdt)
    (x, v, xerror, verror, jac_step, jac_error, dqdt), _ = lax.scan(body, init, (ii, jj))
    return replace(s, x=x, v=v, xerror=xerror, verror=verror, jac_step=jac_step, jac_error=jac_error, dqdt=dqdt)

