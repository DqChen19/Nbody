from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import lax

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
    tmp7n = d.jac_kick @ s.dqdt
    s = replace(s, dqdt=s.dqdt + d.dqdt_kick + tmp7n)
    jac_copy = d.jac_kick @ s.jac_step
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)

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

    def _forward_body(carry, idx):
        i, j = idx
        s_c, d_c = carry
        s_c, d_c = _kepler_step(s_c, d_c, i, j, i * 7, j * 7, sevn, h2, True)
        return (s_c, d_c), None

    (s, d), _ = lax.scan(_forward_body, (s, d), (ii, jj))

    # 4. Kick-pair correction
    q0 = _pack_q(s.x, s.v, s.m, n)   # starting state at this step

    verror0 = jnp.zeros_like(s.v)
    v1, verror1 = phic(s.x, s.v, verror0, h, s.m, s.pair, n)
    v2, verror2 = phisalpha(s.x, v1, verror1, h, two, s.m, s.pair, n)
    s = replace(s, v=v2, verror=verror2)
    
    full_jac = jax.jacfwd(lambda qq: _phi_combined_physics(qq, h, two, s.pair, n))(q0)
    jac_phi  = full_jac - jnp.eye(sevn, dtype=s.x.dtype)
    dqdt_phi = jax.jacfwd(lambda hh: _phi_combined_physics(q0, hh, two, s.pair, n))(h)

    jac_copy = jac_phi @ s.jac_step
    tmp7n    = jac_phi @ s.dqdt
    s = replace(s, dqdt=s.dqdt + dqdt_phi + tmp7n)
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)
    d = replace(d, jac_phi=jac_phi, dqdt_phi=dqdt_phi)

    #5. backward Kepler
    ii_b, jj_b = _solve_pairs_backward(s.pair, n)

    def _backward_body(carry, idx):
        i, j = idx
        s_c, d_c = carry
        s_c, d_c = _kepler_step(s_c, d_c, i, j, i * 7, j * 7, sevn, h2, False)
        return (s_c, d_c), None

    (s, d), _ = lax.scan(_backward_body, (s, d), (ii_b, jj_b))

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
    tmp7n = d.jac_kick @ s.dqdt
    s = replace(s, dqdt=s.dqdt + d.dqdt_kick + tmp7n)
    jac_copy = d.jac_kick @ s.jac_step
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)

    return s, d

def _solve_pairs_forward(pair, n: int):
    """
    static:
    真正需要做 Kepler 求解的 (i, j) 对,顺序与原来的
    `for i in range(n-1): for j in range(i+1, n): if not pair[i][j]` 完全一致。
    pair -> s.pair —— 静态 tuple,trace 时就已知内容。
    """
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

# ─────────────────────────────────────────────────────────────────────────────
# jac_phi/dqdt_phi
# ─────────────────────────────────────────────────────────────────────────────
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

def _phi_combined_physics(q: Array, h: Array, alpha: Array, pair: tuple, n: int) -> Array:
    """
    phic() then phisalpha()
    For jax.jacfwd
    """
    x, v, m = _unpack_q(q, n)
    verror0 = jnp.zeros_like(v)
    v1, _ = phic(x, v, verror0, h, m, pair, n)
    v2, _ = phisalpha(x, v1, verror0, h, alpha, m, pair, n)
    return _pack_q(x, v2, m, n)

# ─────────────────────────────────────────────────────────────────────────────
# Kepler step with Jacobian for one pair (i, j) — wrapped in gm guard
# ─────────────────────────────────────────────────────────────────────────────
def _pair_update_physics(q: Array, h: Array, drift_first: bool) -> Array:
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

    return jnp.concatenate([xi_new, vi_new, mi[None], xj_new, vj_new, mj[None]])

def _kepler_step(s: State, d: Derivatives, i: int, j: int, indi, indj, sevn: int, h: Array, drift_first: bool):
    x0 = s.x[:, i] - s.x[:, j]
    v0 = s.v[:, i] - s.v[:, j]
    gm = GNEWT * (s.m[i] + s.m[j])

    dtype = s.x.dtype
    delxv = _jac_delxv_gamma_no_grad(x0, v0, gm, h, drift_first=drift_first)

    mijinv = 1.0 / (s.m[i] + s.m[j])
    mi = s.m[i] * mijinv
    mj = s.m[j] * mijinv

    xi,  xei  = comp_sum(s.x[:, i], s.xerror[:, i],  mj * delxv[:3])
    xj_, xej  = comp_sum(s.x[:, j], s.xerror[:, j], -mi * delxv[:3])
    vi,  vei  = comp_sum(s.v[:, i], s.verror[:, i],  mj * delxv[3:])
    vj_, vej  = comp_sum(s.v[:, j], s.verror[:, j], -mi * delxv[3:])
    x_new  = s.x.at[:, i].set(xi ).at[:, j].set(xj_)
    xe_new = s.xerror.at[:, i].set(xei).at[:, j].set(xej)
    v_new  = s.v.at[:, i].set(vi ).at[:, j].set(vj_)
    ve_new = s.verror.at[:, i].set(vei).at[:, j].set(vej)

    # ---- jax.jacfwd to get derivatives of (q, h) ----
    mi_row = s.m[i][None]
    mj_row = s.m[j][None]
    q = jnp.concatenate([s.x[:, i], s.v[:, i], mi_row, s.x[:, j], s.v[:, j], mj_row])

    jac_wrt_q, jac_wrt_h = jax.jacfwd(_pair_update_physics, argnums=(0, 1))(q, h, drift_first)

    jac_ij  = jac_wrt_q - jnp.eye(14, dtype=dtype)
    dqdt_ij = jac_wrt_h

    # indi = i*7, indj = j*7
    # dynamic_slice_in_dim(x, start_index=, slice_size=, axis=)
    jac_step_i = lax.dynamic_slice_in_dim(s.jac_step,  indi, 7, axis=0)   
    jac_step_j = lax.dynamic_slice_in_dim(s.jac_step,  indj, 7, axis=0)
    jac_err_i  = lax.dynamic_slice_in_dim(s.jac_error, indi, 7, axis=0)
    jac_err_j  = lax.dynamic_slice_in_dim(s.jac_error, indj, 7, axis=0)
    dqdt_i     = lax.dynamic_slice_in_dim(s.dqdt,      indi, 7, axis=0)
    dqdt_j     = lax.dynamic_slice_in_dim(s.dqdt,      indj, 7, axis=0)

    jac_tmp1  = d.jac_tmp1.at[0:7, :sevn].set(jac_step_i).at[7:14, :sevn].set(jac_step_j)
    jac_err1  = d.jac_err1.at[0:7, :sevn].set(jac_err_i ).at[7:14, :sevn].set(jac_err_j)
    dqdt_tmp1 = d.dqdt_tmp1.at[0:7].set(dqdt_i).at[7:14].set(dqdt_j)

    jac_tmp2 = jac_ij @ jac_tmp1
    jac_tmp1, jac_err1 = comp_sum_matrix(jac_tmp1, jac_err1, jac_tmp2)
    dqdt_ij_new = dqdt_ij * 0.5 + dqdt_tmp1 + jac_ij @ dqdt_tmp1

    jac_step  = lax.dynamic_update_slice_in_dim(s.jac_step,  jac_tmp1[0:7,  :sevn], indi, axis=0)
    jac_step  = lax.dynamic_update_slice_in_dim(jac_step,    jac_tmp1[7:14, :sevn], indj, axis=0)
    jac_error = lax.dynamic_update_slice_in_dim(s.jac_error, jac_err1[0:7,  :sevn], indi, axis=0)
    jac_error = lax.dynamic_update_slice_in_dim(jac_error,   jac_err1[7:14, :sevn], indj, axis=0)
    dqdt      = lax.dynamic_update_slice_in_dim(s.dqdt,      dqdt_ij_new[0:7],      indi, axis=0)
    dqdt      = lax.dynamic_update_slice_in_dim(dqdt,        dqdt_ij_new[7:14],     indj, axis=0)

    s_out = replace(s, x=x_new, xerror=xe_new, v=v_new, verror=ve_new,
                    jac_step=jac_step, jac_error=jac_error, dqdt=dqdt)
    d_out = replace(d, jac_ij=jac_ij, dqdt_ij=dqdt_ij_new,
                    jac_tmp1=jac_tmp1, jac_err1=jac_err1, dqdt_tmp1=dqdt_tmp1)

    return s_out, d_out

