from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jax import lax

from ...utils import comp_sum, comp_sum_matrix, G3, H1, H2, H3, H5, H6
from ...pre_alloc_arrays import Derivatives
from ..State import State

Array = jax.Array
YEAR  = 365.242
GNEWT = 39.4845 / (YEAR * YEAR)


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point  (matches scheme_grad interface)
# ─────────────────────────────────────────────────────────────────────────────
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
    h2   = h * jnp.asarray(0.5, dtype=dtype)
    h6   = h / jnp.asarray(6.0, dtype=dtype)
    two  = jnp.asarray(2.0, dtype=dtype)

    d = d.zero_out()
    s = replace(state, dqdt=jnp.zeros(sevn, dtype=dtype))

    # ── Step 1: kickfast(h/6) ───────────────────────────────────────────────
    s, d = _kickfast_grad(s, d, h6)
    d = replace(d, dqdt_kick=d.dqdt_kick / jnp.asarray(6.0, dtype=dtype))
    tmp7n = d.jac_kick @ s.dqdt
    s = replace(s, dqdt=s.dqdt + d.dqdt_kick + tmp7n)
    jac_copy = d.jac_kick @ s.jac_step
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)

    # ── Step 2: drift_grad(h/2) ─────────────────────────────────────────────
    s = _drift_grad(s, h2)
    # dqdt update for drift (first occurrence → set, not add)
    dqdt = s.dqdt
    for i in range(n):
        indi = i * 7
        for k in range(3):
            dqdt = dqdt.at[indi + k].set(
                jnp.asarray(0.5, dtype=dtype) * s.v[k, i]
                + h2 * s.dqdt[indi + 3 + k]
            )
    s = replace(s, dqdt=dqdt)

    # ── Step 3: forward Kepler loop ─────────────────────────────────────────
    for i in range(n - 1):
        indi = i * 7
        for j in range(i + 1, n):
            indj = j * 7
            if not s.pair[i][j]:
                s, d = _kepler_step(s, d, i, j, indi, indj, sevn, h2, True)

    # ── Step 4: phic + phisalpha ────────────────────────────────────────────
    s, d = _phic_grad(s, d, h)
    s, d = _phisalpha_grad(s, d, h, two)
    jac_copy = d.jac_phi @ s.jac_step
    tmp7n = d.jac_phi @ s.dqdt
    s = replace(s, dqdt=s.dqdt + d.dqdt_phi + tmp7n)
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)

    # ── Step 5: backward Kepler loop ────────────────────────────────────────
    for i in range(n - 2, -1, -1):
        indi = i * 7
        for j in range(n - 1, i, -1):
            indj = j * 7
            if not s.pair[i][j]:
                s, d = _kepler_step(s, d, i, j, indi, indj, sevn, h2, False)

    # ── Step 6: drift_grad(h/2) ─────────────────────────────────────────────
    s = _drift_grad(s, h2)
    dqdt = s.dqdt
    for i in range(n):
        indi = i * 7
        for k in range(3):
            dqdt = dqdt.at[indi + k].add(
                jnp.asarray(0.5, dtype=dtype) * s.v[k, i]
                + h2 * s.dqdt[indi + 3 + k]
            )
    s = replace(s, dqdt=dqdt)

    # ── Step 7: final kickfast(h/6) ─────────────────────────────────────────
    d = replace(d, dqdt_kick=jnp.zeros(sevn, dtype=dtype))
    s, d = _kickfast_grad(s, d, h6)
    d = replace(d, dqdt_kick=d.dqdt_kick / jnp.asarray(6.0, dtype=dtype))
    tmp7n = d.jac_kick @ s.dqdt
    s = replace(s, dqdt=s.dqdt + d.dqdt_kick + tmp7n)
    jac_copy = d.jac_kick @ s.jac_step
    js, je = comp_sum_matrix(s.jac_step, s.jac_error, jac_copy)
    s = replace(s, jac_step=js, jac_error=je)

    return s, d


# ─────────────────────────────────────────────────────────────────────────────
# Drift with Jacobian
# ─────────────────────────────────────────────────────────────────────────────
def _drift_grad(s: State, h: Array) -> State:
    x, xerror = comp_sum(s.x, s.xerror, h * s.v)
    jac_step  = s.jac_step
    jac_error = s.jac_error
    for i in range(s.n):
        indi = i * 7
        new_sum, new_err = comp_sum_matrix(
            jac_step [indi:indi + 3, :],
            jac_error[indi:indi + 3, :],
            h * jac_step[indi + 3:indi + 6, :],
        )
        jac_step  = jac_step .at[indi:indi + 3, :].set(new_sum)
        jac_error = jac_error.at[indi:indi + 3, :].set(new_err)
    return replace(s, x=x, xerror=xerror, jac_step=jac_step, jac_error=jac_error)


# ─────────────────────────────────────────────────────────────────────────────
# Fast kick with Jacobian
# ─────────────────────────────────────────────────────────────────────────────
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
                v, verror, jac_kick, dqdt_kick = _do_kick(
                    (v, verror, jac_kick, dqdt_kick))

    s = replace(s, v=v, verror=verror)
    d = replace(d, jac_kick=jac_kick, dqdt_kick=dqdt_kick)
    return s, d


# ─────────────────────────────────────────────────────────────────────────────
# Pair correction with Jacobian (phic)
# ─────────────────────────────────────────────────────────────────────────────
def _phic_grad(s: State, d: Derivatives, h: Array):
    n     = s.n
    dtype = s.x.dtype
    eps   = jnp.finfo(dtype).eps
    eye3  = jnp.eye(3, dtype=dtype)
    coeff = h ** 3 * GNEWT / jnp.asarray(36.0, dtype=dtype)

    v        = s.v
    verror   = s.verror
    a        = jnp.zeros((3, n), dtype=dtype)
    dadq     = jnp.zeros((3, n, 4, n), dtype=dtype)
    jac_phi  = jnp.zeros_like(d.jac_phi)
    dqdt_phi = jnp.zeros_like(d.dqdt_phi)

    # ── First pass: 2h/3 kick, accumulate a and dadq ────────────────────────
    for i in range(n - 1):
        indi = i * 7
        for j in range(i + 1, n):
            indj = j * 7

            def _first(carry, _i=i, _j=j, _indi=indi, _indj=indj):
                v_c, ve_c, a_c, dadq_c, jph_c, dqph_c = carry
                rij   = s.x[:, _i] - s.x[:, _j]
                r2    = jnp.maximum(jnp.dot(rij, rij), eps)
                r2inv = 1.0 / r2
                r3inv = r2inv * jnp.sqrt(r2inv)
                fac   = GNEWT * rij * r3inv
                facv  = fac * (2.0 * h / 3.0)

                vi,  vei  = comp_sum(v_c[:, _i], ve_c[:, _i], -s.m[_j] * facv)
                vj_, vej  = comp_sum(v_c[:, _j], ve_c[:, _j],  s.m[_i] * facv)
                v_c  = v_c .at[:, _i].set(vi ).at[:, _j].set(vj_)
                ve_c = ve_c.at[:, _i].set(vei).at[:, _j].set(vej)

                dqph_c = (dqph_c
                    .at[_indi + 3:_indi + 6].add(-(1.0 / h) * s.m[_j] * facv)
                    .at[_indj + 3:_indj + 6].add( (1.0 / h) * s.m[_i] * facv))

                a_c = a_c.at[:, _i].add(-s.m[_j] * fac).at[:, _j].add(s.m[_i] * fac)

                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indj + 6].add(-facv)
                    .at[_indj + 3:_indj + 6, _indi + 6].add( facv))

                facv3 = facv * 3.0 * r2inv
                outer_ij = jnp.outer(facv3, rij)
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( s.m[_j] * outer_ij)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-s.m[_j] * outer_ij)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( s.m[_i] * outer_ij)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-s.m[_i] * outer_ij))

                facv0 = (2.0 * h / 3.0) * GNEWT * r3inv
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add(-facv0 * s.m[_j] * eye3)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add( facv0 * s.m[_j] * eye3)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add(-facv0 * s.m[_i] * eye3)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add( facv0 * s.m[_i] * eye3))

                dadq_c = (dadq_c
                    .at[:, _i, 3, _j].add(-fac)
                    .at[:, _j, 3, _i].add( fac))

                fac3 = fac * 3.0 * r2inv
                outer_dadq = jnp.outer(fac3, rij)
                dadq_c = (dadq_c
                    .at[:, _i, 0:3, _i].add( s.m[_j] * outer_dadq)
                    .at[:, _i, 0:3, _j].add(-s.m[_j] * outer_dadq)
                    .at[:, _j, 0:3, _j].add( s.m[_i] * outer_dadq)
                    .at[:, _j, 0:3, _i].add(-s.m[_i] * outer_dadq))

                fac0 = GNEWT * r3inv
                kidx = jnp.arange(3)
                dadq_c = (dadq_c
                    .at[kidx, _i, kidx, _i].add(-fac0 * s.m[_j])
                    .at[kidx, _i, kidx, _j].add( fac0 * s.m[_j])
                    .at[kidx, _j, kidx, _j].add(-fac0 * s.m[_i])
                    .at[kidx, _j, kidx, _i].add( fac0 * s.m[_i]))

                return v_c, ve_c, a_c, dadq_c, jph_c, dqph_c

            if s.pair[i][j]:
                v, verror, a, dadq, jac_phi, dqdt_phi = _first(
                    (v, verror, a, dadq, jac_phi, dqdt_phi))

    # ── Second pass: higher-order correction ────────────────────────────────
    for i in range(n - 1):
        indi = i * 7
        for j in range(i + 1, n):
            indj = j * 7

            def _second(carry, _i=i, _j=j, _indi=indi, _indj=indj):
                v_c, ve_c, jph_c, dqph_c = carry
                aij   = a[:, _i] - a[:, _j]
                rij   = s.x[:, _i] - s.x[:, _j]
                r2    = jnp.maximum(jnp.dot(rij, rij), eps)
                r1    = jnp.sqrt(r2)
                ardot = jnp.dot(aij, rij)
                fac1  = coeff / (r2 * r2 * r1)
                fac2s = 3.0 * ardot
                fac_vec = fac1 * (rij * fac2s - r2 * aij)

                vi,  vei  = comp_sum(v_c[:, _i], ve_c[:, _i],  s.m[_j] * fac_vec)
                vj_, vej  = comp_sum(v_c[:, _j], ve_c[:, _j], -s.m[_i] * fac_vec)
                v_c  = v_c .at[:, _i].set(vi ).at[:, _j].set(vj_)
                ve_c = ve_c.at[:, _i].set(vei).at[:, _j].set(vej)

                dqph_c = (dqph_c
                    .at[_indi + 3:_indi + 6].add( (3.0 / h) * s.m[_j] * fac_vec)
                    .at[_indj + 3:_indj + 6].add(-(3.0 / h) * s.m[_i] * fac_vec))

                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indj + 6].add( fac_vec)
                    .at[_indj + 3:_indj + 6, _indi + 6].add(-fac_vec))

                fac5 = fac_vec * (5.0 / r2)
                outer5 = jnp.outer(fac5, rij)
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add(-s.m[_j] * outer5)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add( s.m[_j] * outer5)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add(-s.m[_i] * outer5)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add( s.m[_i] * outer5))

                fac_diag = fac1 * fac2s
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( fac_diag * s.m[_j] * eye3)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-fac_diag * s.m[_j] * eye3)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( fac_diag * s.m[_i] * eye3)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-fac_diag * s.m[_i] * eye3))

                fac3_mat = fac1 * (-2.0 * jnp.outer(aij, rij) + 3.0 * jnp.outer(rij, aij))
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( s.m[_j] * fac3_mat)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-s.m[_j] * fac3_mat)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( s.m[_i] * fac3_mat)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-s.m[_i] * fac3_mat))

                diff_dadq = dadq[:, _i, :, :] - dadq[:, _j, :, :]
                dotdadq   = jnp.tensordot(rij, diff_dadq, axes=([0], [0]))
                fac_acc   = -fac1 * r2
                fac_fin   = 3.0 * fac1

                for di in range(n):
                    indd = di * 7
                    jph_c = (jph_c
                        .at[_indi + 3:_indi + 6, indd:indd + 3].add(
                            fac_acc * s.m[_j] * diff_dadq[:, 0:3, di]
                            + fac_fin * s.m[_j] * jnp.outer(rij, dotdadq[0:3, di]))
                        .at[_indj + 3:_indj + 6, indd:indd + 3].add(
                            -fac_acc * s.m[_i] * diff_dadq[:, 0:3, di]
                            - fac_fin * s.m[_i] * jnp.outer(rij, dotdadq[0:3, di]))
                        .at[_indi + 3:_indi + 6, indd + 6].add(
                            fac_acc * s.m[_j] * diff_dadq[:, 3, di]
                            + fac_fin * s.m[_j] * rij * dotdadq[3, di])
                        .at[_indj + 3:_indj + 6, indd + 6].add(
                            -fac_acc * s.m[_i] * diff_dadq[:, 3, di]
                            - fac_fin * s.m[_i] * rij * dotdadq[3, di]))

                return v_c, ve_c, jph_c, dqph_c

            if s.pair[i][j]:
                v, verror, jac_phi, dqdt_phi = _second(
                    (v, verror, jac_phi, dqdt_phi))

    s = replace(s, v=v, verror=verror)
    d = replace(d, jac_phi=jac_phi, dqdt_phi=dqdt_phi, dadq=dadq)
    return s, d


# ─────────────────────────────────────────────────────────────────────────────
# 4th-order correction with Jacobian (phisalpha)
# ─────────────────────────────────────────────────────────────────────────────
def _phisalpha_grad(s: State, d: Derivatives, h: Array, alpha: Array):
    n     = s.n
    dtype = s.x.dtype
    eps   = jnp.finfo(dtype).eps
    eye3  = jnp.eye(3, dtype=dtype)
    coeff = alpha * h ** 3 * GNEWT / jnp.asarray(48.0, dtype=dtype)

    v        = s.v
    verror   = s.verror
    a        = jnp.zeros((3, n), dtype=dtype)
    dadq     = jnp.zeros((3, n, 4, n), dtype=dtype)
    jac_phi  = d.jac_phi
    dqdt_phi = d.dqdt_phi

    # ── First pass: non-pair acceleration + dadq ─────────────────────────────
    for i in range(n - 1):
        for j in range(i + 1, n):

            def _first(carry, _i=i, _j=j):
                a_c, dadq_c = carry
                rij   = s.x[:, _i] - s.x[:, _j]
                r2    = jnp.maximum(jnp.dot(rij, rij), eps)
                r3    = r2 * jnp.sqrt(r2)
                fac2s = GNEWT / r3
                fac   = fac2s * rij

                a_c = a_c.at[:, _i].add(-s.m[_j] * fac).at[:, _j].add(s.m[_i] * fac)

                dadq_c = (dadq_c
                    .at[:, _i, 3, _j].add(-fac)
                    .at[:, _j, 3, _i].add( fac))

                fac3 = fac * (3.0 / r2)
                outer_dadq = jnp.outer(fac3, rij)
                dadq_c = (dadq_c
                    .at[:, _i, 0:3, _i].add( s.m[_j] * outer_dadq)
                    .at[:, _i, 0:3, _j].add(-s.m[_j] * outer_dadq)
                    .at[:, _j, 0:3, _j].add( s.m[_i] * outer_dadq)
                    .at[:, _j, 0:3, _i].add(-s.m[_i] * outer_dadq))

                kidx = jnp.arange(3)
                dadq_c = (dadq_c
                    .at[kidx, _i, kidx, _i].add(-fac2s * s.m[_j])
                    .at[kidx, _i, kidx, _j].add( fac2s * s.m[_j])
                    .at[kidx, _j, kidx, _j].add(-fac2s * s.m[_i])
                    .at[kidx, _j, kidx, _i].add( fac2s * s.m[_i]))

                return a_c, dadq_c

            if not s.pair[i][j]:
                a, dadq = _first((a, dadq))

    # ── Second pass: velocity correction ────────────────────────────────────
    for i in range(n - 1):
        indi = i * 7
        for j in range(i + 1, n):
            indj = j * 7

            def _second(carry, _i=i, _j=j, _indi=indi, _indj=indj):
                v_c, ve_c, jph_c, dqph_c = carry
                aij   = a[:, _i] - a[:, _j]
                rij   = s.x[:, _i] - s.x[:, _j]
                r2    = jnp.maximum(jnp.dot(rij, rij), eps)
                r1    = jnp.sqrt(r2)
                ardot = jnp.dot(aij, rij)
                fac1  = coeff / (r2 * r2 * r1)
                fac2s = 2.0 * GNEWT * (s.m[_i] + s.m[_j]) / r1 + 3.0 * ardot
                fac_vec = fac1 * (rij * fac2s - r2 * aij)

                vi,  vei  = comp_sum(v_c[:, _i], ve_c[:, _i],  s.m[_j] * fac_vec)
                vj_, vej  = comp_sum(v_c[:, _j], ve_c[:, _j], -s.m[_i] * fac_vec)
                v_c  = v_c .at[:, _i].set(vi ).at[:, _j].set(vj_)
                ve_c = ve_c.at[:, _i].set(vei).at[:, _j].set(vej)

                dqph_c = (dqph_c
                    .at[_indi + 3:_indi + 6].add( (3.0 / h) * s.m[_j] * fac_vec)
                    .at[_indj + 3:_indj + 6].add(-(3.0 / h) * s.m[_i] * fac_vec))

                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indj + 6].add( fac_vec)
                    .at[_indj + 3:_indj + 6, _indi + 6].add(-fac_vec))

                fac5 = fac_vec * (5.0 / r2)
                outer5 = jnp.outer(fac5, rij)
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add(-s.m[_j] * outer5)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add( s.m[_j] * outer5)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add(-s.m[_i] * outer5)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add( s.m[_i] * outer5))

                fac_m2 = 2.0 * GNEWT * fac1 * rij / r1
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi + 6].add( fac_m2 * s.m[_j])
                    .at[_indi + 3:_indi + 6, _indj + 6].add( fac_m2 * s.m[_j])
                    .at[_indj + 3:_indj + 6, _indj + 6].add(-fac_m2 * s.m[_i])
                    .at[_indj + 3:_indj + 6, _indi + 6].add(-fac_m2 * s.m[_i]))

                fac_diag = fac1 * fac2s
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( fac_diag * s.m[_j] * eye3)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-fac_diag * s.m[_j] * eye3)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( fac_diag * s.m[_i] * eye3)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-fac_diag * s.m[_i] * eye3))

                fac_dot = -2.0 * fac1 * (rij * GNEWT * (s.m[_i] + s.m[_j]) / (r2 * r1) + aij)
                fac3_mat = jnp.outer(fac_dot, rij) + 3.0 * fac1 * jnp.outer(rij, aij)
                jph_c = (jph_c
                    .at[_indi + 3:_indi + 6, _indi:_indi + 3].add( s.m[_j] * fac3_mat)
                    .at[_indi + 3:_indi + 6, _indj:_indj + 3].add(-s.m[_j] * fac3_mat)
                    .at[_indj + 3:_indj + 6, _indj:_indj + 3].add( s.m[_i] * fac3_mat)
                    .at[_indj + 3:_indj + 6, _indi:_indi + 3].add(-s.m[_i] * fac3_mat))

                diff_dadq = dadq[:, _i, :, :] - dadq[:, _j, :, :]
                dotdadq   = jnp.tensordot(rij, diff_dadq, axes=([0], [0]))
                fac_acc   = -fac1 * r2
                fac_fin   = 3.0 * fac1

                for di in range(n):
                    indd = di * 7
                    jph_c = (jph_c
                        .at[_indi + 3:_indi + 6, indd:indd + 3].add(
                            fac_acc * s.m[_j] * diff_dadq[:, 0:3, di]
                            + fac_fin * s.m[_j] * jnp.outer(rij, dotdadq[0:3, di]))
                        .at[_indj + 3:_indj + 6, indd:indd + 3].add(
                            -fac_acc * s.m[_i] * diff_dadq[:, 0:3, di]
                            - fac_fin * s.m[_i] * jnp.outer(rij, dotdadq[0:3, di]))
                        .at[_indi + 3:_indi + 6, indd + 6].add(
                            fac_acc * s.m[_j] * diff_dadq[:, 3, di]
                            + fac_fin * s.m[_j] * rij * dotdadq[3, di])
                        .at[_indj + 3:_indj + 6, indd + 6].add(
                            -fac_acc * s.m[_i] * diff_dadq[:, 3, di]
                            - fac_fin * s.m[_i] * rij * dotdadq[3, di]))

                return v_c, ve_c, jph_c, dqph_c

            if not s.pair[i][j]:
                v, verror, jac_phi, dqdt_phi = _second(
                    (v, verror, jac_phi, dqdt_phi))

    s = replace(s, v=v, verror=verror)
    d = replace(d, jac_phi=jac_phi, dqdt_phi=dqdt_phi)
    return s, d


# ─────────────────────────────────────────────────────────────────────────────
# Kepler step with Jacobian for one pair (i, j) — wrapped in gm guard
# ─────────────────────────────────────────────────────────────────────────────
def _kepler_step(s: State, d: Derivatives, i: int, j: int,
                 indi: int, indj: int, sevn: int, h: Array, drift_first: bool):
    x0 = s.x[:, i] - s.x[:, j]
    v0 = s.v[:, i] - s.v[:, j]
    gm = GNEWT * (s.m[i] + s.m[j])
    eps = jnp.finfo(s.x.dtype).eps

    def _apply(sd):
        s_in, d_in = sd
        params, delxv = _jac_delxv_gamma(x0, v0, gm, h, drift_first)
        jac_kepler, jac_mass = _compute_jacobian_gamma(params, x0, v0, drift_first)

        mijinv = 1.0 / (s_in.m[i] + s_in.m[j])
        mi = s_in.m[i] * mijinv
        mj = s_in.m[j] * mijinv

        xi,  xei  = comp_sum(s_in.x[:, i], s_in.xerror[:, i],  mj * delxv[:3])
        xj_, xej  = comp_sum(s_in.x[:, j], s_in.xerror[:, j], -mi * delxv[:3])
        vi,  vei  = comp_sum(s_in.v[:, i], s_in.verror[:, i],  mj * delxv[3:])
        vj_, vej  = comp_sum(s_in.v[:, j], s_in.verror[:, j], -mi * delxv[3:])
        x_new  = s_in.x      .at[:, i].set(xi ).at[:, j].set(xj_)
        xe_new = s_in.xerror .at[:, i].set(xei).at[:, j].set(xej)
        v_new  = s_in.v      .at[:, i].set(vi ).at[:, j].set(vj_)
        ve_new = s_in.verror .at[:, i].set(vei).at[:, j].set(vej)

        # Assemble jac_ij (14×14)
        jac_ij = jnp.zeros((14, 14), dtype=s_in.x.dtype)
        jac_ij = (jac_ij
            .at[0:6,  0:6].add( mj * jac_kepler[:, 0:6])
            .at[0:6,  7:13].add(-mj * jac_kepler[:, 0:6])
            .at[7:13, 0:6].add(-mi * jac_kepler[:, 0:6])
            .at[7:13, 7:13].add( mi * jac_kepler[:, 0:6])
            .at[0:6,  6].set( jac_mass * s_in.m[j])
            .at[0:6,  13].set( mi * delxv * mijinv + GNEWT * mj * jac_kepler[:, 6])
            .at[7:13, 6].set(-mj * delxv * mijinv - GNEWT * mi * jac_kepler[:, 6])
            .at[7:13, 13].set(-jac_mass * s_in.m[i]))

        dqdt_ij = jnp.zeros(14, dtype=s_in.x.dtype)
        dqdt_ij = dqdt_ij.at[0:6].set(mj * jac_kepler[:, 7]).at[7:13].set(-mi * jac_kepler[:, 7])

        # copy_submatrix
        jac_tmp1 = (d_in.jac_tmp1
            .at[0:7,  :sevn].set(s_in.jac_step [indi:indi + 7, :sevn])
            .at[7:14, :sevn].set(s_in.jac_step [indj:indj + 7, :sevn]))
        jac_err1 = (d_in.jac_err1
            .at[0:7,  :sevn].set(s_in.jac_error[indi:indi + 7, :sevn])
            .at[7:14, :sevn].set(s_in.jac_error[indj:indj + 7, :sevn]))
        dqdt_tmp1 = (d_in.dqdt_tmp1
            .at[0:7].set(s_in.dqdt[indi:indi + 7])
            .at[7:14].set(s_in.dqdt[indj:indj + 7]))

        # comp_sum_matrix update
        jac_tmp2 = jac_ij @ jac_tmp1
        jac_tmp1, jac_err1 = comp_sum_matrix(jac_tmp1, jac_err1, jac_tmp2)

        # dqdt_ij chain rule
        dqdt_ij_new = dqdt_ij * 0.5 + dqdt_tmp1 + jac_ij @ dqdt_tmp1

        # ypoc_submatrix
        jac_step = (s_in.jac_step
            .at[indi:indi + 7, :sevn].set(jac_tmp1[0:7,  :sevn])
            .at[indj:indj + 7, :sevn].set(jac_tmp1[7:14, :sevn]))
        jac_error = (s_in.jac_error
            .at[indi:indi + 7, :sevn].set(jac_err1[0:7,  :sevn])
            .at[indj:indj + 7, :sevn].set(jac_err1[7:14, :sevn]))
        dqdt = (s_in.dqdt
            .at[indi:indi + 7].set(dqdt_ij_new[0:7])
            .at[indj:indj + 7].set(dqdt_ij_new[7:14]))

        s_out = replace(s_in, x=x_new, xerror=xe_new, v=v_new, verror=ve_new,
                        jac_step=jac_step, jac_error=jac_error, dqdt=dqdt)
        d_out = replace(d_in, jac_ij=jac_ij, dqdt_ij=dqdt_ij_new,
                        jac_kepler=jac_kepler, jac_mass=jac_mass,
                        jac_tmp1=jac_tmp1, jac_err1=jac_err1, dqdt_tmp1=dqdt_tmp1)
        return s_out, d_out

    return lax.cond(jnp.abs(gm) > eps, _apply, lambda sd: sd, (s, d))


# ─────────────────────────────────────────────────────────────────────────────
# Kepler solver (returns delxv and 22-parameter tuple)
# ─────────────────────────────────────────────────────────────────────────────
def _jac_delxv_gamma(x0: Array, v0: Array, k: Array, h: Array, drift_first: bool):
    dtype = x0.dtype
    zero  = jnp.asarray(0.0, dtype=dtype)

    if drift_first:
        rtmp = x0 - h * v0
        r0   = jnp.linalg.norm(rtmp)
        eta  = jnp.dot(rtmp, v0)
    else:
        r0   = jnp.linalg.norm(x0)
        eta  = jnp.dot(x0, v0)

    r0inv   = 1.0 / r0
    beta0   = 2.0 * k * r0inv - jnp.dot(v0, v0)
    beta0inv= 1.0 / beta0
    signb   = jnp.sign(beta0)
    sqb     = jnp.sqrt(signb * beta0)
    zeta    = k - r0 * beta0

    gamma_fallback = h * r0inv * sqb

    def cubic_guess(_):
        zinv = 6.0 / zeta
        return _cubic1(0.5 * eta * sqb * zinv,
                       r0 * signb * beta0 * zinv,
                       -h * signb * beta0 * sqb * zinv)

    def noncubic_guess(_):
        def quad(_):
            reta = r0 / eta
            disc = reta ** 2 + 2.0 * h / eta
            return lax.cond(disc > zero,
                lambda _: sqb * (-reta + jnp.sqrt(disc)),
                lambda _: gamma_fallback, None)
        return lax.cond(eta != zero, quad, lambda _: gamma_fallback, None)

    gamma_guess = lax.cond(zeta != zero, cubic_guess, noncubic_guess, None)

    c2_ = -2.0 * zeta
    c3_ = 2.0 * eta * signb * sqb
    c4_ = -sqb * h * beta0
    c5_ = 2.0 * eta * signb * sqb

    def sincos_hg(gamma):
        xx = 0.5 * gamma
        return lax.cond(beta0 > zero,
            lambda z: (jnp.sin(z),  jnp.cos(z)),
            lambda z: (jnp.sinh(z), jnp.exp(-z) + jnp.sinh(z)),
            xx)

    init = (gamma_guess, 2.0 * gamma_guess, 3.0 * gamma_guess,
            jnp.zeros((), jnp.int32), jnp.asarray(True))

    def cond_fn(carry):
        _, _, _, it, go = carry
        return go & (it < 20)

    def body_fn(carry):
        g, g1, g2, it, _ = carry
        sx, cx = sincos_hg(g)
        num = k * g + c2_ * sx * cx + c3_ * sx ** 2 + c4_
        den = 2.0 * signb * zeta * sx ** 2 + c5_ * sx * cx + r0 * beta0
        gn  = g - num / den
        it2 = it + 1
        go2 = (it2 < 20) & (gn != g2) & (gn != g1)
        return gn, g, g1, it2, go2

    gamma, _, _, _, _ = lax.while_loop(cond_fn, body_fn, init)

    sx, cx = sincos_hg(gamma)
    g1bs = 2.0 * sx * cx / sqb
    g2bs = 2.0 * signb * sx ** 2 * beta0inv
    g0bs = 1.0 - beta0 * g2bs
    g3bs = G3(gamma, beta0, sqb)

    r    = r0 * g0bs + eta * g1bs + k * g2bs
    rinv = 1.0 / r
    dfdt = -k * g1bs * rinv * r0inv

    if drift_first:
        h1_val = zero
        h2_val = zero
        fm1    = -k * r0inv * g2bs
        gmh    = k * r0inv * (h * g2bs - r0 * g3bs)
        dgdtm1 = k * r0inv * rinv * (h * g1bs - r0 * g2bs)
    else:
        h1_val = H1(gamma, beta0)
        h2_val = H2(gamma, beta0, sqb)
        fm1    = k * rinv * (g2bs - k * r0inv * h1_val)
        gmh    = k * rinv * (r0 * h2_val + eta * h1_val)
        dgdtm1 = -k * rinv * g2bs

    delxv  = jnp.concatenate([fm1 * x0 + gmh * v0, dfdt * x0 + dgdtm1 * v0])
    params = (gamma, g0bs, g1bs, g2bs, g3bs, h1_val, h2_val,
              dfdt, fm1, gmh, dgdtm1,
              r0, r, r0inv, rinv, k, h, beta0, beta0inv, eta, sqb, zeta)
    return params, delxv


# ─────────────────────────────────────────────────────────────────────────────
# Analytical Jacobian of Kepler step
# ─────────────────────────────────────────────────────────────────────────────
def _compute_jacobian_gamma(params, x0: Array, v0: Array, drift_first: bool):
    (gamma, g0, g1, g2, g3, h1_kep, h2_kep,
     dfdt, fm1, gmh, dgdtm1,
     r0, r, r0inv, rinv, k, h, beta, betainv, eta, sqb, zeta) = params

    dtype  = x0.dtype
    jk     = jnp.zeros((6, 8), dtype=dtype)
    jmass  = jnp.zeros(6, dtype=dtype)
    idx3   = jnp.arange(3)

    r0inv2 = r0inv ** 2
    r0inv3 = r0inv2 * r0inv
    rinv2  = rinv ** 2
    rinv3  = rinv2 * rinv
    hsq    = h ** 2
    ksq    = k ** 2

    if drift_first:
        d_c  = (h + eta * g2 + 2.0 * k * g3) * betainv
        c1   = d_c - r0 * g3
        c2   = eta * g0 + g1 * zeta
        c3   = d_c * k + g1 * r0 ** 2
        c13  = g1 * h - g2 * r0
        c9   = 2.0 * g2 * h - 3.0 * g3 * r0
        c10  = k * r0inv2 ** 2 * (-g2 * r0 * h + k * c9 * betainv - c3 * c13 * rinv)
        c24  = r0inv3 * (r0 * (2.0 * k * r0inv - beta) * betainv - g1 * c3 * rinv / g2)
        h6v  = H6(gamma, beta)
        h4v  = -H1(gamma, beta) * beta
        h5v  = H5(gamma, beta, sqb)
        h3v  = H3(gamma, beta, sqb)
        h8v  = -2.0 * h3v + 3.0 * h5v

        dfm1dxx = fm1 * c24
        dfm1dxv = -fm1 * (g1 * rinv + h * c24);  dfm1dvx = dfm1dxv
        dfm1dvv = fm1 * rinv * (-r0*g2 + k*h6v*betainv/g2 + h*(2.0*g1 + h*r*c24))
        dfm1dh  = fm1 * (g1*rinv*(1.0/g2 + 2.0*k*r0inv - beta) - eta*c24)
        dfm1dk  = fm1 * (1.0/k + g1*c1*rinv*r0inv/g2 - 2.0*betainv*r0inv)
        dfm1dk2 = r0 * h4v + k * h6v

        dgmhdxx = c10
        dgmhdxv = -g2*k*c13*rinv*r0inv - h*c10;  dgmhdvx = dgmhdxv
        dgmhdvv = (2.0*g2*h*k*c13*rinv*r0inv + hsq*c10 +
                   k*betainv*rinv*r0inv*(r0**2*h8v - beta*h*r0*g2**2 + (h*k + eta*r0)*h6v))
        dgmhdh  = (g2*k*r0inv + k*c13*rinv*r0inv +
                   g2*k*(2.0*k*r0inv - beta)*c13*rinv*r0inv - eta*c10)
        dgmhdk  = r0inv*(k*c1*c13*rinv*r0inv + g2*h - g3*r0 - k*c9*betainv*r0inv)
        dgmhdk2 = (h6v*g3*ksq + eta*r0*(h6v + g2*h4v) + r0**2*g0*h5v +
                   k*eta*g2*h6v + (g1*h6v + g3*h4v)*k*r0)

        jk = jk.at[idx3, idx3].set(fm1).at[idx3, idx3 + 3].set(gmh)
        Ax = dfm1dxx*x0 + dfm1dxv*v0;  Bx = dgmhdxx*x0 + dgmhdxv*v0
        Av = dfm1dvx*x0 + dfm1dvv*v0;  Bv = dgmhdvx*x0 + dgmhdvv*v0
        jk = (jk.at[0:3, 0:3].add(jnp.outer(x0, Ax) + jnp.outer(v0, Bx))
                 .at[0:3, 3:6].add(jnp.outer(x0, Av) + jnp.outer(v0, Bv))
                 .at[0:3, 6].set(dfm1dk*x0 + dgmhdk*v0)
                 .at[0:3, 7].set(dfm1dh*x0 + dgmhdh*v0))
        jmass = jmass.at[0:3].set((GNEWT*r0inv)**2 * betainv * rinv * (dfm1dk2*x0 - dgmhdk2*v0))

        c12  = g0*h - g1*r0
        c17  = r0 - r - g2*k
        c25  = k*rinv*r0inv2*(-g2 + k*(c13-g2*r0)*betainv*r0inv2 - c13*r0inv
                               - c12*c3*rinv*r0inv2 + c13*c2*c3*rinv2*r0inv2
                               - c13*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv2)
        c26  = k*rinv2*r0inv*(-g2*c12 - g1*c13 + g2*c13*c2*rinv)
        c34  = ((-beta*eta**2*g2**2 - eta*k*h8v - h6v*ksq
                  - 2.0*beta*eta*r0*g1*g2 + (g2**2 - 3.0*g1*g3)*beta*k*r0
                  - beta*g1**2*r0**2) * betainv*rinv2
                + (eta*g2**2)*rinv/g1 + k*h8v*betainv*rinv/g1)
        c22  = rinv*(-g1 - g0*g2/g1 + g2*c2*rinv)
        c21  = ((g2*k - r0)*(beta*c3 - k*g1*r)*betainv*rinv2*r0inv3/g1
                + eta*g1*rinv*r0inv2 - 2.0*r0inv2)

        ddfdtdxx = dfdt * c21
        ddfdtdxv = dfdt*(c22 - h*c21);  ddfdtdvx = ddfdtdxv
        ddfdtdvv = dfdt*(c34 - 2.0*h*c22 + hsq*c21)
        ddfdtdk  = dfdt*(1.0/k - betainv*r0inv - c17*betainv*rinv*r0inv
                         - c1*(g1*c2 - g0*r)*rinv2*r0inv/g1)
        ddfdtdk2 = -(g2*k - r0)*(beta*r0*(g3-g1*g2) - beta*eta*g2**2 + k*h3v)*betainv*rinv2*r0inv
        ddfdtdh  = dfdt*(g0*rinv/g1 - c2*rinv2 - (2.0*k*r0inv-beta)*c22 - eta*c21)

        dgm1dxx = c25;  dgm1dxv = c26 - h*c25;  dgm1dvx = dgm1dxv
        h2loc   = H2(gamma, beta, sqb)
        c33     = (d_c*k*rinv3*r0inv*k*(h*g2 - r0*g3) +
                   k*(-eta*k*g1*g2**2 - g1*g2*g3*ksq - r0*eta*beta*g1*g2**2
                      - r0*k*g1*h2loc - beta*g2**2*g0*r0**2)*betainv*rinv2*r0inv)
        dgm1dvv  = c33 - 2.0*h*c26 + hsq*c25
        dgm1dk   = rinv*r0inv*(-k*(c13-g2*r0)*betainv*r0inv + c13
                               - k*c13*c17*betainv*rinv*r0inv + k*c1*c12*rinv*r0inv
                               - k*c1*c2*c13*rinv2*r0inv)
        dgm1dk2  = k*betainv*rinv2*r0inv*(-beta*eta**2*g2**4
                    + eta*g2*(g1*g2**2 + g1**2*g3 - 5.0*g2*g3)*k
                    + g2*g3*h3v*ksq + 2.0*eta*r0*beta*g2**2*(g3 - g1*g2)
                    + (4.0*g3 - g0*g3 - g1*g2)*(g3 - g1*g2)*r0*k
                    + beta*(2.0*g1*g3*g2 - g1**2*g2**2 - g3**2)*r0**2)
        dgm1dh   = (g1*k*rinv*r0inv + k*c12*rinv2*r0inv - k*c2*c13*rinv3*r0inv
                    - (2.0*k*r0inv - beta)*c26 - eta*c25)

        jk = jk.at[idx3+3, idx3].set(dfdt).at[idx3+3, idx3+3].set(dgdtm1)
        Ax2 = ddfdtdxx*x0 + ddfdtdxv*v0;  Bx2 = dgm1dxx*x0 + dgm1dxv*v0
        Av2 = ddfdtdvx*x0 + ddfdtdvv*v0;  Bv2 = dgm1dvx*x0 + dgm1dvv*v0
        jk  = (jk.at[3:6, 0:3].add(jnp.outer(x0, Ax2) + jnp.outer(v0, Bx2))
                  .at[3:6, 3:6].add(jnp.outer(x0, Av2) + jnp.outer(v0, Bv2))
                  .at[3:6, 6].set(ddfdtdk*x0  + dgm1dk*v0)
                  .at[3:6, 7].set(ddfdtdh*x0  + dgm1dh*v0))
        jmass = jmass.at[3:6].set(GNEWT**2 * r0inv * rinv * (ddfdtdk2*x0 + dgm1dk2*v0))

    else:  # kepler first
        h1v  = h1_kep;  h2v = h2_kep
        d_c  = (h + eta*g2 + 2.0*k*g3)*betainv
        c1   = d_c - r0*g3
        c2   = eta*g0 + g1*zeta
        c3   = d_c*k + g1*r0**2
        c6   = (r0*g0 - k*g2)*betainv
        c9   = g2*r  - h1v*k
        c14  = r0*g2 - k*h1v
        c15  = eta*h1v + h2v*r0
        c16  = eta*h2v + g1*gamma*r0/sqb
        c17  = r0 - r - g2*k
        c19  = 4.0*eta*h1v + 3.0*h2v*r0
        c23  = h2v*k - r0*g1
        h6v  = H6(gamma, beta)
        h3v  = H3(gamma, beta, sqb)
        h5v  = H5(gamma, beta, sqb)
        h8v  = -2.0*h3v + 3.0*h5v
        h7v  = beta*g1*g2**2 - g0*h8v

        dfm1dxx = (k*rinv3*betainv*r0inv**4
                   *(k*h1v*r**2*r0*(beta - 2.0*k*r0inv)
                     + beta*c3*(r*c23 + c14*c2) + c14*r*(k*(r-g2*k) + g0*r0*zeta)))
        dfm1dxv = k*rinv2*r0inv*(k*(g2*h2v+g1*h1v) - 2.0*g1*g2*r0 + g2*c14*c2*rinv)
        dfm1dvx = dfm1dxv
        dfm1dvv = (k*r0inv*rinv2*betainv
                   *(2.0*eta*k*(g2*g3-g1*h1v) + (3.0*g3*h2v-4.0*h1v*g2)*ksq
                     + beta*g2*r0*(3.0*h1v*k - g2*r0)
                     + c14*rinv*(-beta*g2**2*eta**2 + eta*k*(2.0*g0*g3-h2v) - h6v*ksq
                                  + (-2.0*eta*g1*g2 + k*(h1v-2.0*g1*g3))*beta*r0
                                  - beta*g1**2*r0**2)))
        dfm1dh  = (g1*k - h2v*ksq*r0inv - k*c14*c2*rinv*r0inv)*rinv2
        dfm1dk  = (rinv*r0inv
                   *(4.0*h1v*ksq*betainv*r0inv - k*h1v - 2.0*g2*k*betainv + c14
                     - k*c14*c17*betainv*rinv*r0inv + k*(g1*r0-k*h2v)*c1*rinv*r0inv
                     - k*c14*c1*c2*rinv2*r0inv))
        dfm1dk2 = (betainv*r0inv*rinv2
                   *(r*(2.0*eta*k*(g1*h1v-g3*g2) + (4.0*g2*h1v-3.0*g3*h2v)*ksq
                         - eta*r0*beta*g1*h1v + (g3*h2v-4.0*g2*h1v)*beta*k*r0
                         + g2*h1v*beta**2*r0**2)
                    - c14*(-eta**2*beta*g2**2 - k*eta*h8v - ksq*h6v
                           - eta*r0*beta*(g1*g2+g0*g3) + 2.0*(h1v-g1*g3)*beta*k*r0
                           - (g2-beta*g1*g3)*beta*r0**2)))

        dgmhdxx = (k*rinv*r0inv
                   *(h2v + k*c19*betainv*r0inv2 - c16*c3*rinv*r0inv2
                     + c2*c3*c15*(rinv*r0inv)**2
                     - c15*(k*(g2*k+r)-g0*r0*zeta)*betainv*rinv*r0inv2))
        dgmhdxv = k*rinv2*(h1v*r - g2*c16 - g1*c15 + g2*c2*c15*rinv)
        dgmhdvx = dgmhdxv
        dgmhdvv = (k*betainv*rinv2
                   *(2.0*eta**2*(g1*h1v-g2*g3) + eta*k*(4.0*g2*h1v-3.0*h2v*g3)
                     + r0*eta*(4.0*g0*h1v-2.0*g1*g3) + 3.0*r0*k*((g1+beta*g3)*h1v-g3*g2)
                     + (g0*h8v-beta*g1*(g2**2+g1*g3))*r0**2
                     - c15*rinv*(beta*g2**2*eta**2 + eta*k*h8v + h6v*ksq
                                  + (2.0*eta*g1*g2-k*(g2**2-3.0*g1*g3))*beta*r0
                                  + beta*g1**2*r0**2)))
        dgmhdk  = rinv*(k*c1*c16*rinv*r0inv + c15 - k*c15*c17*betainv*rinv*r0inv
                        - k*c19*betainv*r0inv - k*c1*c2*c15*rinv2*r0inv)
        dgmhdk2 = (betainv*rinv2
                   *(r*(2.0*eta**2*(g3*g2-g1*h1v) + eta*k*(3.0*g3*h2v-4.0*g2*h1v)
                         + r0*eta*(beta*g3*(g1*g2+g0*g3)-2.0*g0*h6v)
                         + (-h6v*(g1+beta*g3)+g2*(2.0*g3-h2v))*r0*k + (h7v-beta**2*g1*g3**2)*r0**2)
                    - c15*(-beta*eta**2*g2**2 + eta*k*(-h2v+2.0*g0*g3) - h6v*ksq
                           - r0*eta*beta*(h2v+2.0*g0*g3) + 2.0*beta*(2.0*h1v-g2**2)*r0*k
                           + beta*(beta*g1*g3-g2)*r0**2)))
        dgmhdh  = k*rinv3*(r*c16 - c2*c15)

        jk = jk.at[idx3, idx3].set(fm1).at[idx3, idx3+3].set(gmh)
        Ax = dfm1dxx*x0+dfm1dxv*v0;  Bx = dgmhdxx*x0+dgmhdxv*v0
        Av = dfm1dvx*x0+dfm1dvv*v0;  Bv = dgmhdvx*x0+dgmhdvv*v0
        jk = (jk.at[0:3, 0:3].add(jnp.outer(x0, Ax) + jnp.outer(v0, Bx))
                 .at[0:3, 3:6].add(jnp.outer(x0, Av) + jnp.outer(v0, Bv))
                 .at[0:3, 6].set(dfm1dk*x0 + dgmhdk*v0)
                 .at[0:3, 7].set(dfm1dh*x0 + dgmhdh*v0))
        jmass = jmass.at[0:3].set(GNEWT**2 * rinv * r0inv * (dfm1dk2*x0 + dgmhdk2*v0))

        c5  = (r0-k*g2)*rinv/g1
        c7  = g2*(1.0/g1 + c2*rinv)
        c8  = (k*c6 + r*r0 + c3*c5)*r0inv3
        c12 = g0*h - g1*r0
        c20 = k*(g2*k+r) - g0*r0*zeta

        ddfdtdxx = (dfdt*(eta*g1*rinv - 2.0 - g0*c3*rinv*r0inv/g1 + c2*c3*r0inv*rinv2
                          - k*(k*g2-r0)*betainv*rinv*r0inv)*r0inv2)
        ddfdtdxv = -dfdt*(g0*g2/g1 + (r0*g1+eta*g2)*rinv)*rinv
        ddfdtdvx = ddfdtdxv
        ddfdtdvv = (-k*rinv3*r0inv*betainv
                    *((beta*eta*g2**2+k*h8v)*(r0*g0+k*g2)
                      + g1*(-h6v*ksq + (-2.0*eta*g1*g2+(h1v-2.0*g1*g3)*k)*beta*r0
                             - beta*g1**2*r0**2)))
        ddfdtdk  = dfdt*(1.0/k + c1*(r0-g2*k)*r0inv*rinv2/g1 - betainv*r0inv*(1.0+c17*rinv))
        ddfdtdk2 = ((r0-g2*k)*betainv*r0inv*rinv2
                    *(-eta*beta*g2**2 + h3v*k + (g3-g1*g2)*beta*r0))
        ddfdtdh  = dfdt*(r0-g2*k)*rinv2/g1

        dgdm1dxx = (rinv2*r0inv3
                    *((eta*g2+g1*r0)*k*c3*rinv + g2*k*(k*(g2*k-r)-g0*r0*zeta)*betainv))
        dgdm1dxv = k*g2*rinv3*(r*g1 + r0*g1 + eta*g2)
        dgdm1dvx = dgdm1dxv
        dgdm1dvv = (k*betainv*rinv3
                    *(eta**2*beta*g2**3 - eta*k*g2*h3v + 3.0*r0*eta*beta*g1*g2**2
                      + r0*k*(-g0*h6v+3.0*beta*g1*g2*g3) + beta*g2*(g0*g2+g1**2)*r0**2))
        dgdm1dk  = (rinv*r0inv*(-r0*g2 + g2*k*(r+r0-g2*k)*betainv*rinv
                               - k*g1*c1*rinv + k*g2*c1*c2*rinv2))
        dgdm1dk2 = (betainv*rinv2
                    *(-beta*eta**2*g2**3 + eta*k*g2*h3v + eta*r0*beta*g2*(g3-2.0*g1*g2)
                      + (h6v-beta*g2**3)*r0*k + beta*g1*(g3-g1*g2)*r0**2))
        dgdm1dh  = k*rinv3*(g2*c2 - r*g1)

        jk = jk.at[idx3+3, idx3].set(dfdt).at[idx3+3, idx3+3].set(dgdtm1)
        Ax2 = ddfdtdxx*x0+ddfdtdxv*v0;  Bx2 = dgdm1dxx*x0+dgdm1dxv*v0
        Av2 = ddfdtdvx*x0+ddfdtdvv*v0;  Bv2 = dgdm1dvx*x0+dgdm1dvv*v0
        jk  = (jk.at[3:6, 0:3].add(jnp.outer(x0, Ax2) + jnp.outer(v0, Bx2))
                  .at[3:6, 3:6].add(jnp.outer(x0, Av2) + jnp.outer(v0, Bv2))
                  .at[3:6, 6].set(ddfdtdk*x0  + dgdm1dk*v0)
                  .at[3:6, 7].set(ddfdtdh*x0  + dgdm1dh*v0))
        jmass = jmass.at[3:6].set(GNEWT**2 * rinv * r0inv * (ddfdtdk2*x0 + dgdm1dk2*v0))

    return jk, jmass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _cbrt(x):
    return jnp.sign(x) * jnp.abs(x) ** (1.0 / 3.0)


def _cubic1(a, b, c):
    """One real root of x³ + a x² + b x + c = 0."""
    dtype  = jnp.result_type(a, b, c)
    eps    = jnp.finfo(dtype).eps
    a3     = a / 3.0
    q      = a3 * a3 - b / 3.0
    rr     = a3 ** 3 + 0.5 * (-a3 * b + c)
    r2     = rr * rr;  q3 = q ** 3
    safe_b = jnp.where(jnp.abs(b) > eps, b, jnp.where(b >= 0, eps, -eps))

    def three_real(_):
        return -c / safe_b

    def one_real(_):
        disc = jnp.maximum(r2 - q3, 0.0)
        cb   = jnp.abs(rr) + jnp.sqrt(disc)
        ra   = -jnp.sign(rr) * _cbrt(cb)
        rb   = jnp.where(ra == 0.0, 0.0, q / ra)
        return ra + rb - a3

    return lax.cond(r2 < q3, three_real, one_real, None)
