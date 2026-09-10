from __future__ import annotations
import jax
import jax.numpy as jnp

Array  = jax.Array

def calc_bv(s, i, j):
    '''
    Sky-plane relative velocity magnitude and squared impact parameter between bodies i and j
    Parameters
    s:State object containing:
        - s.x: positions, shape (3, n)
        - s.v: velocities, shape (3, n)

        The state object must be a registered JAX PyTree when this function is called inside jax.jit.

    i, j: Body indices.
    return:
    vsky : sky-plane relative velocity speed, sqrt((v_xj - v_xi) ** 2 + (v_yj - v_yi) ** 2)
    bsky2: squared sky-plane separation, (xj - xi) ** 2 + (yj - yi) ** 2
    '''
    dx = s.x[0, j] - s.x[0, i]
    dy = s.x[1, j] - s.x[1, i]
    dvx = s.v[0, j] - s.v[0, i]
    dvy = s.v[1, j] - s.v[1, i]

    vsky = jnp.hypot(dvx, dvy) #sky-plane relative velocity speed
    bsky2 = dx * dx + dy * dy # squared sky-plane separation
    return vsky,bsky2


def g_func(i, j, x, v):
    '''
    Calculate the transit condition

        g = r_sky · v_sky.

    A root g == 0 corresponds to an extremum in the projected
    sky-plane separation. An additional front/back or impact-parameter
    condition may still be needed to identify a physical transit.

    Refer to Fabrycky (2008) - eqs 8 - 10 - Non Keplerian Dynamics
    '''
    dx = x[0, j] - x[0, i]
    dy = x[1, j] - x[1, i]
    dvx = v[0, j] - v[0, i]
    dvy = v[1, j] - v[1, i]

    return dx * dvx + dy * dvy

def gd_func(i, j, x, v, dqdt):
    '''
    Time derivative of g_func. 
    
    dqdt: (7 * n, ) [x, y, z, vx, vy, vz, m]
    '''
    indi = 7 * i
    indj = 7 * j

    dx = x[0, j] - x[0, i]
    dy = x[1, j] - x[1, i]

    dvx = v[0, j] - v[0, i]
    dvy = v[1, j] - v[1, i]

    ddx_dt = dqdt[indj + 0] - dqdt[indi + 0]
    ddy_dt = dqdt[indj + 1] - dqdt[indi + 1]

    ddvx_dt = dqdt[indj + 3] - dqdt[indi + 3]
    ddvy_dt = dqdt[indj + 4] - dqdt[indi + 4]
    return  dx * ddvx_dt + dy * ddvy_dt + dvx * ddx_dt + dvy * ddy_dt

def calc_dbvdq(s, i, j):
    """
    Calculate vsky, bsky2, and their derivatives with respect to the initial phase-space state.
    The derivative includes the implicit dependence of transit time through the transit condition g(t, q) = 0.

    Parameters
    ----------
    s: State object containing:
        - x: shape (3, n)
        - v: shape (3, n)
        - dqdt: shape (7*n,)
        - jac_step: shape (7*n, 7*n)

    i, j: Body indices.

    Returns
    -------
    vsky: Relative sky-plane speed.

    bsky2: Squared sky-plane separation.

    dbvdq: Derivatives with shape (2, 7, n):
        - dbvdq[0] = d(vsky)/dq
        - dbvdq[1] = d(bsky2)/dq
    """
    vsky, bsky2 = calc_bv(s, i, j)

    gsky = g_func(i, j, s.x, s.v)
    gdot = gd_func(i, j, s.x, s.v, s.dqdt)

    xi, xj = s.x[0, i], s.x[0, j]
    yi, yj = s.x[1, i], s.x[1, j]
    vxi, vxj = s.v[0, i], s.v[0, j]
    vyi, vyj = s.v[1, i], s.v[1, j]
    
    dx = xj - xi
    dy = yj - yi
    dvx = vxj - vxi
    dvy = vyj - vyi

    indi = 7 * i
    indj = 7 * j
    
    # Relative acceleration
    adx = s.dqdt[indj + 3] - s.dqdt[indi + 3]
    ady = s.dqdt[indj + 4] - s.dqdt[indi + 4]
    # Time derivatives of the relative sky-plane speed
    dvdt = (dvx * adx + dvy * ady)/vsky

    # Since bsky2 = dx**2 + dy**2:
    # d(bsky2)/dt = 2 * (dx*dvx + dy*dvy) = 2*g
    dbdt = 2.0 * gsky

    #(7n, 7n) Jacobian s.jac_step for x, y, vx, vy of bodies i, j
    # Derivatives of the relative state at the current integration
    # time with respect to the initial state.
    djx = s.jac_step[indj + 0, :] - s.jac_step[indi + 0, :] #\p(xj - xi) / \p(q0), (7n, )
    djy = s.jac_step[indj + 1, :] - s.jac_step[indi + 1, :]  #\p(yj - yi) / \p(q0), (7n, )
    djvx = s.jac_step[indj + 3, :] - s.jac_step[indi + 3, :]  #\p(vxj - vxi) / \p(q0), (7n, )
    djvy = s.jac_step[indj + 4, :] - s.jac_step[indi + 4, :]  #\p(vyj - vyi) / \p(q0), (7n, )

    # From implicit differentiation of
    #     g(t(q), q) = 0:
    #     dt/dq = -(partial g / partial q) / (dg/dt).
    dtdq = -(djx * dvx + djy * dvy + djvx * dx + djvy * dy) / gdot
    # Explicit derivative at fixed time, plus the implicit
    # transit-time contribution.
    dvsky_dq = (djvx * dvx + djvy * dvy)/ vsky + dvdt * dtdq #i, j
    dbsky2_dq = 2.0 * (djx * dx + djy * dy) + dbdt * dtdq #i, j

    n = s.jac_step.shape[0]//7
    
    # dbvdq = \partial(v_sky, b^2_sky) / \partial(q_0) -> (2, 7, n)
    dbvdq = jnp.stack((dvsky_dq.reshape(n, 7).T,dbsky2_dq.reshape(n, 7).T), axis = 0)

    return vsky, bsky2, dbvdq

def calc_dbvdelements(dbvdq0, jac_init):
    """
    Convert derivatives with respect to the Cartesian initial state
    into derivatives with respect to the initial element vector.

    Parameters
    ----------
    dbvdq0
        Shape (2, n, nt, 7, n).

        For each observable (2, vsky, bsky2, axis = 0), body (n, axis = 1), and transit (axis = 2, transit event):

            dbvdq0[ibv, body, transit]

        has shape (7, n).
        row0: derivatives of x to every body
        row1: derivatives of y to every body
        ...
        row6: derivatives of m to every body

    jac_init
        Initial-condition Jacobian, shape (7*n, 7*n).

    Returns
    -------
    dbvdelements
        Shape (2, n, nt, 7, n).
    """
    dbvdq0 = jnp.asarray(dbvdq0)
    jac_init = jnp.asarray(jac_init)

    if dbvdq0.ndim != 5:
        raise ValueError(
            "dbvdq0 must have shape (2, n, nt, 7, n); "
            f"got {dbvdq0.shape}."
        )

    n = dbvdq0.shape[-1]
    expected_size = 7 * n
    # Perform this for all ibv, bodies, and transits simultaneously.
    dq0_flat = jnp.swapaxes(dbvdq0,-1,-2,).reshape(*dbvdq0.shape[:-2],expected_size,)

    result_flat = jnp.einsum("...p,pq->...q",dq0_flat,jac_init,)

    return jnp.swapaxes(result_flat.reshape(*dbvdq0.shape[:-2], n, 7,),-1,-2,)