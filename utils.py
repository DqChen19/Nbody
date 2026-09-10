from __future__ import annotations

import jax.numpy as jnp
import jax

ITMAX = 100

# ================================ Compensated Sum =================================#
@jax.jit
def comp_sum(sum_value, sum_error, addend):
    """
    Kahan (1965) compensated summation for a single float.

    INPUT:
    sum_value : Current value of the sum.
    sum_error : Error from truncation/rounding accumulated from prior steps.
    addend    : New value to be added to the sum.

    RETURN:
    (sum_value, sum_error) updated.
    """
    new_error = sum_error + addend
    new_sum = sum_value + new_error
    new_error = new_error + sum_value - new_sum
    return new_sum, new_error

@jax.jit
def comp_sum_matrix(sum_value, sum_error, addend):
    """
    Kahan (1965) compensated summation for arrays.

    Mutates `sum_value` and `sum_error` in place (mirrors the Julia `!`
    naming convention for `comp_sum_matrix!`), element-wise, vectorized
    with numpy instead of the explicit `@inbounds for` loop used in Julia.

    INPUT:
    sum_value : Current value of the sum. Modified in place.
    sum_error : Error accumulated from prior steps. Modified in place.
    addend    : New values to be added to the sum.
    """
    corrected = sum_error + addend
    new_sum = sum_value + corrected
    new_error = corrected + sum_value - new_sum
    return new_sum, new_error


# ================================ Integrator Utilities ==================================#
# NOTE: `state.py` doesn't define `State` yet, so `s`/`d` below are duck-typed:
# `s` is expected to expose `.jac_step`, `.jac_error`, `.dqdt` (each a
# (7n,7n)/(7n,7n)/(7n,) array), and `d` a `Derivatives`-like object
# (see pre_alloc_arrays.py) exposing `.jac_tmp1`, `.jac_err1`, `.dqdt_tmp1`,
# `.dqdt_ij` ((14,7n)/(14,7n)/(14,)/(14,) arrays).
def copy_submatrix(s, d, indi, indj, sevn):
    """
    Copies the rows of the Jacobian (and current time derivatives)
    belonging to bodies `i` (rows indi:indi+7) and `j` (rows indj:indj+7)
    out of the global (7n,7n) Jacobian `s.jac_step`/`s.jac_error` into the
    (14, sevn) scratch buffers `d.jac_tmp1`/`d.jac_err1`, so a local 14x14
    (or 14xsevn) computation can be done without touching the full matrix.
    """
    jac_tmp1 = d.jac_tmp1.at[0:7, :sevn].set(s.jac_step[indi:indi + 7, :sevn])
    jac_tmp1 = jac_tmp1.at[7:14, :sevn].set(s.jac_step[indj:indj + 7, :sevn])

    jac_err1 = d.jac_err1.at[0:7, :sevn].set(s.jac_error[indi:indi + 7, :sevn])
    jac_err1 = jac_err1.at[7:14, :sevn].set(s.jac_error[indj:indj + 7, :sevn])

    dqdt_tmp1 = d.dqdt_tmp1.at[0:7].set(s.dqdt[indi:indi + 7])
    dqdt_tmp1 = dqdt_tmp1.at[7:14].set(s.dqdt[indj:indj + 7])

    d.jac_tmp1 = jac_tmp1
    d.jac_err1 = jac_err1
    d.dqdt_tmp1 = dqdt_tmp1

def ypoc_submatrix(s, d, indi: int, indj: int, sevn: int) -> None:
    """
    Copies back: the inverse of `copy_submatrix`. Writes the (updated)
    14-row scratch buffers `d.jac_tmp1`/`d.jac_err1` back into the rows of
    bodies `i` and `j` in the global Jacobian `s.jac_step`/`s.jac_error`,
    and copies `d.dqdt_ij` (filled elsewhere, after `copy_submatrix`, by
    the local Kepler/drift step) back into `s.dqdt`.
    """
    jac_step = s.jac_step.at[indi:indi+7, :sevn].set(d.jac_tmp1[0:7, :sevn])
    jac_step = s.jac_step.at[indj:indj+7, :sevn].set(d.jac_tmp1[7:14, :sevn])

    jac_error = s.jac_error.at[indi:indi+7, :sevn].set(d.jac_err1[0:7, :sevn])
    jac_error = s.jac_error.at[indj:indj+7, :sevn].set(d.jac_err1[7:14, :sevn])

    dqdt = s.dqdt.at[indi:indi + 7, :sevn].set(d.dqdt_ij[0:7])
    dqdt = s.dqdt.at[indj:indj + 7, :sevn].set(d.dqdt_ij[7:14])
    s.jac_step = jac_step
    s.jac_error = jac_error
    s.dqdt = dqdt######

@jax.jit
def G3(gamma, beta, sqb, gc = 0.5):
    def use_series(_):
       return G3_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (gamma - jnp.sin(gamma)) / (sqb * beta),
            lambda _: (gamma - jnp.sinh(gamma)) / (sqb * beta),
            operand = None
        )
    return jax.lax.cond( gamma < gc, use_series, use_closed_series, operand = None)    

@jax.jit
def G3_series(gamma, beta, sqb):
    # Computes G_3(beta,s) using a series tailored to the precision of s.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(1.0, dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
        jnp.asarray(2.0, dtype=dtype),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )
    def cond_fun(state):
        _, g3, g31, g32, _, it = state
        return ((it < ITMAX) & (g3 != g32) & (g3 != g31))
    
    def body_fun(state):
        term, g3, g31, _, n, it = state

        new_g32 = g31
        new_g31 = g3
        new_n = n + 1

        new_term = (term * x2 / ((2 * new_n + 3) * (2 * new_n + 2)))
        new_g3 = g3 + new_term

        return (new_term,new_g3,new_g31,new_g32,new_n,it + 1,)

    _, g3, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state) #condition fn, body fn, and init value

    return g3 * (-x2 * gamma) / (6.0 * beta * sqb)

@jax.jit
def H1(gamma, beta, gc = 0.5) -> float:
    def use_series(_):
       return H1_series(gamma, beta)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (4 * jnp.sin(0.5 * gamma) ** 2 - gamma * jnp.sin(gamma)) / beta ** 2,
            lambda _: (-4 * jnp.sinh(0.5 * gamma) ** 2 + gamma * jnp.sinh(gamma)) / beta ** 2,
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)    

@jax.jit
def H1_series(gamma, beta):
    # Computes H_1(beta,s) using a series tailored to the precision of s.
    dtype = jnp.result_type(gamma, beta)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0, dtype=dtype), # term
        jnp.asarray(1.0, dtype=dtype), # h1
        jnp.asarray(2.0, dtype=dtype), # h11
        jnp.asarray(2.0, dtype=dtype), # h12
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
    )

    def cond_fun(state):
        _, h1, h11, h12, _, it = state
        return ((it < ITMAX) & (h1 != h11) & (h1 != h12))

    def body_fun(state):
        term, h1, h11, _, n, it = state
        new_h12 = h11
        new_h11 = h1
        new_n = n + 1
        new_term = term * x2 * (new_n + 1) / ((2 * new_n + 4) * (2 * new_n + 3) * new_n)
        new_h1 = h1 + new_term
        return (new_term, new_h1, new_h11, new_h12, new_n, it + 1, )
    
    _, h1, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state) #condition fn, body fn, and init value
    return h1 * x2 ** 2 / (12 * beta ** 2)

@jax.jit
def H2(gamma, beta, sqb, gc = 0.5):
    def use_series(_):
        return H2_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (jnp.sin(gamma) - gamma * jnp.cos(gamma)) / (sqb * beta),
            lambda _: (jnp.sinh(gamma) - gamma * jnp.cosh(gamma)) / (sqb * beta),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series,use_closed_series, operand = None)

@jax.jit
def H2_series(gamma, beta, sqb):
    # Computes H_2(beta,s) using a series tailored to the precision of s.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0, dtype=dtype), # term
        jnp.asarray(1.0, dtype=dtype), # h2
        jnp.asarray(2.0, dtype=dtype), # h21
        jnp.asarray(2.0, dtype=dtype), # h22
        jnp.asarray(0, dtype=jnp.int32), # n
        jnp.asarray(0, dtype=jnp.int32), # it
    )

    def cond_fun(state):
        _, h2, h21, h22, _, it = state
        return ((it < ITMAX) & (h2 != h21) & (h2 != h22))
    
    def body_fun(state):
        term, h2, h21, _, n, it = state
        new_h22 = h21
        new_h21 = h2
        new_n = n + 1
        new_term = term * x2 / ((4 * new_n + 6) * new_n)
        new_h2 = h2 + new_term
        return (new_term, new_h2, new_h21, new_h22, new_n, it + 1, )
    
    
    _, h2, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state) #condition fn, body fn, and init value
    return h2 * (-x2) * gamma / (3 * beta * sqb)

@jax.jit
def H3(gamma, beta, sqb, gc = 0.5):
    # This is H_3 = G_1 G_2 - 3 G_3
    def use_series(_):
       return H3_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (4 * jnp.sin(gamma) - jnp.sin(gamma) * jnp.cos(gamma) - 3 * gamma) / (beta * sqb),
            lambda _: (4 * jnp.sinh(gamma) - jnp.sinh(gamma) * jnp.cosh(gamma) - 3 * gamma) / (beta * sqb),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)


@jax.jit
def H3_series(gamma, beta, sqb):
    # Computes H_3(beta,s) using a series tailored to the precision of gamma.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0 / 30.0, dtype=dtype), # term
        jnp.asarray(0.1, dtype=dtype), # h3
        jnp.asarray(0.2, dtype=dtype), # h31
        jnp.asarray(0.2, dtype=dtype), # h32
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(4.0, dtype=dtype),# four2n
    )
    def cond_fun(state):
        _, h3, h31, h32, _, it, _ = state
        return ((it < ITMAX) & (h3 != h31) & (h3 != h32))
    
    def body_fun(state):
        term, h3, h31, _, n, it, four2n= state
        new_h32 = h31
        new_h31 = h3
        new_n = n + 1
        new_term = term * x2 / ((2 * new_n + 4) * (2 * new_n + 5))
        new_four2n = 4 * four2n
        new_h3 = h3 + new_term * (new_four2n - 1)
        return (new_term, new_h3, new_h31, new_h32, new_n, it + 1, new_four2n)
    
    _, h3, _, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state) #condition fn, body fn, and init value
    return h3 * (-x2** 2) * gamma / (beta * sqb)

@jax.jit
def H5(gamma, beta, sqb, gc = 0.5):
    # This is G_1 G_2 - (2+G_0) G_3 = H_2 - 2 G_3
    def use_series(_):
       return H5_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (3 * jnp.sin(gamma) - 2 * gamma - gamma * jnp.cos(gamma)) / (beta * sqb),
            lambda _: (3 * jnp.sinh(gamma) - 2 * gamma - gamma * jnp.cosh(gamma)) / (beta * sqb),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)    

@jax.jit
def H5_series(gamma, beta, sqb):
    # Computes H_5(beta,s) using a series tailored to the precision of gamma.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0 / 60.0, dtype=dtype), # term
        jnp.asarray(1.0 / 60.0, dtype=dtype), # h5
        jnp.asarray(2 * 1.0 / 60.0, dtype=dtype), # h51
        jnp.asarray(2 * 1.0 / 60.0, dtype=dtype), # h52
        jnp.asarray(0, dtype=jnp.int32), #n
        jnp.asarray(0, dtype=jnp.int32), #it
    )

    def cond_fun(state):
        _, h5, h51, h52, _, it = state
        return ((it < ITMAX) & (h5 != h51) & (h5 != h52))
    
    def body_fun(state):
        term, h5, h51, _, n, it = state
        new_h52 = h51
        new_h51 = h5
        new_n = n + 1
        new_term = term * x2 * (new_n + 1) / ((2 * new_n + 5) * (2 * new_n + 4) * new_n)
        new_h5 = h5 + new_term
        return (new_term, new_h5, new_h51, new_h52, new_n, it + 1, )
    _, h5, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state) #condition fn, body fn, and init value
    
    return h5 * (-x2 ** 2) * gamma / (beta * sqb)

@jax.jit
def H6(gamma, beta, gc = 0.5) -> float:
    # This is 2 G_2^2 - 3 G_1 G_3
    def use_series(_):
       return H6_series(gamma, beta)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (9 - 8 * jnp.cos(gamma) - jnp.cos(2 * gamma) - 6 * gamma * jnp.sin(gamma)) / (2 * beta ** 2),
            lambda _: (9 - 8 * jnp.cosh(gamma) - jnp.cosh(2 * gamma) + 6 * gamma * jnp.sinh(gamma)) / (2 * beta ** 2),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)    


@jax.jit
def H6_series(gamma, beta):
    # Computes H_6(beta,s) using a series tailored to the precision of gamma.
    dtype = jnp.result_type(gamma, beta)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(1.0 / 360.0, dtype=dtype), # term
        jnp.asarray(1.0 / 40.0, dtype=dtype), # h6
        jnp.asarray(2.0 / 40.0, dtype=dtype), # h61
        jnp.asarray(2.0 / 40.0, dtype=dtype), # h62
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(16.0, dtype=dtype) #four2n
    )
    def cond_fun(state):
        _, h6, h61, h62, _, it, _ = state
        return ((it < ITMAX) & (h6 != h61) & (h6 != h62))
    
    def body_fun(state):
        term, h6, h61, _, n, it, four2n = state
        new_h62 = h61
        new_h61 = h6
        new_n = n + 1
        new_term = term * x2 / ((2 * new_n + 5) * (2 * new_n + 6))
        new_four2n = 4 * four2n
        new_h6 = h6 + new_term * (new_four2n - 3 * new_n - 7)
        return (new_term, new_h6, new_h61, new_h62, new_n, it + 1, new_four2n)
    
    _, h6, _, _, _, _, _ = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return h6 * (-x2)** 3 / (beta ** 2)

@jax.jit
def H7(gamma, beta, sqb, gc = 0.5):
    # This is H_7 = G_1 G_2 (1 - 2 G_0) + 3 G_0^2 G_3
    def use_series(_):
       return H7_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (3 * jnp.cos(gamma) * (gamma * jnp.cos(gamma) - jnp.sin(gamma)) + jnp.sin(gamma) ** 3) / (beta * sqb),
            lambda _: (3 * jnp.cosh(gamma) * (gamma * jnp.cosh(gamma) - jnp.sinh(gamma)) - jnp.sinh(gamma) ** 3) / (beta * sqb),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)    

@jax.jit
def H7_series(gamma: float, beta: float, sqb: float) -> float:
    # Computes H_7(beta,s) using a series tailored to the precision of gamma.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2
    init_state = (
        jnp.asarray(-(3.0 / 20160.0) * x2, dtype=dtype), # term
        jnp.asarray(1.0 / 10.0 - (11.0 / 840.0) * x2, dtype=dtype), # h7
        jnp.asarray((1.0 / 10.0 - (11.0 / 840.0) * x2) * 2.0, dtype=dtype), # h71
        jnp.asarray((1.0 / 10.0 - (11.0 / 840.0) * x2) * 2.0, dtype=dtype), # h72
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(128.0, dtype=dtype), # four2n
        jnp.asarray(729.0, dtype=dtype), # nine2n
    )
    def cond_fun(state):
        _, h7, h71, h72, _, it, _, _ = state
        return ((it < ITMAX) & (h7 != h71) & (h7 != h72))
    
    def body_fun(state):
        term, h7, h71, _, n, it, four2n, nine2n = state
        new_h72 = h71
        new_h71 = h7
        new_n = n + 1
        new_term = term * x2 /((2 * new_n + 6) * (2 * new_n + 7))
        new_four2n = 4 * four2n
        new_nine2n = 9 * nine2n
        new_h7 = h7 + new_term * (new_nine2n - 1 - (5 + 2 * new_n) * new_four2n)
        return (new_term, new_h7, new_h71, new_h72, new_n, it + 1, new_four2n, new_nine2n)
    
    _, h7, _, _, _, _, _,_  = jax.lax.while_loop(cond_fun,body_fun,init_state)
    return h7 * x2 ** 2 * gamma / (beta * sqb)

@jax.jit
def H8(gamma: float, beta: float, sqb: float, gc: float = 0.5) -> float:
    # This is H_8 = G_1 G_2 - 3 G_0 G_3
    def use_series(_):
       return H8_series(gamma, beta, sqb)
    def use_closed_series(_):
        return jax.lax.cond(
            beta >= 0,
            lambda _: (-3 * gamma * jnp.cos(gamma) + jnp.sin(gamma) + jnp.sin(2 * gamma)) / (beta * sqb),
            lambda _: (-3 * gamma * jnp.cosh(gamma) + jnp.sinh(gamma) + jnp.sinh(2 * gamma)) / (beta * sqb),
            operand = None
        )
    return jax.lax.cond(gamma < gc, use_series, use_closed_series, operand = None)    

@jax.jit
def H8_series(gamma, beta, sqb):
    # Computes H_8(beta,s) using a series tailored to the precision of gamma.
    dtype = jnp.result_type(gamma, beta, sqb)
    x2 = -jnp.sign(beta) * gamma ** 2

    init_state = (
        jnp.asarray(1.0 / 120.0, dtype=dtype), # term
        jnp.asarray(3.0 / 20.0, dtype=dtype), # h8
        jnp.asarray(2 * 3.0 / 20.0, dtype=dtype), # h81
        jnp.asarray(2 * 3.0 / 20.0, dtype=dtype), # h82
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(0, dtype=jnp.int32),
        jnp.asarray(32.0, dtype=dtype), # four2n
    )
    def cond_fun(state):
        _, h8, h81, h82, _, it, _ = state
        return ((it < ITMAX) & (h8 != h81) & (h8 != h82))
    
    def body_fun(state):
        term, h8, h81, _, n, it, four2n = state
        new_h82 = h81
        new_h81 = h8
        new_n = n + 1
        new_term = term * x2 /((2 * new_n + 4) * (2 * new_n + 5))
        new_four2n = 4 * four2n
        new_h8 = h8 + new_term * (new_four2n - 14 - 6 * new_n)
        return (new_term, new_h8, new_h81, new_h82, new_n, it+1, new_four2n)
    
    _, h8, _, _, _, _, _ = jax.lax.while_loop(cond_fun,body_fun,init_state)
    return h8 *  x2 ** 2 * gamma / (beta * sqb)


# ================================ Computation fast =================================#
@jax.jit
def dot_fast(a, b=None):
    if b is None:
        return jnp.dot(a, a)
    return jnp.dot(a, b)



@jax.jit
def cubic1(a, b, c):
    """Analytic solver for a depressed cubic x^3 + a*x^2 + b*x + c = 0."""
    a3 = a * (1.0/3.0)
    Q = a3 ** 2 - b * (1.0/3.0)
    R = a3 ** 3 + 0.5 * (-a3 * b + c)
    R2 = R ** 2
    Q3 = Q ** 3
    
    def small_discriminant(_):
        return -c / b
    def large_discriminant(_):
        A = -jnp.sign(R) * jnp.cbrt(jnp.abs(R) + jnp.sqrt(R2 - Q3))
        B = jnp.where(A == 0.0, 0.0, Q / A)
        return A + B - a3
    return jax.lax.cond(R2 < Q3, small_discriminant, large_discriminant, operand = None)
