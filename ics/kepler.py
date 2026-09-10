import jax
import jax.numpy as jnp
# ======================================
# Original Julia code adapted
# ======================================
def ekepler(m, ecc):
    '''
    Input : 
    m - Mean Anomaly [radians]
    ecc - eccentricity

    OUTPUT :
    Eccentric Anomaly E
    '''
    KEPLER_TOL  = 1e-8
    ITMAX = 20

    def zero_case(_):
        return jnp.array(0.0)
    
    def solve_case(_):
        ms = jnp.fmod(m, 2 * jnp.pi)
    
        # Initial guess:
        de0 = ecc * 0.85 * jnp.copysign(1.0, ms)
        de1 = 2 * de0
        #carry = (cur_de, pre_de, iter)
        init = (de0, de1, jnp.asarray(0, dtype=jnp.int32))

        def cond_fun(carry):
            current, previous, iter = carry
            not_converged = (jnp.abs(current - previous) > KEPLER_TOL)
            return (iter < ITMAX) & not_converged

        def body_fun(carry):
            current, _, iter = carry
            f3 = ecc * jnp.cos(current + ms)
            f2 = ecc * jnp.sin(current + ms)
            new_de = (f2 - current * f3) / (1 - f3)
            return new_de, current, iter + 1
        de_final, _, _ = jax.lax.while_loop(cond_fun,body_fun,init)

        return de_final + m

    return jax.lax.cond(m == 0.0, zero_case, solve_case, operand = None)

@jax.jit
def kepler(m, ecc) -> float:
    '''
    Input : 
    m - Mean Anomaly [radians]
    ecc - eccentricity

    OUTPUT :
    True Anomaly [radians]
    '''
    ekep = ekepler(m, ecc)
    return 2.0 * jnp.atan2(jnp.sqrt(1.0 + ecc) * jnp.sin(0.5 * ekep), jnp.sqrt(1.0 - ecc) * jnp.cos(0.5 * ekep))


if __name__ == '__main__':

    #test
    f = kepler(100, 0.21)
    print(f)