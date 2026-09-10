import jax.numpy as jnp
from .initial_conditions import ElementsIC, CartesianIC
from .kepler_init import kepler_init

NDIM    = 3

def kronecker_delta(i, j): 
    ''' N * N Kronecker Delta function ''' 
    return 1.0 if i == j else 0.0

def kepcalc(ic: ElementsIC):
    '''
    Compute Kepler's problem for each pair of bodies in the system

    returns:
    rkepler: shape (n,3) matrix of initial position vectors for each keplerian
    rdotkepler: shape (n,3) matrix of intial velocity vectors for each keplerian
    jac_init: shape (7n, 7n) matrix Derivatives of the A-matrix and cartesian positions 
            and velocities with respect to the masses of each object.
    '''
    n = ic.nbody
    rkepler = jnp.zeros((n, NDIM), dtype = ic.m.dtype) #(n,3)
    rdotkepler = jnp.zeros((n, NDIM), dtype = ic.m.dtype) #(n,3)

    if ic.der:
        #For each oribit, 6 row for (x, y, z, vx, vy, vz) and 7 parameters for [P, t0, ecos_omega, esin_omega, I, Omega] + m
        jac_kepler  = jnp.zeros((6 * n, 7 * n), dtype = ic.m.dtype)  # sparse
        #jac_21 = jnp.zeros((7, 7), dtype = ic.m.dtype)
    #Compute Kepler's problem for each binary
    b = 0

    for i in range(n-1):
        ind = ic.epsilon[i,:] != 0 # pick pair at this hierarchy
        mu = jnp.sum(jnp.where(ind, ic.m, 0.0)) #mass of this oribit

        if not bool(ind[0]): #ind[0] = 0, check for a new binary
            b += 1

        elem_row = ic.elements[i+1+b, 1:]
        
        #Convert orbital elements to Cartesian position and velocity.
        r, rdot, jac21 = kepler_init(ic.t0, mu, elem_row, ic.der)

        rkepler = rkepler.at[i, :].set(r)
        rdotkepler = rdotkepler.at[i, :].set(rdot)

        if ic.der:
            jac_kepler = jac_kepler.at[i * 6: i * 6 + 6, (i + 1) * 7: (i + 1) * 7 + 6,].set(jac21[:6,:6])
            for j in range(n):
                if bool(ic.epsilon[i, j] != 0):
                    jac_kepler = jac_kepler.at[ i * 6: i * 6 + 6, j * 7 + 6].set(jac21[:6, 6])
        if b > 0:
            b -= 2
        elif b < 0:
            b = 0
    
    if ic.der:
        jac_init = d_dm(ic, rkepler, rdotkepler, jac_kepler)
        return rkepler, rdotkepler, jac_init
    else:
        return rkepler, rdotkepler, jnp.nan
    

def d_dm(ic, rkepler, rdotkepler, jac_kepler):
    '''
    Computes derivatives of A-matrix, position, and velocity with respect to the masses of each object
    INPUT:
    ic: elements
    rkepler:  (n, 3)
    rdotkepler: (n, 3)
    jac_kepler: (6n, 7n)

    RETURN 
    Jacobian matrix for initial conditions
    jac_init shape (7n, 7n)
    '''
    N = ic.nbody
    m = ic.m
    epsilon = ic.epsilon
    dtype = m.dtype

    jac_init = jnp.zeros((7 * N, 7 * N), dtype = dtype)
    dAdm = jnp.zeros((N, N, N), dtype = dtype)
    Ainv = jnp.linalg.inv(ic.amat)

    for k in range(N):
        for i in range(N):
            for j in range(N):
                sm = sigma_m_init(m, i, j, epsilon)
                if sm!= 0.0:
                    # Differentiate A matrix wrt the mass of each body
                    dAdm = dAdm.at[i, j, k].set((kronecker_delta(k, j) * epsilon[i, j] / sm 
                                                    - kronecker_delta(epsilon[i,j], epsilon[i,k]) * epsilon[i,j] * m[j]/ (sm **2)))
    
    dAinvdm = jnp.zeros((N, N, N), dtype = dtype)

    for k in range(N):
        derivative = -Ainv @ dAdm[:, :, k] @ Ainv
        dAinvdm = dAinvdm.at[:,:,k].set(derivative)
    
    # Contributions propagated through the Kepler Jacobian.
    for i in range(N):
        for k in range(N):
            jac_init = jac_init.at[i * 7: i * 7 + 3,:].add(Ainv[i, k] * jac_kepler[k * 6: k * 6 + 3, :]) #postion 
            jac_init = jac_init.at[i * 7 + 3: i * 7 + 6, :].add(Ainv[i, k] * jac_kepler[k * 6 + 3: k * 6 + 6, :]) #velcity
    
    # Contributions from derivative of A^{-1}.
    for i in range(N):
        for k in range(N):
            # (N, N) @ (N, 3) -> (N, 3) -> .T -> (3, N)
            dxdm_mat = (dAinvdm[:, :, k] @ rkepler).T
            dvdm_mat = (dAinvdm[:, :, k] @ rdotkepler).T

            jac_init = jac_init.at[i * 7 : i * 7 + 3, k * 7 + 6].add(dxdm_mat[:, i])
            jac_init = jac_init.at[i * 7 + 3 : i * 7 + 6, k * 7 + 6].add(dvdm_mat[:, i])

        jac_init = jac_init.at[i * 7 + 6, i * 7 + 6].set(1.0) 
    
    return jac_init

#not necessary to be jitted
def init_nbody(ic):
    '''
    Convert initial orbital elements into Cartesian coordinates
    
    Outputs:
    x: shape (3, n) position
    v: shape (3, n) velocity
    jac_init (7n, 7n) Jacobian matrix
    '''
    if isinstance(ic, ElementsIC):
        return _init_nbody_elements(ic)
    elif isinstance(ic, CartesianIC):
        return _init_nbody_cartesian(ic)
    else:
        raise TypeError('Not support this type!!')
    
def _init_nbody_elements(ic):
    rkepler, rdotkepler, jac_init = kepcalc(ic)
    Ainv = jnp.linalg.inv(ic.amat)

    x = (Ainv @ rkepler).T
    v = (Ainv @ rdotkepler).T

    return x, v, jac_init

def _init_nbody_cartesian(ic):
    x = jnp.asarray(ic.x)
    v = jnp.asarray(ic.v)
    n = x.shape[1]
    jac_init = jnp.eye(7 * n, dtype=x.dtype) 

    return x, v, jac_init

def sigma_m_init(mass, i, j, epsilon):
    '''
    Sums masses in current Keplerian system
    epsilon - from Jacobi coordinates, copy from hierarchy matrix
    e.g.
    [[-1.  1.  0.]
    [-1. -1.  1.]
    [-1. -1. -1.]]

    '''
    mask = epsilon[i, :] == epsilon[i,j]
    return jnp.sum(mass[mask])