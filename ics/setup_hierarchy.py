import numpy as np
import jax.numpy as jnp

def hierarchy(ic_vec):
    '''
    INPUT:
    ic_vec : array_like
    e.g [5,2,1,1]

    OUTPUT: hierarchy array matrix
    [[-1.  1.  0.  0.  0.]
    [ 0.  0. -1.  1.  0.]
    [-1. -1.  1.  1.  0.]
    [-1. -1. -1. -1.  1.]
    [-1. -1. -1. -1. -1.]]
    N = nbody
    Shape (N, N)
    Total # of binaries should = N - 1

    '''
    ic_vec = np.asarray(ic_vec, dtype = np.int64)
    if ic_vec.ndim != 1 or ic_vec.size < 2:
        raise ValueError("ic_vec must be a one-dimensional array of length >= 2")
    
    nbody  = int(ic_vec[0]) # num of body
    bins = ic_vec[1:].copy()

    level = len(bins) if bins[-1] == 1 else len(bins) - 1
    
    if bins[0] > nbody / 2 or bins[0] > 2**level:
        raise ValueError("Invalid hierarchy")
    if np.sum(bins[:level]) != nbody - 1:
        raise ValueError(
            "Invalid hierarchy: total number of binaries must equal nbody - 1"
        )
    h = np.zeros((nbody, nbody))
    _bottom_level(nbody, bins, h, iter = 1)
    h[-1,:] = -1
    hierarchy_jax = jnp.asarray(h)
    return hierarchy_jax

def _bottom_level(nbody, bins, h, iter):
    h[0, 0] = -1.0
    h[0, 1] = 1.0

    if bins[0] > 1:
        j = 1 if bins[-1] == 1 else int(nbody/2) - 1
        for i in range(1, bins[0]):
            if j + 3 <= nbody:
                h[i, j + 1] = -1
                h[i, j + 2] = 1
                j += 2
    
    binsp = int(bins[0])
    row = binsp + 1
    bodies = int(bins[0]) * 2

    if bins[-1] > 1:
        _symmetric_nest(nbody, bodies, bins[:-1], binsp, row, h, iter + 1)
    elif 2 * binsp == nbody:
        _symmetric(nbody, bodies, bins, binsp, row, h, iter + 1)
    else:
        _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)

def _nlevel(nbody, bodies, bins, binsp, row, h, iter):
    if iter > len(bins):
        return
    
    if nbody != bodies:
        if bins[iter - 1] == binsp:
            bodies = bodies +  bins[iter - 1]
        elif bins[iter - 1] > binsp:
            bodies = bodies + 2 * bins[iter - 1] - 1
    elif nbody == bodies and row == nbody - 1:
        if bodies % 4 > 2:
            h[row - 1, 0:row - 2] = -1
            h[row - 1, row - 2: bodies] = 1
        else:
            h[row - 1, 0:row - 1] = -1
            h[row - 1, row - 1: bodies] = 1
        return

    if nbody < bodies:
        return

    if binsp == 1:
        if bins[iter - 1] == 1:
            h[row - 1, 0:bodies - binsp] = -1
            h[row - 1, bodies - binsp: bodies] = 1
            row += binsp
            binsp = int(bins[iter - 1])
            _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)

        elif bins[iter - 1] == 2:
            h[row - 1, 0:bodies - 2*binsp - 1] = -1   # Julia: h[row, 1:bodies-(2*binsp)-1]
            h[row - 1, bodies - 2*binsp - 1]   = 1    # Julia: h[row, bodies-(2*binsp)]
            h[row,     bodies - 2*binsp]        = -1   # Julia: h[row+1, bodies-(2*binsp)+1]
            h[row,     bodies - 1]              = 1    # Julia: h[row+1, bodies]
            row  += 2
            binsp = int(bins[iter - 1])
            _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)
        
    elif binsp == 2:
        if bins[iter - 1] == 1:
            h[row - 1, 0:row - 1] = -1                # Julia: h[row, 1:row-1]
            h[row - 1, row - 1:bodies] = 1            # Julia: h[row, row:bodies]
            row  += 1
            binsp = int(bins[iter - 1])
            _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)

        elif bins[iter - 1] == 2:
            h[row - 1, 0:row - 1] = -1                         # Julia: h[row, 1:row-1]
            h[row - 1, row - 1:bodies - 2*(binsp-1)] = 1       # Julia: h[row, row:bodies-2*(binsp-1)]
            h[row,     bodies - 2*(binsp-1)]          = -1     # Julia: h[row+1, bodies-2*(binsp-1)+1]
            h[row,     bodies - 1]                    = 1      # Julia: h[row+1, bodies]
            row  += 2
            binsp = int(bins[iter - 1])
            _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)

    elif binsp == 3:
        if bins[iter - 1] == 2:
            h[row - 1, 0:bodies//2 - 1]         = -1  # Julia: h[row, 1:Int64(bodies/2)-1]
            h[row - 1, bodies//2 - 1:2*(bodies//3)] = 1  # Julia: h[row, Int64(bodies/2):2*Int64(bodies/3)]
            h[row,     2*(bodies//3):bodies]     = -1  # Julia: h[row+1, 2*Int64(bodies/3)+1:bodies]
            h[row,     bodies]                   = 1   # Julia: h[row+1, bodies+1]
            row   += 2
            binsp  = int(bins[iter - 1])
            bodies += 1
            _nlevel(nbody, bodies, bins, binsp, row, h, iter + 1)

def _symmetric(nbody: int, bodies: int, bins: np.ndarray,
               binsp: int, row: int, h: np.ndarray, iter: int):
    j = 0
    while row < nbody - 1:
        h[row - 1, j:j + 2]     = -1   # Julia: h[row, 1+j:2+j]
        h[row - 1, j + 2:j + 4] =  1   # Julia: h[row, j+3:j+4]
        j   += 4
        row += 1
    h[row - 1, 0:binsp]   = -1         # Julia: h[row, 1:binsp]
    h[row - 1, binsp:nbody] = 1        # Julia: h[row, binsp+1:nbody]

def _symmetric_nest(nbody: int, bodies: int, bins: np.ndarray,
                    binsp: int, row: int, h: np.ndarray, iter: int):
    
    hlf = nbody // 2
    j = 0
    while row < nbody - 1:
        h[row - 1, 0:2 + j]          = -1  # Julia: h[row, 1:2+j]
        h[row - 1, 2 + j]            =  1  # Julia: h[row, 3+j]
        h[row,     hlf:hlf + 2 + j]  = -1  # Julia: h[row+1, hlf+1:hlf+2+j]
        h[row,     hlf + 2 + j]      =  1  # Julia: h[row+1, hlf+3+j]
        j   += 1
        row += 2
    h[row - 1, 0:hlf]    = -1              # Julia: h[row, 1:hlf]
    h[row - 1, hlf:nbody] = 1             # Julia: h[row, hlf+1:nbody]


if __name__ == '__main__':
    H = 3 #[3,1,1]
    h = hierarchy(H)
    print(h.size, h.dtype, h)