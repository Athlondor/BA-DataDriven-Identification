# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Collection of external routines
# ========================================================================

import numpy as np

# Convert tensor to Kelvin notation
def kelvin(T):

    # Input checking
    assert isinstance(T, np.ndarray), "Tensor T must be a numpy array."
    assert T.shape in ((3,3), (3,3,3,3)), \
        "T must be a 2nd  or 4th order tensor in 3d."
    assert checkSymmetric(T), "Tensor T must be symmetric."
    
    # I: Index in Kelvin notation, i,j: indices in standard notation,
    # c: factor associsated with the index pair i,j
    #       I  i  j  c
    ind = ((0, 0, 0, 1.0),
           (1, 1, 1, 1.0),
           (2, 2, 2, 1.0),
           (3, 0, 1, np.sqrt(2.0)),
           (4, 1, 2, np.sqrt(2.0)),
           (5, 0, 2, np.sqrt(2.0)))
    
    # Conversion
    if T.shape == (3,3):
        # T is 2nd order
        TKel = np.zeros(6)
        for i, k, l, c in ind:
            TKel[i] = c*T[k,l]
        
    elif T.shape == (3,3,3,3):
        # T is 4th order
        TKel = np.zeros((6,6))
        for i, k, l, c in ind:
            for j, m, n, d in ind:
                TKel[i,j] = c*d*T[k,l,m,n]
    
    return TKel


# Convert Kelvin to regular tensor notation
def unkelvin(TKel):
    
    assert isinstance(TKel, np.ndarray), "Tensor T must be a numpy array."
    assert TKel.shape in ((6,), (6,6)), \
        "TKel must be a 6x1 vector or a 6x6 matrix."
    # I: Index in Kelvin notation, i,j: indices in standard notation,
    # c: factor associsated with the index pair i,j
    #       I  i  j  c
    ind = ((0, 0, 0, 1.0),
           (1, 1, 1, 1.0),
           (2, 2, 2, 1.0),
           (3, 0, 1, np.sqrt(0.5)),
           (4, 1, 2, np.sqrt(0.5)),
           (5, 0, 2, np.sqrt(0.5)))
    
    # Conversion
    if TKel.shape == (6,):
        # T is 2nd order
        T = np.zeros((3,3))
        for i, k, l, c in ind:
            T[k,l] = T[l,k] = c*TKel[i]
        
    elif TKel.shape == (6,6):
        # T is 4th order
        T = np.zeros((3,3,3,3))
        for i, k, l, c in ind:
            for j, m, n, d in ind:
                T[k,l,m,n] = T[l,k,m,n] = T[k,l,n,m] = T[l,k,n,m] = \
                    c*d*TKel[i,j]
    
    return T


# Compute 4th order identity tensors
def identityTensors(tdm):
    I = np.identity(tdm)
    IdyI = np.zeros((tdm,tdm,tdm,tdm))
    Isym = np.zeros((tdm,tdm,tdm,tdm))
    
    for i in range(tdm):
        for j in range(tdm):
            for k in range(tdm):
                for l in range(tdm):
                    IdyI[i,j,k,l] = I[i,j]*I[k,l]
                    Isym[i,j,k,l] = 0.5*(I[i,k]*I[j,l] + I[i,l]*I[j,k])

    return (IdyI, Isym)


# Compute dyadic product of two second order tensors
def dyad(a,b):
    """Dyadic product of two 2nd order tensors."""
    c = np.zeros((3,3,3,3))
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    c[i,j,k,l] = a[i,j]*b[k,l]
                    
    return c

# Compute odyadic product of two second order tensors
def odyad(a,b,ndm):
    """Dyadic product of two 2nd order tensors."""
    c = np.zeros((ndm,ndm,ndm,ndm))
    for i in range(ndm):
        for j in range(ndm):
            for k in range(ndm):
                for l in range(ndm):
                    c[i,j,k,l] = a[i,k]*b[j,l]
                    
    return c

# Check if a tensor `A` is numerically symmetric.
def checkSymmetric(A, rtol=1e-05, atol=1e-12):
    
    assert isinstance(A, np.ndarray), "A must be a numpy array."
    assert A.shape in ((3,3), (3,3,3,3)), "A must be of 2nd or 4th order."
    
    if A.shape == (3,3):
        # 2nd order tensors / matrices
        return  np.allclose(A, A.T, rtol=rtol, atol=atol)
    
    elif A.shape == (3,3,3,3):
        # 4th order tensors
        
        # Check for the minor symmetries in indices ij and kl, respectively
        symij = np.allclose(A, np.swapaxes(A, 0, 1), rtol=rtol, atol=atol)
        symkl = np.allclose(A, np.swapaxes(A, 2, 3), rtol=rtol, atol=atol)
        
        return (symij and symkl)




