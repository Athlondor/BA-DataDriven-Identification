# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for computing the Jacobian matrix
# ========================================================================

import numpy as np

# Subroutine : Jacobian 'matrix'
def jacobian2d(xe,gamma,nen,ndm):

    # Jacobian matrix at the quadrature points q
    # Hint : For 2d-case the Jacobian is a 2x2 matrix
    # Jq = xe * gamma
    
    Jq = np.zeros((ndm,ndm))
    for i in range (ndm):
        for j in range (ndm):
            for A in range (nen):
                Jq[i,j] += xe[i,A]*gamma[A,j]
                
    # Determinant of Jacobian
    detJq = np.linalg.det(Jq)
    
    # Inverse of Jacobian
    invJq = np.linalg.inv(Jq)
    return detJq,invJq