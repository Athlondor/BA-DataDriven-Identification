# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for determining DOF-types based on input data
# ========================================================================

import numpy as np

def dofSeparation(nnp, ndf, drlt, neum):
    # separate dof types
    allDofs = np.arange(0, nnp*ndf)
    
    # dofs with Dirichlet boundary conditions
    numDrltDofs = drlt.shape[0]
    drltDofs = np.zeros(numDrltDofs, dtype=np.int32)
    
    for i in range(numDrltDofs):
        node = drlt[i,0]
        ldof = drlt[i,2]
        drltDofs[i] = (node-1)*ndf + ldof - 1
    
    
    # free dofs
    freeDofs = np.setdiff1d(allDofs, drltDofs)
    
    # dofs with Neumann boundary conditions
    numNeumDofs = neum.shape[0]
    neumDofs = np.zeros(numNeumDofs, dtype=np.int32)
    for i in range(numNeumDofs):
        node = neum[i,0]
        ldof = neum[i,2]
        neumDofs[i] = (node-1)*ndf + ldof - 1
    
    return drltDofs, freeDofs, neumDofs