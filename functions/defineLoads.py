# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for determining loads/BCs as a function of time
# ========================================================================

import numpy as np

def defineLoads(time, drlt, neum, neumDofs, loadcurve, fsur):
    
    numNeumDofs = neum.shape[0]
    neumValue   = np.zeros(numNeumDofs)
    for i in range(numNeumDofs):
        lcID         = neum[i,1]-1
        scale        = neum[i,3] * loadcurve[int(lcID)].scalefactor
        neumValue[i] = scale*np.interp(time, loadcurve[int(lcID)].time, loadcurve[int(lcID)].value)
    fsur[neumDofs] = neumValue

    # Dirichlet boundary conditions
    numDrltDofs = drlt.shape[0]
    drltValue = np.zeros(numDrltDofs)
    for i in  range(numDrltDofs):
        lcID         = drlt[i,1]-1
        scale        = drlt[i,3] * loadcurve[int(lcID)].scalefactor
        drltValue[i] = scale*np.interp(time, loadcurve[int(lcID)].time, loadcurve[int(lcID)].value)
        
    return fsur, drltValue