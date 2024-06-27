# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for calculating the volume average of SDVs regarding all nqp
# Gauss points
# ========================================================================

import numpy as np

# Volume average of state variables
def computeAverage(elem):
    
    # Determine constants
    nel = len(elem)
    [nqp,nsv] = elem[0].stateVar.shape
    
    
    # Store data
    class clldt():
        def __init__(self, nsv):
            self.stateVar = np.zeros(nsv)
    
    celldata = [clldt(nsv) for i in range(nel)]
    
    Vlist = 0
    # Calculate average sdv values
    for e in range(nel):
        
        # Average
        V = 0
        for q in range(nqp):
            V += elem[e].dV[q]
            celldata[e].stateVar += elem[e].stateVar[q,:] * elem[e].dV[q]
        celldata[e].stateVar = celldata[e].stateVar / V
        
        Vlist += V
    
    
    
    return celldata, Vlist