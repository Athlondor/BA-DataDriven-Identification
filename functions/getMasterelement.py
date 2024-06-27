# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for determining the masterelement for the chosen types of shape
# functions and integration scheme
# ========================================================================

import numpy as np

from Subroutines.gauss2d        import gauss2d
from Subroutines.shape2d        import shape2d


# Set up masterelement
def getMasterelement(nen, ndm, nqp):
    
    # Define a class for storing Gauss point related data
    class mstrElmnt():
        def __init__(self):
            N     = np.zeros(nen)
            gamma = np.zeros((nen,ndm))
            w8    = np.zeros(nqp)
    
    # Initialize array for the Gauss points
    masterElement = [mstrElmnt() for q in range(nqp)]
    
    # Determine Gauss points and weights for combination of nqp, ndm
    [xi, w8] = gauss2d(nqp, ndm)
    
    # Fill masterElement for Gauss points
    for q in range(nqp):
        # Shape functions and derivatives
        [masterElement[q].N, masterElement[q].gamma] = shape2d(xi[:,q], nen, ndm)
        
        # Weighting factor
        masterElement[q].w8 = w8[q]
    
    
    
    return masterElement