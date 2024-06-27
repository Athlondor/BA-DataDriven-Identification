# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for defining the Gauss integration points
# ========================================================================

import numpy as np

# Subroutine : Gauss quadrature points and weights
def gauss2d(nqp,ndm):

    # Initialization
    xi = np.zeros((ndm,nqp))
    w8 = np.zeros(nqp)
    
    
    # 4 Gauss points for quadrilaterals
    if ((nqp == 4) and (ndm == 2)):
        xi[0,0] = -np.sqrt(1.0/3.0)
        xi[1,0] = -np.sqrt(1.0/3.0)
        xi[0,1] =  np.sqrt(1.0/3.0)
        xi[1,1] = -np.sqrt(1.0/3.0)
        xi[0,2] = -np.sqrt(1.0/3.0)
        xi[1,2] =  np.sqrt(1.0/3.0)
        xi[0,3] =  np.sqrt(1.0/3.0)
        xi[1,3] =  np.sqrt(1.0/3.0)

        w8[0] = 1
        w8[1] = 1
        w8[2] = 1
        w8[3] = 1
    
    
    # 9 Gauss points for quadrilaterals
    elif ((nqp == 9) and (ndm == 2)):
        xi[0,0] = -np.sqrt(3.0/5.0)
        xi[1,0] = -np.sqrt(3.0/5.0)
        xi[0,1] =  0.0
        xi[1,1] = -np.sqrt(3.0/5.0)
        xi[0,2] =  np.sqrt(3.0/5.0)
        xi[1,2] = -np.sqrt(3.0/5.0)
        xi[0,3] = -np.sqrt(3.0/5.0)
        xi[1,3] =  0.0
        xi[0,4] =  0.0
        xi[1,4] =  0.0
        xi[0,5] =  np.sqrt(3.0/5.0)
        xi[1,5] =  0.0
        xi[0,6] = -np.sqrt(3.0/5.0)
        xi[1,6] =  np.sqrt(3.0/5.0)
        xi[0,7] =  0.0
        xi[1,7] =  np.sqrt(3.0/5.0)
        xi[0,8] =  np.sqrt(3.0/5.0)
        xi[1,8] =  np.sqrt(3.0/5.0)
        
        w8[0] = 25.0/81.0
        w8[1] = 40.0/81.0
        w8[2] = 25.0/81.0
        w8[3] = 40.0/81.0
        w8[4] = 64.0/81.0
        w8[5] = 40.0/81.0
        w8[6] = 25.0/81.0
        w8[7] = 40.0/81.0
        w8[8] = 25.0/81.0
    
    
    else:
        raise BaseException('gauss2d: Unknown combination of nqp, ndm! STOP...!!!')

    return xi,w8