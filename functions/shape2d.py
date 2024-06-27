# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for defining the shape functions and evaluating those at the 
# Gauss point xi
# ========================================================================

import numpy as np

# Subroutine : Shape functions N and their derivatives gamma
def shape2d(xi,nen,ndm):

    # Initialization
    N = np.zeros((nen,1))
    gamma = np.zeros((nen,ndm))
    
    
    # 4 node quadrilateral
    if ((nen == 4) and (ndm == 2)):
        N[0] = 0.25*(1-xi[0])*(1-xi[1])
        N[1] = 0.25*(1+xi[0])*(1-xi[1])
        N[2] = 0.25*(1+xi[0])*(1+xi[1])
        N[3] = 0.25*(1-xi[0])*(1+xi[1])

        gamma[0,0] = -0.25*(1-xi[1])
        gamma[0,1] = -0.25*(1-xi[0])
        gamma[1,0] =  0.25*(1-xi[1])
        gamma[1,1] = -0.25*(1+xi[0])
        gamma[2,0] =  0.25*(1+xi[1])
        gamma[2,1] =  0.25*(1+xi[0])
        gamma[3,0] = -0.25*(1+xi[1])
        gamma[3,1] =  0.25*(1-xi[0])
    
    
    # 8 node serendipity quadrilateral
    elif ((nen == 8) and (ndm == 2)):
        N[4] = 0.5*(1-xi[0]**2)*(1-xi[1])
        N[5] = 0.5*(1-xi[1]**2)*(1+xi[0])
        N[6] = 0.5*(1-xi[0]**2)*(1+xi[1])
        N[7] = 0.5*(1-xi[1]**2)*(1-xi[0])
        
        N[0] = 0.25*(1-xi[0])*(1-xi[1]) - 0.5*(N[4] + N[7])
        N[1] = 0.25*(1+xi[0])*(1-xi[1]) - 0.5*(N[4] + N[5])
        N[2] = 0.25*(1+xi[0])*(1+xi[1]) - 0.5*(N[5] + N[6])
        N[3] = 0.25*(1-xi[0])*(1+xi[1]) - 0.5*(N[6] + N[7])
        
        gamma[4,0] = -xi[0]*(1-xi[1])
        gamma[4,1] = -0.5*(1-xi[0]**2)
        gamma[5,0] =  0.5*(1-xi[1]**2)
        gamma[5,1] = -xi[1]*(1+xi[0])
        gamma[6,0] = -xi[0]*(1+xi[1])
        gamma[6,1] =  0.5*(1-xi[0]**2)
        gamma[7,0] = -0.5*(1-xi[1]**2)
        gamma[7,1] = -xi[1]*(1-xi[0])
        
        gamma[0,0] = -0.25*(1-xi[1]) - 0.5*(gamma[4,0] + gamma[7,0])
        gamma[0,1] = -0.25*(1-xi[0]) - 0.5*(gamma[4,1] + gamma[7,1])
        
        gamma[1,0] =  0.25*(1-xi[1]) - 0.5*(gamma[4,0] + gamma[5,0])
        gamma[1,1] = -0.25*(1+xi[0]) - 0.5*(gamma[4,1] + gamma[5,1])
        
        gamma[2,0] =  0.25*(1+xi[1]) - 0.5*(gamma[5,0] + gamma[6,0])
        gamma[2,1] =  0.25*(1+xi[0]) - 0.5*(gamma[5,1] + gamma[6,1])
        
        gamma[3,0] = -0.25*(1+xi[1]) - 0.5*(gamma[6,0] + gamma[7,0])
        gamma[3,1] =  0.25*(1-xi[0]) - 0.5*(gamma[6,1] + gamma[7,1])
    
    
    # 9 node quadrilateral 
    elif ((nen == 9) and (ndm == 2)):
        
        N[0] = (xi[0]-1)*(xi[1]-1)*xi[0]*xi[1]/4.0
        N[1] = (xi[0]+1)*(xi[1]-1)*xi[0]*xi[1]/4.0
        N[2] = (xi[0]+1)*(xi[1]+1)*xi[0]*xi[1]/4.0
        N[3] = (xi[0]-1)*(xi[1]+1)*xi[0]*xi[1]/4.0
        
        N[4] = (1-xi[0]**2)*(xi[1]-1)*xi[1]/2.0
        N[5] = (1-xi[1]**2)*(xi[0]+1)*xi[0]/2.0
        N[6] = (1-xi[0]**2)*(xi[1]+1)*xi[1]/2.0
        N[7] = (1-xi[1]**2)*(xi[0]-1)*xi[0]/2.0
        
        N[8] = (1-xi[0]**2)*(1-xi[1]**2)
        
        gamma[0,0] = (2*xi[0]-1)*(xi[1]-1)*xi[1]/4.0
        gamma[0,1] = (2*xi[1]-1)*(xi[0]-1)*xi[0]/4.0
        gamma[1,0] = (2*xi[0]+1)*(xi[1]-1)*xi[1]/4.0
        gamma[1,1] = (2*xi[1]-1)*(xi[0]+1)*xi[0]/4.0
        gamma[2,0] = (2*xi[0]+1)*(xi[1]+1)*xi[1]/4.0
        gamma[2,1] = (2*xi[1]+1)*(xi[0]+1)*xi[0]/4.0
        gamma[3,0] = (2*xi[0]-1)*(xi[1]+1)*xi[1]/4.0
        gamma[3,1] = (2*xi[1]+1)*(xi[0]-1)*xi[0]/4.0
        
        gamma[4,0] = -xi[0]*(xi[1]-1)*xi[1]
        gamma[4,1] = (xi[1]-0.5)*(1-xi[0]**2)
        gamma[5,0] = (xi[0]+0.5)*(1-xi[1]**2)
        gamma[5,1] = -xi[1]*(xi[0]+1)*xi[0]
        gamma[6,0] = -xi[0]*(xi[1]+1)*xi[1]
        gamma[6,1] = (xi[1]+0.5)*(1-xi[0]**2)
        gamma[7,0] = (xi[0]-0.5)*(1-xi[1]**2)
        gamma[7,1] = -xi[1]*(xi[0]-1)*xi[0]
        
        gamma[8,0] = -2*xi[0]*(1-xi[1]**2)
        gamma[8,1] = -2*xi[1]*(1-xi[0]**2)
    
    else:
        raise BaseException('shape2d: Unknown combination of nen, ndm! STOP...!!!')

    return N,gamma
