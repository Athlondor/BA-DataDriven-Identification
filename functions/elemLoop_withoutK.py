# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Element loop without computation of stiffness matrix 
# (e.g. for global numerical tangent)
# ========================================================================

import numpy as np

from Subroutines.jacobian2d import jacobian2d


def elemLoop_withoutK(x,u,elem,masterElement,material_law,mat,flag_tangent,b,fint,fvol,ndm,ndf,tdm,nel,nen,nqp,dt):
# Loop over all elements e
    for e in range(nel):
        
        # Determine global dof numbers of the nodes of element e (gdof): gdof of size nen*ndf
        gdof = np.zeros(nen*ndf, dtype=int)
        for node in range(nen):
            gdof[node*ndf:(node+1)*ndf] = [(elem[e].cn[node]-1)*ndf, (elem[e].cn[node]-1)*ndf+1]
        
        
        # Coordinates of the element nodes (ndm x nen)
        xe = (x[gdof]).reshape(nen,ndm).transpose()
        
        # Displacements of the element nodes (ndf x nen)
        ue = (u[gdof]).reshape(nen,ndf).transpose()
        
        
        # Initialize Ke, finte, fvole
        finte = np.zeros(nen*ndf)
        fvole = np.zeros(nen*ndf)
        
        # Loop over the gauss points, summation over all gauss points 'q'
        for q in range(nqp):
            
            # Get evaluated shape functions, their derivatives and weights
            # at Gauss point q
            N     = masterElement[q].N
            gamma = masterElement[q].gamma
            w8    = masterElement[q].w8
            
            
            # Determinant of Jacobian and the inverse of Jacobian
            # at the quadrature points q
            [detJq,invJq] = jacobian2d(xe,gamma,nen,ndm)
            
            # Gradient of the shape functions w.r.t. to x: G
            G = np.dot(gamma,invJq)
            
            # Determine strains epsilon
            H = np.dot(ue, G)
            eps = np.zeros((tdm, tdm))
            eps[0:ndm,0:ndm] = (H + np.transpose(H))/2.0
            
            # Determine sigma and C4 from material routine
            [sig, C4, elem[e].stateVar[q,:]] = material_law(mat, eps, elem[e].stateVar0[q,:], flag_tangent,dt)
            
            
            # Determine the volume contribution and store it
            dV = detJq*w8
            elem[e].dV[q] = dV
            
            
            # Loop over the number of nodes A
            for A in range(nen):
                
                # Gradient of the shape functions of node A
                G_A = G[A,:]
                
                # Internal force contribution
                fintA = np.dot(G_A, sig[0:ndm,0:ndm])*dV
                finte[A*ndf:A*ndf+ndm] += fintA
                
                # Volume force contribution: fvolA
                fvolA = N[A]*b*dV
                fvole[A*ndf:A*ndf+ndm] += fvolA
                
                
        # End of integration loop (q)
        
        
        # Assemble element contributions into global K, fint, fvol
        for i, gdof_i in enumerate(gdof):
            fint[gdof_i] += finte[i]
            fvol[gdof_i] += fvole[i]
        
        
    return fint, fvol, elem