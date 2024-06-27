# ========================================================================
#
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Finite Element Code for 2d (plane strain)
# ========================================================================


# ------------------------------------------------------------------------
# Imports
# ------------------------------------------------------------------------

# External libraries
import numpy as     np
import time
from   copy  import deepcopy

import matplotlib.pyplot as plt
import matplotlib.collections


# Preprocessing
from Subroutines.dofSeparation     import dofSeparation
from Subroutines.defineLoads       import defineLoads
from Subroutines.getMasterelement  import getMasterelement

# Main analysis
from Subroutines.elemLoop          import elemLoop
from Subroutines.elemLoop_withoutK import elemLoop_withoutK

# Postprocessing
from Subroutines.computeAverage    import computeAverage
from Subroutines.plotResults2d     import plotResults2d
from Subroutines.vtkOutput         import vtkOutput



# ------------------------------------------------------------------------
# Set input and flags
# ------------------------------------------------------------------------

# Select input routine
# from Input.input_notchedPlate import input_notchedPlate as inputData
from Input.input_singleElem   import input_singleElem   as inputData
from Input.input_beamMacro    import input_beamMacro    as inputData


# Choose flag for tangent calculation:
# 0 --> analytic
# 1 --> local numerical
# 2 --> global numerical
flag_tangent = 0


# Flags for witching on/off certain functions
flag_plot_vM   = True
flag_vtkOutput = False


# ------------------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------------------
print('Preprocessing...', end='')


# Read input
[x,nnp,ndm,ndf,conn,elem,nel,nen,nqp,tdm,mat,flag_mat,drlt,neum,loadcurve,t_end,b,fileName,vtkCellType] = inputData()

# Load the corresponding material routine
if (flag_mat == 0): # Hooke
    from Material.LinElasticity_2d   import LinElasticity_2d as material_law
elif (flag_mat == 1): # Kauderer
    from Material.Kauderer           import kauderersLaw     as material_law
elif (flag_mat == 2): # von Mises plasticity
    from Material.vonMisesPlasticity import vonMises         as material_law
elif (flag_mat == 3): # viso-elastoplasticity
    from Material.viscoPlasti        import viscoPlasti      as material_law
elif (flag_mat == 4): # FE^2
    from Material.microFEM           import microFEM         as material_law
else:
    raise TypeError("Unknown type of material!")


# Divide dof types in freeDofs and drltDofs
[drltDofs, freeDofs, neumDofs] = dofSeparation(nnp, ndf, drlt, neum)

# Determine masterelement (properties for each integration point q)
masterElement = getMasterelement(nen, ndm, nqp)


print('finished.')



# ------------------------------------------------------------------------
# FE analysis
# ------------------------------------------------------------------------
print('FE analysis...')

# Initialize displacements u
u = np.zeros(nnp*ndf)

# Initialize time/load step loop
step_cnt  = 0
dt = 1.0
t  = 0.0

# initialize storage for plots
plot_sig, plot_eps, plot_t = [0], [0], [0]


# Time/load step loop
while (t < t_end):
    
    # Wall time measurement
    walltime_start = time.time()
    
    
    # Increment update
    step_cnt += 1
    t += dt
    print('\nStep %d, time=%4.2f (%3d%%)'%(step_cnt, t, int(t/t_end*100)))
    
    # Get BCs for current step/time
    fsur             = np.zeros(ndf*nnp)
    fsur, drltValues = defineLoads(t, drlt, neum, neumDofs, loadcurve, fsur)
    
    
    # Initialize iteration loop 
    iter_cnt = 0
    iter_max = 20
    rsn = 1.0
    tol = 1e-8
    
    # Iteration loop
    while ((rsn > tol) and (iter_cnt < iter_max)):
        
        # Print iteration counter
        print ('  iter = %d, '%iter_cnt, end='')
        
        
        # --------------------------------------------------------------------
        # Determine system of equations
        # --------------------------------------------------------------------
        
        # Initialisation of global vectors and stiffness matrix
        K    = np.zeros((nnp*ndf, nnp*ndf))
        fint = np.zeros(nnp*ndf)
        fvol = np.zeros(nnp*ndf)
        frea = np.zeros(nnp*ndf)
        
        
        # Run element loop for local tangent
        if ((flag_tangent == 0) or (flag_tangent == 1)):
            [K, fint, fvol, elem] = elemLoop(x,u,elem,masterElement,material_law,mat,flag_tangent,b,K,fint,fvol,ndm,ndf,tdm,nel,nen,nqp,dt)
        
        
        # Run element loop for global numerical tangent
        elif (flag_tangent == 2):
            
            # Calculate nodal forces and residual for default configuration
            [fint, fvol, elem] = elemLoop_withoutK(x,u,elem,masterElement,material_law,mat,flag_tangent,b,fint,fvol,ndm,ndf,tdm,nel,nen,nqp,dt)
            rsd_full = fint- fvol - fsur
            
            
            # Copy elem in order to not change the original
            elem_pert = deepcopy(elem)
            
            # Perturbation value
            delta = 1e-8
            
            # Loop over perturbation in etries of displacements u
            for pertDOF in range(nnp*ndf):
                
                # (Re-)initialize nodal forces for perturbated results
                fint_pert = np.zeros(nnp*ndf)
                fvol_pert = np.zeros(nnp*ndf)
            
                # Set up perturbated displacements
                u_pert = deepcopy(u)
                u_pert[pertDOF] += delta
                
                # Calculate nodal forces and residual for pertubated u
                [fint_pert, fvol_pert, elem_pert] = elemLoop_withoutK(x,u_pert,elem_pert,masterElement,material_law,mat,flag_tangent,b,fint_pert,fvol_pert,ndm,ndf,tdm,nel,nen,nqp,dt)
                rsd_pert = fint_pert - fvol_pert - fsur
                
                # Determine stiffness matrix from FDM
                K[:,pertDOF] = (rsd_pert - rsd_full)/delta
        
        else:
            raise TypeError(f'Unknown type of tangent flag_tangent={flag_tangent}!')
        
        
        
        
        # --------------------------------------------------------------------
        # Solve equations
        # --------------------------------------------------------------------
        
        # Split stiffness matrix in Kff, Kdd, Kfd, Kdf
        Kff = K[freeDofs,:][:,freeDofs]
        Kdd = K[drltDofs,:][:,drltDofs]
        Kfd = K[freeDofs,:][:,drltDofs]
        Kdf = np.transpose(Kfd)
        
        
        # Determine residual norm
        rsd = fint[freeDofs] - fvol[freeDofs] - fsur[freeDofs]
        rsn = np.linalg.norm(rsd) + np.linalg.norm(drltValues - u[drltDofs])
        print('rsn = %e'%rsn)
        
        # Calculate update for displacements
        if (rsn > tol):
            
            if (iter_cnt == 0):
                rhs = -rsd - np.dot(Kfd, (drltValues - u[drltDofs]))
                u[drltDofs] = drltValues
            else:
                rhs = -rsd
            
            du = np.zeros(u.shape)
            du[freeDofs] = np.linalg.solve(Kff, rhs)
            u[freeDofs] += du[freeDofs]
        
        
        # Update iteration counter
        iter_cnt += 1
    
    
    # End of iteration loop
    
    
    
    # Calculate reaction forces (after convergence was met)
    frea[drltDofs] = fint[drltDofs] - fvol[drltDofs]
    
    
    # store data for plots
    plot_t.append(t)
    plot_sig.append(elem[0].stateVar[3,tdm**2])
    plot_eps.append(elem[0].stateVar[3,0])
    
    
    # Check whether maximum number of iterations was exceeded
    if (iter_cnt >= iter_max):
        raise Exception(f'Algorithm did not converge within iter_max={iter_max} iterations!')
    
    
    
    # ------------------------------------------------------------------------
    # Post-Processing
    # ------------------------------------------------------------------------
    
    # Update state variables for next time step
    for e in range(nel):
        elem[e].stateVar0 = deepcopy(elem[e].stateVar)
    
    
    # Determine averaged data for the elements
    if (flag_plot_vM or flag_vtkOutput):
        [cellData, __] = computeAverage(elem)
    
    # plot von Mises stresses if required
    if (flag_plot_vM):
        # Calculated element wise von Mises stresses
        sig_vM = np.zeros(nel)
        for e in range (nel):
            sig = cellData[e].stateVar[tdm**2:2*tdm**2].reshape((tdm,tdm))
            
            sig_vM[e] = np.sqrt(0.5*((sig[0,0]-sig[1,1])**2 
                                   + (sig[1,1]-sig[2,2])**2 
                                   + (sig[2,2]-sig[0,0])**2)
                              + 3.0*(sig[0,1]**2 + sig[1,2]**2 + sig[2,0]**2))
        
        # Plot von Mises stresses
        plotResults2d(x, nnp, ndm, conn, u, sig_vM, title = "equivalent von Mises stress $\sigma_{vM}$")
    
    
    # Generate vtk-Output
    if (flag_vtkOutput):
        vtkOutput(cellData,fint,frea,fvol,fsur,step_cnt,t,ndm,ndf,nnp,nel,nqp,tdm,x,u,elem,fileName,vtkCellType)
    
    
    
    
    walltime_end = time.time()
    print ('Wall time for this step: %5.3fs' %(walltime_end - walltime_start))
    
    
# End of time/load step loop



print("FEM-Analysis finished.")



# # Plotting load-displacement curve

# # Set plot resolution
# matplotlib.rcParams["figure.dpi"] = 300

# # Generate plot
# fig, axis = plt.subplots()
# axis=plt.gca()


# # A: strains over time
# # plt.plot(plot_t, plot_eps)
# # axis.set_xlabel('Time $t$ [s]')
# # axis.set_ylabel('Strains $\epsilon_{11}$ [-]')

# # B: stresses over time
# plt.plot(plot_t, plot_sig)
# axis.set_xlabel('Time $t$ [s]')
# axis.set_ylabel('Stresses $\sigma_{11}$ [MPa]')

# # C: stresses over strains
# # axis.set_xlabel('Strains $\epsilon_{11}$ [-]')
# # axis.set_ylabel('Stresses $\sigma_{11}$ [MPa]')
# # plt.plot(plot_eps,plot_sig)


# # scale axes and activate grid
# axis.autoscale()
# matplotlib.pyplot.grid(b=None, which='major', axis='both')


# # Print everything
# plt.show()




