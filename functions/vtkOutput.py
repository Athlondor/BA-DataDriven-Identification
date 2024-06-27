# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for writing vkt ouput into an external (list of) files
# ========================================================================

import numpy as np
from pathlib import Path

# Write vtk output into file
def vtkOutput(cellData,fint,frea,fvol,fsur,step,time,ndm,ndf,nnp,nel,nqp,tdm,x,u,elem,fname,vtkCellType):
    
    # Filename
    Path("Output/"+ fname).mkdir(parents=True, exist_ok=True)
    vtk_filename = (fname + "_" + str("%06d" %step) + ".vtk")
    vtk_filename = ("Output/"+ fname + '/' + vtk_filename)
    
    # Write file as text file
    fid = open(vtk_filename,'wt')
    
    # "HEADER" of vtk-file
    fid.write('# vtk DataFile Version 3.0\n')
    fid.write('%s\n' %vtk_filename)
    
    fid.write('ASCII\n')
    fid.write('DATASET UNSTRUCTURED_GRID\n')
    
    # Use field data to store time and cycle information
    fid.write('FIELD FieldData 2\n')
    fid.write('TIME 1 1 double\n')
    fid.write('%e\n' %time)
    fid.write('CYCLE 1 1 int\n')
    fid.write('%d\n' %step)
    
    # "POINTS": node points
    fid.write('POINTS %d double\n' %nnp)
    x_aux = np.reshape(x, (nnp, ndf))
    for i in range(nnp):
        coord        = np.zeros((1,ndm))
        coord[0:ndm-1] = x_aux[i,:]
        fid.write('%24.12e %24.12e %24.12e\n' %(coord[0,0],coord[0,1],0))
    
    # "CELLS": write element connectivity
    number = 0
    for e in range(nel):
        cn = elem[e].cn
        nen = cn.shape[0]
        number = number + nen + 1
    
    fid.write('CELLS %d %d\n' %(nel,number))
    
    for e in range(nel):
        nen = elem[e].cn.shape[0]
        fid.write('%d ' %nen)
        cn = elem[e].cn
        
        for A in range(nen):
            # vtk uses C arrays
            # numbering in C starts with 0 not with 1
            fid.write(' %8d' %int(cn[A]-1))
        fid.write('\n')
    
    # "CELL_TYPES": write type of each element (BAR2, QUAD4, etc.)
    # at the moment only quad4 is used -> vtkElType = 10... see manual
    fid.write('CELL_TYPES %d\n' %nel)
    
    for e in range(nel):
        fid.write('%d\n' % vtkCellType) # TODO: variable element type
    
    # "POINT_DATA": write SCALARS, VECTORS, TENSORS (in that order!)
    fid.write('POINT_DATA %d\n' %nnp)
    
    # "VECTORS": write displacements
    fid.write('VECTORS DSPL double\n')
    for i in range(nnp):
        uline = np.zeros((1,ndm))
        
        uline[0,0] = u[2*i]
        uline[0,1] = u[2*i+1]
        
        fid.write('%24.12e %24.12e %24.12e\n' %(uline[0,0],uline[0,1],0))
    
    # "VECTORS": write forces
    fid.write('VECTORS FINT double\n')
    for jnp in range(nnp):
        f = np.zeros((1,ndm))
        
        f[0,0] = fint[2*jnp]
        f[0,1] = fint[2*jnp+1]
        
        fid.write('%24.12e %24.12e %24.12e\n' %(f[0,0],f[0,1],0))
    
    # "VECTORS": write forces
    fid.write('VECTORS FREA double\n')
    for jnp in range(nnp):
        f = np.zeros((1,ndm))
        
        f[0,0] = frea[2*jnp]
        f[0,1] = frea[2*jnp+1]
        
        fid.write('%24.12e %24.12e %24.12e\n' %(f[0,0],f[0,1],0))
    
    fid.write('VECTORS FVOL double\n')
    for jnp in range(nnp):
        f = np.zeros((1,ndm))
        
        f[0,0] = fvol[2*jnp]
        f[0,1] = fvol[2*jnp+1]
        
        fid.write('%24.12e %24.12e %24.12e\n' %(f[0,0],f[0,1],0))
        
    fid.write('VECTORS FSUR double\n')
    for jnp in range(nnp):
        f = np.zeros((1,ndm))
        
        f[0,0] = fsur[2*jnp]
        f[0,1] = fsur[2*jnp+1]
        
        fid.write('%24.12e %24.12e %24.12e\n' %(f[0,0],f[0,1],0))
    
    
    # "CELL_DATA": write SCALARS, VECTORS, TENSORS (in that order!!)
    fid.write('CELL_DATA %d\n' %nel)
        
    # Write stresses
    for j in range (tdm*tdm):
        
        if j==0:
            varName = 'SIG_11'
        elif j==1:
            varName = 'SIG_12'
        elif j==2:
            varName = 'SIG_13'
        elif j==3:
            varName = 'SIG_21'
        elif j==4:
            varName = 'SIG_22'
        elif j==5:
            varName = 'SIG_23'
        elif j==6:
            varName = 'SIG_31'
        elif j==7:
            varName = 'SIG_32'
        elif j==8:
            varName = 'SIG_33'
        else:
            varName = "FEHLER"
        
        fid.write('SCALARS %s double\n' %varName)
        fid.write('LOOKUP_TABLE default\n')
        
        
        for e in range(nel):
            Pi = cellData[e].stateVar[tdm*tdm : 2*tdm*tdm]
            P = Pi[j]
            fid.write('%24.12e\n' %P)


    # von Mises stresses
    fid.write('SCALARS SIG_MISES double\n')
    fid.write('LOOKUP_TABLE default\n')
    for e in range(nel):
            Pi = cellData[e].stateVar[tdm*tdm : 2*tdm*tdm]
            sig = Pi.reshape((tdm,tdm))
            P = np.sqrt(0.5*((sig[0,0]-sig[1,1])**2 
                               + (sig[1,1]-sig[2,2])**2 
                               + (sig[2,2]-sig[0,0])**2)
                          + 3.0*(sig[0,1]**2 + sig[1,2]**2 + sig[2,0]**2))
            fid.write('%24.12e\n' %P)


    # Write strains
    for j in range(tdm*tdm):
        
        if j==0:
            varName = 'EPS_11'
        elif j==1:
            varName = 'EPS_12'
        elif j==2:
            varName = 'EPS_13'
        elif j==3:
            varName = 'EPS_21'
        elif j==4:
            varName = 'EPS_22'
        elif j==5:
            varName = 'EPS_23'
        elif j==6:
            varName = 'EPS_31'
        elif j==7:
            varName = 'EPS_32'
        elif j==8:
            varName = 'EPS_33'
        else:
            varName = "FEHLER"
        
        fid.write('SCALARS %s double\n' %varName)
        fid.write('LOOKUP_TABLE default\n')
        
        for e in range(nel):
            Pi = cellData[e].stateVar[0:tdm*tdm]
            P = Pi[j]
            fid.write('%24.12e\n' %P)
    
    fid.close()