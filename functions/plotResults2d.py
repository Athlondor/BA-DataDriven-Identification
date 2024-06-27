# ========================================================================
# 
# Finite Element Methods II
# Institute of Mechanics
#
# TU Dortmund University
#
# ========================================================================
# Routine for plotting an element based values on a 2d FE mesh
# 
# Created with the help of StackOverflow
# https://stackoverflow.com/questions/52202014/how-can-i-plot-2d-fem-results-using-matplotlib
# ========================================================================


import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np


def plotResults2d(x, nnp, ndm, conn, u, values, title):
    
    # rearange connectivity if necessary
    nen = conn.shape[1]
    
    if ((ndm == 2) and ((nen == 3) or (nen == 4))): # Tri3, Quad4
        # no rearangement required
        pass
    elif ((ndm == 2) and ((nen == 8) or (nen == 9))): # Quad8, Quad9
        conn=conn[:,[0,4,1,5,2,6,3,7]]
    else:
        raise TypeError('Unknown type of element for plot!')
    
    
    # Plot updated node position
    x_new = (x+u).reshape(nnp,ndm)
    
    # Set plot resolution
    matplotlib.rcParams["figure.dpi"] = 300

    # Generate plot
    fig, axis = plt.subplots()
    axis.set_aspect('equal')
    axis=plt.gca()
    pc = matplotlib.collections.PolyCollection(x_new[np.asarray(conn-1)], edgecolor="black", linewidth=0.2, cmap="rainbow")
    pc.set_array(values)
    axis.add_collection(pc)
    axis.autoscale()

    # Add colorbar, title, etc.
    fig.colorbar(pc, ax=axis)        
    axis.set(title = title)
    
    # Print everything
    plt.show()
