"""
This file is the main program that is the functions that, when given a 
    structure that contains the data on the position of the particles within
    the simulation, exports an image, set of images, or some other graphics
    visualization file or technique for seeing the simulation in action, or 
    at the end.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from . import Particle as Hoku_Part


def crude_display_simulation_final_3D(particle_history,
                                      display_axis=True,
                                      file_export=None):
    """
    This function takes in a final particle history (a particle array),
        expecting the very last position of the particles in the simulation,
        and makes a crude 3d plot. This function can only do 3D 
        visualization.
    This display using matplotlib, it is considered to be a crude way of 
        visualizing the values.

    """

    # Check that the dimension of this simulation is 3d, if it is not, then
    #   bark back as this only works for 3d.

    # Define the figure space.
    figure = plt.figure()
    axis = figure.add_subplot(111, projection='3d')

    # From every particle, extract the array of the x,y,z positional values.
    pos_array = Hoku_Part.extract_position_array(particle_history)

    x_pos_array = pos_array[:,0]
    y_pos_array = pos_array[:,1]
    z_pos_array = pos_array[:,2]

    # Make the scatter plot of all of the particles.
    axis.scatter(x_pos_array, y_pos_array, z_pos_array, c='r', marker='o')

    # Check to see if the user does, or does not, want the axis.
    if not (display_axis):
        plt.axis('off')

    # Check if the user wanted the file to be saved to disk. If they wanted
    #   it saved to the disk, then save it as a general named file.
    if (file_export != None):
        figure.savefig(file_export + '.png', dpi=fig.dpi)
    else:
        plt.show(block=False)







