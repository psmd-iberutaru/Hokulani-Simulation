"""
This file contains all of the simulation functionality. Including collisions
    and the progression of the simulation over time.
"""

import numpy as np

from .Execptions import *
from . import Configuration as CONFIG
from . import Particle as Hoku_Part
from . import Mechanics as Hoku_Mech


def simulate_particles(particle_array,
                       return_particle_history=False,
                       # Simulation wide.
                       delta_time=CONFIG.DELTA_TIME):
    """
    This function completes the main part of the simulation. If the user wants
        the entire history of the particles, then return that instead of the
        last state of the particles. 
    """

    # To be used if the user wants the entire history of the particle stats.
    #   Warn the user that this is memory intensive.
    if (return_particle_history):
        Hokulani_Warning(Hokulani_SanityWarning,('Return particle history is '
                                                 'set to true. Be aware that '
                                                 'this task becomes very '
                                                 'memory intensive.'))
        particle_history = []
    
    
    # Begin to cycle through the simulation.
    for interationdex in range(CONFIG.TOTAL_ITERATION_COUNT):
        
        # For ease and speed (if that is really a thing in this entire 
        #   simulation, keep record of the total number of particles.
        n_particles = len(particle_array)


        # Search through the entire particle list to see if there are any
        #   particles that should be scanned for collisions. Append these 
        #   particles to the main particle array.
        particle_array = Hoku_Mech.compute_particle_collision(particle_array)
        

        # Compute the basic kinematic evolution of the particles.
        particle_array = simulate_particle_kinematics(particle_array,
                                                      delta_time=delta_time)

        # Scan all of the particles and clean up all particles that have been
        #   tagged for deletion for whatever reason.
        particle_array = Hoku_Part.purge_particle_array(particle_array)


        # If the user wants the entire history of the particle arrays, then store
        #   this particular state.
        if (return_particle_history):
            particle_history.append(particle_array)

        print(interationdex,'    ',n_particles)

    # Return the value that the user desires, based on the initial values.
    if (return_particle_history):
        # The user desires the entire history of the particles.
        return np.array(particle_history)
    else:
        # By default, just return the final position.
        return particle_array


def simulate_particle_kinematics(particle_array,
                                 seperation_array = None,
                                 delta_time=CONFIG.DELTA_TIME):
    """
    This function simulates a change in the position, velocity, and
        acceleration of the particles based on Newtonian and Maxwell 
        mechanics. 
    """
    # Find the total number of particles.
    n_particles = len(particle_array)

    # Allow for the recycling of the separation array, if it is given, use it
    #   else, regenerate it. The values of the position and derivative 
    #   positions are still needed.
    if not (seperation_array is None):
        # Extract the needed property arrays that are needed to compute the 
        #   progression of this particle system.
        pos_array = Hoku_Part.extract_position_array(particle_array)
        vel_array = Hoku_Part.extract_velocity_array(particle_array)
        accel_array = Hoku_Part.extract_acceleration_array(particle_array)

        sep_array = seperation_array
    else:
        # Extract the needed property arrays that are needed to compute the 
        #   progression of this particle system.
        pos_array = Hoku_Part.extract_position_array(particle_array)
        vel_array = Hoku_Part.extract_velocity_array(particle_array)
        accel_array = Hoku_Part.extract_acceleration_array(particle_array)

        sep_array = Hoku_Mech.compute_seperation_array(pos_array)


    # Compute the force of the array. The distance, separation, between each
    #   particle is given, the mass of each must also be given.
    mas_array = Hoku_Part.extract_mass_array(particle_array)
    force_array = Hoku_Mech.compute_force_array(sep_array,mas_array)

    # Update the acceleration, velocity, and position array, in that order, 
    #   based on the new force vectors.
    accel_array = Hoku_Mech.compute_acceleration_array(force_array,mas_array)
    vel_array = Hoku_Mech.compute_velocity_array(vel_array,accel_array,
                                                 delta_time=delta_time)
    pos_array = Hoku_Mech.compute_position_array(pos_array,vel_array,accel_array,
                                                 delta_time=delta_time)

    # Rebind the new arrays into new particles.
    particle_array = Hoku_Part.reparcel_particle_array(particle_array,
                                                       position_array=pos_array,
                                                       velocity_array=vel_array,
                                                       acceleration_array=accel_array,
                                                       n_particles=n_particles)

    return np.array(particle_array)