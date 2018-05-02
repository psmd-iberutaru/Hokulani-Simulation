import numpy as np
import scipy as sp
import scipy.constants as CONST

from .Execptions import *
from . import Configuration as CONFIG
from . import Validation as Hoku_Valid
from . import Particle as Hoku_Part


def generate_particle(system_tempeture=CONFIG.INITAL_TEMPETURE,
                      maximum_mass=CONFIG.MAX_GENERATE_PARTICLE_MASS,
                      dark_matter_fraction=CONFIG.DARK_MATTER_FRACTION,
                      # Simulation wide settings.
                      space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS,
                      simulation_size=CONFIG.SIMULATION_BOX_SIZE):
    """
    This function generates a single particle. 
    """
    particle_array = Hoku_Part.instantiate_particle_array(n_particles=1,
                               system_tempeture=system_tempeture,
                               maximum_mass=maximum_mass,
                               dark_matter_fraction=dark_matter_fraction,
                               # Simulation wide settings.
                               space_dimensions=space_dimensions,
                               simulation_size=simulation_size)

    return particle_array[0]


def generate_particle_array(n_particles=CONFIG.NUMBER_OF_PARTICLES,
                            system_tempeture=CONFIG.INITAL_TEMPETURE,
                            maximum_mass=CONFIG.MAX_GENERATE_PARTICLE_MASS,
                            dark_matter_fraction=CONFIG.DARK_MATTER_FRACTION,
                            # Simulation wide settings.
                            space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS,
                            simulation_size=CONFIG.SIMULATION_BOX_SIZE):
    """
    This function generates a particle array. This function has almost 
        the exact similar function of instantiate_particle_array(...).
        However, this function is preferred as it checks the initial conditions.
    """

    # Check that the initial conditions are valid.
    particles = Hoku_Valid.validate_number_of_particles(n_particles)
    sys_temp = Hoku_Valid.validate_inital_tempeture(system_tempeture)
    max_mass = Hoku_Valid.validate_max_generate_particle_mass(maximum_mass)
    dark_m_frac = Hoku_Valid.validate_dark_matter_fraction(dark_matter_fraction)

    space_dim = Hoku_Valid.validate_number_of_dimensions(space_dimensions)
    simulate_size = Hoku_Valid.validate_simulation_box_size(simulation_size)



    particle_array = Hoku_Part.instantiate_particle_array(n_particles=particles,
                               system_tempeture=sys_temp,
                               maximum_mass=max_mass,
                               dark_matter_fraction=dark_m_frac,
                               # Simulation wide settings.
                               space_dimensions=space_dim,
                               simulation_size=simulate_size)

    return particle_array




def generate_particle_position_array(n_particles,
                                     simulation_size=CONFIG.SIMULATION_BOX_SIZE,
                                     user_pos_array=False):
    """
    This function generates the position vectors of the particles. It is 
        assumed that the simulation is contained within some box with width 
        given by the simulation's box size.
    """
    # If the user has defined their own positions then, just return it to them.
    #   however, check that the acceleration array that they gave is valid.
    if (user_pos_array):
        if ((len(user_pos_array) == n_particles) and 
            (len(user_pos_array[0]) == 3)):
            return np.array(user_pos_array)
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined position array has invalid'
                              'dimensions. Default methods are being used.'))

    # The default position vectors are all randomized, assuming that the 
    #   initial state of the system is inertial.
    pos_array = np.random.uniform(-simulation_size,simulation_size,
                                  size=[n_particles,3])
    
    return pos_array.astype(CONFIG.NP_FLOAT_DTYPE)

def generate_particle_velocity_array(n_particles,mass_array,sys_temperature,
                                     user_vel_array=False,
                                     maxwell_boltzman_distribution=False,
                                     # Simulation wide
                                     space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    Generate all of the velocities of the particles based on a 
        Maxwell-Boltzmann distribution. The user specifies the temperature
        of the entire system for the velocities to be generated out of.
        All three components of the velocity are all generated per particle
        at this step.
    """
    
    # If the user has defined their own positions then, just return it to them.
    #   however, check that the acceleration array that they gave is valid.
    if (user_vel_array):
        if ((len(user_vel_array) == n_particles) and 
            (len(user_vel_array[0]) == 3)):
            return np.array(user_vel_array)
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined position array has invalid '
                              'dimensions. Default methods are being used.'))


    # Execute Maxwell-Boltzmann distribution if wanted.
    if (maxwell_boltzman_distribution):
        # Generate a Gaussian distribution for a number of particles
        #   over all 3 dimensions for all 3 dimensions for space. 
        randomized_3_gaussian = np.random.normal(size=(n_particles,space_dimensions))

        # Spread the masses of the particles over all three dimensions.
        spread_masses = np.transpose(np.tile(mass_array,(space_dimensions,1)))

        # Convert into velocities that a Maxwell-Boltzmann distribution would 
        # return.
        vel_array = (randomized_3_gaussian 
                     * np.sqrt((CONST.Boltzmann*sys_temperature)
                               / spread_masses))

    # Default is to have the velocity array be all zeros.
    else:
        vel_array = np.zeros([n_particles,space_dimensions])

    return vel_array.astype(CONFIG.NP_FLOAT_DTYPE)


def generate_particle_acceleration_array(n_particles,user_accel_array=False,
                                         # Simulation wide
                                         space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function generates the acceleration vectors of the particles. 
        However, the default is to generate zero vectors assuming that the 
        system starts at a inertial state. However, it is possible for a 
        user to add a their own acceleration vector array to use instead.
    """

    # If the user has defined their own accelerations then, just return it to 
    #   them however, check that the acceleration array that they gave is 
    #   valid.
    if (user_accel_array):
        if ((len(user_accel_array) == n_particles) and 
            (len(user_accel_array[0]) == space_dimensions)):
            return np.array(user_accel_array)
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined acceleration array has invalid '
                              'dimensions. Default methods are being used.'))

    # The default acceleration vectors are all zero, assuming that the initial 
    #   state of the system is inertial.
    accel_array = np.zeros([n_particles,space_dimensions])

    return accel_array.astype(CONFIG.NP_FLOAT_DTYPE)

def generate_particle_mass_array(n_particles,
                                 maximum_mass=CONFIG.MAX_GENERATE_PARTICLE_MASS,
                                 user_mass_array=False):
    """
    This function generates a random distribution of the mass of particles.
        By default, the distribution is uniform.
    """
    
    # Check if the user has defined their own particle masses. Perform a
    #   very basic check to see if the values are somewhat seen as numbers.
    #   If not, go back to defaults.
    if (user_mass_array):
        if ((len(user_mass_array) == n_particles) 
            and ((type(user_mass_array[0]) == float 
                  or type(user_mass_array[0]) == int))):
            return user_mass_array
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined mass array has invalid dimensions '
                              'or values. Default methods are being used.'))

    mass_array = np.random.uniform(0,maximum_mass,size=[n_particles])

    return mass_array.astype(CONFIG.NP_FLOAT_DTYPE)


def generate_particle_variety_array(n_particles,user_variety_array=False):
    """
    This function determines the variety of each of the particles. By
        default, the variety of each particle is a gas variety.
    """

    try:
        validate_variety_code_array(variety_array,n_particles)
    except Exception:
        Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined variety array has invalid '
                              'dimensions or values. Default methods are '
                              'being used.'))
        variety_array = np.full(n_particles,CONFIG.DEFAULT_VARIETY_CODE,
                                dtype=np.uint8)

    return variety_array


def generate_particle_dark_matter_array(n_particles,
                                        dark_matter_fraction=CONFIG.DARK_MATTER_FRACTION,
                                        user_variety_array=False):
    """
    This function determines the if or if not the particle is considered to be
        dark matter. For visualization purposes, the particles is not shown. 
        The default being a random distribution based on the dark matter
        fraction default.
    """

    # If the user provides a valid array, just use the array instead.
    if (user_variety_array):
        if ((len(user_variety_array) == n_particles) and 
            (type(user_accel_array[0]) == bool)):
            return user_variety_array
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined dark matter array has invalid '
                              'dimensions or values. Default methods are '
                              'being used.'))

    dark_matter_prob_values = np.random.uniform(0,1,size=[n_particles])

    dark_matter_array = dark_matter_prob_values <= dark_matter_fraction
    return dark_matter_array


def generate_particle_deletion_array(n_particles,user_deletion_array=False):
    """
    This function defines the deletion values for all of the particles in 
        the simulation. By default, all of these values are false. 
    """

    if (user_deletion_array):
        if ((len(user_deletion_array) == n_particles) and 
            (user_deletion_array == False)):
            return user_deletion_array
        else:
            Hokulani_Warning(Hokulani_TypeWarning, 
                             ('User defined deletion array has invalid '
                              'dimensions or values. Default methods are '
                              'being used.'))
            
    deletion_array = np.full([n_particles],False,dtype=bool)

    return deletion_array



