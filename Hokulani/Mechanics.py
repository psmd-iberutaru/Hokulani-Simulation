"""
This file contains all of the needed mathematically rigorous computations 
    run the entire simulation (ex. kinematics, array manipulation, etc).

Every single function in here should be somewhat sped up by the numba module.

"""

import numpy as np
import scipy.constants as CONST
import numba
import copy

from .Execptions import *
from . import Configuration as CONFIG
from . import Particle as Hoku_Part


@numba.jit
def compute_position_array(position_array,velocity_array,acceleration_array,
                           delta_time=CONFIG.DELTA_TIME):
    """
    This function computes the new position vector for an array of particles. 
        This is mostly done by the classical mechanical method using the
        solutions to the differential equations of motion. In particular,
        using the generalized equations for constant linear acceleration.
    """

    updated_position_array = (position_array 
                              + velocity_array * delta_time
                              + 0.5 * acceleration_array * delta_time**2)

    return np.array(updated_position_array)


@numba.jit
def compute_velocity_array(velocity_array,acceleration_array,
                           delta_time=CONFIG.DELTA_TIME):
    """
    This function computes the new velocity vector for an array of particles.
        This is mostly done by the classical mechanical method using the
        solutions to the differential equations of motion. In particular,
        using the generalized equations for constant linear acceleration.
    """

    updated_velocity_array = velocity_array + acceleration_array * delta_time

    return np.array(updated_velocity_array)


@numba.jit
def compute_acceleration_array(force_array,mass_array,
                               space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function computes the new acceleration vectors for an array of 
        particles. This is done mostly by the classical mechanical method
        of Newton's second law: F=ma. If the particle array is provided,
        then automatically update the particle array. Else, just return
        the new acceleration array.
    """

    # Assume that the mass array is always correct.
    n_particles = len(mass_array)

    # Using Newton's second law.
    updated_acceleration_array = (force_array 
                                  / (mass_array.repeat(space_dimensions)
                                     .reshape(n_particles,space_dimensions)))

    return np.array(updated_acceleration_array)


@numba.jit
def compute_seperation_array(position_array):
    """
    This function creates the separation array, a matrix of the vectors
        that describe the separation vector between any two particles. This
        is mostly to be used for the distance between two points.
    """

    temp_position_array = np.array([position_array])
    n_particles = len(position_array)

    # Create a 2D array for each particle vector (each a 1D array). Transpose  
    #   this two array such that the particles are transposed. Over one axis,
    #   the list of all particles, the second axis is also all particles. The
    #   array is similar to that of a multiplication table. 
    position_array_2 = np.repeat(temp_position_array,n_particles,axis=0)

    position_array_1 = np.transpose(position_array_2,axes=(1,0,2))

    # The subtraction of these arrays gives the separation array for all 
    #   particles vs all other particles. The array values is the vector 
    #   [x,y,z] from the particle (axis 0) to the n particle in the array,
    #   the value n being the location in axis 1. Axis 2 contains the values
    #   of the [x,y,z] vector.
    seperation_array = position_array_2 - position_array_1

    return np.array(seperation_array,dtype=CONFIG.NP_FLOAT_DTYPE)


# @numba.jit
def compute_seperation_magnitude_array(seperation_array):
    """
    This function computes the magnitude of the separation array given, 
        returning the absolute distance as given by the separation vector.
    """

    n_particles = int(np.sqrt(seperation_array.size 
                              // CONFIG.NUMBER_OF_DIMENSIONS))

    # Obtain the magnitude of the vectors within the separation array. By
    #   transforming the array into a list of vectors, finding their dot 
    #   product, then rearranging them back into the original shape before 
    #   taking the square-root of the values.
    reshape_sep_array = np.reshape(seperation_array,
                                   (n_particles**2,CONFIG.NUMBER_OF_DIMENSIONS),
                                   order='C')

    reshape_mag_sep_array = np.sqrt(np.sum((reshape_sep_array**2).T[:],axis=0))

    seperation_magnitude_array = np.reshape(reshape_mag_sep_array,
                                            (n_particles,n_particles,1),
                                            order='C')

    return seperation_magnitude_array.astype(CONFIG.NP_FLOAT_DTYPE)


def compute_force_array(seperation_array,mass_array,
                        space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function computes the force of gravity between all particles, 
        between any two given particles. This model of gravity includes only
        the Newtonian equation of gravity. This function returns the net
        force vector felt by each particle from the gravitational force.
    """

    # Define useful values.
    if (len(seperation_array) == len(mass_array)):
        n_particles = len(mass_array)
    else:
        raise Hokulani_ShapeError(('The length of the separation array and '
                                  'mass array are different.'))

    # Obtain the magnitude of the vectors within the separation array.
    seperation_mag_array = compute_seperation_magnitude_array(seperation_array)

    def compute_gravity_force_array(mass_array, sep_array, sep_mag_array):
        # Arrange an array containing the product of any two given masses for 
        #   each particle. The array should look similar to multiplication
        #   tables when they are multiplied. Then transform the mass array to
        #   add another dimension for compatibility with the other arrays.
        mass_product = np.outer(mass_array,mass_array)
        mass_product = np.reshape(mass_product,(n_particles,n_particles,1),
                                  order='C')

        # Use Newton's universal law of gravitation to generate the force array.
        gravity_magnitude_array = ((CONST.gravitational_constant * mass_product)
                                   / sep_mag_array**2)

        # Handle the case where a zero separation magnitude array leads to an
        #   infinite gravitational force. An infinity number is just a large
        #   number. The zero direction vector handles it going to zero.
        gravity_magnitude_array = np.nan_to_num(gravity_magnitude_array)

        # Scale directional unitary vectors by the magnitude of the 
        #   gravitational force to find the force vector
        temp_seperation_direction_array = (sep_array / sep_mag_array)

        # Handle the case where the division by zero returns Nan. A zero magnitude
        #   vector is a zero vector, thus its components are zero.
        seperation_direction_array = np.nan_to_num(temp_seperation_direction_array)

        gravity_vector_array = gravity_magnitude_array * seperation_direction_array

        net_gravity_vector_array = np.sum(gravity_vector_array,axis=1)

        return np.array(net_gravity_vector_array,dtype=CONFIG.NP_FLOAT_DTYPE)

    # Compute the force due to gravity.
    gravity_force = compute_gravity_force_array(mass_array,
                                                seperation_array,
                                                seperation_mag_array)

    return np.array(gravity_force,dtype=CONFIG.NP_FLOAT_DTYPE)
    


def compute_particle_collision(particle_array,
                               seperation_array=None,
                               seperation_threshold=CONFIG.SEPERATION_THRESHOLD,
                               n_particles=CONFIG.NUMBER_OF_PARTICLES):

    """
    This function searches for the indices of all particles that are 
        to have collided. The two metrics used to this is calculation is
        separation distance and velocity. Inelastic collisions occur if the
        particles are to be combined, elastic collisions are used otherwise.
    Both the separation array and the particle array can be worked. The 
        separation array is considered faster.
    """

    if (seperation_array is not None):
        Hokulani_Warning(Hokulani_PhysicalWarning,('Using the separation '
                                                   'array may be faster, but '
                                                   'it is considered to be '
                                                   'less accurate.'))

    # This function computes a single iteration of particle collisions. 
    def compute_particle_collision_iteration(particle_array,
                                             seperation_array=None,
                                             seperation_threshold=CONFIG.SEPERATION_THRESHOLD,
                                             n_particles=CONFIG.NUMBER_OF_PARTICLES):
        
        # It is important that the particle array is a numpy array. Double 
        #   check that it is, or convert.
        if not (isinstance(particle_array,np.ndarray)):
            try:
                particle_array = np.array(particle_array)
            except Exception:
                raise Hokulani_TypeError('Particle array input must be an '
                                         'array or an object that can be '
                                         'turned into an array.')
    
        # Check to see if the user specified a separation array instead of a
        #   particle array. If they did, recycle it.
        if not (seperation_array is None):
            sep_mag_array = compute_seperation_magnitude_array(seperation_array)
        else:
            pos_array = Hoku_Part.extract_position_array(particle_array)
            seperation_array = compute_seperation_array(pos_array)
            sep_mag_array = compute_seperation_magnitude_array(seperation_array)
    
        # See where the separation of the particles is less than the threshold.
        #   Keep track of where in the main particle list these particles are. 
        #   This returns a pair of parallel arrays containing the particles to 
        #   be collided.
        # The last element of collision pair can be discarded as the separation 
        #   magnitude array is 3D for compatibility purposes when it should be 2D.
        collision_pair = np.nonzero(sep_mag_array <= seperation_threshold)
        collision_pair = np.array(collision_pair[:-1])
    
        # Reorder this collision list based on the separation value, from 
        #   lowest to highest.
        pair_sep_array = []
        for pairdex in collision_pair.T:
            particle_sep_vect = (particle_array[pairdex[1]].position 
                                 - particle_array[pairdex[0]].position)
            particle_sep_mag = np.sqrt(np.dot(particle_sep_vect,particle_sep_vect))
            pair_sep_array.append(particle_sep_mag)
    
        sort_index = np.argsort(pair_sep_array)
    
        collision_pair = collision_pair[:,sort_index]
    
        # Begin colliding particles.
        collided_particles = []
        for collisiondex in collision_pair.T:
            # Test to see if both particles still exist, that is, their deletion
            #   values are not true. If they are, pass to the next iteration.
            if (particle_array[collisiondex[0]].delete or 
                particle_array[collisiondex[1]].delete):
                continue
    
            # Test to see if both particles are actually the same particle. It 
            #   does not make sense to collide a particle into itself.
            if (collisiondex[0] == collisiondex[1]):
                continue
            
            # Dark matter is considered to be weakly interacting, and thus 
            #   does not 'collide'. If either of these particles are dark 
            #   matter, then deny the collision.
            if (particle_array[collisiondex[0]].dark_matter or
                particle_array[collisiondex[1]].dark_matter):
                continue
    
            # Collide the current pair of particles and record the final 
            #   particle.
            temp_collided_particle = compute_inelastic_particle_collision(
                particle_array[collisiondex[0]],particle_array[collisiondex[1]])
    
            collided_particles.append(temp_collided_particle)
    
            # The two particles that just collided should be tagged for 
            #   deletion.
            particle_array[collisiondex[0]].delete = True
            particle_array[collisiondex[1]].delete = True
    
        
        return np.array(collided_particles)

    # Execute the function for a single iteration of collisions. Loop over to
    #   check for any new collisions that may have been as a result of 
    #   the particle combination (do this within this function to prevent the
    #   next iteration of the simulation.

    # Default to there always being a particle collision
    available_collision = True

    # Run through all of the particle collisions. 
    while (available_collision):
       # Complete an iteration of particle collisions.
        collided_particles = compute_particle_collision_iteration(particle_array,
                                             seperation_array=seperation_array,
                                             seperation_threshold=seperation_threshold,
                                             n_particles=n_particles)

        # If there is no collided particles returns then there are no 
        #   particle collisions to do afterwards.
        if (collided_particles.size == 0):
            available_collision = False
            
            # An assurance.
            continue  
        else:
            # There should still be particles to execute for collisions here.
            #   Do a check to make sure that the particles returned are 
            #   actually particles. 
            if (isinstance(collided_particles[0],Hoku_Part.Particle)):
                particle_array = np.append(particle_array,collided_particles)
            else:
                # The list of new post-collision particles seems to be non-
                #   empty, but it does not contain particles. Alert the user.
                raise Hokulani_UnknownError('The result of colliding particles '
                                            'did not return a empty array or a '
                                            'filled array with Particles.')

            # Check and purge the particle array for any deleted particles.
            particle_array = Hoku_Part.purge_particle_array(particle_array)
        
    # After the loop, the particles should be completely collided, and there
    #   exists no other possible collisions based on the current criteria.
    return particle_array



def compute_inelastic_particle_collision(particle1,particle2):
    """
    This function returns a single particle based on the collision of two
        particles. This should be only called if two particles are detected
        to have collided. All this function uses is linear momentum 
        conservation.

    This function currently does not allow for dark matter collision.
    """

    # Check that both particles are either dark matter or not. It is assumed
    #   that dark matter does not collide with normal matter. If the user 
    #   decides to override this and allow foreign collisions, let them.
    if (particle1.dark_matter != particle2.dark_matter):
        raise Hokulani_PhysicalError('Calculating dark matter collision with '
                                     'normal matter is considered as beyond '
                                     'this simulation\'s abilities.')
    elif (particle1.dark_matter or particle2.dark_matter):
        raise Hokulani_PhysicalError('Dark matter is more or less '
                                     'non-interacting with respect to '
                                     'collision mechanics.')
    else:
        new_particle_dark_m = False

    # Check to see that the particles are actually not supposed to have been
    #   previously deleted.
    if (particle1.delete or particle2.delete):
        raise Hokulani_Warning(Hokulani_ValueWarning,
                               ('One of the delete values of the particles '
                                'is true prior to collision. This should not '
                                'be the case. Some collision may have already '
                                'happened. Please double check.'))
    elif (particle1.delete and particle2.delete):
        raise Hokulani_ValueError('Both of the particles prior to collision '
                                  'have been marked for deletion. It may be '
                                  'that these pair of particles have already '
                                  'collided.')

    # Check which of the variety wins over in a collision of the two particles.
    new_particle_vary = particle_variety_comparison(particle1.variety,
                                                    particle2.variety)



    # Find the separation of the two particles. Use this to place the final
    #   resulting particle after collision
    sep_vector = particle2.position - particle1.position
    new_particle_pos = particle1.position + sep_vector/2

    # The collision is inelastic, therefore by the conservation of mass.
    total_particle_mass = particle1.mass + particle2.mass

    # The conservation of linear momentum drives inelastic particle.
    total_momentum = (particle1.return_momentum_vector() 
                      + particle2.return_momentum_vector())
    new_particle_vel = total_momentum / total_particle_mass

    # The particle, at the end of the collision, is assumed to not have any
    #   force acting on it, other than gravity which is to be calculated
    #   later on.
    new_particle_accel = np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)

    # Assign the new particle the new properties based on the collision. It
    #   should be assumed that this new post-collision particle is real.
    new_particle = Hoku_Part.parcel_particle(new_particle_pos,
                                             new_particle_vel,
                                             new_particle_accel,
                                             total_particle_mass,
                                             new_particle_vary,
                                             new_particle_dark_m,
                                             False)

    # The old particles should be considered 'destroyed'. Check if or if they
    #   are not.
    if not (particle1.delete):
        particle1.delete = True
    if not (particle2.delete):
        particle2.delete = True

    return new_particle



def compute_inelastic_array_collision(particle1_array,particle2_array,
                                      n_particles=CONFIG.NUMBER_OF_PARTICLES):
    """
    This function is to calculate the inelastic collision of a given array
        of particles. This is not the most efficient method of carrying out
        the math. It is expected in later versions that this function and the
        functions it depends on will be optimized.
    """

    # Check that particle 1 array and particle 2 array is the same size. If
    #   it is not, a particle cannot collide with itself. Also check if the
    #   number of particles provided may be applied.
    if (len(particle1_array) != len(particle2_array)):
        raise Hokulani_ShapeError('The length of the colliding particle '
                                  'arrays are not the same. Collisions must'
                                  'involve at least a pair of particles.')
    elif (len(particle1_array) != n_particles):
        raise Hokulani_Warning(Hokulani_ShapeWarning,
                               ('The number of particles given is not the '
                                'same as the length of the particle arrays. '
                                'The length of the particle arrays are '
                                'consistent and will be used instead.'))
        n_particles = len(particle1_array)

    # Loop over all particles, making the new particles 
    new_particle_array = []
    for particledex in range(n_particles):
        new_particle_array.append(
            compute_inelastic_particle_collision(particle1_array[particledex],
                                                 particle2_array[particledex]))

    # Return the new particles.
    return np.array(new_particle_array)


@numba.jit
def particle_variety_comparison(variety1,variety2,
                                particle1=None,particle2=None):
    """
    This function tests two different varieties, and 
        returns a new variety value or array with the new varieties after 
        a given reaction that would have combined the two values or 
        particles. If particles are given, the entire function uses the
        particles instead and disregards the mandatory values.
    """

    # Test to see if the user gave particles instead of values.
    if ((particle1 != None) and (particle2 != None)):
        variety1 = np.array([particle1.variety])
        variety2 = np.array([particle2.variety])
    else:
        variety1 = np.array([variety1])
        variety2 = np.array([variety2])

    # Store the variety values in deep copied variables.
    variety1_original = copy.deepcopy(variety1)
    variety2_original = copy.deepcopy(variety2)
    
    # Test to see if the two different variety arrays are the same size, or
    #   if they are arrays.
    if ((len(variety1) != len(variety2)) and 
        (isinstance(variety1,np.ndarray) and 
         (isinstance(variety2,np.ndarray)))):
        raise Hokulani_ShapeError('Both variety inputs are detected to be '
                                  'arrays; however, the sizes of these '
                                  'are not the same.')

    # Find the correct variety value(s) and return it.
    final_variety = np.where(variety1 < variety2,variety1_original,
                                                  variety2_original)

    # If the input was a single value, then return a single value, else
    #   return the array. The result should be a single value if the input
    #   was also a single value.
    if (len(final_variety) == 1):
        return final_variety[0]
    else:
        return final_variety
    
    


