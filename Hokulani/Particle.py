import numpy as np
import numba
import multiprocessing.dummy as multi_pro

from .Execptions import *
from . import Configuration as CONFIG
from . import Validation as Hoku_Valid
from . import Generation as Hoku_Gen
from . import Varieties as Hoku_Vary

class Particle:
    """
    Particle class: a class to store the information required for particles. 
        There are also specific functions for the derived information 
        of these particles.


    Object properties, these are all slotted properties:
        self.position       -- The position of the particle using a numpy 
                            array for the (x,y,z) point of the particle.
        self.velocity       -- The velocity of the particle, storing the 
                            (x,y,z) magnitude of the velocity vector.
        self.acceleration   -- The acceleration of the particle, storing the 
                            (x,y,z) magnitude of the acceleration vector.
        self.mass           -- The mass of the particle
        self.variety        -- The classification variety of the particle. 
        self.dark_matter    -- A boolean to tell if or if not this particle
                            is to be dark matter, if it is shown in 
                            visualization processes.
        self.delete         -- A boolean value to determine if this particle 
                            is to be deleted.

    Object methods/functions:
        return_position_scalar()
            -- Returns the magnitude, distance, of the position vector.
        return_position_direction()
            -- Returns the direction of the position vector of the particle.
        return_velocity_scalar()
            -- Returns the magnitude of the velocity vector.
        return_velocity_direction()
            -- Return the direction of the velocity vector, that is, it 
            returns the unitary vector of the velocity vector.
        return_acceleration_scalar()
            -- Return the magnitude of the acceleration vector.
        return_acceleration_direction()
            -- Return the direction of the acceleration vector, that is,
            it returns the unitary vector of the acceleration vector.
        return_momentum_vector()
            -- Returns the momentum vector of the particle. Assuming Newtonian
            mechanics, not including special relativity.
        return_momentum_scalar()
            -- Returns the magnitude of the momentum vector. 
        return_momentum_direction()
            -- Returns the direction of the momentum vector, that is, it 
            returns the unitary vector of the momentum vector.
        return_kinetic_energy()
            -- Returns the value of the kinetic energy of the particle. Only
            Newtonian mechanics are considered for the kinetic energy.
        
        print_info()
            -- Prints the information stored by the particle in a nicely 
            formatted system.
      
    """
    
    # Define the slots of the particle class object.
    __slots__ = ('position','velocity','acceleration',
                 'mass','variety',
                 'dark_matter','delete')
    
    def __init__(self, position, velocity, acceleration, 
                 mass, variety, dark_matter, delete):
        # Note, this should not be used by the user. Please use the 
        #   instantiate_particle(...) function or the  
        #   instantiate_particle_array(...) instead.

        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.acceleration = np.array(acceleration)
        self.mass = mass
        self.variety = variety
        self.dark_matter = dark_matter
        self.delete = delete
        
    # Beginning the object's functions.
    def return_position_scalar(self):
        """
        Function returns the magnitude of the position vector as a 
            scalar number.
        """
        return np.sqrt(np.dot(self.position,self.position))

    def return_position_direction(self):
        """
        Function returns the direction vector of the position vector.
        """
        pos_scalar = self.return_position_scalar()

        # If the magnitude of the vector is zero, then the direction vector
        #   is zero.
        if (pos_scalar == 0):
            return np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
        else:
            try: 
                pos_direction_vector = self.position/pos_scalar
            except ZeroDivisionError: 
                # Catch if the magnitude of the velocity is zero, by default 
                #    this implies that the velocity direction is a zero vector.
                pos_direction_vector = np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
            return np.array(pos_direction_vector) 

    def return_velocity_scalar(self):
        """
        Function returns the magnitude of the velocity vector as a 
            scalar number.
        """
        return np.sqrt(np.dot(self.velocity,self.velocity))

    def return_velocity_direction(self):
        """
        Function returns the direction vector of the velocity vector.
        """
        vel_scalar = self.return_velocity_scalar()
        
        # If the magnitude of the vector is zero, then the direction vector
        #   is zero.
        if (vel_scalar == 0):
            return np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
        else:
            try:
                vel_direction_vector = self.velocity/vel_scalar
            except ZeroDivisionError:
                # Catch if the magnitude of the velocity is zero, by default
                #   this implies that the velocity direction is a zero vector.
                vel_direction_vector = np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
            return np.array(vel_direction_vector) 
    
    def return_acceleration_scalar(self):
        """
        Function returns the magnitude of the acceleration vector as a 
            scalar number.
        """
        return np.sqrt(np.dot(self.acceleration,self.acceleration))
    
    def return_acceleration_direction(self):
        """
        Function returns the direction vector of the acceleration vector.
        """
        accel_scalar = self.return_acceleration_scalar()
        
        # If the magnitude of the vector is zero, then the direction vector
        #   is zero.
        if (accel_scalar == 0):
            return np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
        else:
            try:
                accel_direction_vector = self.acceleration/accel_scalar
            except ZeroDivisionError:
                # Catch if the magnitude of the velocity is zero, by default
                #   this implies that the velocity direction is a zero vector.
                accel_direction_vector = np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
            return np.array(accel_direction_vector) 

    def return_momentum_vector(self):
        """
        Returns the momentum vector of the particle. Assuming Newtonian
        mechanics, not including special relativity.
        """
        return self.mass * self.velocity
    
    def return_momentum_scalar(self):
        """
        Returns the magnitude of the momentum vector. 
        """
        momentum_vector = self.return_momentum_vector()
        
        return np.sqrt(np.dot(momentum_vector,momentum_vector))

    def return_momentum_direction(self):
        """
        Function returns the direction vector of the acceleration vector.
        """
        momentum_vector = self.return_momentum_vector()
        momentum_scalar = self.return_momentum_scalar()

        # If the magnitude of the vector is zero, then the direction vector
        #   is zero.
        if (momentum_scalar == 0):
            return np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
        else:
            try:
                momentum_direction_vector = momentum_vector/momentum_scalar
            except ZeroDivisionError:
                momentum_direction_vector = np.zeros(CONFIG.NUMBER_OF_DIMENSIONS)
            return np.array(momentum_direction_vector) 

    def return_kinetic_energy(self): 
        """
        Returns the value of the kinetic energy of the particle. Only
        Newtonian mechanics are considered for the kinetic energy.
        """
        return 0.5 * self.mass * return_velocity_scalar()**2

    def print(self,particle_number=1):
        # Convert the variety code to a string.
        vary_string = Hoku_Vary.convert_variety_code_to_string(self.variety)


        # Print out the information of the particle
        print('Printing Particle {num} Information:'
              .format(num=particle_number))
        print('Position:  {pos}    Velocity:  {vel}    Acceleration:  {accel}'
              .format(pos=self.position,
                      vel=self.velocity,
                      accel=self.acceleration))
        print('Mass:  {mas}'
              .format(mas=self.mass))
        print('Variety:  {vary}    Dark Matter:  {dark_m}'
              .format(vary=vary_string,
                      dark_m=self.dark_matter))
        print('Delete:  << {deletion} >>'
              .format(deletion=self.delete))


# Define the particle instantiation functions for particles. These functions
#   create an array of particles based on the array data that was fed into
#   the functions. 
###############

def instantiate_particle(position_val,velocity_val,acceleration_val,
                             mass_val,variety_val,dark_matter_val,
                             delete_val,
                             # Other simulation wide terms.
                             space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function makes a single particle object given a single value, or
        set in the case of the vectorial properties, per property. Advanced 
        type checking is performed to ensure that the particle contains 
        all of the correct elements.

    This should be used instead of the Particle class's inherent initializer.
    """

    # Parcel particle and instantiate particle have the same function.
    particle = parcel_particle(position_val,velocity_val,acceleration_val,
                             mass_val,variety_val,dark_matter_val,
                             delete_val,
                             # Other simulation wide terms.
                             space_dimensions)
    return particle

def instantiate_particle_array(n_particles=CONFIG.NUMBER_OF_PARTICLES,
                               system_tempeture=CONFIG.INITAL_TEMPETURE,
                               maximum_mass=CONFIG.MAX_GENERATE_PARTICLE_MASS,
                               dark_matter_fraction=CONFIG.DARK_MATTER_FRACTION,
                               # Simulation wide settings.
                               space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS,
                               simulation_size=CONFIG.SIMULATION_BOX_SIZE):
    """
    This function produces an array of particles for manipulation later on. 
        The particle properties are generated using the default methods for
        each particle. Basic information is still required.

        This function does not write an array of particle attributes into a 
        particle array. Please use parcel_particle_array(...) for that 
        purpose.
    """

    # First, define the mass of the particles. Mostly because the velocity 
    #   depends on the masses of the particles.
    mas_array = Hoku_Gen.generate_particle_mass_array(n_particles,
                                                       maximum_mass)

    # Next, define the kinematic arrays.
    pos_array = Hoku_Gen.generate_particle_position_array(n_particles,
                                                          simulation_size)

    vel_array = Hoku_Gen.generate_particle_velocity_array(n_particles,
                                                          mas_array,
                                                          system_tempeture)

    accel_array = Hoku_Gen.generate_particle_acceleration_array(n_particles)

    # Other particle properties.
    vary_array = Hoku_Gen.generate_particle_variety_array(n_particles)

    dark_m_array = Hoku_Gen.generate_particle_dark_matter_array(n_particles,
                                                                dark_matter_fraction)

    # Main particle properties.
    deletion_array = Hoku_Gen.generate_particle_deletion_array(n_particles)

    particle_array = parcel_particle_array(pos_array,
                                           vel_array,
                                           accel_array,
                                           mas_array,
                                           vary_array,
                                           dark_m_array,
                                           deletion_array,
                                           # Simulation wide parameters.
                                           space_dimensions)

    return particle_array


# Define the parceling functions. The parceling functions do the opposite 
#   of the extraction functions. The parceling functions create an array of 
#   particles given arrays of the properties of the particles.

def parcel_particle(position_val,velocity_val,acceleration_val,
                             mass_val,variety_val,dark_matter_val,
                             delete_val,
                             # Other simulation wide terms.
                             space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function makes a single particle object given a single value, or
        set in the case of the vectorial properties, per property. Advanced 
        type checking is performed to ensure that the particle contains 
        all of the correct elements.
    """

    # Type check all of the variables, if there is any false value, then bark
    #   at the user if the value is unworkable.
    pos_val = Hoku_Valid.validate_position_value(position_val,
                                                 space_dimensions)

    vel_val = Hoku_Valid.validate_velocity_value(velocity_val,
                                                 space_dimensions)

    accel_val = Hoku_Valid.validate_acceleration_value(acceleration_val,
                                                       space_dimensions)

    mas_val = Hoku_Valid.validate_mass_value(mass_val)

    vary_val = Hoku_Valid.validate_variety_code(variety_val)

    dark_m_val = Hoku_Valid.validate_dark_matter_value(dark_matter_val)


    del_val = Hoku_Valid.validate_delete_value(delete_val)

    # If all of the checks have passed, make a particle.
    parcelled_particle = Particle(pos_val,vel_val,accel_val,
                                  mas_val,vary_val,dark_m_val,
                                  del_val)

    return parcelled_particle


def parcel_particle_array(position_array,velocity_array,acceleration_array,
                          mass_array,variety_array,dark_matter_array,
                          delete_array,
                          # Simulation wide parameters.
                          space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function takes the different array properties of each different 
        particle. These values are then arranged into an array of particles 
        instead, all containing the same information within the particle 
        class's organization.
    """
    
    # Assume that the deletion array is always correct.
    n_particles = len(delete_array)

    # Checking different areas helps ensure correctness.
    if (space_dimensions != len(position_array[0]) or
        space_dimensions != len(velocity_array[-1]) or 
        space_dimensions != len(acceleration_array[0])):
        # Check if the values of the length of the position, velocity, or
        #   acceleration array are actually the same.
        if not (len(position_array[0]) 
                == len(velocity_array[-1]) 
                == len(acceleration_array[0])):
            raise Hokulani_ShapeError(('The number of dimensions of the '
                                      'position array, velocity array, and '
                                      'the acceleration array are not equal.'))
        else:
            Hokulani_Warning(Hokulani_ShapeWarning,
                             ('The number of space dimensions inputted'
                              'conflicts with the data. Constructing a new'
                              'value from the data.'))
            space_dimensions = len(position_array[0])

    # Second, do advanced type checking to make sure that the inputs are 
    #   workable. 
    pos_array = Hoku_Valid.validate_position_array(position_array,
                                                   n_particles,
                                                   space_dimensions)

    vel_array = Hoku_Valid.validate_velocity_array(velocity_array,
                                                   n_particles,
                                                   space_dimensions)

    accel_array = Hoku_Valid.validate_acceleration_array(acceleration_array,
                                                         n_particles,
                                                         space_dimensions)
    
    mas_array = Hoku_Valid.validate_mass_array(mass_array,n_particles)

    vary_array = Hoku_Valid.validate_variety_code_array(variety_array,
                                                        n_particles)

    dark_m_array = Hoku_Valid.validate_dark_matter_array(dark_matter_array,
                                                         n_particles)

    deletion_array = Hoku_Valid.validate_delete_array(delete_array,
                                                      n_particles)


    # Use multi-threading. This value is hard coded for testing.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_list = pool.starmap(Particle,
                                     zip(pos_array,vel_array,accel_array,
                                         mas_array,vary_array,
                                         dark_m_array,deletion_array))
    else:
        # Combine these arrays into the particle's values. A priori it 
        #   seems to be faster to use python lists then convert to array. 
        particle_list = []
        for particledex in range(n_particles):
            particle_list.append(
                Particle(pos_array[particledex],
                         vel_array[particledex],
                         accel_array[particledex],
                         mas_array[particledex],
                         vary_array[particledex], 
                         dark_m_array[particledex],
                         deletion_array[particledex]))

    # Return the particle list as a numpy array by conversion.
    return np.array(particle_list)

def reparcel_particle_array(particle_array,
                            position_array=None,
                            velocity_array=None,
                            acceleration_array=None,
                            mass_array=None,
                            variety_array=None,
                            dark_matter_array=None,
                            delete_array=None,
                            # Simulation wide parameters.
                            n_particles=CONFIG.NUMBER_OF_PARTICLES,
                            space_dimensions=CONFIG.NUMBER_OF_DIMENSIONS):
    """
    This function takes optional inputs, a array of particle value inputs to
        update. This should be only used if some value of all particles in an
        array should be updated, but everything is to remain the same.
    If the entire particle array is to be rebuilt from scratch, or all values
        have arrays to change the particle array, it is suggested to re-parcel
        the array using parcel_particle_array().
        
    """

    # Type check all of the arrays that the user provided, ensure that they
    #   are compatible with the usage of this simulation.
    if not (position_array is None):
        position_array = Hoku_Valid.validate_position_array(position_array,
                                                            n_particles,
                                                            space_dimensions)
    if not (velocity_array is None):
        velocity_array = Hoku_Valid.validate_velocity_array(velocity_array,
                                                            n_particles,
                                                            space_dimensions)
    if not (acceleration_array is None):
        acceleration_array = Hoku_Valid.validate_acceleration_array(acceleration_array,
                                                                    n_particles,
                                                                    space_dimensions)
    if not (mass_array is None):
        mass_array = Hoku_Valid.validate_mass_array(mass_array,
                                                    n_particles)
    if not (variety_array is None):
        variety_array = Hoku_Valid.validate_variety_array(variety_array,
                                                          n_particles)
    if not (dark_matter_array is None):
        dark_matter_array = Hoku_Valid.validate_dark_matter_array(dark_matter_array,
                                                                  n_particles)
    if not (delete_array is None):
        delete_array = Hoku_Valid.validate_delete_array(delete_array,
                                                        n_particles)

    # Of the arrays that are valid and do seem to exist, replace the values.
    #   Assume that the values of the array is in the right index for the
    #   re-parceling function.
    for particledex in range(n_particles):
        if not (position_array is None):
            particle_array[particledex].position = position_array[particledex]
        if not (velocity_array is None):
            particle_array[particledex].velocity = velocity_array[particledex]
        if not (acceleration_array is None):
            particle_array[particledex].acceleration = acceleration_array[particledex]
        if not (mass_array is None):
            particle_array[particledex].mass = mass_array[particledex]
        if not (variety_array is None):
            particle_array[particledex].variety = variety_array[particledex]
        if not (dark_matter_array is None):
            particle_array[particledex] = dark_matter_array[particledex]

    # The particle array should have all of its information updated.
    return np.array(particle_array)

# Define the extraction functions. The extraction functions extract, 
#   from the particle class, a numpy array of the property of the 
#   function.
###############


def extract_position_array(particle_array):
    """
    Return the position array of the particles. The position array is the
        position vector of all the particles. This is repeated over
        all particles. This is an inefficient function.
    """

    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_pos_list = pool.map(lambda particle: particle.position, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        n_particles = len(particle_array)
        particle_pos_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_pos = particle_array[particledex].position

            particle_pos_list.append(particle_pos)

    return np.array(particle_pos_list)
    

def extract_velocity_array(particle_array):
    """
    Return the velocity array of the particles. The velocity array is the
        velocity vector of all the particles. This is repeated over
        all particles. This is an inefficient function.
    """
    
    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_vel_list = pool.map(lambda particle: particle.velocity, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_vel_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_vel = particle_array[particledex].velocity

            particle_vel_list.append(particle_vel)

    return np.array(particle_vel_list)


def extract_acceleration_array(particle_array):
    """
    Return the acceleration array of the particles. The acceleration  
        array is the acceleration vector of all the particles. This is 
        repeated over all particles. This is an inefficient function.
    """
    
    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_accel_list = pool.map(lambda particle: particle.acceleration, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_accel_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_accel = particle_array[particledex].acceleration

            particle_accel_list.append(particle_accel)

    return np.array(particle_accel_list)


def extract_mass_array(particle_array):
    """
    Return the mass array of the particles. The mass array 
        is the mass of all the particles. This is repeated over
        all particles. This is an inefficient function.
    """
    
    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_mass_list = pool.map(lambda particle: particle.mass, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_mass_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_mass = particle_array[particledex].mass

            particle_mass_list.append(particle_mass)

    return np.array(particle_mass_list)


def extract_variety_array(particle_array):
    """
    Return the variety array of the particles. The variety array 
        is the variety of all the particles. This is repeated over
        all particles. This is an inefficient function.
    """

    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_variety_list = pool.map(lambda particle: particle.variety, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_variety_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_variety = particle_array[particledex].variety

            particle_variety_list.append(particle_variety)

        return np.array(particle_variety_list)


def extract_dark_matter_array(particle_array):
    """
    Return the dark matter array of the particles. The dark matter  
        array is the dark matter value of all the particles. This is
        repeated over all particles. This is an inefficient function.
    """

    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_dark_list = pool.map(lambda particle: particle.dark_matter, 
                                      particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_dark_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_dark = particle_array[particledex].dark_matter

            particle_dark_list.append(particle_dark)

    return np.array(particle_dark_list)


def extract_delete_array(particle_array):
    """
    Return the deletion array of the particles. The deletion  
        array is the delete value of all the particles. This is
        repeated over all particles. This is an inefficient function.
    """
    
    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_delete_list = pool.map(lambda particle: particle.delete, 
                                         particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_delete_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_delete = particle_array[particledex].delete

            particle_delete_list.append(particle_delete)

    return np.array(particle_delete_list)


def extract_momentum_array(particle_array):
    """
    Return the momentum array of the particles. The momentum  
        array is the momentum vector value of all the particles. This is
        repeated over all particles. This is an inefficient function.
    """

    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_momen_list = pool.map(
            lambda particle: particle.return_momentum_vector(),
            particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_momen_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_momen = particle_array[particledex].return_momentum_vector()

            particle_momen_list.append(particle_momen)

    return np.array(particle_momen_list)

def extract_kinetic_energy_array(particle_array):
    """
    Return the kinetic energy array of the particles. The kinetic energy  
        array is the kinetic energy value of all the particles. This is
        repeated over all particles. This is an inefficient function.
    """

    # See if the user wants to use multi-threading. Generally, this is slower.
    if (CONFIG.MULTITHREAD_COUNT > 1):
        # Generate a multiprocessing pool.
        pool = multi_pro.Pool(CONFIG.MULTITHREAD_COUNT)
    
        # Execute multi-threaded extraction of the particle position's array. 
        #   Retrieving the position must be done by a lambda function as the 
        #   input of the pool map function requires a function.
        particle_kinerg_list = pool.map(
            lambda particle: particle.return_kinetic_energy(),
            particle_array)

        # Close the pool and join.
        pool.close()
        pool.join()
    else:
        # Obtaining variables.
        n_particles = len(particle_array)
        particle_kinerg_list = []

        # List the position values of all particles.
        for particledex in range(n_particles):
            particle_kinerg = particle_array[particledex].return_kinetic_energy()

            particle_kinerg_list.append(particle_kinerg)

    return np.array(particle_kinerg_list)



# For particles that have the deletion tag, remove them from the particle
#   array. 
def purge_particle_array(particle_array):
    """
    This function removes all particles in a particle array that have been 
        tagged for deletion. This accomplish the task by only returning 
        a particle array only with particles not tagged for deletion.
    """

    # Extract the delete array from the particle array.
    delete_array = extract_delete_array(particle_array)

    # Find those particles that are not considered to be deleted, keep track
    #   of their indices.
    real_particles_location = np.nonzero(np.logical_not(delete_array))
    real_particles_location = np.array(real_particles_location).flatten()


    # Raise an error if all of the particles have vanished, this would be a 
    #   problem with the simulation.
    if (real_particles_location.size == 0):
        raise Hokulani_SanityError('It has been detected that all particles '
                                   'have been tagged for deletion and all '
                                   'particles in this simulation should be '
                                   'deleted.')
    
    # Return the valid particles only.
    return np.array(particle_array[real_particles_location])

