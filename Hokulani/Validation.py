import numpy as np

from .Execptions import *
from . import Particle as Hoku_Part

# Begin defining the validation functions for the configuration information.
def validate_simulation_box_size(box_size):
    """
    This functions validates that the box size of the simulation is a logical
        value. If it is, then return such value, if not, bark at the user.
    """

    if not (isinstance(box_size,(int,float))):
        raise Hokulani_TypeError(('The input type of the simulation box '
                                  'is invalid. Expected: {type1} or {type2}.')
                                  .format(type1=type(int),type2=type(float)))
    elif (box_size <= 0):
        raise Hokulani_PhysicalError(('The size of the simulation box cannot '
                                     'be a negative number or a zero.'))
    else:
        return box_size


def validate_number_of_dimensions(space_dimensions):
    """
    This functions validates that the number of spatial dimensions of the 
        simulation is a logical value. If it is, then return such value, 
        if not, bark at the user.
    """

    if not (isinstance(space_dimensions,int)):
        raise Hokulani_TypeError(('The input type of the number of '
                                  'dimensions is invalid. Expected: '
                                  '{type1}.')
                                  .format(type1=type(int)))
    elif (space_dimensions <= 0):
        raise Hokulani_PhysicalError(('The number of dimensions cannot '
                                     'be a negative number or a zero.'))
    else:
        return space_dimensions 


def validate_number_of_particles(n_particles):
    """
    This functions validates that the number of particles in the 
        simulation is a logical value. If it is, then return such value, 
        if not, bark at the user.
    """

    if not (isinstance(n_particles,int)):
        raise Hokulani_TypeError(('The input type of the number of '
                                  'particles is invalid. Expected: '
                                  '{type1}.')
                                  .format(type1=type(int)))
    elif (n_particles <= 0):
        raise Hokulani_PhysicalError(('The number of particles cannot '
                                     'be a negative number or a zero.'))
    elif (n_particles == 1):
        Hokulani_Warning(Hokulani_PhysicalWarning,
                         ('There is only one particle. A simulation might be '
                          'a bit pointless.'))
    else:
        return n_particles 


def validate_max_generate_particle_mass(particle_mass):
    """
    This functions validates that the number of for the maximum particle 
        mass that can be generated of the simulation is a logical value. 
        If it is, then return such value, if not, bark at the user.
    """

    if not (isinstance(particle_mass,(int,float))):
        raise Hokulani_TypeError(('The input type of the number of '
                                  'dimensions is invalid. Expected: '
                                  '{type1} or {type2}.')
                                  .format(type1=type(int),
                                          type2=type(float)))
    elif (particle_mass <= 0):
        raise Hokulani_PhysicalError(('The maximum generated particle mass '
                                     'cannot be a negative number or a zero.'))
    else:
        return particle_mass 


def validate_inital_tempeture(tempeture):
    """
    This function validates that the value of the initial temperature is valid.
        If it is, return it to the user, if not, bark at the user.
    """

    if not (isinstance(tempeture,(int,float))):
        raise Hokulani_TypeError(('The input type of the initial temperature '
                                  'is invalid. Expected: {type1} or {type2}.')
                                  .format(type1=type(int),type2=type(float)))
    elif (tempeture <= 0):
        raise Hokulani_PhysicalError(('The value of the temperature may not be '
                                     'a negative or zero value.'))
    else:
        return tempeture


def validate_dark_matter_fraction(dark_matter_fraction):
    """
    This function validates that the value of the dark matter fraction is 
        valid. If it is, return it to the user, if not, bark at the user.
    """

    if not (isinstance(dark_matter_fraction,(int,float))):
        raise Hokulani_TypeError(('The input type of the dark matter fraction '
                                  'is invalid. Expected: {type1} or {type2}.')
                                  .format(type1=type(int),type2=type(float)))
    elif (dark_matter_fraction < 0):
        raise Hokulani_PhysicalError(('The value of the dark matter fraction '
                                     'may not be a negative value.'))
    else:
        return dark_matter_fraction


# Begin defining the validation functions for a single value of data
#   (mostly for a single particle).
def validate_particle(particle,space_dimensions):
    """
    This function validates a single particle to see if the particle is an
        object derived from the Particle class within this simulation, or if 
        the values contained within the particle is also valid.
    """

    # Check if the particle actually is a particle. 
    if not (isinstance(particle,Hoku_Part.Particle)):
        raise Hokulani_TypeError('The inputted particle is not an object '
                                 'derived from the Particle class from the '
                                 'simulation.')

    # Check to see if the values of the particle is correct.
    particle.position = validate_position_value(particle.position,
                                                space_dimensions)
    particle.velocity = validate_position_value(particle.velocity,
                                                space_dimensions)
    particle.acceleration = validate_position_value(particle.acceleration,
                                                    space_dimensions)

    particle.mass = validate_mass_value(particle.mass,
                                        space_dimensions)

    particle.variety = validate_variety_value(particle.variety)
    particle.dark_matter = validate_dark_matter_value(particle.dark_matter)
    particle.delete = validate_delete_value(particle.delete)

    return particle

def validate_position_value(position_val,space_dimensions):
    """
    This function validates that the position value array or list is 
        the right type and that the values of the list or array are valid. 
        If it is not correct, or it cannot be turned into the correct format, 
        bark at the user. If it can be, return the data correctly formatted.
    """
    try:
        pos_val = np.array(position_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted position value cannot '
                                  'be turned into a numpy array.'))

    try:
        len_pos_val = len(pos_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted position value does not '
                                  'have a length element, and therefore '
                                  'cannot be processed by other functions.'))

    if not (isinstance(pos_val,(np.ndarray,list))):
        raise Hokulani_TypeError(('Position value input is not the right '
                                 'type. Expected: {type1} or {type2}.')
                                 .format(type1=type(np.ndarray),
                                         type2=type(list)))
    elif (len_pos_val != space_dimensions):
        raise Hokulani_ShapeError(('Position value input is not the right '
                                  'shape. Expected: {size1}')
                                 .format(size1=np.shape(
                                     np.zero([space_dimensions]))))
    else:
        return pos_val


def validate_velocity_value(velocity_val,space_dimensions):
    """
    This function validates that the velocity value array or list is 
        the right type and that the values of the list or array are valid. 
        If it is not correct, or it cannot be turned into the correct format, 
        bark at the user. If it can be, return the data correctly formatted.
    """
    try:
        vel_val = np.array(velocity_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted velocity value cannot be '
                                 'turned into a numpy array.'))

    try:
        vel_len_val = len(vel_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted velocity value does not '
                                  'have a length element, and therefore '
                                  'cannot be processed by other functions.'))

    if not (isinstance(vel_val,(np.ndarray,list))):
        raise Hokulani_TypeError(('Velocity value input is not the right '
                                 'type. Expected: {type1} or {type2}.')
                                 .format(type1=type(np.ndarray),
                                         type2=type(list)))
    elif (vel_len_val != space_dimensions):
        raise Hokulani_ShapeError(('Velocity value input is not the right '
                                  'shape. Expected: {size1}')
                                 .format(size1=np.shape(
                                     np.zero([space_dimensions]))))
    else:
        return vel_val


def validate_acceleration_value(acceleration_val,space_dimensions):
    """
    This function validates that the acceleration value array or list is 
        the right type and that the values of the list or array are valid. 
        If it is not correct, or it cannot be turned into the correct format, 
        bark at the user. If it can be, return the data correctly formatted.
    """
    try:
        accel_val = np.array(acceleration_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted position value cannot be '
                                 'turned into a numpy array.'))

    try:
        accel_len_val = len(accel_val)
    except Exception:
        raise Hokulani_TypeError(('The inputted acceleration value does not '
                                  'have a length element, and therefore '
                                  'cannot be processed by other functions.'))

    if not (isinstance(accel_val,(np.ndarray,list))):
        raise Hokulani_TypeError(('Acceleration value input is not the right '
                                 'type. Expected: {type1} or {type2}.')
                                 .format(type1=type(np.ndarray),
                                         type2=type(list)))
    elif (accel_len_val != space_dimensions):
        raise Hokulani_ShapeError(('Acceleration value input is not the '
                                  'right shape. Expected: {size1}')
                                 .format(size1=np.shape(
                                     np.zero([space_dimensions]))))
    else:
        return accel_val


def validate_mass_value(mass_val):
    """
    This function validates that the mass value is the right type and that
        the value is valid. If it is not correct, or it cannot be turned
        into the correct format, bark at the user. If it can be, return the 
        data correctly formatted.
    """
    if not (isinstance(mass_val,(int,float,CONFIG.NP_FLOAT_DTYPE))):
        raise Hokulani_TypeError('Mass value input is not the right '
                                 'type.')
    elif (mass_val <= 0):
        raise Hokulani_PhysicalError('Mass value is detected to be negative.')
    else:
        return np.array(mass_val,dtype=CONFIG.NP_FLOAT_DTYPE)


def validate_variety_code(variety_code):
    """
    This function validates that the variety code given is a valid code
        that can be processed. If it is not correct, or if it cannot be 
        turned into the correct format, bark at the user. If it can be, 
        return the data correctly formatted.
    """

    # Test if the entry given is actually a integer. Attempt to turn it into
    #   one if it is not.
    if (type(variety_code) is np.dtype(np.uint8)):
        pass
    else:
        try:
            variety_code = np.uint8(variety_code)
        except Exception:
            raise Hokulani_TypeError('Variety code cannot be converted into '
                                     'a integer value.')

    # Test to see if the entry is within the range of the variety codes. It
    #   is assumed that the variety codes start at 0.
    if not (0 <= variety_code < len(CONFIG.VARIETY_DICTIONARY)):
        raise Hokulani_ValueError('Variety code is not within a range '
                                  'covered by the current variety dictionary.')

    return variety_code


def validate_variety_string(variety_string):
    """
    This function validates that the variety string value is the right type 
        and that the value is valid. If it is not correct, or it cannot 
        be turned into the correct format, bark at the user. If it can be, 
        return the data correctly formatted.
    """

    # First, test that the value of the variety string to make sure that it 
    #   is a string.
    if (isinstance(variety_string,str)):
        pass
    else:
        try:
            variety_string = str(variety_string)
        except Exception:
            raise Hokulani_TypeError('Variety string is not detected to be a '
                                     'string, nor can it be turned into one.')

    return variety_string

    
def validate_dark_matter_value(dark_matter_val):
    """
    This function validates that the dark matter value is the right type 
        and that the value is valid. If it is not correct, or it cannot 
        be turned into the correct format, bark at the user. If it can be, 
        return the data correctly formatted.
    """
    if (isinstance(dark_matter_val,np.bool_)):
        pass
    else:
        try:
            dark_matter_val = np.bool_(dark_matter_val)
        except Exception:
            raise Hokulani_TypeError(('Dark matter value input is not the right '
                                      'type. Expected: {type1}.')
                                      .format(type1=type(np.bool_)))
    return dark_matter_val


def validate_delete_value(delete_val):
    """
    This function validates that the delete value is the right type and 
        that the value is valid. If it is not correct, or it cannot be 
        turned into the correct format, bark at the user. If it can be,  
        return the data correctly formatted.
    """
    if (isinstance(delete_val,np.bool_)):
        pass
    else:
        try:
            delete_val = np.bool_(delete_val)
        except Exception:
            raise Hokulani_TypeError(('Delete value input is not the right '
                                      'type. Expected: {type1}.')
                                      .format(type1=type(np.bool_)))
    if (delete_val):
        Hokulani_Warning(Hokulani_TypeWarning,('Delete value is {val1}, '
                         'which is currently detected to be true.')
                         .format(val1=delete_val))

    return delete_val


# Begin defining the validation functions for arrays of data (mostly for
#   arrays of particles).
##############

def validate_particle_array(particle_array,n_particles,space_dimensions,
                            validate_all_particles=False,):
    """
    This function validates that the particle array inputted is a valid 
        numpy array of particles. It barks back at the use if it is not.
        Also, this function contains the functionality of calling a related
        function to scan all particles within that array.
    """

    # Check to see that particle array is a numpy array, of, if it is
    #   not, check to see if it can be converted.
    if not (isinstance(particle_array,np.ndarray)):
        try: 
            particle_array = np.array(particle_array)
        except Exception:
            raise Hokulani_TypeError('Particle array input is not a numpy '
                                     'array, nor can it be turned into a '
                                     'numpy array.')

    # Check to see if the total number of particles is the same as the 
    #   length of the particle array.
    if (len(particle_array) != n_particles):
        raise Hokulani_ShapeError('Particle array and the inputted number '
                                  'of particles does not seem to be the same.')

    # Check to see if the objects within the particle array are of the 
    #   Particle class made in this simulation.
    if not (isinstance(particle_array[0],Hoku_Part.Particle)):
        raise Hokulani_TypeError('Particle array is not an array of Particles '
                                 'derived from this simulation\'s Particle '
                                 'class.')

    # If the user also wanted all particles to be evaluated, check all of
    #   the particles in the particle array.
    if (validate_all_particles):
        for index,particledex in enumerate(particle_array):
            particle_array[index] = validate_particle(particledex,
                                                      space_dimensions)
    
    return np.array(particle_array)


def validate_position_array(position_array,n_particles,space_dimensions):
    """
    This function validates that the position array inputted is a 
        valid data type and size. It barks back at the user if it is not 
        the right size or type (and thus suboptimal usage at best, at worse, 
        it breaks the program). If it passes the type checking or it is  
        able to be turned into the correct type and size, then it returns the 
        data in the correct way.
    """
    try:
        pos_array = np.array(position_array,dtype=CONFIG.NP_FLOAT_DTYPE)
    except Exception:
        raise Hokulani_TypeError(('The inputted position array cannot be '
                                 'turned into a numpy array.'))

    if (pos_array.dtype != CONFIG.NP_FLOAT_DTYPE):
        raise Hokulani_TypeError(('The position array does not contain the '
                                 'right value types. Expected: {type1}.')
                                 .format(type1=CONFIG.NP_FLOAT_DTYPE))
    elif (pos_array.shape != (n_particles,space_dimensions)):
        raise Hokulani_ShapeError(('The position array does not have the '
                                  'right shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}.')
                                  .format(cur_shape=pos_array.shape,
                                          exp_shape=(n_particles,
                                                     space_dimensions)))
    else:
        return pos_array.astype(CONFIG.NP_FLOAT_DTYPE)


def validate_velocity_array(velocity_array,n_particles,space_dimensions):
    """
    This function validates that the velocity array inputted is a valid 
        data type and size. It barks back at the user if it is not the right 
        size or type (and thus suboptimal usage at best, at worse, it breaks 
        the program). If it passes the type checking or it is able to be 
        turned into the correct type and size, then it returns the data 
        in the correct way.
    """
    try:
        vel_array = np.array(velocity_array,dtype=CONFIG.NP_FLOAT_DTYPE)
    except Exception:
        raise Hokulani_TypeError(('The inputted velocity array cannot be '
                                 'turned into a numpy array.'))

    if (vel_array.dtype != CONFIG.NP_FLOAT_DTYPE):
        raise Hokulani_TypeError(('The velocity array does not contain the '
                                 'right value types. Expected: {type1}.')
                                 .format(type1=CONFIG.NP_FLOAT_DTYPE))
    elif (vel_array.shape != (n_particles,space_dimensions)):
        raise Hokulani_ShapeError(('The velocity array does not have the '
                                  'right shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}.')
                                  .format(cur_shape=vel_array.shape,
                                          exp_shape=(n_particles,
                                                     space_dimensions)))
    else:
        return vel_array.astype(CONFIG.NP_FLOAT_DTYPE)


def validate_acceleration_array(acceleration_array,n_particles,space_dimensions):
    """
    This function validates that the acceleration array inputted is 
        a valid data type and size. It barks back at the user if it is not 
        the right size or type (and thus suboptimal usage at best, at worse, 
        it breaks the program). If it passes the type checking or it is able 
        to be turned into the correct type and size, then it returns the 
        data in the correct way.
    """
    try:
        accel_array = np.array(acceleration_array,dtype=CONFIG.NP_FLOAT_DTYPE)
    except Exception:
        raise Hokulani_TypeError(('The inputted acceleration array cannot '
                                 'be turned into a numpy array.'))

    if (accel_array.dtype != CONFIG.NP_FLOAT_DTYPE):
        raise Hokulani_TypeError(('The acceleration array does not contain '
                                 'the right value types. Expected: {type1}.')
                                 .format(type1=CONFIG.NP_FLOAT_DTYPE))
    elif (accel_array.shape != (n_particles,space_dimensions)):
        raise Hokulani_ShapeError(('The acceleration array does not have the '
                                  'right shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}.')
                                  .format(cur_shape=accel_array.shape,
                                          exp_shape=(n_particles,
                                                     space_dimensions)))
    else:
        return accel_array.astype(CONFIG.NP_FLOAT_DTYPE)


def validate_mass_array(mass_array,n_particles):
    """
    This function validates that the mass array inputted is a valid 
        data type and size. It barks back at the user if it is not the right 
        size or type (and thus suboptimal usage at best, at worse, it breaks 
        the program). If it passes the type checking or it is able to be 
        turned into the correct type and size, then it returns the data in 
        the correct way.
    """
    try:
        mas_array = np.array(mass_array,dtype=CONFIG.NP_FLOAT_DTYPE)
    except Exception:
        raise Hokulani_TypeError(('The inputted mass array cannot be turned '
                                 'into a numpy array.'))

    if (mas_array.dtype != CONFIG.NP_FLOAT_DTYPE):
        raise Hokulani_TypeError(('The mass array does not contain the right '
                                 'value types. Expected: {type1}.')
                                 .format(type1=CONFIG.NP_FLOAT_DTYPE))
    elif (mas_array.shape != (n_particles,)):
        raise Hokulani_ShapeError(('The mass array does not have the right '
                                  'shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}')
                                  .format(cur_shape=mas_array.shape,
                                          exp_shape=(n_particles,)))
    elif (np.any(mas_array < 0)):
        raise Hokulani_PhysicalError(('The mass array has at least one value '
                                     'that is detected to be negative.'))
    else:
        return mas_array.astype(CONFIG.NP_FLOAT_DTYPE)


def validate_variety_code_array(variety_array,n_particles):
    """
    This function validates that the variety array inputted is a valid 
        data type and size. It barks back at the user if it is not the right 
        size or type (and thus suboptimal usage at best, at worse, it breaks 
        the program). If it passes the type checking or it is able to be 
        turned into the correct type and size, then it returns the data 
        in the correct way.
    """
    try:
        vary_array = np.array(variety_array,dtype=np.uint8)
    except Exception:
        raise Hokulani_TypeError(('The inputted variety array cannot be '
                                 'turned into a numpy array.'))

    if (vary_array.dtype.kind != np.dtype(np.uint8).kind):
        raise Hokulani_TypeError(('The variety array does not contain the '
                                 'right value types. Expected: {type1}.')
                                 .format(type1=np.uint8))
    elif (vary_array.shape != (n_particles,)):
        raise Hokulani_ShapeError(('The variety array does not have the '
                                  'right shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}')
                                  .format(cur_shape=mas_array.shape,
                                          exp_shape=(n_particles,)))
    else:
        return vary_array


def validate_dark_matter_array(dark_matter_array,n_particles):
    """
    This function validates that the dark matter array inputted is 
        a valid data type and size. It barks back at the user if it is 
        not the right size or type (and thus suboptimal usage at best, 
        at worse, it breaks the program). If it passes the type checking 
        or it is able to be turned into the correct type and size, then 
        it returns the data in the correct way.
    """
    try:
        dark_m_array = np.array(dark_matter_array,dtype=bool)
    except Exception:
        raise Hokulani_TypeError(('The inputted dark matter array cannot '
                                 'be turned into a numpy array.'))

    if (dark_m_array.dtype != bool):
        raise Hokulani_TypeError(('The dark matter array does not contain '
                                 'the right value types. Expected: {type1}.')
                                 .format(type1=bool))
    elif (dark_m_array.shape != (n_particles,)):
        raise Hokulani_ShapeError(('The dark matter array does not have the '
                                  'right shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}')
                                  .format(cur_shape=dark_m_array.shape,
                                          exp_shape=(n_particles,)))
    else:
        return dark_m_array


def validate_delete_array(delete_array,n_particles):
    """
    This function validates that the delete array inputted is a valid 
        data type and size. It barks back at the user if it is not the right 
        size or type (and thus suboptimal usage at best, at worse, it breaks 
        the program). If it passes the type checking or it is able to be 
        turned into the correct type and size, then it returns the data in 
        the correct way.
    """

    try:
        deletion_array = np.array(delete_array,dtype=bool)
    except Exception:
        raise Hokulani_TypeError(('The inputted delete array cannot be '
                                 'turned into a numpy array.'))

    if (deletion_array.dtype != bool):
        raise Hokulani_TypeError(('The delete array does not contain the '
                                 'right value types. Expected: {type1}.')
                                 .format(type1=bool))
    elif (deletion_array.shape != (n_particles,)):
        raise Hokulani_ShapeError(('The delete array does not have the right '
                                  'shape. Current: {cur_shape}   '
                                  'Expected: {exp_shape}')
                                  .format(cur_shape=deletion_array.shape,
                                          exp_shape=(n_particles,)))
    elif (np.any(deletion_array)):
        Hokulani_Warning(Hokulani_TypeWarning,('Delete array currently '
                         'contains one detected true value.'))
    else:
        return deletion_array

