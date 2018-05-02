import warnings

from . import Configuration as CONFIG

# First, define all of the errors that stop the program from continuing.
##########

class Hokulani_Error(Exception):
    """
    This is the general exception class for all of Hokulani's
    operations and custom defined exceptions. By default, they
    all contain a header of Hokulani.
    """
    pass

class Hokulani_SanityError(Hokulani_Error):
    """
    This is a error that tells the user that something that they instigated
        might make little sense or might lead to very unexpected results. An
        example may be the removal of all particles.
    """

    def __init__(self,message):
        self.message = message

class Hokulani_ValueError(Hokulani_Error):
    """
    This error is triggered if the user inputs incorrect values for the 
    parameters for the Hokulani simulation's modules.
    """

    def __init__(self, message):
        self.message = message

class Hokulani_TypeError(Hokulani_Error):
    """
    This error is triggered if the user inputs incorrect parameters
    for the Hokulani simulation's modules.
    """

    def __init__(self, message):
        self.message = message


class Hokulani_ShapeError(Hokulani_Error):
    """
    This error is triggered if the user inputs, or if the program tries to
        manipulate an array that does not have the expected shape. An example 
        is if the number of particles is less than expected.
    """

    def __init__(self, message):
        self.message = message


class Hokulani_PhysicalError(Hokulani_Error):
    """
    This error is triggered if the user inputs incorrect parameters
    for the Hokulani simulation's modules. This is triggered if the user
    input some physically impossible value (ex. negative mass).
    """

    def __init__(self, message):
        self.message = message

class Hokulani_UnknownError(Hokulani_Error):
    """
    This error is triggered if the module tries and do something that does 
        not make any sense. It should be that this message should be used
        if the program, and not the user, did something wrong.

    Add to the error this may be a bug, and to constant the developer.
    """

    def __init__(self,message):
        self.message = message

    def __str__(self):
        error_message = (self.message 
                        + ('\n\n'
                           '    Attention! This is not your fault. \n'
                           '      = + = + = + = + = + = \n'
                           '    If this error comes up, this is likely \n'
                           '    means there is something wrong with \n'
                           '    Hokulani itself. \n'
                           '    Please contact the developer.'))
        return error_message

# Define all of the warnings that will arise, though shall not stop,
#     throughout the program.
##########

def Hokulani_Warning(warning,message):
    """
    This is the main warning function to print warnings to the user. These
        are not printed if the user decided to suppress warnings. The
        suppression is done in this function call rather than every individual
        function.
    """

    # Check to see if the user wanted to suppress warnings, if so, don't do
    #   anything. The default is always to warn the user.
    if (CONFIG.SUPPRESS_WARNINGS):
        pass
    else:
        warnings.warn(message, warning, stacklevel=2)

class Hokulani_SanityWarning(UserWarning):
    """
    This is a warning that tells the user that something that they instigated
        might make little sense or might lead to very unexpected results. An
        example may be the potential for memory overloads or very long 
        simulations.
    """

class Hokulani_ValueWarning(UserWarning):
    """
    This is a warning that tells the user that one or more of the inputs that 
        they gave is invalid. However, the program should be able to use 
        defaults provided instead. 
    """

class Hokulani_TypeWarning(UserWarning):
    """
    This is a warning that tells the user that one or more of the inputs that 
        they gave is invalid. However, the program should be able to use 
        defaults provided instead. 
    """

class Hokulani_ShapeWarning(UserWarning):
    """
    This is a warning that tells the user that one or more of the inputs that
        they gave, or the program gave, is invalid, but is still handleable by
        the program. 
    """

class Hokulani_PhysicalWarning(UserWarning):
    """
    This is a warning that some element of the program is no longer considered
        to be physical (ex. a velocity greater than light). However, the 
        program should have already dealt with it; only calling this to inform 
        the user.
    """