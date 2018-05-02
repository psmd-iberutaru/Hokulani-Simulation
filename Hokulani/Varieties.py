"""
This document maintains all of the needed functionality of different particle
    varieties, or any property that relies on flag codes that can be 
    represented as a numerical value.
"""
import numpy as np

from .Execptions import *
from . import Configuration as CONFIG
from . import Validation as Hoku_Valid


def construct_variety_dictionary(variety_list=CONFIG.LIST_OF_VARIETIES):
    """
    This function constructs the variety dictionary out of either the default
        values or the values that a user specified.
    """

    # Generate parallel arrays of the varieties and their codes. 
    variety_list = np.array(variety_list)

    n_varieties = len(variety_list)
    variety_codes = np.fromiter(range(n_varieties),int)
    
    # Compile the parallel arrays into a dictionary.
    return dict(zip(variety_codes,variety_list))



def convert_variety_code_to_string(variety_code):
    """
    This function converts the numerical variety code into a string for
        better reading and handling by the user.
    """

    # Ensure that the variety code entered is valid.
    variety_code = Hoku_Valid.validate_variety_code(variety_code)

    # Use a switch case dictionary as a replacement for the non-support for
    #   switches. Format new entries as  'String': code
    # Generally, lower values indicate superseding properties.
    try:
        vary_string = CONFIG.VARIETY_DICTIONARY[variety_code]
    except KeyError: 
        raise Hokulani_ValueError('An attempt to convert a variety code into '
                                  'a variety string failed. Code could not be '
                                  'found in the variety dictionary.')
    except Exception:
        raise Hokulani_UnknownError('The attempt to convert a variety code '
                                    'into a string failed with an exception '
                                    'other than a KeyError.')
    
    return vary_string 

def convert_variety_string_to_code(variety_string):
    """
    This function converts the string entry of the variety to a integer for 
        better computer handling in the program.
    """

    # Ensure that the value given for the variety string is valid.
    variety_string = Hoku_Valid.validate_variety_string(variety_string)

    # Extract the code from the dictionary. This is adapted from a post on
    #   Stack Exchange.
    return (list(CONFIG.VARIETY_DICTIONARY.keys())
            [list(CONFIG.VARIETY_DICTIONARY.values()).index(variety_string)])