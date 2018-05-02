"""
The initial conditions of the simulation are mostly stipulated here.
    This is the area for the configuration values. However, the simulation is
    still run by the default module.
"""

import numpy as np
import scipy as sp

# This defines the axis of the simulation's box. As in, this establishes
#   the width of the box that the simulation runs in. The length of an 
#   axis of the box is half this value (i.e. -L/2 <= x <= L/2).
SIMULATION_BOX_SIZE = 1

# This establishes the length of the simulation in simulation time. The unit
#   is expected to be in seconds. 
RELETIVE_SIMULATION_LENGTH = 10

# This establishes the total number of iterations of the simulation. Please
#   note that the absolute time that the simulation runs for is of the order 
#   O(n^2) for an increasing value.
TOTAL_ITERATION_COUNT = int(5e5)

# This defines the number of spatial dimensions the simulation will run 
#   over. It does not make much sense to have more than three.
NUMBER_OF_DIMENSIONS = 3

# This is the default value for the total number of particles in the 
#   simulation.
NUMBER_OF_PARTICLES = 100

# This is the maximum mass that a particle may be generated with.
MAX_GENERATE_PARTICLE_MASS = 1e8

# This is the minimum distance two particles can be before they are tagged
#   for collision.
SEPERATION_THRESHOLD = 0.05

# This is the overall initial temperature of the system before the simulation 
#   starts.
INITAL_TEMPETURE = 1e21

# This establishes the fraction of the particles that are dark matter.
DARK_MATTER_FRACTION = 0.0

# This number establishes the number of threads that should be used for
#   multi-threading.
MULTITHREAD_COUNT = 1

# This value establishes the data type of all of the float values used. 
#   This is done because memory might be too small for the most accurate
#   float values.
NP_FLOAT_DTYPE = np.float32

# This value establishes the data type of all of the uint values used. 
#   This is done because memory might be too small for the most accurate
#   float values.
NP_UINT_DTYPE = np.uint8

# This suppresses all warnings if true, both Hokulani's warning and any
#   other of the module's warnings.
SUPPRESS_WARNINGS = False

# This is the string list of all of the varieties, the order matters.  
#   Generally, lower index inputs are considered to be of higher importance.
LIST_OF_VARIETIES = np.array([
    'Blackhole',
    'Star',
    'Planet',
    'Moon',
    'Gas'
    ])

# This is the default variety type of all of the particles during the initial
#   simulation. Please use the string instead of the code (although Hokulani
#   can adapt for this).
DEFAULT_VARIETY_STRING = 'Gas'

##########
# These are values explicitly calculated as a result of the values above. 

from . import Varieties as Hoku_Vary

# This value establishes the change in time between each and every iteration 
#   of the simulation.
DELTA_TIME = RELETIVE_SIMULATION_LENGTH / TOTAL_ITERATION_COUNT

# This is the variety dictionary. Although it is made in a different module,
#   is considered a constant at the point of runtime, so, add this to the 
#   list of constants. The any variety dictionary should be accessed from here.
VARIETY_DICTIONARY =  Hoku_Vary.construct_variety_dictionary()

# This is the default variety code, if the user specified a code instead of a
#   string, just switch the two roles.
DEFAULT_VARIETY_CODE = Hoku_Vary.convert_variety_string_to_code(
    DEFAULT_VARIETY_STRING)


# If the user desired the warnings were suppressed, suppress the warnings.
if (SUPPRESS_WARNINGS):
    # Ignore numpy's warnings.
    np.warnings.filterwarnings('ignore')
    # Ignore scipy's warnings.
    sp.warnings.filterwarnings('ignore')
