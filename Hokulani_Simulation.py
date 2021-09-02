import numpy as np
import datetime
import copy
import matplotlib.pyplot as plt
 
import Hokulani.Generation as Hoku_Gen
import Hokulani.Particle as Hoku_Part
import Hokulani.Mechanics as Hoku_Mech
import Hokulani.Visualization as Hoku_Visual
import Hokulani.Execptions as Hoku_Execpt

from Hokulani import Configuration as Hoku_Config
from Hokulani import Simulation as Hoku_Sim
from Hokulani import Mechanics as Hoku_Mech

def Sparrow():
    particles = Hoku_Gen.generate_particle_array()
    
    pos_array = Hoku_Part.extract_position_array(particles)
    vel_array = Hoku_Part.extract_velocity_array(particles)
    accel_array = Hoku_Part.extract_acceleration_array(particles)
    
    mas_array = Hoku_Part.extract_mass_array(particles)
    
    vary_array = Hoku_Part.extract_variety_array(particles)

    dark_m_array = Hoku_Part.extract_dark_matter_array(particles)
    deletion_array = Hoku_Part.extract_delete_array(particles)
    
    new_particles = []
    
    
    for timedex in range(Hoku_Config.TIME_STEP):
        sep_array = Hoku_Mech.compute_seperation_array(pos_array)
        grav_force_array = Hoku_Mech.compute_force_gravity_array(sep_array,mas_array)
    
        accel_array = Hoku_Mech.compute_acceleration_array(grav_force_array,mas_array)
        vel_array = Hoku_Mech.compute_velocity_array(vel_array,accel_array)
        pos_array = Hoku_Mech.compute_position_array(pos_array,vel_array,accel_array)
    
        temp_particle_list = Hoku_Part.parcel_particle_array(
            pos_array,vel_array,accel_array,
            mas_array,vary_array,dark_m_array,
            deletion_array)
    
        new_particles.append(temp_particle_list)
    
    new_particles = np.array(new_particles)

    return new_particles
    

if (__name__ == '__main__'):


    print('Start!')

    # Ignore warnings
    Hoku_Config.SUPPRESS_WARNINGS = True

    particles = Hoku_Gen.generate_particle_array()
    #true_particles = copy.deepcopy(particles)

    print('Begin!')

    Hoku_Visual.crude_display_simulation_final_3D(particles)

    start_time = datetime.datetime.now()
    print(start_time)


    new_particles = Hoku_Sim.simulate_particles(particles,
                                                return_particle_history=False)

    end_time = datetime.datetime.now()

    
    # for timedex in range(Hoku_Config.TOTAL_ITERATION_COUNT):
    #     for particledex in range(Hoku_Config.NUMBER_OF_PARTICLES):
    #         new_particles[timedex,particledex].print(particle_number=particledex)
    #     print('\n\n')
    # 
    #     print(timedex)
    

    Hoku_Visual.crude_display_simulation_final_3D(new_particles)

    for particledex in range(len(new_particles)):
        particles[particledex].print(particledex)
        new_particles[particledex].print(particledex)
        print('\n\n =============== \n\n')

    print('Total time {delta_time} .'
          .format(delta_time=end_time-start_time))

    plt.show()

