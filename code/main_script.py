from auxiliary_funcs import *
from simulation_funcs import *
from init_fig_4 import *     #set initial condition
from initialization import * #params about time and space discretization are in initialization

## RUN SIMULATION
a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = Simulation_Wz_C(basic_args_for_loop, parameters_args, variable_args, initial_cond_args)

## SAVE RESULTS
folder_name = "lam_2e_1_fig4"
store_simulation_results(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1, folder_name)


## Load results
a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = load_simulation_results(folder_name, (N, num_z, num_r, num_theta))

'''No Need to Truncate'''
# this function truncates the part after which the simulation is already finished
# a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = truncate_excess_part(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1)


## MAKE A MOVIE
#make_video(a, "video_wz_c_new.mp4")

Plot = True
if Plot:
    # plot the cylinder at different time
    plot_cylinder2_at_diff_times(a)
    
    view_angle = 0.5
    
    # plot a1 w.r.t.z at different time
    plot_against_z_at_different_time(c, "c", view_angle)
    
    # plot concentration w.r.t t at different z
    plot_against_t_at_different_z(c,"c", t1, view_angle)
    
    # plot a w.r.t t at different position
    plot_against_t_at_different_z(a,"a", t1, view_angle)

    # plot the total tissue growth
    total_tissue_growth(a, Plot)
    
    #Vertical View Plot
    plot_from_above(a)
    
    
    #plot_against_t_at_different_z(c,"c", t1)
    #plot_against_t_at_different_z(a,"a", t1, view_angle=0.3)

