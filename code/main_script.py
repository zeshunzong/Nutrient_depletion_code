from auxiliary_funcs import *
from simulation_funcs import *
from initialization import *

## RUN SIMULATION
a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = Simulation_Wz_C()

## SAVE RESULTS
#store_simulation_results(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1, "storage_results")


## Load results
#a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = load_simulation_results("storage_results", (N, num_z, num_r, num_theta))


## MAKE A MOVIE
#make_video(a, "video_wz_c_new.mp4")


# plot the cylinder at different time
#plot_cylinder2_at_diff_times(a)

view_angle = 0.3

# plot a1 w.r.t.z at different time
plot_against_z_at_different_time(a1, "a1", view_angle)

# plot concentration w.r.t t at different z
plot_against_t_at_different_z(c,"c", t1, view_angle)

# plot a w.r.t t at different position
plot_against_t_at_different_z(a,"a", t1, view_angle)

# plot the total tissue growth
#total_tissue_growth(a)

#Vertical View Plot
#plot_from_above(a)
