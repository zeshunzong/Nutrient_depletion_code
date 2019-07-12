import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import simps
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import cv2
import os
from os.path import isfile, join
from matplotlib import cm
import scipy.io
from auxiliary_funcs import *
from simulation_funcs import *




'''Step 1, find the corresponding initialization file you need, import it'''
from initialization_for_fig4 import *



'''Step 2, we run the simulation, we store the result, we make a movie'''
'''WARNING: PLEASE MAKE SURE THERE ARE DIRECTORIES data_for_video AND storage_results UNDER THE CURRENT FOLDER '''
'''WARNING: IF RERUN THE SIMULATION, MAKE SURE THE ABOVE DIRECTORIES ARE EMPTY '''
#############################
# Simulation
#a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = Simulation_Wz_C(basic_args_for_loop, parameters_args, variable_args, initial_cond_args)
#############################
# Movie
#make_video(a, "video.mp4")
#############################
# Store
#store_simulation_results(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1, "storage_results")
#############################

'''Step 3, we load the matrices, and then do whatever plots we want'''
'''WARNING: PLEASE MAKE SURE, WHEN STEP 2 IS EXECUTED, STEP 1 HAS ALREADY BEEN EXECUTED AND HAS BEEN COMMENTTED OUT'''
#############################
# Load results
a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = load_simulation_results("storage_results", (N, num_z, num_r, num_theta))


plot_against_t_at_different_z(c,"c", endtime = t1)
