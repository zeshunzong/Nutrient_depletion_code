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
import pickle

from auxiliary_funcs import transpose, f, df, d2f, integrate_z_1, find_boundary_idx
from auxiliary_funcs import plot_cylinder2, convert_frames_to_video, make_cylinder_video, plot_cylinder2_at_diff_times
from auxiliary_funcs import plot_against_z_at_different_time
from simulation_funcs import Simulation_Wz_C, Simulation_No_C


# time discretization
t1         = 0.04                     #end time
dt         = 0.005                    #time interval delta t
N          = int(t1/dt+1)             #the num of steps of each path
t          = np.linspace(0, t1, N)   #time steps (0, t1, t1/h+1)

##parameters
R_hat      = 0.5                     #fluid-cell-layer interface radius, unit mm
Qi_hat     = 1                       #inlet flux, unit ml/min
L_hat      = R_hat/0.2                       #fluid-cell-layer interface length, unit mm
##################
##################
Sigma_hat  = 1e-2*64                     #??? Coefficient in Q = Sigma DivC
Lambda_hat = .05                     #??? Coefficient at the bundary
##################
##################
e          = R_hat/L_hat             #epsilon
n          = 4                       #number of the angles
pe_star    = L_hat*e**2*Qi_hat/(np.pi*R_hat**2*Sigma_hat)
pe         = pe_star/e
lam_star   = Lambda_hat*np.pi*R_hat**2/(Qi_hat*e)

num_r      = 100#1                     #discretize the radius in r direction
num_theta  = 60#1                     #discretize the angle in theta direction
num_z      = 300#1                     #discretize the length in Z direction
a          = R_hat/(L_hat*e)         #non-dimensional a (â0 = ^R)
r          = np.reshape(np.linspace(0,a,num_r),[1,num_r])
#r          = np.reshape(np.linspace(0,1,num_r),[1,num_r])
theta      = np.reshape(np.linspace(0,2*np.pi,num_theta),[1,num_theta])
z          = np.reshape(np.linspace(0,1,num_z),[1,num_z])
                                     #z is 1*num_z matrix, the values of z are from 0~1
Extand_Mat = np.ones([num_z,num_r,num_theta])  # a helpful matrix that later will help convert an array to a tensor
r_mat      = r.T*Extand_Mat
theta_mat  = theta*Extand_Mat
z_mat      = np.reshape(z,[num_z,1,1])*Extand_Mat

##initial condition
'''For all variance, it has three dimension: r,theta,z
discretize on r direction,
discretize on theta direction
discretize on z direction
each with num of points:
num_r, num_theta, num_z.

Notice: Here I used to representation
a0 is 1*num_z, which indicates that a0 is only a function of z
a0_mat is a 3d variable: a0_mat[i,j,k] represents the value of a0
at z=i/num_z,r=j/num_r*a,theta=k/num_theta*(2pi)
The same notation used also for other variables'''

## initial condition on a
#a0          = 0.9*np.ones([1,num_z])                       #Fig.4 in Tissue Paper
a1          = -z-0.5
Lambda2     = -z+2
Gamma2      = -z+2
da1_dz      = -1*np.ones([1,num_z])
Lambda2_mat = Extand_Mat*np.reshape(Lambda2,[num_z,1,1])
Lambda_Cos  = Lambda2_mat*np.cos(n*theta_mat)                  #calculate Lambda2*cos(n*theta)
Gamma2_mat  = Extand_Mat*np.reshape(Gamma2,[num_z,1,1])
a0_mat      = Extand_Mat*0.9                               #convert a0 to 3d variable
a1_mat      = Extand_Mat*np.reshape(a1,[num_z,1,1])        #a1 is constant everywhere with same
a2_mat      = Lambda_Cos+Gamma2_mat                        #a2 = Lambda2*cos(n*theta)+Gamma2
a_mat       = a0_mat+e*a1_mat+e**2*a2_mat                        #only for initial condition
da1_dz_mat  = Extand_Mat*np.reshape(da1_dz,[num_z,1,1])

print(lam_star)
print(pe)

basic_args_for_loop = (t1, dt, N, t, num_z, num_r, num_theta)
parameters_args = (R_hat, Qi_hat, L_hat, Sigma_hat, Lambda_hat, e, n, pe_star, pe, lam_star)
variable_args = (a, r, theta, z, Extand_Mat, r_mat, theta_mat, z_mat)
initial_cond_args = (a1, Lambda2, Gamma2, da1_dz, Lambda2_mat, Lambda_Cos, Gamma2_mat, a0_mat, a1_mat, a2_mat, a_mat, da1_dz_mat)

a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = Simulation_Wz_C(basic_args_for_loop, parameters_args, variable_args, initial_cond_args)

#plot_cylinder2_at_diff_times(a)


'''make video'''
#os.chdir(os.path.dirname(__file__))
# the above command makes sure that we are working in the current dir
#pathIn = './data_for_video/'
#pathOut = 'video.mp4'
#make_cylinder_video(pathIn, pathOut, a[:,:,0,:])


'''pickle your data'''

'''
os.chdir(os.path.dirname(__file__)) # make sure your are at the current working directory
result_tuple = (a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2)
pickle.dump(result_tuple, open('./result_tuple_enter_file_name.pkl', 'wb'))
'''


'''unpickle your data'''
'''
os.chdir(os.path.dirname(__file__))
result_tuple = pickle.load(open('./result_tuple_enter_file_name.pkl', 'rb'))
a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = result_tuple
'''

plot_against_z_at_different_time(a1, "a1")
