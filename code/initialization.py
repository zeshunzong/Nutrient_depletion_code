import numpy as np

# time discretization
t1         = 0.40                     #end time
dt         = 0.005                    #time interval delta t
N          = int(t1/dt+1)             #the num of steps of each path
t          = np.linspace(0, t1, N)    #time steps (0, t1, t1/h+1)

#space discretization
num_r      = 50#1                     #discretize the radius in r direction
num_theta  = 60#1                     #discretize the angle in theta direction
num_z      = 40#1                     #discretize the length in Z direction

##parameters
R_hat      = 0.5                      #fluid-cell-layer interface radius, unit mm
Qi_hat     = 1                        #inlet flux, unit ml/min
e          = 0.2                      #epsilon
L_hat      = R_hat/e                  #fluid-cell-layer interface length, unit mm
a          = R_hat/(L_hat*e)          #non-dimensional a (â0 = ^R)
r          = np.reshape(np.linspace(0,a,num_r),[1,num_r])
theta      = np.reshape(np.linspace(0,2*np.pi,num_theta),[1,num_theta])
z          = np.reshape(np.linspace(0,1,num_z),[1,num_z])
                                               #z is 1*num_z matrix, the values of z are from 0~1
Extand_Mat = np.ones([num_z,num_r,num_theta])  # a helpful matrix that later will help convert an array to a 3d array
r_mat      = r.T*Extand_Mat
theta_mat  = theta*Extand_Mat
z_mat      = np.reshape(z,[num_z,1,1])*Extand_Mat