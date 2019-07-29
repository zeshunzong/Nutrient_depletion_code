# Initial parameters are same as paper fig.4
# New parameters (which are not existed in the paper): Lambda_hat, Sigma_hat
import numpy as np
from initialization import *   #params about time and space discretization are in initialization

##################
##Important param that we need to modify and plot
Sigma_hat  = 1e-2*64                  #??? Coefficient in Q = Sigma DivC
Lambda_hat = 0                        #??? Coefficient at the bundary
##################
##################
n          = 4                        #number of the angles
pe_star    = L_hat*e**2*Qi_hat/(np.pi*R_hat**2*Sigma_hat)
pe         = pe_star/e
lam_star   = Lambda_hat*np.pi*R_hat**2/(Qi_hat*e)
##Constant param used in Bio Growth function (paper fig.2)
sigma1     = 7                        
sigma2     = 15 
F1         = 1
F2         = 3

##Initial condition
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

#a0          = 0.9*np.ones([1,num_z])                       #Fig.4 in Tissue Paper
a1          = -z-0.5
Lambda2     = -z+2
Gamma2      = -z+2
da1_dz      = -1*np.ones([1,num_z])
Lambda2_mat = Extand_Mat*np.reshape(Lambda2,[num_z,1,1])
Lambda_Cos  = Lambda2_mat*np.cos(n*theta_mat)                #calculate Lambda2*cos(n*theta)
Gamma2_mat  = Extand_Mat*np.reshape(Gamma2,[num_z,1,1])
a0_mat      = Extand_Mat*0.9                                 #convert a0 to 3d variable
a1_mat      = Extand_Mat*np.reshape(a1,[num_z,1,1])          #a1 is constant everywhere with same
a2_mat      = Lambda_Cos+Gamma2_mat                          #a2 = Lambda2*cos(n*theta)+Gamma2
a_mat       = a0_mat+e*a1_mat+e**2*a2_mat                    #only for initial condition
da1_dz_mat  = Extand_Mat*np.reshape(da1_dz,[num_z,1,1])

## Wrap the params
basic_args_for_loop = (t1, dt, N, t, num_z, num_r, num_theta)
parameters_args = (R_hat, Qi_hat, L_hat, Sigma_hat, Lambda_hat, e, n, pe_star, pe, lam_star,sigma1,sigma2,F1,F2)
variable_args = (a, r, theta, z, Extand_Mat, r_mat, theta_mat, z_mat)
initial_cond_args = (a1, Lambda2, Gamma2, da1_dz, Lambda2_mat, Lambda_Cos, Gamma2_mat, a0_mat, a1_mat, a2_mat, a_mat, da1_dz_mat)
