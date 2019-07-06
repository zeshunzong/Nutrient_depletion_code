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

from auxiliary_funcs import transpose, f, df, d2f, integrate_z_1, find_boundary_idx
from auxiliary_funcs import plot_cylinder2, convert_frames_to_video, make_cylinder_video, plot_cylinder2_at_diff_times



def Simulation_Wz_C(basic_args_for_loop, parameters_args, variable_args, initial_cond_args):

    t1, dt, N, t, num_z, num_r, num_theta = basic_args_for_loop
    R_hat, Qi_hat, L_hat, Sigma_hat, Lambda_hat, e, n, pe_star, pe, lam_star = parameters_args
    a, r, theta, z, Extand_Mat, r_mat, theta_mat, z_mat = variable_args
    a1, Lambda2, Gamma2, da1_dz, Lambda2_mat, Lambda_Cos, Gamma2_mat, a0_mat, a1_mat, a2_mat, a_mat, da1_dz_mat = initial_cond_args

    ## allocate space for recording statistics
    a0_tensor        = np.zeros([N,num_z,num_r,num_theta])
    a1_tensor        = np.zeros([N,num_z,num_r,num_theta])
    Lambda2_tensor   = np.zeros([N,num_z,num_r,num_theta])
    Gamma2_tensor    = np.zeros([N,num_z,num_r,num_theta])
    a_tensor         = np.zeros([N,num_z,num_r,num_theta])
    sigmaS0_tensor   = np.zeros([N,num_z,num_r,num_theta])
    sigmaS1_tensor   = np.zeros([N,num_z,num_r,num_theta])
    c_tensor         = np.zeros([N,num_z,num_r,num_theta])
    c0_tensor        = np.zeros([N,num_z,num_r,num_theta])
    c1_tensor        = np.zeros([N,num_z,num_r,num_theta])
    c2_tensor        = np.zeros([N,num_z,num_r,num_theta])

    #with concentration
    for i in range(N):## Time Loop

        ## update xi
        #  xi as the pressure at the pore inlet, should be constant everywhere
        xi0_mat     = 8/a0_mat**4

        ## update w -- velocity in z direction
        #  w0, w1 is a function of r,z, w2 is a function of theta, z, r
        w0_mat      = xi0_mat/4*(a0_mat**2-r_mat**2)
        w1_mat      = a1_mat*xi0_mat/(2*a0_mat)*(2*r_mat**2-a0_mat**2)
        w2_mat      = (Lambda2_mat/2*(a0_mat**(1-n))*(r_mat**n)*np.cos(theta_mat*n)+ \
                       1/(a0_mat**4) *(-3* a0_mat**2 * a1_mat**2 + a0_mat**3 * Gamma2_mat - a1_mat**4 / 4) \
                       *(r_mat**2 - a0_mat**2)-7/4*a1_mat**2 + 1/2*a0_mat*Gamma2_mat)*xi0_mat
        w_mat       = w0_mat+e*w1_mat+e**2*w2_mat

        ## update v -- velocity in theta direction
        v_mat       = 0*Extand_Mat

        ## update u -- velocity in r direction
        u1          = da1_dz_mat*xi0_mat/(4*a0_mat)*r_mat*(a0_mat**2-r_mat**2)
        u_mat       = e*u1

        ## update C -- nutri concetration
        #  function of z
        c0_mat      = np.exp(-2*lam_star*a0_mat*z_mat)
        c1_mat      = 4*a0_mat**4/pe*lam_star**2*z_mat*np.exp(-2*a0_mat*lam_star*z_mat)
        c2_mat      = np.exp(-2*a0_mat*lam_star*z_mat)*(z_mat*(-16*a0_mat**7*lam_star**3/pe**2-\
                            4*a0_mat**2*lam_star/pe*da1_dz_mat-\
                            4*a1_mat**2*lam_star/a0_mat+2*a1_mat**4*lam_star/a0_mat**3)+\
                            0.5*z_mat**2*16*a0_mat**8*lam_star**4/pe**2)
        c_mat       = c0_mat+e*c1_mat+e**2*c2_mat

        ## update sigmaS
        sigmaS0_mat = a0_mat/2*xi0_mat
        sigmaS1_mat = 2*a1_mat*xi0_mat
        sigmaS2a_mat= n*Lambda2_mat*xi0_mat/2
        sigmaS2b_mat= 1/a0_mat**3*(-6*a0_mat**2*a1_mat**2+2*a0_mat**3*Gamma2_mat-a1_mat**4/2)*xi0_mat

        ## update kappa
        #  not so sure whether we can use kappa from tissue engineering paper directly?
        k0_mat      = 1/a0_mat
        k1_mat      = 0*Extand_Mat
        k2_mat      = n**2*Lambda2_mat/a0_mat**2

        ## update f and f' and f''
        f_mat       = f(sigmaS0_mat)
        df_mat      = 0*Extand_Mat
        df2_mat     = 0*Extand_Mat

        ## update a
        a0_mat      = a0_mat - c0_mat*k0_mat*f_mat*dt
        a1_mat      = a1_mat - (c0_mat*k0_mat*sigmaS1_mat*df_mat+c0_mat*k1_mat*f_mat+c1_mat*k0_mat*f_mat)*dt
        da1_dz      = np.reshape(np.gradient(a1_mat[:,0,0],z[0]),[1,num_z])
        da1_dz_mat  = Extand_Mat*np.reshape(da1_dz,[num_z,1,1])
        Lambda2_mat = Lambda2_mat - (f_mat*k2_mat*c0_mat+df_mat*sigmaS2a_mat*c0_mat*k0_mat)*dt
        #Lambda2_mat = 0*Extand_Mat
        Gamma2_mat  = Gamma2_mat  - ((c1_mat*k1_mat+c2_mat*k0_mat)*f_mat + \
                                    (0.5*sigmaS1_mat**2*df2_mat+sigmaS2b_mat*df_mat)*c0_mat*k0_mat+\
                                    (k1_mat*c0_mat+k0_mat*c1_mat)*sigmaS1_mat*df_mat)*dt
        a_mat       = a0_mat+e*a1_mat+e**2*(Lambda2_mat*np.cos(theta_mat*n)+Gamma2_mat)     #only for initial condition

        a0_tensor[i,:,:,:]      = a0_mat
        a1_tensor[i,:,:,:]      = a1_mat
        Lambda2_tensor[i,:,:,:] = Lambda2_mat
        Gamma2_tensor[i,:,:,:]  = Gamma2_mat
        a_tensor[i,:,:,:]       = a_mat
        sigmaS0_tensor[i,:,:,:] = sigmaS0_mat
        sigmaS1_tensor[i,:,:,:] = sigmaS1_mat
        c_tensor[i,:,:,:]       = c_mat
        c0_tensor[i,:,:,:]      = c0_mat
        c1_tensor[i,:,:,:]      = c1_mat
        c2_tensor[i,:,:,:]      = c2_mat

        #print(c1_mat*k0_mat*f_mat)


    return a0_tensor,a1_tensor,Lambda2_tensor,Gamma2_tensor,a_tensor,sigmaS0_tensor,sigmaS1_tensor,c_tensor,c0_tensor,c1_tensor,c2_tensor



def Simulation_No_C(n,e,a0_mat,a1_mat,Lambda2_mat,Gamma2_mat,da1_dz_mat ):
    '''simulation with C is a constant'''
    ## allocate space for recording statistics
    a0_tensor        = np.zeros([N,num_z,num_r,num_theta])
    a1_tensor        = np.zeros([N,num_z,num_r,num_theta])
    Lambda2_tensor   = np.zeros([N,num_z,num_r,num_theta])
    Gamma2_tensor    = np.zeros([N,num_z,num_r,num_theta])
    a_tensor         = np.zeros([N,num_z,num_r,num_theta])
    sigmaS0_tensor   = np.zeros([N,num_z,num_r,num_theta])
    sigmaS1_tensor   = np.zeros([N,num_z,num_r,num_theta])
    c_tensor         = np.zeros([N,num_z,num_r,num_theta])
    c0_tensor        = np.zeros([N,num_z,num_r,num_theta])
    c1_tensor        = np.zeros([N,num_z,num_r,num_theta])
    c2_tensor        = np.zeros([N,num_z,num_r,num_theta])

    #without concentration
    for i in range(N):## Time Loop

        ## update xi
        #  xi as the pressure at the pore inlet, should be constant everywhere
        xi0_mat     = 8/a0_mat**4

        ## update w -- velocity in z direction
        #  w0, w1 is a function of r,z, w2 is a function of theta, z, r
        w0_mat      = xi0_mat/4*(a0_mat**2-r_mat**2)
        w1_mat      = a1_mat*xi0_mat/(2*a0_mat)*(2*r_mat**2-a0_mat**2)
        w2_mat      = (Lambda2_mat/2*(a0_mat**(1-n))*(r_mat**n)*np.cos(theta_mat*n)+ \
                       1/(a0_mat**4) *(-3* a0_mat**2 * a1_mat**2 + a0_mat**3 * Gamma2_mat - a1_mat**4 / 4) \
                       *(r_mat**2 - a0_mat**2)-7/4*a1_mat**2 + 1/2*a0_mat*Gamma2_mat)*xi0_mat
        w_mat       = w0_mat+e*w1_mat+e**2*w2_mat

        ## update v -- velocity in theta direction
        v_mat       = 0*Extand_Mat

        ## update u -- velocity in r direction
        u1          = da1_dz_mat*xi0_mat/(4*a0_mat)*r_mat*(a0_mat**2-r_mat**2)
        u_mat       = e*u1

        ## update C -- nutri concetration
        #  function of z
        c0_mat      = 1*Extand_Mat
        c1_mat      = 0*Extand_Mat
        c2_mat      = 0*Extand_Mat
        c_mat       = c0_mat+e*c1_mat+e**2*c2_mat

        ## update sigmaS
        sigmaS0_mat = a0_mat/2*xi0_mat
        sigmaS1_mat = 2*a1_mat*xi0_mat
        sigmaS2a_mat= n*Lambda2_mat*xi0_mat/2
        sigmaS2b_mat= 1/a0_mat**3*(-6*a0_mat**2*a1_mat**2+2*a0_mat**3*Gamma2_mat-a1_mat**4/2)*xi0_mat

        ## update kappa
        #  not so sure whether we can use kappa from tissue engineering paper directly?
        k0_mat      = 1/a0_mat
        k1_mat      = 0*Extand_Mat
        k2_mat      = n**2*Lambda2_mat/a0_mat**2

        ## update f and f' and f''
        f_mat       = f(sigmaS0_mat)
        df_mat      = 0*Extand_Mat
        df2_mat     = 0*Extand_Mat

        ## update a
        a0_mat      = a0_mat - c0_mat*k0_mat*f_mat*dt
        a1_mat      = a1_mat - (c0_mat*k0_mat*sigmaS1_mat*df_mat+c0_mat*k1_mat*f_mat+c1_mat*k0_mat*f_mat)*dt
        da1_dz      = np.reshape(np.gradient(a1_mat[:,0,0],z[0]),[1,num_z])
        da1_dz_mat  = Extand_Mat*np.reshape(da1_dz,[num_z,1,1])
        Lambda2_mat = Lambda2_mat - (f_mat*k2_mat*c0_mat+df_mat*sigmaS2a_mat*c0_mat*k0_mat)*dt
        #Lambda2_mat = 0*Extand_Mat
        Gamma2_mat  = Gamma2_mat  - ((c1_mat*k1_mat+c2_mat*k0_mat)*f_mat + \
                                    (0.5*sigmaS1_mat**2*df2_mat+sigmaS2b_mat*df_mat)*c0_mat*k0_mat+\
                                    (k1_mat*c0_mat+k0_mat*c1_mat)*sigmaS1_mat*df_mat)*dt
        a_mat       = a0_mat+e*a1_mat+e**2*(Lambda2_mat*np.cos(theta_mat*n)+Gamma2_mat)     #only for initial condition

        a0_tensor[i,:,:,:]      = a0_mat
        a1_tensor[i,:,:,:]      = a1_mat
        Lambda2_tensor[i,:,:,:] = Lambda2_mat
        Gamma2_tensor[i,:,:,:]  = Gamma2_mat
        a_tensor[i,:,:,:]       = a_mat
        sigmaS0_tensor[i,:,:,:] = sigmaS0_mat
        sigmaS1_tensor[i,:,:,:] = sigmaS1_mat
        c_tensor[i,:,:,:]       = c_mat
        c0_tensor[i,:,:,:]      = c0_mat
        c1_tensor[i,:,:,:]      = c1_mat
        c2_tensor[i,:,:,:]      = c2_mat


    return a0_tensor,a1_tensor,Lambda2_tensor,Gamma2_tensor,a_tensor,sigmaS0_tensor,sigmaS1_tensor,c_tensor,c0_tensor,c1_tensor,c2_tensor
