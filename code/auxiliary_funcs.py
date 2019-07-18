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
from initialization import *
def transpose(x):
    '''a function used to take transpose of each matrix inside a tensor x
     x is a tensor n*d*c, return n*c*d'''
    d = len(x[0,:,0])                       #num of rows of each matrix
    c = len(x[0,0,:])                       #num of columns of each matrix
    n = len(x[:,0,0])                       #num of matrices in the tensor x

    X_T = np.zeros([n,c,d])  #allocate space
    for i in range(d):
        X_T[:,:,i]=x[:,i,:]
    return X_T

def f(s):
    '''function in Tissue Paper (43)'''
    F1 = 1
    F2 = 3
    m  = 5000
    sigma1 = 7
    sigma2 = 15
    fs = F1+(F2-F1)*(1+np.tanh(m*(s-sigma1)))/2-F2*(1+np.tanh(m*(s-sigma2)))/2
    return fs

def df(s):
    '''the derivative function of f(s)'''
    F1 = 1
    F2 = 3
    m  = 5000
    sigma1 = 7
    sigma2 = 15
    dfs = m*(F2-F1)*(1-np.tanh(m*(s-sigma1))**2)/2-m*F2*(1-np.tanh(m*(s-sigma2))**2)/2
    return dfs

def d2f(s):
    '''the second derivative of f(s)'''
    F1 = 1
    F2 = 3
    m  = 5000
    sigma1 = 7
    sigma2 = 15
    d2fs   = m*(F2-F1)*-2*m*(1-np.tanh(m*(s-sigma1))**2)/2-m*F2*-2*m*(1-np.tanh(m*(s-sigma2))**2)/2
    return d2fs

def integrate_z_1(data,z):
    '''a function used to integrate from z to 1
    @parameter: data,  1*num_z matrix
    @return: integral, 1*num_z matrix'''

    l = len(data[0])                               #the number of data points inside data
    integral = np.zeros([1,l])                     #allocate space
    for i in range(l):
        integral[0,i]=simps(data[0][i:], z[0][i:]) #integrate from z to 1
    return integral

def find_boundary_idx(r_mat, a_mat):
    '''find the index of each matrix where r values is the boundary
    @return: num_z*1*num_theta'''

    idx = (np.abs(r_mat - a_mat)).argmin(axis=1)
    return idx

# for plot use only
def plot_cylinder2(a_mat_trunc):
    fig = plt.figure(figsize=(12,8))
    #figsize=(12,8)
    ax = fig.gca(projection='3d')

    num_z, num_theta = np.shape(a_mat_trunc)
    z_metric = np.linspace(0,-1,num_z)

    #2D matrix
    rr_mat = a_mat_trunc
    # rr_mat[i,j] corresponds to the radius of a point at height z[i] and angle theta[j]
    # we plot the first half of the cylinder, which corresponds to theta ranging from 0 to pi, or y>0
    first_half = rr_mat[:,0:int(num_theta/2)+1]
    # there are altogether num_z*(int(num_theta/2)+1) points
    # the z_coordinate of each point in first_half[i,j] = np.linspace(0,1, num_z)[i]
    # the x_coordinate of each point in first_half[i,j] = rcostheta = first_half[i,j] * cos(np.linspace(0, 2pi, num_theta)[j])
    # the y_coordinate ... = first_half[i,j] * sin(np.linspace(0, 2pi, num_theta)[j])
    ZMAT = np.outer(z_metric,np.ones(int(num_theta/2)+1))
    XMAT = first_half * np.outer(np.ones(num_z),np.cos(np.linspace(0, 2*np.pi, num_theta)[:int(num_theta/2)+1]))
    YMAT = first_half * np.outer(np.ones(num_z),np.sin(np.linspace(0, 2*np.pi, num_theta)[:int(num_theta/2)+1]))
    ax.plot_surface(XMAT, YMAT, ZMAT, linewidth = 0.02, alpha = 0.8, cmap=cm.coolwarm)


    # plot the second half of the cylinder, theta from pi to 2pi
    second_half = rr_mat[:,int(num_theta/2):]
    ZMAT2 = np.outer(z_metric,np.ones(num_theta - int(num_theta/2)))
    XMAT2 = second_half * np.outer(np.ones(num_z),np.cos(np.linspace(0, 2*np.pi, num_theta)[int(num_theta/2):]))
    YMAT2 = second_half * np.outer(np.ones(num_z),np.sin(np.linspace(0, 2*np.pi, num_theta)[int(num_theta/2):]))

    surf = ax.plot_surface(XMAT2, YMAT2, ZMAT2, linewidth = 0.02, alpha = 0.8, cmap=cm.coolwarm)
    fig.colorbar(surf, shrink = 1, aspect=2)

    ax.set_zlim(-1,0)
    ax.set_xlim(-1.1,1.1)
    ax.set_ylim(-1.1,1.1)
    ax.set_xlabel('$x$ axis')
    ax.set_ylabel('$y$ axis')
    ax.set_zlabel('$z$ axis')
    #ax.view_init(elev=90, azim=0)
    ax.set_aspect('equal')

# for video only
def convert_frames_to_video(pathIn,pathOut,fps):
    if os.path.exists('./data_for_video/.DS_Store'):
        os.remove('./data_for_video/.DS_Store')

    frame_array = []
    files = []
    for data_file in sorted(os.listdir(pathIn)):
        if isfile(join(pathIn, data_file)):
            files.append(data_file)

    #files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

    #for sorting the file names properly
    #files.sort(key = lambda x: int(x[5:-4]))

    for i in range(len(files)):
        filename=pathIn + files[i]
        #print(filename)
        #reading each files
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)

        #inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()

# for video only
def make_cylinder_video(pathIn, pathOut, a_as_t, fps=5):

    # demo pathIn: './data_for_video/'
    # demo pathOut: 'video.mp4'
    # a_as_t: a matrix that records how a changes as t goes
    # fps: the smaller, the longer the video

    #a_as_t = a_as_t[:, :, 0,:]
    num_t, useless1, useless2 = np.shape(a_as_t)

    # save the frames as pictures
    for n in range(num_t):
        plot_cylinder2(a_as_t[n,:,:])
        if n < 10:
            pic_name = pathIn + 'frame00' + str(n) + '.png'
        elif n < 100:
            pic_name = './data_for_video/frame0' + str(n) + '.png'
        else:
            pic_name = './data_for_video/frame' + str(n) + '.png'
        plt.savefig(pic_name, bbox_inches='tight')
        plt.close()

    convert_frames_to_video(pathIn,pathOut,fps)

    for n in range(num_t):
        if n < 10:
            pic_name = pathIn + 'frame00' + str(n) + '.png'
        elif n < 100:
            pic_name = './data_for_video/frame0' + str(n) + '.png'
        else:
            pic_name = './data_for_video/frame' + str(n) + '.png'
        os.remove(pic_name)

'''
pathIn = './data_for_video/'
pathOut = 'video.mp4'
make_cylinder_video(pathIn, pathOut, a0_tensor[:,:,0,:])
'''
def plot_cylinder2_at_diff_times(a_as_t):
    fig = plt.figure(figsize=(10,6))
    #figsize=(12,8)
    #ax = fig.gca(projection='3d')

    num_t, num_z, num_r, num_theta = np.shape(a_as_t)
    z_metric = np.linspace(0,-1,num_z)

    #3D matrix
    rr_mat = a_as_t[[0, int(num_t/4), int(num_t/2), num_t-1],:,0,:]

    t_title_vec = [0, 0.25, 0.5, 1]

    for j in range(0,4):
        ax = fig.add_subplot(2, 2, j+1, projection='3d')
        first_half = rr_mat[j,:,0:int(num_theta/2)+1]
        ZMAT = np.outer(z_metric,np.ones(int(num_theta/2)+1))
        XMAT = first_half * np.outer(np.ones(num_z),np.cos(np.linspace(0, 2*np.pi, num_theta)[:int(num_theta/2)+1]))
        YMAT = first_half * np.outer(np.ones(num_z),np.sin(np.linspace(0, 2*np.pi, num_theta)[:int(num_theta/2)+1]))
        ax.plot_surface(XMAT, YMAT, ZMAT, linewidth = 0.02, alpha = 0.8, cmap=cm.coolwarm)
        second_half = rr_mat[j,:,int(num_theta/2):]
        ZMAT2 = np.outer(z_metric,np.ones(num_theta - int(num_theta/2)))
        XMAT2 = second_half * np.outer(np.ones(num_z),np.cos(np.linspace(0, 2*np.pi, num_theta)[int(num_theta/2):]))
        YMAT2 = second_half * np.outer(np.ones(num_z),np.sin(np.linspace(0, 2*np.pi, num_theta)[int(num_theta/2):]))
        surf = ax.plot_surface(XMAT2, YMAT2, ZMAT2, linewidth = 0.02, alpha = 0.8, cmap=cm.coolwarm)
        fig.colorbar(surf, shrink = 1, aspect=1.5)

        ax.set_zlim(-1,0)
        ax.set_xlim(-1.1,1.1)
        ax.set_ylim(-1.1,1.1)
        ax.set_xlabel('$x$ axis')
        ax.set_ylabel('$y$ axis')
        ax.set_zlabel('$z$ axis')
        #ax.view_init(elev=90, azim=0)
        ax.set_aspect('equal')
        ax.title.set_text(r'$t=$'+str(t_title_vec[j]) + '$t_f$')
    plt.show()


'''This function automatically generates a plot of the variable you want,
against z, at six different times, 0%, 20%, 40%, 60%, 80%, 100%. The angle of
of plot is by default theta=0, can be changed manually'''
def plot_against_z_at_different_time(data_mat, variable_name, view_angle = 0):
    # data_mat is a 4d matrix, variable_name is the string name of the data, used for title, should be a or c
    num_t, num_z, num_r, num_theta = np.shape(data_mat)
    angle = int(np.floor(view_angle/2/np.pi*num_theta))
    z_vec = np.linspace(0,1,num_z)
    t_position = np.zeros(6)
    for i in range(6):
        t_position[i] = int(num_t * i/5)

    plt.plot(z_vec, data_mat[int(t_position[0]), :, 0, angle], label = r"$t=0$")
    plt.plot(z_vec, data_mat[int(t_position[1]), :, 0, angle], label = r"$t=0.2t_f$")
    plt.plot(z_vec, data_mat[int(t_position[2]), :, 0, angle], label = r"$t=0.4t_f$")
    plt.plot(z_vec, data_mat[int(t_position[3]), :, 0, angle], label = r"$t=0.6t_f$")
    plt.plot(z_vec, data_mat[int(t_position[4]), :, 0, angle], label = r"$t=0.8t_f$")
    plt.plot(z_vec, data_mat[-1, :, 0, angle], label = r"$t=t_f$")
    plt.legend()
    plt.title(variable_name + " against z, at different times, "+ r"$\theta=$" + str(view_angle))
    plt.show()

'''This function automatically generates a plot of the variable you want,
against t, at six different z's, z= 0, 0.2, 0.40, 0.6, 0.80, 1. The angle of
of plot is by default theta=0, can be changed manually, shoulbe be  between 0 and 2pi'''
def plot_against_t_at_different_z(data_mat, variable_name, endtime, view_angle = 0):
    # data_mat is a 4d matrix, variable_name is the string name of the data, used for title, should be a or c
    # endtime is the time t1 in the main script
    num_t, num_z, num_r, num_theta = np.shape(data_mat)
    angle = int(np.floor(view_angle/2/np.pi*num_theta))
    t_vec = np.linspace(0, endtime ,num_t)
    z_position = np.zeros(6)
    for i in range(6):
        z_position[i] = int(num_z * i/5)

    plt.plot(t_vec, data_mat[:,int(z_position[0]), 0, angle], label = r"$z=0$")
    plt.plot(t_vec, data_mat[:,int(z_position[1]), 0, angle], label = r"$z=0.2$")
    plt.plot(t_vec, data_mat[:,int(z_position[2]), 0, angle], label = r"$z=0.4$")
    plt.plot(t_vec, data_mat[:,int(z_position[3]), 0, angle], label = r"$z=0.6$")
    plt.plot(t_vec, data_mat[:,int(z_position[4]), 0, angle], label = r"$z=0.8$")
    plt.plot(t_vec, data_mat[:,-1, 0, angle], label = r"$z=1$")
    plt.legend()
    plt.title(variable_name + " against t, at different z's, " + r"$\theta=$" + str(view_angle))
    plt.show()



'''This function automatically generates a plot of the variable you want,
against z, at six different times, 0%, 20%, 40%, 60%, 80%, 100%. The angle of
of plot is by default theta=0, can be changed manually'''
def plot_against_z_at_different_time(data_mat, variable_name, view_angle = 0):
    # data_mat is a 4d matrix, variable_name is the string name of the data, used for title, should be a or c
    num_t, num_z, num_r, num_theta = np.shape(data_mat)
    angle = int(np.floor(view_angle/2/np.pi*num_theta
                         ))
    z_vec = np.linspace(0,1,num_z)
    t_position = np.zeros(6)
    for i in range(6):
        t_position[i] = int(num_t * i/5)

    plt.plot(z_vec, data_mat[int(t_position[0]), :, 0, angle], label = r"$t=0$")
    plt.plot(z_vec, data_mat[int(t_position[1]), :, 0, angle], label = r"$t=0.2t_f$")
    plt.plot(z_vec, data_mat[int(t_position[2]), :, 0, angle], label = r"$t=0.4t_f$")
    plt.plot(z_vec, data_mat[int(t_position[3]), :, 0, angle], label = r"$t=0.6t_f$")
    plt.plot(z_vec, data_mat[int(t_position[4]), :, 0, angle], label = r"$t=0.8t_f$")
    plt.plot(z_vec, data_mat[-1, :, 0, angle], label = r"$t=t_f$")
    plt.legend()
    plt.title(variable_name + " against z, at different times, "+ r"$\theta=$" + str(view_angle))
    plt.show()

'''This function automatically generates a plot of the variable you want,
against t, at six different z's, z= 0, 0.2, 0.40, 0.6, 0.80, 1. The angle of
of plot is by default theta=0, can be changed manually, shoulbe be  between 0 and 2pi'''
def plot_against_t_at_different_z(data_mat, variable_name, endtime, view_angle = 0):
    # data_mat is a 4d matrix, variable_name is the string name of the data, used for title, should be a or c
    # endtime is the time t1 in the main script
    num_t, num_z, num_r, num_theta = np.shape(data_mat)
    angle = int(np.floor(view_angle/2/np.pi*num_theta))
    t_vec = np.linspace(0, endtime ,num_t)
    z_position = np.zeros(6)
    for i in range(6):
        z_position[i] = int(num_z * i/5)

    plt.plot(t_vec, data_mat[:,int(z_position[0]), 0, angle], label = r"$z=0$")
    plt.plot(t_vec, data_mat[:,int(z_position[1]), 0, angle], label = r"$z=0.2$")
    plt.plot(t_vec, data_mat[:,int(z_position[2]), 0, angle], label = r"$z=0.4$")
    plt.plot(t_vec, data_mat[:,int(z_position[3]), 0, angle], label = r"$z=0.6$")
    plt.plot(t_vec, data_mat[:,int(z_position[4]), 0, angle], label = r"$z=0.8$")
    plt.plot(t_vec, data_mat[:,-1, 0, angle], label = r"$z=1$")
    plt.legend()
    plt.title(variable_name + " against t, at different z's, " + r"$\theta=$" + str(view_angle))
    plt.show()


'''
Observe that our matrices for saving data are too large. Here we try to reduce
the size by removing duplicates.
Note: a0, a1, a are changing only with t, z, and theta, so can be reduced to 3D matrix
Note: c0, c1, c2, c are changing only with t and z
Note: Lambda2, Gamma2, sigmaS0, sigmaS1 are only changing with t and z. So reduced to 2D matrix
'''
def reduce_mat_size_to_3D(mat4d):
    # this function applies to a0, a1, and a
    return mat4d[:,:,0,:]

def reduce_mat_size_to_2D(mat4d):
    # this function applies to c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1
    return mat4d[:,:,0,0]

def reconstruct_4dmat_from_3d(mat3d, num_t, num_z, num_r, num_theta):
    # this function applies to reduced version of a0, a1, and a
    mat4d = np.zeros([num_t, num_z, num_r, num_theta])
    for i in range(num_r):
        mat4d[:,:,i,:] = mat3d
    return mat4d

def reconstruct_4dmat_from_2d(mat2d, num_t, num_z, num_r, num_theta):
    # this function applies to reduced version of c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1
    mat4d = np.zeros([num_t, num_z, num_r, num_theta])
    for i in range(num_r):
        for j in range(num_theta):
            mat4d[:,:,i,j] = mat2d

    return mat4d

def get_reduced_mats(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1):
    # return a sequence of 2d and 3d matrices
    reduced_a0 = reduce_mat_size_to_3D(a0)
    reduced_a1 = reduce_mat_size_to_3D(a1)
    reduced_a = reduce_mat_size_to_3D(a)
    reduced_c0 = reduce_mat_size_to_2D(c0)
    reduced_c1 = reduce_mat_size_to_2D(c1)
    reduced_c2 = reduce_mat_size_to_2D(c2)
    reduced_c = reduce_mat_size_to_2D(c)
    reduced_Lambda2 = reduce_mat_size_to_2D(Lambda2)
    reduced_Gamma2 = reduce_mat_size_to_2D(Gamma2)
    reduced_sigmaS0 = reduce_mat_size_to_2D(sigmaS0)
    reduced_sigmaS1 = reduce_mat_size_to_2D(sigmaS1)
    return reduced_a0, reduced_a1, reduced_a, reduced_c0, reduced_c1, reduced_c2, reduced_c, reduced_Lambda2, reduced_Gamma2, reduced_sigmaS0, reduced_sigmaS1

def get_back_mats(reduced_a0, reduced_a1, reduced_a, reduced_c0, reduced_c1, reduced_c2, reduced_c, reduced_Lambda2, reduced_Gamma2, reduced_sigmaS0, reduced_sigmaS1, sizes):
    # return a sequence of 4d matrices
    num_t, num_z, num_r, num_theta = sizes
    a0 = reconstruct_4dmat_from_3d(reduced_a0, num_t, num_z, num_r, num_theta)
    a1 = reconstruct_4dmat_from_3d(reduced_a1, num_t, num_z, num_r, num_theta)
    a = reconstruct_4dmat_from_3d(reduced_a, num_t, num_z, num_r, num_theta)
    c0 = reconstruct_4dmat_from_2d(reduced_c0, num_t, num_z, num_r, num_theta)
    c1 = reconstruct_4dmat_from_2d(reduced_c1, num_t, num_z, num_r, num_theta)
    c2 = reconstruct_4dmat_from_2d(reduced_c2,num_t, num_z, num_r, num_theta)
    c = reconstruct_4dmat_from_2d(reduced_c, num_t, num_z, num_r, num_theta)
    Lambda2 = reconstruct_4dmat_from_2d(reduced_Lambda2,num_t, num_z, num_r, num_theta)
    Gamma2 = reconstruct_4dmat_from_2d(reduced_Gamma2,num_t, num_z, num_r, num_theta)
    sigmaS0 = reconstruct_4dmat_from_2d(reduced_sigmaS0,num_t, num_z, num_r, num_theta)
    sigmaS1 = reconstruct_4dmat_from_2d(reduced_sigmaS1,num_t, num_z, num_r, num_theta)

    return a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1

def store_simulation_results(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1, folder_name):
    # store the results as matrices
    os.chdir(os.path.dirname(__file__))
    os.chdir("./" + folder_name + "/")
    reduced_a0, reduced_a1, reduced_a, reduced_c0, reduced_c1, reduced_c2, reduced_c, reduced_Lambda2, reduced_Gamma2, reduced_sigmaS0, reduced_sigmaS1 = get_reduced_mats(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1)
    scipy.io.savemat("a0_mat", mdict = {"a0": reduced_a0})
    scipy.io.savemat("a1_mat", mdict = {"a1": reduced_a1})
    scipy.io.savemat("Lambda2_mat", mdict = {"Lambda2": reduced_Lambda2})
    scipy.io.savemat("Gamma2_mat", mdict = {"Gamma2": reduced_Gamma2})
    scipy.io.savemat("a_mat", mdict = {"a": reduced_a})
    scipy.io.savemat("sigmaS0_mat", mdict = {"sigmaS0": reduced_sigmaS0})
    scipy.io.savemat("sigmaS1_mat", mdict = {"sigmaS1": reduced_sigmaS1})
    scipy.io.savemat("c_mat", mdict = {"c": reduced_c})
    scipy.io.savemat("c0_mat", mdict = {"c0": reduced_c0})
    scipy.io.savemat("c1_mat", mdict = {"c1": reduced_c1})
    scipy.io.savemat("c2_mat", mdict = {"c2": reduced_c2})

def load_simulation_results(folder_name, sizes):
    os.chdir(os.path.dirname(__file__))
    os.chdir("./" + folder_name + "/")

    reduced_a0 = scipy.io.loadmat("a0_mat.mat")["a0"]
    reduced_a1 = scipy.io.loadmat("a1_mat.mat")["a1"]
    reduced_a = scipy.io.loadmat("a_mat.mat")["a"]

    reduced_c0 = scipy.io.loadmat("c0_mat.mat")["c0"]
    reduced_c1 = scipy.io.loadmat("c1_mat.mat")["c1"]
    reduced_c2 = scipy.io.loadmat("c2_mat.mat")["c2"]
    reduced_c = scipy.io.loadmat("c_mat.mat")["c"]

    reduced_Lambda2 = scipy.io.loadmat("Lambda2_mat.mat")["Lambda2"]
    reduced_Gamma2 = scipy.io.loadmat("Gamma2_mat.mat")["Gamma2"]
    reduced_sigmaS0 = scipy.io.loadmat("sigmaS0_mat.mat")["sigmaS0"]
    reduced_sigmaS1 = scipy.io.loadmat("sigmaS1_mat.mat")["sigmaS1"]

    return get_back_mats(reduced_a0, reduced_a1, reduced_a, reduced_c0, reduced_c1, reduced_c2, reduced_c, reduced_Lambda2, reduced_Gamma2, reduced_sigmaS0, reduced_sigmaS1, sizes)



def make_video(a, video_name = 'video.mp4'):
    '''make video'''
    os.chdir(os.path.dirname(__file__))
    # the above command makes sure that we are working in the current dir
    pathIn = './data_for_video/'
    pathOut = video_name
    make_cylinder_video(pathIn, pathOut, a[:,:,0,:])



def total_tissue_growth(a):
    '''parameter: a from function Simulation_Wz_C()
    return: the total tissue growth of this simulation'''
    th = theta_mat[:,0,:] #take the first row of all theta_mat for each z

    a_initial = a[0,:,0,:] #get the initial value, only need the first row, since each row are the same for same z
    a_x0 = a_initial*np.cos(th) # convert r,theta to x,y representation
    a_y0 = a_initial*np.sin(th)

    I_list = []
    for i in range(N): #time loop
        a_final = a[i,:,0,:]  #take the a at time i
        a_x1 = a_final*np.cos(th)
        a_y1 = a_final*np.sin(th)
        I = np.sum(simps(a_x0,a_y0)-simps(a_x1,a_y1))/num_z
        I_list.append(I)
    plt.plot(t,I_list)
    plt.title("total tissue growth")
    plt.show()

def plot_from_above(a):
    """Vertical view Plot, at final time and the initial time, with different z"""
    plt.figure(figsize=(5,5))
    ax = plt.subplot(111, projection='polar')
    ax.plot(0,1)
    ax.plot(theta[0], a[0,0,  0,:] ,'-.',label="z=0",   linewidth=0.7)
    ax.plot(theta[0], a[0,-1,0,:] ,'--',label="z=0.2", linewidth=0.7)
    ax.plot(theta[0], a[-1,0,  0,:] ,'-.',label="z=0",   linewidth=0.7)
    ax.plot(theta[0], a[-1,-1,0,:] ,'--',label="z=0.2", linewidth=0.7)
    #ax.plot(theta[0], a[-1,80,0,:] ,'-',label="z=0.4", linewidth=1)
    #ax.plot(theta[0], a[-1,120,0,:] ,'-.',label="z=0.6", linewidth=1.4)
    #ax.plot(theta[0], a[-1,160,0,:] ,'--',label="z=0.8", linewidth=1.7)
    #ax.plot(theta[0], a[-1,199,0,:] ,'-',label="z=1",   linewidth=2.0)

    #ax.set_rticks([0,0.5,1])
    ax.tick_params(direction='out', length= 6, width = 0.1, colors='k')
    plt.legend(loc = 'upper left')
    ax.grid(True)
    #ax.axis('off')
    plt.show()
