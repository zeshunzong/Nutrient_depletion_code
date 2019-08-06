from auxiliary_funcs import *
from simulation_funcs import *
from initialization import *


def store_one_simulation(folder_name):
    # change to current working directory
    os.chdir(os.path.dirname(__file__))
    # make a folder
    os.mkdir(folder_name)
    # run simulation
    a0,a1,Lambda2,Gamma2,a,sigmaS0,sigmaS1,c,c0,c1,c2 = Simulation_Wz_C()
    # store results in the folder just created
    store_simulation_results(a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1, folder_name)
    os.chdir(os.path.dirname(__file__))
    os.chdir(folder_name)
    file1 = open("parameters_for_this_simulation.txt","w")
    file1.write("t1: " + str(t1) + "\n")
    file1.write("dt: " + str(dt)+ "\n")
    file1.write("R_hat: " + str(R_hat)+ "\n")
    file1.write("Qi_hat: " + str(Qi_hat)+ "\n")
    file1.write("Sigma_hat: " +str(Sigma_hat) + "\n" )
    file1.write("Lambda_hat: " + str(Lambda_hat)+ "\n")
    file1.write("e: " + str(e)+ "\n")
    file1.write("n: " + str(n)+ "\n")
    file1.write("pe_star: " + str(pe_star)+ "\n")
    file1.write("pe: " + str(pe)+ "\n")
    file1.write("lam_star: " + str(lam_star)+ "\n")
    file1.write("num_r: " + str(num_r)+ "\n")
    file1.write("num_theta: "  + str(num_theta)+ "\n")
    file1.write("num_z: "+ str(num_z)+ "\n")
    os.chdir(os.path.dirname(__file__))


# PLEASE CHANGE FOLDER NAME, EVERYTIME YOU RUN
#folder_name = "saved_matrices_v1"
#store_one_simulation(folder_name)


# next you can load the result, after performing all the experiments you want
goal_folder_name = "saved_matrices_v1"
a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = load_simulation_results(goal_folder_name, (N, num_z, num_r, num_theta))
