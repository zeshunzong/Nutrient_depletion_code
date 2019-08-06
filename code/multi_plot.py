'''functions here are used, if you need to load several simulation result, then plot them together'''

from auxiliary_funcs import total_tissue_growth,load_simulation_results
import matplotlib.pyplot as plt
from initialization import *

def plot_total_tissue():
    """plot several simulation result"""
    plt.figure(1)
    folder_list = ["lam_0_fig4","lam_1e_2_fig4","lam_1e_1_fig4","lam_2e_1_fig4","lam_5e_1_fig4"]
    lam_list = [0, 0.01, 0.1, 0.2, 0.5]*np.pi*R_hat**2/(Qi_hat*e)
    label_list = []
    for i in range(len(lam_list)):
        label_list.append(str(lam_list[i]))
    for i in range(len(folder_list)):
        a0, a1, a, c0, c1, c2, c, Lambda2, Gamma2, sigmaS0, sigmaS1 = load_simulation_results(folder_list[i], (N, num_z, num_r, num_theta))
        # calculate total tissue growth
        ttg = total_tissue_growth(a,Plot = False) 
        plt.plot(t,ttg,label = "$\eta^{*} = $"+label_list[i])
    plt.xlabel("time")
    plt.ylabel("total tissue growth")
    plt.legend()
    plt.show()