import sys
import numpy as np
import para_new
import matplotlib.pyplot as plt
import std_v3
import feasible
import Mech_related
import Mech_draw
import Algorithm_2_in_our_paper
import TrackingADMM
import DPMM
import IPLUX
import DCADMM
import Algorithm_1_in_our_paper_for_improvement_comparison
import Algorithm_2_in_our_paper_for_improvement_comparison
import performance_plot



def mechanism_design_comparison(examplenow=1):
    """Compare the performance between two machanisms.

    Args:
        examplenow: ID of example, examplenow is always 1.
    """

    envpara = para_new.Param(example = examplenow)

    # check the feasibility of parameters
    feasible = feasible.feasible_check(envpara)
    if feasible == False:
        print("Parameters have some problem.")
        sys.exit()


    results1, results2, results3 = Mech_related.task1(envpara)

    fig_d = Mech_draw.draw_task1d(results1)
    fig_m = Mech_draw.draw_task1m(results2)
    fig_c = Mech_draw.draw_task1c(results3)

    ind_result0 = Mech_related.task2_N0(envpara)
    ind_result1 = Mech_related.task2_N1(envpara)
    ind_result2 = Mech_related.task2_N2(envpara)
    ind_result3 = Mech_related.task2_N3(envpara)

    fig_line0 = Mech_draw.draw_task2_0(ind_result0)
    fig_line1 = Mech_draw.draw_task2_1(ind_result1)
    fig_line2 = Mech_draw.draw_task2_2(ind_result2)
    fig_line3 = Mech_draw.draw_task2_3(ind_result3)

    return

                
def distributed_alg_comparison(examplenow):
    """ Compare the performance between our algorithm and other four algorithms.

    Args:
        examplenow: ID of example, examplenow is in {1,2,3}, representing small-scale, midium-scale, large-scale scenarios respectively.
    """
    envpara = para_new.Param(example = examplenow)
    opt_per = std_v3.std(envpara)
    # distributed_alg_comparison(examplenow, envpara)
    rho_ours = [0.05, 0.05, 0.5]
    sigma_ours = [0.05, 0.05, 0.5]
    config_ours = {
        "best_obj": opt_per,
        "maxT": 10000,
        "savedir": "savedata/alg_ours_example" + str(examplenow) + "_",
        "rho": rho_ours[examplenow - 1],
        "sigma": sigma_ours[examplenow - 1]
    }
    perf_ours = Algorithm_2_in_our_paper.alg(envpara, config_ours)

    sigma_TrackingADMM = [0.1, 0.5, 0.5]
    config_TrackingADMM = {
        "best_obj": opt_per,
        "maxT": 10000,
        "savedir": "savedata/alg_trackingADMM_example" + str(examplenow) + "_",
        "sigma": sigma_TrackingADMM[examplenow - 1]
    }
    perf_TrackingADMM = TrackingADMM.alg(envpara, config_TrackingADMM)

    gamma_DPMM = [0.5, 0.8, 0.8]
    beta_DPMM = [0.1, 0.1, 0.1]
    config_DPMM = {
        "best_obj": opt_per,
        "maxT": 10000,
        "savedir": "savedata/alg_DPMM_example" + str(examplenow) + "_",
        "gamma": gamma_DPMM[examplenow - 1],
        "beta": beta_DPMM[examplenow - 1]
    }
    perf_DPMM = DPMM.alg(envpara, config_DPMM)

    alpha_IPLUX = [0.5, 0.5, 0.5]
    rho_IPLUX = [0.5, 0.5, 0.5]
    config_IPLUX = {
        "best_obj": opt_per,
        "maxT": 10000,
        "savedir": "savedata/alg_IPLUX_example" + str(examplenow) + "_",
        "alpha": alpha_IPLUX[examplenow - 1],
        "rho": rho_IPLUX[examplenow - 1]
    }
    perf_IPLUX = IPLUX.alg(envpara, config_IPLUX)

    sigma_DCADMM = [0.1, 0.1, 0.1]
    config_DCADMM = {
        "best_obj": opt_per,
        "maxT": 10000,
        "savedir": "savedata/alg_DCADMM_example" + str(examplenow) + "_",
        "sigma": sigma_DCADMM[examplenow - 1]

    }
    perf_DCADMM = DCADMM.alg(envpara, config_DCADMM)

    totaldir = [config_ours["savedir"], config_TrackingADMM["savedir"], config_DPMM["savedir"], config_IPLUX["savedir"], config_DCADMM["savedir"]]
    fig1, fig2 = performance_plot.draw(examplenow = examplenow, envpara = envpara, type = 0, dir = totaldir)
    dir1 = "value_example"+str(examplenow)+".png"
    dir2 = "error_example"+str(examplenow)+".png"
    fig1.savefig(dir1)
    fig2.savefig(dir2)

    return




def improvement_comparison(examplenow):
    """ Compare the performance between Algorithm 1 (Consensus-Tracking-ADMM) and Algorithm 2 (Improved-Consensus-Tracking-ADMM).

    Args:
        examplenow: ID of example, examplenow is in {4,5,6}, representing small-scale, midium-scale, large-scale scenarios respectively.
    """
    envpara = para_new.Param(example = examplenow)
    opt_per = std_v3.std(envpara)
    rho_improvement = [0.05, 0.1, 0.3]
    sigma_improvement = [0.05, 0.1, 0.3]
    maxTime_improvement = [100, 1000, 15000]

    config_improvement = {
        "best_obj": opt_per,
        "maxT": 150000,
        "maxTime": maxTime_improvement[examplenow - 4],
        "savedir": "savedata/alg_improvement_example" + str(examplenow) + "_",
        "rho": rho_improvement[examplenow - 4],
        "sigma": sigma_improvement[examplenow - 4]
    }
    perf_improvement = Algorithm_2_in_our_paper_for_improvement_comparison.alg(envpara, config_improvement)



    config_noimprovement = {
        "best_obj": opt_per,
        "maxT": 10000,
        "maxTime": maxTime_improvement[examplenow - 4],
        "savedir": "savedata/alg_noimprovement_example" + str(examplenow) + "_",
        "rho": rho_improvement[examplenow - 4],
        "sigma": sigma_improvement[examplenow - 4]

    }
    perf_noimprovement = Algorithm_1_in_our_paper_for_improvement_comparison.alg(envpara, config_noimprovement)


    totaldir = [config_improvement["savedir"], config_noimprovement["savedir"]]
    fig1, fig2 = performance_plot.draw(examplenow = examplenow, envpara = envpara, type = 1, dir = totaldir)
    dir1 = "value_example"+str(examplenow)+".png"
    dir2 = "error_example"+str(examplenow)+".png"
    fig1.savefig("value_example"+str(examplenow)+".png")
    fig2.savefig("error_example"+str(examplenow)+".png")






# ---------------------------------------------------------------------------

# examplenow = 1 for mechanism_design_comparison
# examplenow = 1/2/3 for distributed_alg_comparison
# examplenow = 4/5/6 for acceleration_comparison

#mechanism_design_comparison(examplenow = 1)
#distributed_alg_comparison(examplenow = 1)
#improvement_comparison(examplenow = 4)


