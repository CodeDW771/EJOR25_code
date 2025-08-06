import sys
import numpy as np
import para_new
import matplotlib.pyplot as plt
import std_v3


# ---------------------------------------------------------------------------

def draw(examplenow, envpara, type, dir):
    '''Draw two figures of performance: the comparison of function value and constraint error.

    Args:
        examplenow: Example ID, 1/2/3 for distributed algorithm comparison, 4/5/6 for acceleration comparison, representing small-scale, midium-scale, large-scale scenarios respectively.
        envpara: Generated from examplenow.
        type: 0 for distributed algorithm comparison, 1 for acceleration comparison.
        dir: Data storage directory.
    Reture:
        Two figure of of performance, fig_value is the comparison of function value, fig_constraint is the comparison of constraint error.
    '''
    opt_per = std_v3.std(envpara)

    # distributed algorithm comparison
    if type == 0:
        T = 10000
        perf1 = np.load(dir[0]+"perf.npy")
        perf2 = np.load(dir[1]+"perf.npy")
        perf3 = np.load(dir[2]+"perf.npy")
        perf4 = np.load(dir[3]+"perf.npy")
        perf5 = np.load(dir[4]+"perf.npy")

        con1 = np.load(dir[0]+"con.npy")
        con2 = np.load(dir[1]+"con.npy")
        con3 = np.load(dir[2]+"con.npy")
        con4 = np.load(dir[3]+"con.npy")
        con5 = np.load(dir[4]+"con.npy")

        xlim = [i for i in range(T+1)]
        fig_value, ax1 = plt.subplots(figsize = (11,9))

        ax1.plot(xlim, np.log10(np.abs(perf1[:T+1] - opt_per) / opt_per), label="Our algorithm", zorder = 2, linewidth=2.5)
        ax1.plot(xlim, np.log10(np.abs(perf3[:T+1] - opt_per) / opt_per), label="DPMM(2023)", zorder = 1, linewidth=2.5)
        ax1.plot(xlim, np.log10(np.abs(perf4[:T+1] - opt_per) / opt_per), label="IPLUX(2023)", zorder = 1, linewidth=2.5)
        ax1.plot(xlim, np.log10(np.abs(perf2[:T+1] - opt_per) / opt_per), label="Tracking-ADMM(2020)", zorder = 1, linewidth=2.5)
        ax1.plot(xlim, np.log10(np.abs(perf5[:T+1] - opt_per) / opt_per), label="DC-ADMM(2016)", zorder = 1, linewidth=2.5)
        #plt.plot(xlim, np.log10(np.abs(perf6[:T+1] - opt_per) / opt_per), label="Our algorithm()", zorder = 1)
        #plt.savefig("test2.png")

        ax1.set_xlabel("Iteration step", fontsize=20)
        ax1.set_ylabel("Relative error of function value (log10)", fontsize=20)
        ax1.set_xlim(0,T)
        ax1.set_xticks(np.arange(0, 10001, 2000))     
        ax1.set_xticklabels(np.arange(0, 10001, 2000),fontsize=20) 
        ax1.tick_params(axis='y', labelsize=20)
        #plt.title("Performance Comparison (log10)")
        if examplenow == 1:
            ax1.legend(fontsize=20)
        else:
            ax1.legend(fontsize=20, loc='lower left')
        #plt.savefig("function value of " + example + ".jpg")
        ax1.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)


        fig_constraint, ax2 = plt.subplots(figsize = (11,9))
        ax2.plot(xlim, np.log10(con1[:T+1]), label="Our algorithm", zorder = 2, linewidth=2.5)
        ax2.plot(xlim, np.log10(con3[:T+1]), label="DPMM(2023)", zorder = 1, linewidth=2.5)
        ax2.plot(xlim, np.log10(con4[:T+1]), label="IPLUX(2023)", zorder = 1, linewidth=2.5)
        ax2.plot(xlim, np.log10(con2[:T+1]), label="Tracking-ADMM(2020)", zorder = 1, linewidth=2.5)
        ax2.plot(xlim, np.log10(con5[:T+1]), label="DC-ADMM(2016)", zorder = 1, linewidth=2.5)
        #plt.plot(xlim, np.log10(con6[:T+1]), label="Our algorithm()", zorder = 1)
        #plt.savefig("test2.png")

        ax2.set_xlabel("Iteration step", fontsize=20)
        ax2.set_ylabel("Constraint violation (log10)", fontsize=20)
        ax2.set_xlim(0,T)
        ax2.set_xticks(np.arange(0, 10001, 2000))    
        ax2.set_xticklabels(np.arange(0, 10001, 2000),fontsize=20) 
        ax2.tick_params(axis='y', labelsize=20)
        #plt.xlim(0,T)
        #plt.title("Performance Comparison (log10)")
        if examplenow == 1:
            ax2.legend(fontsize=20)
        else:
            ax2.legend(fontsize=20, loc='lower left')
        #plt.savefig("constraint erro of " + example + ".jpg")
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)

        return fig_value, fig_constraint

    # acceleration comparison
    elif type == 1:
        perf1 = np.load(dir[0]+"perf.npy")
        perf6 = np.load(dir[1]+"perf.npy")

        con1 = np.load(dir[0]+"con.npy")
        con6 = np.load(dir[1]+"con.npy")

        times1 = np.load(dir[0]+"time.npy")
        times6 = np.load(dir[1]+"time.npy")
        times1[0] = 0
        times6[0] = 0

        fig_value, ax1 = plt.subplots(figsize=(11, 9))
        fig_constraint, ax2 = plt.subplots(figsize=(11,9))
        if examplenow == 4:
            T1 = np.where(con1 == 0)[0][0] - 1
            T2 = np.where(con6 == 0)[0][0] - 1
            Timelimit = 25
            ax1.set_xticks(np.arange(0, 26, 5))
            ax2.set_xticks(np.arange(0, 26, 5))
        elif examplenow == 5:
            T1 = np.where(con1 == 0)[0][0] - 1
            T2 = np.where(con6 == 0)[0][0] - 1
            Timelimit = 600
            ax1.set_xticks(np.arange(0, 601, 100))
            ax2.set_xticks(np.arange(0, 601, 100))
        elif examplenow == 6:
            T1 = np.where(con1 == 0)[0][0] - 1
            T2 = np.where(con6 == 0)[0][0] - 1
            Timelimit = 15000
            ax1.set_xticks(np.arange(0, 15001, 3000))
            ax2.set_xticks(np.arange(0, 15001, 3000))
        #print(T1, T2)
        ax1.plot(times1[:T1+1], np.log10(np.abs(perf1[:T1+1] - opt_per) / opt_per), label="Algorithm 2", zorder = 2, linewidth=2.5)
        ax1.plot(times6[:T2+1], np.log10(np.abs(perf6[:T2+1] - opt_per) / opt_per), label="Algorithm 1", zorder = 1, linewidth=2.5)

        ax1.set_xlabel("Time(s)", fontsize=20)
        ax1.set_ylabel("Relative error of function value (log10)", fontsize=20)
        ax1.set_xlim(0,Timelimit)
        #plt.title("Performance Comparison (log10)")
        ax1.legend(fontsize=20)
        #plt.savefig("function value of " + example + "_2.jpg")
        ax1.grid(True, alpha=0.3)
        ax1.set_axisbelow(True)
        #plt.show()
        


        
        ax2.plot(times1[:T1+1], np.log10(con1[:T1+1]), label="Algorithm 2", zorder = 2, linewidth=2.5)
        ax2.plot(times6[:T2+1], np.log10(con6[:T2+1]), label="Algorithm 1", zorder = 1, linewidth=2.5)

        ax2.set_xlabel("Time(s)", fontsize=20)
        ax2.set_ylabel("Constraint violation (log10)", fontsize=20)
        ax2.set_xlim(0,Timelimit)
        #plt.title("Performance Comparison (log10)")
        ax2.legend(fontsize=20)
        #plt.savefig("constraint erro of " + example + "_2.jpg")
        ax2.grid(True, alpha=0.3)
        ax2.set_axisbelow(True)
        #plt.show()

        return fig_value, fig_constraint
        
