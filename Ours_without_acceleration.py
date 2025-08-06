import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import time
# import pandas as pd
####
# 对应文章中的Algorithm 1(Section 5)
####

def alg(envpara, config):
    """Run our algorithm Consensus-Tracking-ADMM (without acceleration)."""
    begin = time.time()
    
    #envpara = para.Param()
    N, M, K, A, B, C,D, Q, W, L, St, I, c0, c, theta = envpara.out()
    

    suminv = np.zeros(envpara.edgecnt)
    Qtuda = np.zeros((N,envpara.edgecnt)) 
    for i in range(N):
        Qtuda[i] = Q[i] @ np.ones(M*K*I)
        suminv = suminv + Qtuda[i]
        #Q[i] = coo_matrix(Q[i])
    for i in range(envpara.edgecnt):
        if suminv[i] > 1e-5:
            suminv[i] = 1.0 / suminv[i]
    suminv = np.diag(suminv) 
    
    Btuda = coo_matrix(np.vstack((B,C)))
    A = coo_matrix(A)
    
    
    Qtudatuda = []
    H = []
    Atuda = np.zeros((N,M*K,N*M*K*I))
    Atuda_s = []
    Atuda_t = []
    Btuda_s = np.zeros((N, M+K , N*M*K*I))
    Btuda0 = []
    Phi = []
    Dp = []
    Dptuda = []
    Theta = np.zeros((envpara.edgecnt,N*M*K*I))
    Mtuda = []
    
    
    for i in range(N):
        Qtudatuda.append(coo_matrix(np.diag(suminv @ Qtuda[i]))) 
        H.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(N*M*K*I)).T)) 
        Atuda[i][:,i*M*K*I:(i+1)*M*K*I] = A.toarray()
        Atuda_s.append(coo_matrix(Atuda[i]))
        Atuda_t.append(coo_matrix(Atuda[i].T @ Atuda[i]))
        Btuda_s[i][:,i*M*K*I:(i+1)*M*K*I] = Btuda.toarray()
        Btuda0.append(coo_matrix(Btuda_s[i]))
        Phi.append(vstack((Atuda_s[i],H[i])))
        Dp.append(D[i])  
        Dptuda.append(np.concatenate((Dp[i],np.zeros(N*N*M*K*I))))
        Theta[:,i*M*K*I:(i+1)*M*K*I] = Q[i]
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    
    Theta = coo_matrix(Theta) 
    
    
    Sigma = []
    Psi = np.zeros((N,N*M*K*I))
    
    for i in range(N):
        Sigma.append(coo_matrix(2 * c0 * Theta.T @ Qtudatuda[i] @ Theta))
        Psi[i][i*M*K*I: (i+1)*M*K*I] = c[i] @ Q[i]
        
    # Preparation complete.
    
    # Step 0 parameters 
    
    rho = config["rho"]   # penalty coefficient of consensus part
    sigma_0 = config["sigma"]  # penalty coefficient of coupling constraint part
    
    # Step 1 generate the required matrix
    Sigma_tensor = []
    temp = coo_matrix(np.identity((N*M*K*I)))
    for i in range(N):
        Sigma_tensor.append(coo_matrix(Sigma[i] + rho * envpara.Lap[i][i] * temp + sigma_0 * Atuda_t[i]))
       
    
    # Step 2 solve with Gurobi
    
    def stopcon(t, tim):
        """Stop criteria."""
        if t >= maxT:
            return False
        if tim >= maxTime:
            return False
        return True
    
    def solve(i, Delta):
        """Solve subproblem for each agent."""
        model = grb.Model("alg2")
        model.setParam(grb.GRB.Param.OutputFlag, 0)
        model.setParam(grb.GRB.Param.NumericFocus, 2)
        model.setParam(grb.GRB.Param.BarConvTol, 1e-12)
        model.setParam("Method", 2)
        model.setParam("OptimalityTol", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
    
        X = model.addMVar(N*M*K*I, vtype = grb.GRB.CONTINUOUS, name = "X", lb = float('-inf'))
        
        
        model.update()
        
        model.setObjective(0.5* X.T @ Sigma_tensor[i] @ X + Delta @ X, sense = grb.GRB.MINIMIZE)
        
        model.addConstr(Btuda @ X[i*M*K*I:(i+1)*M*K*I] <= Mtuda[i])
        model.addConstr(X[i*M*K*I:(i+1)*M*K*I] >= np.zeros(M*K*I))


        model.optimize()

    
        if model.status == grb.GRB.OPTIMAL:
            #print("Success.")
            pass
        elif model.status == grb.GRB.INFEASIBLE:
            print("Infeasible")
        elif model.status == grb.GRB.UNBOUNDED:
            print("Unbounded")
        else:
            print("Fail, status: ", model.status)
        
        Xval = X.X
       
        return np.array(list(Xval))
      
    
    # Step 3 design the initial value
    
    maxT = config["maxT"] 
    maxTime = config["maxTime"]
    y = [[] for i in range(N)]
    lamb = [[] for i in range(N)]
    d = [[] for i in range(N)]
    v = [[] for i in range(N)]
    perf_alg2 = [0 for i in range(maxT+2)]
    abs_con_vio = [0 for i in range(maxT+2)]
    times = [0 for i in range(maxT+1)]
    

    
    for i in range(N):
        y[i].append(np.zeros(N*M*K*I))   # y_i^0
        d[i].append(Atuda_s[i] @ y[i][0] - Dp[i])  # d_i^0
        lamb[i].append(np.zeros(M*K))   # lambda_i^0
        
    WW = []  
    for i in range(N):  
        WW.append([1 if x != 0 else x for x in W[i]])
        WW[i][i]=0
        temp = 0
        for j in range(N):
            temp = temp + WW[i][j] * y[j][0]
        v[i].append(0.5* y[i][0] + 0.5/envpara.Lap[i][i] * temp)
       
    abs_con_vio[0]= np.linalg.norm(D[0]*N)
    

    timepre = time.time()
    times[0] = 0
    print("pre complete", timepre-begin)
    
    t=0
    print(t, perf_alg2[t], abs_con_vio[t], time.time()-timepre)
    

    # Step 4 solve
    while stopcon(t, time.time() - timepre):
        
        l = []
        delta = []
        Delta_tensor = []
        
        for i in range(N):
            l.append(np.zeros(M*K))
            delta.append(np.zeros(M*K))
            for j in range(N):
                l[i] = l[i] + W[i][j] * lamb[j][0]
                delta[i] = delta[i] + W[i][j] * d[j][0]
            
            Delta_tensor.append(
                Psi[i] - rho * envpara.Lap[i][i] * v[i][0] + Atuda_s[i].T @ l[i] -
                sigma_0 * Atuda_t[i] @ y[i][0] + sigma_0 * Atuda_s[i].T @  delta[i]
                )
            
            xsolu = solve(i, Delta_tensor[i])
            
            y[i].append(xsolu)
            
            
            d[i].append(delta[i] + Atuda_s[i] @ (y[i][1] - y[i][0]) )
            lamb[i].append(l[i] + sigma_0 * d[i][1])
            
        
        for i in range(N):
            temp = 0
            for j in range(N):
                temp = temp + WW[i][j] * ( y[j][1] - 0.5* y[j][0] )
                    
            v[i].append( v[i][0] + 1/envpara.Lap[i][i] * temp - 0.5 * y[i][0])
        

        y = [sublist[1:] for sublist in y]
        d = [sublist[1:] for sublist in d]
        lamb = [sublist[1:] for sublist in lamb]
        v = [sublist[1:] for sublist in v] 
    
    
        # objective function value
        perf_alg2[t+1] = sum( 0.5 * y[i][0] @ Sigma[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N)) 
        
        # absolute value of coupling constraint violation
        abs_con_vio[t+1] = np.linalg.norm(sum( Phi[i] @ y[i][0] - Dptuda[i] for i in range(N)) )
        
        times[t+1] = time.time() - timepre
        if (t+1) % 50 == 0:
            print(t+1, perf_alg2[t+1], abs_con_vio[t+1], time.time()-timepre)
        t = t + 1
    

    
    np.save(config["savedir"] + "perf.npy", perf_alg2)
    np.save(config["savedir"] + "con.npy", abs_con_vio)
    np.save(config["savedir"] + "time.npy", times)
    
    perf2_9_o = {}
    perf2_9_o["perf"] = perf_alg2[maxT]
    perf2_9_o["con_vio"] = abs_con_vio[maxT]
    return perf2_9_o