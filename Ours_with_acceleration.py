
import numpy as np
import matplotlib.pyplot as plt
# import para
import gurobipy as grb
#import scipy
#import scipy.linalg
from scipy.sparse.linalg import factorized
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import time
import itertools


def alg(envpara, config):
    """Run our algorithm Accelerated-Consensus-Tracking-ADMM.
    
    Compared with alg in Ours.py, we add some limitations in gurobi solver 
        to ensure consistency between accelerated and non-accelerated comparison results.
    """
    begin = time.time()
    
    # envpara = para.Param()
    N, M, K, A, B, C,D, Q, W, L, St, I, c0, c, theta = envpara.out()
    

    suminv = np.zeros(envpara.edgecnt) 
    Qtuda = np.zeros((N,envpara.edgecnt)) 
    for i in range(N):
        Qtuda[i] = Q[i] @ np.ones(M*K*I)
        suminv = suminv + Qtuda[i]
    for i in range(envpara.edgecnt):
        if suminv[i] > 1e-5:
            suminv[i] = 1.0 / suminv[i]
    suminv = np.diag(suminv)
    
    
    Btuda = coo_matrix(np.vstack((B,C)))
    A = coo_matrix(A)
    AT = coo_matrix(A.T)
    ATA = coo_matrix(A.T @ A)
    
    
    Qtudatuda = []
    H = []
    Atuda = np.zeros((N,M*K,N*M*K*I))
    Atuda_s = []
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
        Phi.append(vstack((Atuda_s[i],H[i]))) 
        Dp.append(D[i])
        Dptuda.append(np.concatenate((Dp[i],np.zeros(N*N*M*K*I))))
        Theta[:,i*M*K*I:(i+1)*M*K*I] = Q[i]
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    del H,Atuda
    
    Theta = coo_matrix(Theta) 
    
    
    Sigma = []
    Sigma_core = []
    Sigma_minus = []  
    Sigma_plus = [] 
    Psi = np.zeros((N,N*M*K*I))
    
    
    for i in range(N):
        Sigma.append(coo_matrix(2 * c0 * Theta.T @ Qtudatuda[i] @ Theta)) 
        row0=slice(i*M*K*I,(i+1)*M*K*I)
        row=list(range(i*M*K*I,(i+1)*M*K*I))
        Sigma_core.append(coo_matrix(Sigma[i].toarray()[row0,row0]))
        Sigma_minus.append(coo_matrix(np.delete(Sigma[i].toarray()[row0,:],row,axis=1)))
        Sigma_plus.append(coo_matrix(np.delete(np.delete(Sigma[i].toarray(),row,axis=0),row,axis=1)))
        Psi[i][i*I*M*K:(i+1)*I*M*K] = c[i] @ Q[i] 
    
    # Preparation complete.
    
    # Step 0 parameters 
    
    rho = config["rho"]   # penalty coefficient of consensus part
    sigma_0 = config["sigma"]  # penalty coefficient of coupling constraint part
    
    
    # Step 1 generate the required matrix
    Sigma_core_b = []
    Sigma_plus_b = [] 
    Sigma_plus_b_inverse = []
    solve_fn = []
    
    for i in range(N):
        Sigma_core_b.append(coo_matrix(Sigma_core[i]+rho*envpara.Lap[i][i]*np.identity((M*K*I))))
        Sigma_plus_b.append(coo_matrix(Sigma_plus[i]+rho*envpara.Lap[i][i]*np.identity(((N-1)*M*K*I))))
        solve_fn.append(factorized(Sigma_plus_b[i].tocsc()))
        Sigma_plus_b_inverse.append(solve_fn[i](np.identity(((N-1)*M*K*I))))
    
    Sigma_p_i_m_b = []
    Sigma_c_m_p_m_b = []
    Sigma_tensor = []
    for i in range(N):
        Sigma_p_i_m_b.append(coo_matrix(Sigma_plus_b_inverse[i] @ Sigma_minus[i].T))
        temp = Sigma_minus[i] @ Sigma_plus_b_inverse[i] @ Sigma_minus[i].T
        Sigma_c_m_p_m_b.append(coo_matrix(Sigma_core_b[i] - temp))
        Sigma_tensor.append(coo_matrix(Sigma_c_m_p_m_b[i]+sigma_0*ATA))
    
    
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
        #model.setParam(grb.GRB.Param.BarHomogeneous, 0) 
        model.setParam(grb.GRB.Param.BarConvTol, 1e-12)
        model.setParam("Method", 2) 
        model.setParam("OptimalityTol", 1e-9)
        model.setParam("FeasibilityTol", 1e-9)
        
        X = model.addMVar(M*K*I, vtype = grb.GRB.CONTINUOUS, name = "X", lb = 0)
       
        model.update()
        
        #timein1 = time.time()
        #model.setMObjective(Sigma0tuda,Bigphituda,0, sense = grb.GRB.MINIMIZE)
        model.setObjective(0.5* X.T @ Sigma_tensor[i] @ X + Delta @ X, sense = grb.GRB.MINIMIZE)
        # timein2 = time.time()
        
        model.addConstr(Btuda @ X <= Mtuda[i])
        #model.addConstr( X[M*K*I: M*K*I + M*K] <= 50000 )
    
        #timein3 = time.time()
        model.optimize()
        #timein4 = time.time()
        
        
        if model.status == grb.GRB.OPTIMAL:
            #print("Success.")
            pass
        elif model.status == grb.GRB.INFEASIBLE:
            print("Infeasible")
        elif model.status == grb.GRB.UNBOUNDED:
            print("Unbounded")
        else:
            print("Fail, status: ", model.status)
        
        
        
        #print(timein2-timein1)
        #print(timein3-timein2)
        #print(timein4-timein3)
        Xval = X.X
       
        return np.array(list(Xval))
      
    
    # Step 3 design the initial value
    
    maxT = config["maxT"]
    maxTime = config["maxTime"]
    
    x = [[] for i in range(N)]
    x_minus = [[] for i in range(N)]
    y = [[] for i in range(N)]
    lamb = [[] for i in range(N)]
    d = [[] for i in range(N)]
    v = [[] for i in range(N)]
    perf_alg2 = [0 for i in range(maxT+2)]
    abs_con_vio = [0 for i in range(maxT+2)]

    times = [0 for i in range(maxT+1)]
    
    
    for i in range(N):
        x[i].append(np.zeros(M*K*I))
        x_minus[i].append(np.zeros((N-1)*M*K*I))
        y[i].append(np.zeros(N*M*K*I)) 
        d[i].append(A @ x[i][0] - Dp[i])
        lamb[i].append(np.zeros(M*K))
        
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
    times[0] = timepre-begin
    print("pre complete", timepre-begin)
    
    t=0
    print(t, perf_alg2[t], abs_con_vio[t], time.time()-timepre)
    
    # Step 4 solve
    while stopcon(t, time.time() - timepre):
        
        l = []
        delta = []
        Delta_x = []
        
        for i in range(N):
            l.append(np.zeros(M*K))
            delta.append(np.zeros(M*K))
            for j in range(N):
                l[i] = l[i] + W[i][j] * lamb[j][0]
                delta[i] = delta[i] + W[i][j] * d[j][0]
            # for j in range(N):
            #      l[i] = l[i] + 1/N * lamb[j][0]
            #      delta[i] = delta[i] + 1/N * d[j][0]
            
            
            Delta_x.append(
                Q[i].T @ c[i] + 
                A.T @ l[i] + 
                sigma_0 * A.T @ (delta[i]- A @ x[i][0]) +
                rho*envpara.Lap[i][i]*(-v[i][0][i*M*K*I:(i+1)*M*K*I]+Sigma_p_i_m_b[i].T @ np.delete(v[i][0],slice(i*M*K*I,(i+1)*M*K*I)))
                )
            
            xsolu = solve(i, Delta_x[i])
            
            x[i].append(xsolu[0 : M*K*I])
            
            x_minus[i].append( 
                - Sigma_p_i_m_b[i] @ x[i][1] + 
                rho * envpara.Lap[i][i] * Sigma_plus_b_inverse[i] @ np.delete(v[i][0],slice(i*M*K*I,(i+1)*M*K*I))
                )
            y[i].append(np.concatenate((x_minus[i][1][:i*M*K*I],x[i][1],x_minus[i][1][i*M*K*I:])))
            
            
            d[i].append(delta[i] + Atuda_s[i] @ (y[i][1] - y[i][0]))
            lamb[i].append(l[i] + sigma_0 * d[i][1])
            
        
        for i in range(N):
            temp = 0
            for j in range(N):
                temp = temp + WW[i][j] * ( y[j][1] - 0.5* y[j][0] )
                    
            v[i].append( v[i][0] + 1/envpara.Lap[i][i] * temp - 0.5 * y[i][0])
        
       # perf_alg2[t+1] = funcg()

        x = [sublist[1:] for sublist in x]
        x_minus = [sublist[1:] for sublist in x_minus]
        y = [sublist[1:] for sublist in y]
        d = [sublist[1:] for sublist in d]
        lamb = [sublist[1:] for sublist in lamb]
        v = [sublist[1:] for sublist in v] 
    
    
        # objective function value
        perf_alg2[t+1] = sum( 0.5 * y[i][0] @ Sigma[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N))
        
        # absolute value of coupling constraint violation
        tempsum = sum( Phi[i] @ y[i][0] - Dptuda[i] for i in range(N))
        abs_con_vio[t+1] = np.linalg.norm(tempsum)

        

        times[t+1] = time.time()-timepre
        if (t+1) % 50 == 0:
            print(t+1, perf_alg2[t+1], abs_con_vio[t+1], time.time()-timepre)
        t = t + 1
    

    
    np.save(config["savedir"] + "perf.npy", perf_alg2)
    np.save(config["savedir"] + "con.npy", abs_con_vio)

    np.save(config["savedir"] + "time.npy", times)
    
    perf2_9 = {}
    perf2_9["perf"] = perf_alg2[maxT]
    perf2_9["con_vio"] = abs_con_vio[maxT]
    return perf2_9
