import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import time
import itertools

def alg(envpara, config):
    """Run comparison algorithm DC-ADMM, paper (A Proximal Dual Consensus ADMM Method for Multi-Agent Constrained Optimization)."""
    begin = time.time()
    
    #envpara = para.Param()
    N, M, K, A, B, C,D, Q, W, L, St, I, c0, c, theta= envpara.out()
    

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
    HTH = []
    ATA2 = []
    Atuda = np.zeros((N,M*K,N*M*K*I))
    Atuda_s = []
    Dp = []
    Theta = np.zeros((envpara.edgecnt,N*M*K*I))
    Mtuda = []
    Phi = []
    
    
    for i in range(N):
        Qtudatuda.append(coo_matrix(np.diag(suminv @ Qtuda[i])))   
        H.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(N*M*K*I)).T))  
        HTH.append(coo_matrix(H[i].T @ H[i]))
        Atuda[i][:,i*M*K*I:(i+1)*M*K*I] = A.toarray()  
        ATA2.append(coo_matrix(Atuda[i][:,:N*M*K*I].T @ Atuda[i][:,:N*M*K*I]))
        Atuda_s.append(coo_matrix(Atuda[i]))
        Phi.append(vstack((Atuda_s[i],H[i])))  
        Dp.append(np.concatenate((D[i],np.zeros((N*N*M*K*I)))))
        Theta[:,i*M*K*I:(i+1)*M*K*I] = Q[i]
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    del H,Atuda
    
    Theta = coo_matrix(Theta)   
    
    Sigma = []
    Psi = np.zeros((N,N*M*K*I))
    #Phituda = []
    
    for i in range(N):
        Sigma.append(coo_matrix(2 * c0 * Theta.T @ Qtudatuda[i] @ Theta))
        Psi[i][i*I*M*K:(i+1)*I*M*K] = c[i] @ Q[i]  # 生成 Psi_i 矩阵
    
    
    # Preparation complete

    # Step 0 parameters 
    
    sigma_0= config["sigma"]
    
    # Step 1 generate the required matrix
    
    Sigma_b = []
    Sigma_core_b = []
    Sigma_minus_b = []
    Sigma_plus_b = [] 
    Sigma_plus_b_inverse = []
    
    
    for i in range(N):
        Sigma_b.append(Sigma[i] +  (ATA2[i] + HTH[i])/(2 * sigma_0 * envpara.Lap[i][i]))
        row0=slice(i*M*K*I,(i+1)*M*K*I)
        row=list(range(i*M*K*I,(i+1)*M*K*I))
        Sigma_core_b.append(coo_matrix(Sigma_b[i].toarray()[row0,row0]))
        Sigma_minus_b.append(coo_matrix(np.delete(Sigma_b[i].toarray()[row0,:],row,axis=1)))
        Sigma_plus_b.append(coo_matrix(np.delete(np.delete(Sigma_b[i].toarray(),row,axis=0),row,axis=1)))
        temp=coo_matrix(np.linalg.inv(Sigma_plus_b[i].toarray()))  
        Sigma_plus_b_inverse.append(coo_matrix(0.5*(temp.T+temp))) 
    
        
    Sigma_p_i_m_b = []
    Sigma_tensor = []
    for i in range(N):
        Sigma_p_i_m_b.append(coo_matrix(Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T))
        temp = Sigma_minus_b[i] @ Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T
        Sigma_tensor.append(coo_matrix(Sigma_core_b[i] - 0.5*(temp+temp.T)))
    
    
    # Step 2 solve with Gurobi
    
    def stopcon(t):
        """Stop criteria."""
        return t < maxT
    
    def solve(i, Delta):
        """Solve subproblem for each agent."""
        model = grb.Model("alg2")
        model.setParam(grb.GRB.Param.OutputFlag, 0)
        X = model.addMVar(M*K*I, vtype = grb.GRB.CONTINUOUS, name = "X", lb = 0)
       
        model.update()
        
        #timein1 = time.time()
        #model.setMObjective(Sigma0tuda,Bigphituda,0, sense = grb.GRB.MINIMIZE)
        model.setObjective(0.5* X.T @ Sigma_tensor[i] @ X + Delta @ X, sense = grb.GRB.MINIMIZE)
        # timein2 = time.time()
        
        model.addConstr(Btuda @ X <= Mtuda[i])
    
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
        
        Xval = X.X
       
        return np.array(list(Xval))
      
    #-----------------------------------------------------------------------------
    
    # Step 3 design the initial value
    maxT = config["maxT"]   # max iteration step
    
    x = [[] for i in range(N)]
    x_minus = [[] for i in range(N)]
    y = [[] for i in range(N)]
    lamb = [[] for i in range(N)]
    p = [[] for i in range(N)]
    
    
    perf_alg2 = [0 for i in range(maxT+1)]
    abs_con_vio = [0 for i in range(maxT+1)]
    times = [0 for i in range(maxT+1)]
    
    WW = []  
    for i in range(N):
        x[i].append(np.zeros(M*K*I))
        x_minus[i].append(np.zeros((N-1)*M*K*I))
        y[i].append(np.zeros(N*M*K*I))   # y_i^0，即初值 
        lamb[i].append(np.zeros(M*K+N*N*M*K*I)) 
        p[i].append(np.zeros(M*K+N*N*M*K*I))
        WW.append([1 if x != 0 else x for x in W[i]])
        WW[i][i]=0
        #ax_z[i].append(np.zeros(M*K+N*N*M*K*I+N*M*K)) 
    
    perf_alg2[0] = sum( 0.5 * y[i][0] @ Sigma[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N))
    abs_con_vio[0] = np.linalg.norm(sum( Phi[i] @ y[i][0] - Dp[i] for i in range(N)))
    
    
    timepre = time.time()
    print("pre complete", timepre-begin)
    
    t=0
    times[0] = time.time() - timepre
    print(t, perf_alg2[t], abs_con_vio[t], time.time()-timepre)
    
    # Step 4 solve

    while stopcon(t):
        
        ll = []
        delta = []
        Gamma = []
        Gamma_core = []
        Gamma_minus = []
        Delta_x = []
        
        
        for i in range(N):
            
            ll.append( sum( WW[i][j] * (lamb[i][0] + lamb[j][0]) for j in range(N)) )
            
            Gamma.append( Psi[i] -  ( Phi[i].T @ ( Dp[i] + p[i][0] - sigma_0 * ll[i] ))/(2 * sigma_0 * envpara.Lap[i][i] )  )
            Gamma_core.append(Gamma[i][i*M*K*I:(i+1)*M*K*I])
            Gamma_minus.append(np.delete(Gamma[i],slice(i*M*K*I,(i+1)*M*K*I)))
            
            
            Delta_x.append(
                Gamma_core[i] - Sigma_p_i_m_b[i].T @ Gamma_minus[i]
                )
            
            
            xsolu = solve(i, Delta_x[i])
            
            x[i].append(xsolu)
            
            #x_minus[i].append(np.maximum(0,-AA_inverserBBT[i] @ x[i][1] - AA_inverse[i] @ CC[i]))
            x_minus[i].append( 
                - Sigma_p_i_m_b[i] @ x[i][1] - Sigma_plus_b_inverse[i] @ Gamma_minus[i]
                )
            
            y[i].append(np.concatenate((x_minus[i][1][:i*M*K*I],x[i][1],x_minus[i][1][i*M*K*I:])))
            
            lamb[i].append(  ( sigma_0 * ll[i] - p[i][0] + Phi[i] @ y[i][1] - Dp[i]) / (2 * sigma_0 * envpara.Lap[i][i])) 
            
            
        for i in range(N):
            
            delta.append( sigma_0 *  sum( WW[i][j] * (lamb[i][1] - lamb[j][1]) for j in range(N)) )
            
            p[i].append( p[i][0] + delta[i] )
    
                
        x = [sublist[1:] for sublist in x]
        y = [sublist[1:] for sublist in y]
        x_minus = [sublist[1:] for sublist in x_minus]
        lamb = [sublist[1:] for sublist in lamb]
        p = [sublist[1:] for sublist in p]
    
       
        # objective function value
        perf_alg2[t+1] = sum( 0.5 * y[i][0] @ Sigma[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N))
        
        # absolute value of coupling constraint violation
        abs_con_vio[t+1] = np.linalg.norm(sum( Phi[i] @ y[i][0] - Dp[i] for i in range(N)))
        times[t+1] = time.time() - timepre
        if (t+1) % 50 == 0:
            print(t+1, perf_alg2[t+1], abs_con_vio[t+1], time.time()-timepre)
        t = t + 1
    
    
    xl = [i for i in range(maxT+1)]
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(xl, perf_alg2)
    plt.axhline(y=config["best_obj"], color='r', linestyle='--')
    plt.xlabel('Iteration steps')
    plt.ylabel('objective function value')
    plt.title( 'No parameter')
    # plt.xlim(0,5001)
    # plt.ylim(0,6800000)
    
    
    plt.figure(figsize=(8, 6))
    plt.plot(xl, abs_con_vio, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Iteration steps')
    plt.ylabel('violations')
    plt.title( 'No parameter')
    '''
    np.save(config["savedir"] + "perf.npy", perf_alg2)
    np.save(config["savedir"] + "con.npy", abs_con_vio)
    #np.save(config["savedir"] + "time.npy", times)
    
    return perf_alg2