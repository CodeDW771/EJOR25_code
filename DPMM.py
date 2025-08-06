import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import time

# import pandas as pd

def alg(envpara, config):
    """Comparison algorithm DPMM (paper: Decentralized Proximal Method of Multipliers for Convex Optimization with Coupled Constraints)"""
    begin = time.time()
    
    #envpara = para.Param()
    N, M, K, A, B, C,D, Q, W, L, St, I,  c0, c, theta = envpara.out()
    
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
    
    
    theta=np.concatenate(theta)
    Btuda = coo_matrix(np.vstack((B,C)))
    Btuda0 = hstack((Btuda, coo_matrix(np.zeros((M+K, M*K)))))
    A = coo_matrix(A)
    AT = coo_matrix(A.T)
    ATA = coo_matrix(A.T @ A)
    
        
    Qtudatuda = []
    H = []
    HTH = []
    G = []
    GTG = []
    ATA2 = []
    Atuda = np.zeros((N,M*K,N*M*K*I+M*K))
    Atuda_s = []
    Dp = [] 
    Theta = np.zeros((envpara.edgecnt,N*M*K*I))
    Mtuda = []
    Phi = []
    
    #temp11=coo_matrix(np.identity(M*K)*(1/N))
    temp12=coo_matrix(np.zeros((N*N*M*K*I, M*K)))
    temp13=coo_matrix(np.zeros((N*M*K, N*M*K*I)))
    
    for i in range(N):
        Qtudatuda.append(coo_matrix(np.diag(suminv @ Qtuda[i])))  
        H.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(N*M*K*I)).T))
        HTH.append(coo_matrix(H[i].T @ H[i]))
        G.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(M*K)).T))
        GTG.append(coo_matrix(G[i].T @ G[i]))
        Atuda[i][:,i*M*K*I:(i+1)*M*K*I] = A.toarray()
        ATA2.append(coo_matrix(Atuda[i][:,:N*M*K*I].T @ Atuda[i][:,:N*M*K*I]))
        Atuda[i][:,N*M*K*I: N*M*K*I+M*K] = np.identity((M*K))*(1/N)
        Atuda_s.append(coo_matrix(Atuda[i]))
        Phi.append(vstack((Atuda_s[i],hstack((H[i],temp12)),hstack((temp13,G[i])))))
        Dp.append(np.concatenate((D[i],np.zeros((N*N*M*K*I + N*M*K)))))
        Theta[:,i*M*K*I:(i+1)*M*K*I] = Q[i]
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    del H,G,Atuda,temp12,temp13
    
    Theta = coo_matrix(Theta)   
    
    Sigma = np.zeros((N,N*M*K*I + M*K,N*M*K*I + M*K))
    Sigma_0 = []
    Sigma_1 = []
    Psi = np.zeros((N,N*M*K*I+M*K))
    # Phituda = []
    
    for i in range(N):
        Sigma_1.append(coo_matrix(2 * c0 * Theta.T @ Qtudatuda[i] @ Theta))
        Sigma[i][0:N*M*K*I, 0:N*M*K*I] = Sigma_1[i].toarray()
        Sigma_0.append(coo_matrix(Sigma[i])) 
        Psi[i][i*I*M*K:(i+1)*I*M*K] = c[i] @ Q[i] 
        Psi[i][N*M*K*I: ] = (1/N) * theta
    
    
    # Preparation complete.
    
    # Step 0 parameters 
    
    gamma_0 = config["gamma"]   # penalty coefficient
    alpha_0 = 1   #  penalty coefficient
    theta_0 = 1
    beta_0 = config["beta"]    # step size
    
    
    # Step 1 generate the required matrix
    
    Sigma_b = []
    Sigma_core_b = []
    Sigma_minus_b = []
    Sigma_plus_b = [] 
    Sigma_plus_b_inverse = []
    
    for i in range(N):
        Sigma_b.append(coo_matrix(Sigma_1[i] + gamma_0 *(ATA2[i]+HTH[i]) + np.eye((N*M*K*I))/alpha_0))
        row0=slice(i*M*K*I,(i+1)*M*K*I)
        row=list(range(i*M*K*I,(i+1)*M*K*I))
        Sigma_core_b.append(coo_matrix(Sigma_b[i].toarray()[row0,row0]))
        Sigma_minus_b.append(coo_matrix(np.delete(Sigma_b[i].toarray()[row0,:],row,axis=1)))
        Sigma_plus_b.append(coo_matrix(np.delete(np.delete(Sigma_b[i].toarray(),row,axis=0),row,axis=1)))
        temp=coo_matrix(np.linalg.inv(Sigma_plus_b[i].toarray()))
        Sigma_plus_b_inverse.append(coo_matrix(0.5*(temp.T+temp)))
    
    
    # Sigma_pre = coo_matrix(vstack((hstack((np.zeros, 1/N * AT)),hstack((1/N * A , 1/(N*N)* coo_matrix(np.eye((M*K))))))))
        
    Sigma_p_i_m_b = []
    Sigma_c_m_p_m_b = []
    Sigma_tensor_p = np.zeros((N, M*K+M*K*I, M*K+M*K*I))
    Sigma_tensor = []
    for i in range(N):
        Sigma_p_i_m_b.append(coo_matrix(Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T))
        temp = Sigma_minus_b[i] @ Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T
        Sigma_c_m_p_m_b.append(coo_matrix(Sigma_core_b[i] - 0.5*(temp+temp.T)))
        Sigma_tensor_p[i][0:M*K*I,0:M*K*I] = Sigma_c_m_p_m_b[i].toarray()
        Sigma_tensor_p[i][M*K*I : M*K*I+M*K , M*K*I : M*K*I+M*K] = gamma_0 * (GTG[i].toarray())+(gamma_0/(N*N)+1/alpha_0)*np.eye((M*K))
        Sigma_tensor_p[i][0:M*K*I,M*K*I:M*K*I+M*K] = 1/N * AT.toarray()
        Sigma_tensor_p[i][M*K*I:M*K*I+M*K,0:M*K*I] = 1/N * A.toarray()
        Sigma_tensor.append(coo_matrix(Sigma_tensor_p[i]))
    
    
    # Step 2 solve with Gurobi
    
    def stopcon(t):
        """Stop criteria."""
        return t < maxT
    
    def solve(i, Delta):
        """Solve subproblem for each agent."""
        model = grb.Model("alg2")
        model.setParam(grb.GRB.Param.OutputFlag, 0)
        X = model.addMVar(M*K*I + M*K, vtype = grb.GRB.CONTINUOUS, name = "X", lb = 0)
       
        model.update()
        
        #timein1 = time.time()
        #model.setMObjective(Sigma0tuda,Bigphituda,0, sense = grb.GRB.MINIMIZE)
        model.setObjective(0.5* X.T @ Sigma_tensor[i] @ X + Delta @ X, sense = grb.GRB.MINIMIZE)
        # timein2 = time.time()
        
        model.addConstr(Btuda @ X[: M*K*I] <= Mtuda[i])
        model.addConstr(X[M*K*I: M*K*I + M*K] == 0 )
    
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
      
    
    # Step 3 design the initial value
    
    maxT=config["maxT"]   # max iteration step
    
    
    x = [[] for i in range(N)]
    w = [[] for i in range(N)]
    x_minus = [[] for i in range(N)]
    y = [[] for i in range(N)]
    lamb = [[] for i in range(N)]
    mu = [[] for i in range(N)]
    
    perf_alg2 = [0 for i in range(maxT+1)]
    abs_con_vio = [0 for i in range(maxT+1)]
    times = [0 for i in range(maxT+1)]

    
    for i in range(N):
        x[i].append(np.zeros(M*K*I))
        x_minus[i].append(np.zeros((N-1)*M*K*I))
        w[i].append(np.zeros(M*K))
        y[i].append(np.zeros(N*M*K*I+M*K))   # y_i^0
        lamb[i].append(np.zeros(M*K+N*N*M*K*I+N*M*K)) 
        mu[i].append(np.zeros(M*K+N*N*M*K*I+N*M*K)) 
    
    perf_alg2[0] = sum( 0.5 * y[i][0] @ Sigma_0[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N))
    abs_con_vio[0] = np.linalg.norm(sum( Phi[i] @ y[i][0] - Dp[i] for i in range(N)))

    timepre = time.time()
    print("pre complete", timepre-begin)
    
    t=0
    times[0] = time.time()-timepre
    print(t, perf_alg2[t], abs_con_vio[t], time.time()-timepre)
    
    while stopcon(t):
        
        y_hat = []
        lamb_hat = []
        Gamma = []
        Gamma_core = []
        Gamma_w = []
        Gamma_minus = []
        Delta_x = []
        Delta_w = []
        Delta_tensor = []
        
        for i in range(N):
            
            Gamma.append( Psi[i] + Phi[i].T @ (lamb[i][0] - gamma_0 * Dp[i] - gamma_0 * mu[i][0])-y[i][0]/alpha_0  )
            Gamma_core.append(Gamma[i][i*M*K*I:(i+1)*M*K*I])
            Gamma_w.append(Gamma[i][N*M*K*I:])
            Gamma_minus.append(np.delete(Gamma[i][:N*M*K*I],slice(i*M*K*I,(i+1)*M*K*I)))
            
            
            Delta_x.append(
                Gamma_core[i] - Sigma_p_i_m_b[i].T @ Gamma_minus[i]
                )
            
            Delta_w.append(
                Gamma_w[i]
                )
            
            Delta_tensor.append(np.concatenate((Delta_x[i], Delta_w[i])))
            
            
            xsolu = solve(i, Delta_tensor[i])
            
            x[i].append(xsolu[0 : M*K*I])
            w[i].append(xsolu[M*K*I : M*K*I + M*K])
            
            #x_minus[i].append(np.maximum(0,-AA_inverserBBT[i] @ x[i][1] - AA_inverse[i] @ CC[i]))
            x_minus[i].append( 
                - Sigma_p_i_m_b[i] @ x[i][1] - Sigma_plus_b_inverse[i] @ Gamma_minus[i]
                )
            y_hat.append(np.concatenate((x_minus[i][1][:i*M*K*I],x[i][1],x_minus[i][1][i*M*K*I:],w[i][1])))
            
            lamb_hat.append(lamb[i][0] + gamma_0 * (Phi[i] @ y_hat[i] - Dp[i] - mu[i][0]))
            y[i].append((1-theta_0) * y[i][0] + theta_0 * y_hat[i])
            
        for i in range(N):
            temp = 0
            for j in range(N):
                temp = temp + envpara.Lap[i][j] * lamb_hat[j]
                    
            mu[i].append( mu[i][0] + beta_0 * temp )
            
            lamb[i].append(lamb_hat[i] + gamma_0 * (mu[i][0]-mu[i][1]) )
            
    
       # perf_alg2[t+1] = funcg()
        x = [sublist[1:] for sublist in x]
        w = [sublist[1:] for sublist in w]
        y = [sublist[1:] for sublist in y]
        x_minus = [sublist[1:] for sublist in x_minus]
        lamb = [sublist[1:] for sublist in lamb]
        mu = [sublist[1:] for sublist in mu]
      
        
        # objective function value
        perf_alg2[t+1] = sum( 0.5 * y[i][0] @ Sigma_0[i] @ y[i][0] + Psi[i] @ y[i][0] for i in range(N))
        
        # absolute value of coupling constraint violation
        abs_con_vio[t+1] = np.linalg.norm(sum( Phi[i] @ y[i][0] - Dp[i] for i in range(N)))
        times[t+1] = time.time() - timepre
        if (t+1)%50 == 0:
            print(t+1, perf_alg2[t+1], abs_con_vio[t+1], time.time()-timepre)
        t = t + 1
    
    
    
    xl = [i for i in range(maxT+1)]
    '''
    plt.figure(figsize=(8, 6))
    plt.plot(xl, perf_alg2)
    plt.axhline(y=config["best_obj"], color='r', linestyle='--')
    plt.xlabel('Iteration steps')
    plt.ylabel('objective function value')
    plt.show()
    # plt.title( f"Value of penalty factor")
    # plt.xlim(0,5001)
    # plt.ylim(0,6800000)
    
    plt.figure(figsize=(8, 6))
    plt.plot(xl, np.log10(np.abs(perf_alg2 - config["best_obj"])))
#    plt.axhline(y=config["best_obj"], color='r', linestyle='--')
    plt.xlabel('Iteration steps')
    plt.ylabel('objective function value')
    plt.show()
    
    plt.figure(figsize=(8, 6))
    plt.plot(xl, abs_con_vio, color='orange')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Iteration steps')
    plt.ylabel('violations')
    plt.show()
    # plt.title( f"Value of penalty factor: rho= 0, sigma= {sigma_0}")
    '''

    np.save(config["savedir"] + "perf.npy", perf_alg2)
    np.save(config["savedir"] + "con.npy", abs_con_vio)
    #np.save(config["savedir"] + "time.npy", times)
    
    return perf_alg2