import numpy as np
import matplotlib.pyplot as plt
import gurobipy as grb
from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack
import time
import itertools
# import pandas as pd

def alg(envpara, config):
    """Run comparison algorithm Tracking-ADMM, paper (Tracking-ADMM for distributed constraint-coupled optimization)."""
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
    Atuda = np.zeros((N,M*K,N*M*K*I+M*K))
    Atuda_s = []
    Dp = [] 
    Theta = np.zeros((envpara.edgecnt,N*M*K*I))
    Mtuda = []
    
    for i in range(N):
        Qtudatuda.append(coo_matrix(np.diag(suminv @ Qtuda[i])))   
        H.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(N*M*K*I)).T))   
        HTH.append(coo_matrix(H[i].T @ H[i]))
        G.append(coo_matrix(np.kron(envpara.Lap[i], np.identity(M*K)).T))
        GTG.append(coo_matrix(G[i].T @ G[i]))
        Atuda[i][:,i*M*K*I:(i+1)*M*K*I] = A.toarray()  
        Atuda[i][:,N*M*K*I: N*M*K*I+M*K] = np.identity((M*K))*(1/N)
        Atuda_s.append(coo_matrix(Atuda[i]))
        Dp.append(D[i]) # (D, 0) 
        Theta[:,i*M*K*I:(i+1)*M*K*I] = Q[i]
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    Theta = coo_matrix(Theta)   
    
    
    Sigma = []
    
    for i in range(N):
        Sigma.append(coo_matrix(2 * c0 * Theta.T @ Qtudatuda[i] @ Theta)) 
    
    
    # Preparation complete.
    
    # Step 0 parameters 
    
    sigma_0=config["sigma"]   # penalty coefficient of the augmented part
    
    
    # Step 1 generate the required matrix
    
    Sigma_b = []
    Sigma_core_b = []
    Sigma_minus_b = []
    Sigma_plus_b = [] 
    Sigma_plus_b_inverse = []
    
    for i in range(N):
        Sigma_b.append(coo_matrix(Sigma[i] + sigma_0 * HTH[i] )) # Sigma_i matrix 
        row0=slice(i*M*K*I,(i+1)*M*K*I)
        row=list(range(i*M*K*I,(i+1)*M*K*I))
        Sigma_core_b.append(coo_matrix(Sigma_b[i].toarray()[row0,row0]))
        Sigma_minus_b.append(coo_matrix(np.delete(Sigma_b[i].toarray()[row0,:],row,axis=1)))
        Sigma_plus_b.append(coo_matrix(np.delete(np.delete(Sigma_b[i].toarray(),row,axis=0),row,axis=1)))
        temp=coo_matrix(np.linalg.inv(Sigma_plus_b[i].toarray()))  
        Sigma_plus_b_inverse.append(coo_matrix(0.5*(temp.T+temp))) # floating-point precision may introduce slight errors, hence forced symmetry
    
    
    Sigma_pre = sigma_0 *coo_matrix(vstack((hstack((ATA, 1/N * AT)),hstack((1/N * A , 1/(N*N)* coo_matrix(np.eye((M*K))))))))
        
    Sigma_p_i_m_b = []
    Sigma_c_m_p_m_b = []
    Sigma_tensor_p = np.zeros((N, M*K+M*K*I, M*K+M*K*I))
    Sigma_tensor = []
    for i in range(N):
        Sigma_p_i_m_b.append(coo_matrix(Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T))
        temp = Sigma_minus_b[i] @ Sigma_plus_b_inverse[i] @ Sigma_minus_b[i].T
        Sigma_c_m_p_m_b.append(coo_matrix(Sigma_core_b[i] - 0.5*(temp+temp.T)))
        Sigma_tensor_p[i][0:M*K*I,0:M*K*I] = Sigma_c_m_p_m_b[i].toarray()
        Sigma_tensor_p[i][M*K*I : M*K*I+M*K , M*K*I : M*K*I+M*K] = sigma_0 * (GTG[i].toarray())
        Sigma_tensor.append(coo_matrix(Sigma_tensor_p[i]+Sigma_pre))
    
    
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
        model.addConstr( X[M*K*I : M*K*I + M*K] == 0 )
    
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
    lamb1 = [[] for i in range(N)]
    lamb2 = [[] for i in range(N)]
    lamb3 = [[] for i in range(N)]
    d1 = [[] for i in range(N)]
    d2 = [[] for i in range(N)]
    d3 = [[] for i in range(N)]
    
    perf_alg2 = [0 for i in range(maxT+1)]
    abs_con_vio = [0 for i in range(maxT+1)]
    times = [0 for i in range(maxT+1)]
    
    #-----------------------------------------------------------------------------
    #initial value
    
    for i in range(N):
        x[i].append(np.zeros(M*K*I))
        x_minus[i].append(np.zeros((N-1)*M*K*I))
        w[i].append(np.zeros(M*K))
        y[i].append(np.zeros(N*M*K*I))   # y_i^0
        d1[i].append(A @ x[i][0] + 1/N * w[i][0] - Dp[i])  # d_i^0
        d2[i].append( H[i] @ y[i][0])
        d3[i].append( G[i] @ w[i][0])
        lamb1[i].append(np.zeros(M*K)) 
        lamb2[i].append(np.zeros(N*N*M*K*I)) 
        lamb3[i].append(np.zeros(N*M*K)) 
    
    abs_con_vio[0] = np.linalg.norm(D[0]*N)
    
    timepre = time.time()
    print("pre complete", timepre-begin)
    
    t=0
    times[0] = time.time()-timepre
    print(t, perf_alg2[t], abs_con_vio[t], time.time()-timepre)
    
    while stopcon(t):
        
        l1= []
        l2= []
        l3= []
        delta1= []
        delta2= []
        delta3= []
        Gamma = []
        Gamma_core = []
        Gamma_minus = []
        Delta_x = []
        Delta_w = []
        Delta_tensor = []
        
        for i in range(N):
            l1.append(np.zeros(M*K))
            l2.append(np.zeros(N*N*M*K*I))
            l3.append(np.zeros(N*M*K))
            delta1.append(np.zeros(M*K))
            delta2.append(np.zeros(N*N*M*K*I))
            delta3.append(np.zeros(N*M*K))
            
            for j in range(N):
                l1[i] = l1[i] + W[i][j] * lamb1[j][0]
                l2[i] = l2[i] + W[i][j] * lamb2[j][0]
                l3[i] = l3[i] + W[i][j] * lamb3[j][0]
                delta1[i] = delta1[i] + W[i][j] * d1[j][0]
                delta2[i] = delta2[i] + W[i][j] * d2[j][0]
                delta3[i] = delta3[i] + W[i][j] * d3[j][0]
            # for j in range(N):
            #      l[i] = l[i] + 1/N * lamb[j][0]
            #      delta[i] = delta[i] + 1/N * d[j][0]
            
            Gamma.append( H[i].T @ (lamb2[i][0] + sigma_0 * delta2[i] - sigma_0 * H[i] @ y[i][0])  )
            Gamma_core.append(Gamma[i][i*M*K*I:(i+1)*M*K*I])
            Gamma_minus.append(np.delete(Gamma[i],slice(i*M*K*I,(i+1)*M*K*I)))
            
            
            Delta_x.append(
                Q[i].T @ c[i] + 
                A.T @ lamb1[i][0] + 
                sigma_0 * A.T @ (delta1[i]- A @ x[i][0]- w[i][0]/N) +
                Gamma_core[i] - Sigma_p_i_m_b[i].T @ Gamma_minus[i]
                )
            
            Delta_w.append(
                theta/N + lamb1[i][0]/N + G[i].T @ lamb3[i][0] +
                sigma_0/N * (delta1[i] - A @ x[i][0] - w[i][0]/N) +
                sigma_0 * G[i].T @ ( delta3[i] - G[i] @ w[i][0])
                )
            
            Delta_tensor.append(np.concatenate((Delta_x[i], Delta_w[i])))
            
            
            xsolu = solve(i, Delta_tensor[i])
            
            x[i].append(xsolu[0 : M*K*I])
            w[i].append(xsolu[M*K*I : M*K*I + M*K])
            
            #x_minus[i].append(np.maximum(0,-AA_inverserBBT[i] @ x[i][1] - AA_inverse[i] @ CC[i]))
            x_minus[i].append( 
                - Sigma_p_i_m_b[i] @ x[i][1] - Sigma_plus_b_inverse[i] @ Gamma_minus[i]
                )
            y[i].append(np.concatenate((x_minus[i][1][:i*M*K*I],x[i][1],x_minus[i][1][i*M*K*I:])))
            
            
            d1[i].append(delta1[i] + A @ (x[i][1] - x[i][0]) + 1/N * (w[i][1] - w[i][0]))
            d2[i].append(delta2[i] + H[i] @ (y[i][1] - y[i][0]) )
            d3[i].append(delta3[i] + G[i] @ (w[i][1] - w[i][0]) )
            
            lamb1[i].append(l1[i] + sigma_0 * d1[i][1])
            lamb2[i].append(l2[i] + sigma_0 * d2[i][1])
            lamb3[i].append(l3[i] + sigma_0 * d3[i][1])
            
    
       # perf_alg2[t+1] = funcg()
        d1 = [sublist[1:] for sublist in d1]
        d2 = [sublist[1:] for sublist in d2]
        d3 = [sublist[1:] for sublist in d3]
        lamb1 = [sublist[1:] for sublist in lamb1]
        lamb2 = [sublist[1:] for sublist in lamb2]
        lamb3 = [sublist[1:] for sublist in lamb3]
        x = [sublist[1:] for sublist in x]
        w = [sublist[1:] for sublist in w]
        y = [sublist[1:] for sublist in y]
        x_minus = [sublist[1:] for sublist in x_minus]
    
       
        x_real=[]
        w_real=[]
        for i in range(N):
            x_real.append(x[i][0])
            w_real.append(w[i][0])
    
        
        # objective function value
        x_real_long = list(itertools.chain.from_iterable(x_real))
        perf_alg2[t+1] = c0 * (Theta @ x_real_long).T @ (Theta @ x_real_long) + sum(c[i] @ Q[i] @ x_real[i] for i in range(N)) + sum( (1/N) * theta @ w_real[i] for i in range(N))
        
        # absolute value of coupling constraint violation
        abs_con_vio[t+1] = np.linalg.norm(sum( A @ x_real[i] + w_real[i]/N for i in range(N)) - (D[0]*N) )
        times[t+1] = time.time()-timepre
        if (t+1)%10 == 0:
            print(t+1, perf_alg2[t+1], abs_con_vio[t+1], time.time()-timepre)
        t = t + 1
    
    
    
    xl = [i for i in range(maxT+1)]
    
    
    np.save(config["savedir"] + "perf.npy", perf_alg2)
    np.save(config["savedir"] + "con.npy", abs_con_vio)
    #np.save(config["savedir"] + "time.npy", times)
    
    return perf_alg2