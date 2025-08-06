import numpy as np
import matplotlib.pyplot as plt
#import para
import gurobipy as gp
from gurobipy import GRB

from scipy.sparse import coo_matrix, csr_matrix, hstack, vstack


def feasible_check(envpara):
    """Check whether the example is feasible, that is, whether the problem is feasible when one individual is absent.

    Args:
        envpara: Instance of Param.
    Return:
        True/False: Whether this envpara is feasible.
    """
    # envpara = para.Param()
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    suminv = np.zeros(envpara.edgecnt)
    
        
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    theta=np.concatenate(theta)
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    #--------------------------------------------------------------------------------
    
    def solve_P():
        """Solve problems for N individuals, get the optimal value of lambda, and get the optimal price under the VCG mechanism."""
        model = gp.Model("std")
        model.setParam(gp.GRB.Param.OutputFlag, 0)
        model.Params.MIPGap = 0
        X = model.addMVar((N, M * K * I), vtype = gp.GRB.CONTINUOUS, name = "X", lb = 0)
        # w = model.addMVar( M*K , vtype = gp.GRB.CONTINUOUS, name="w", lb = 0)
        model.update()

        model.setObjective( c0 * sum(Q[i] @ X[i] for i in range(N)) @ sum(Q[i] @ X[i] for i in range(N))
                           + sum(c[i] @ Q[i] @ X[i] for i in range(N)),
                           sense = gp.GRB.MINIMIZE)
    
        mu0 = model.addConstr( (A @ sum(X[i] for i in range(N))) == b, name = "dual_sum")
    
        mu=[[] for i in range(N)]
        for i in range(N):
            mu[i]=model.addConstr(Btuda @ X[i] <= Mtuda[i], name = f"dual_{i}" )
                
    
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            print("Fail. Status: ", model.status)
            return False, np.array([]), np.array([]), np.array([])
        
        
        Xval = X.X
        # wval = w.X
        
        dual_lambda=mu0.Pi
        dual_mu=[]
        for i in range(N):
            dual_mu.append(mu[i].Pi)
        
        return True, np.array(list(Xval)), np.array(list(dual_lambda)), np.array(dual_mu)
    
    def funcg_P(X):
        """Calculate objective function value."""
        f = c0 * sum(Q[j] @ X[j] for j in range(N)) @ sum(Q[k] @ X[k] for k in range(N))  + sum(c[i] @ Q[i] @ X[i] for i in range(N)) 
        return f
    
    def price_P(X,d_lambda):
        """Calculate the price according to variable value and dual variable value."""
        v = []
        u = []
        temp = sum(Q[k] @ X[k] for k in range(N))
        for i in range(N):
            v.append(A.T @ d_lambda - c0 * Q[i].T @ (temp - Q[i] @ X[i]))
            u.append(v[i] @ X[i])
        return u
    
    state, x, d_lambda, d_mu = solve_P()
    if state == False:
        return False
    
    X = x.reshape(N, M * K * I)
    
    N_f_value = funcg_P(X)
    
    LMP_value = price_P(X,d_lambda)
    LMP_Value = sum(LMP_value)

    
    
    def solve_SP(i):
        """Solve problems for N-1 individuals.

        Args:
            i: Individual i is absent.
        Return:
            True/False: Whether this subproblem has optimal solution
            Xval: The variable value of subproblem
            duam_lambda: The dual variable value of subproblem
        """
        model = gp.Model("std")
        model.setParam(gp.GRB.Param.OutputFlag, 0)
        model.Params.MIPGap = 0
        X = model.addMVar((N, M * K * I), vtype = gp.GRB.CONTINUOUS, name = "X", lb = 0)
        #w = model.addMVar( M*K , vtype = gp.GRB.CONTINUOUS, name="w", lb = 0)
        model.update()
        
        model.setObjective( c0 * sum(Q[i] @ X[i] for i in range(N)) @ sum(Q[i] @ X[i] for i in range(N))
                           + sum(c[i] @ Q[i] @ X[i] for i in range(N)),
                           sense = gp.GRB.MINIMIZE)
    
        mu0 = model.addConstr( (A @ sum(X[i] for i in range(N))) == b, name = "dual_sum")
        
        model.addConstr( X[i] == np.zeros( M*K*I ), name = "add_con_i")
    
        for i in range(N):
            model.addConstr(Btuda @ X[i] <= Mtuda[i], name = f"dual_{i}" )
                
    
        model.optimize()
        
        if model.status != GRB.OPTIMAL:
            print("Fail. Status: ", model.status)
            return False, np.array([]), np.array([])
        
        
        Xval = X.X
       
        dual_lambda=mu0.Pi
        
        return True, np.array(list(Xval)), np.array(list(dual_lambda))
    
    
    VCG_lambda = []
    VCG_f_value = []
    
    for i in range(N):
        state, x, d_lambda = solve_SP(i)
        if state == False:
            return False
        xx = x.reshape(N, M * K * I)
        VCG_lambda.append(d_lambda)
        VCG_f_value.append ( funcg_P(xx))
    
    
    VCG_Value = sum(VCG_f_value) - (N-1)* N_f_value
    
    return True



