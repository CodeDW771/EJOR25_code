import numpy as np
import gurobipy as gp
from gurobipy import GRB


def std(envpara):
    """Use gurobi solve centralized optimization problem.

    Args:
        envpara: Instance of Param
    Return:
        f_value: optimal objective function value
    """
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta= envpara.out()
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    #theta=np.concatenate(theta)

    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))

    x, d_lambda, d_mu = solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b)
    X = x.reshape(N, M * K * I)
    f_value = funcg_P(N, X, Q, c0, c)

    return f_value


    

def solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b):
    """Solve centralized primal problem."""
    model = gp.Model("std")
    # model.setParam(grb.GRB.Param.OutputFlag, 0)
    model.Params.MIPGap = 0
    X = model.addMVar((N, M * K * I), vtype = gp.GRB.CONTINUOUS, name = "X", lb = 0)
    #w = model.addMVar( M*K , vtype = gp.GRB.CONTINUOUS, name="w", lb = 0)
    model.update()
    # 
    model.setObjective( c0 * sum(Q[i] @ X[i] for i in range(N)) @ sum(Q[i] @ X[i] for i in range(N))
                       + sum(c[i] @ Q[i] @ X[i] for i in range(N)),
                       sense = gp.GRB.MINIMIZE)

    mu0 = model.addConstr( (A @ sum(X[i] for i in range(N))) == b, name = "dual_sum")

    mu=[[] for i in range(N)]
    for i in range(N):
        mu[i]=model.addConstr(Btuda @ X[i] <= Mtuda[i], name = f"dual_{i}" )
            

    model.optimize()
    
    if model.status == GRB.OPTIMAL:
        print("Success.")
    elif model.status == GRB.INFEASIBLE:
        print("Unfeasible.")
    elif model.status == GRB.UNBOUNDED:
        print("Unbounded.")
    else:
        print("Fail. Status: ", model.status)
    
    
    Xval = X.X
    
    dual_lambda=mu0.Pi
    dual_mu=[]
    for i in range(N):
        dual_mu.append(mu[i].Pi)
    
    return np.array(list(Xval)), np.array(list(dual_lambda)), np.array(dual_mu)



def funcg_P(N, X, Q, c0, c):
    """Calculate value of objective function."""
    f1 = c0 * sum(Q[j] @ X[j] for j in range(N)) @ sum(Q[k] @ X[k] for k in range(N))
    f2 = sum(c[i] @ Q[i] @ X[i] for i in range(N))
    print(f1, f2)
    return f1 + f2

