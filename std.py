import numpy as np
import matplotlib.pyplot as plt
import para
import gurobipy as grb

t = 0
maxT = 2000
envpara = para.Param()
N, M, K, A, B, C,D, P, Q, W, Dbar, L, St, I, path, p, q, c0, c = envpara.out()
suminv = np.zeros(envpara.edgecnt)
Qtuda = np.zeros((N,envpara.edgecnt))
for i in range(N):
    Qtuda[i] = Q[i] @ np.ones(M*K*I)
    suminv = suminv + Qtuda[i]
for i in range(envpara.edgecnt):
    if suminv[i] > 1e-5:
        suminv[i] = 1.0 / suminv[i]
suminv = np.diag(suminv)
Qtudatuda = np.zeros((N,envpara.edgecnt, envpara.edgecnt))
for i in range(N):
    Qtudatuda[i] = np.diag(suminv @ Qtuda[i])
    
# St 为原问题的M，与边数重复
b = np.zeros(M * K)
for i in range(N):
    b = b + D[i]
    
def solve():
    model = grb.Model("std")
    # model.setParam(grb.GRB.Param.OutputFlag, 0)
    model.Params.MIPGap = 0
    X = model.addMVar((N, M * K * I), vtype = grb.GRB.CONTINUOUS, name = "X", ub = Dbar, lb = 0)
    y = model.addMVar((N, K), vtype = grb.GRB.CONTINUOUS, name="y", lb = 0)
    z = model.addMVar((N, M), vtype = grb.GRB.CONTINUOUS, name="z", lb = 0)
    model.update()
    # 
    model.setObjective( c0 * sum(sum(Q[j] @ X[j] for j in range(N)) @ Qtudatuda[i] @ sum(Q[k] @ X[k] for k in range(N)) for i in range(N)) 
                       + sum(c[i] @ Q[i] @ X[i] for i in range(N)) 
                       + sum(p[i] @ y[i] for i in range(N)) 
                       + sum(q[i] @ z[i] for i in range(N)), sense = grb.GRB.MINIMIZE)

    model.addConstr(sum(A[0] @ X[i] for i in range(N)) == b, name = "dual")

    
    for i in range(N):
        model.addConstr(B[0] @ X[i] - y[i] <= St[i])
            
    for i in range(N):
        model.addConstr(C[0] @ X[i] - z[i] <= L[i])
    
    
    model.optimize()
    
    Xval = X.X
    yval = y.X
    zval = z.X
    cnt = 0
    
    mu = [0 for i in range((M)*K)]
    for j in model.getConstrs():
        if j.constrName[0] == 'd':
            mu[cnt] = j.Pi
            cnt += 1
            # print(f'{c.constrName}: Dual variable (Pi) = {c.Pi}')
    
    #print(mu)
    
    return np.array(list(Xval)), np.array(list(yval)), np.array(list(zval)), np.array(mu)

def funcg(X,y,z):
    f = c0 * sum(sum(Q[j] @ X[j] for j in range(N)) @ Qtudatuda[i] @ sum(Q[k] @ X[k] for k in range(N)) for i in range(N)) + sum(c[i] @ Q[i] @ X[i] for i in range(N)) + sum(p[i] @ y[i] for i in range(N)) + sum(q[i] @ z[i] for i in range(N))
    return f

x, y, z, mu = solve()
x = x.reshape(N, M * K * I)
y = y.reshape(N, K)
z = z.reshape(N, M)
print(x)
print(funcg(x,y,z))
print(mu)
