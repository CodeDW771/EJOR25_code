import numpy as np
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
from tabulate import tabulate
import sys



def task1(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
        
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    index_b = [0.4, 0.6, 0.8, 1, 1.2]
    index_m = [0.8, 0.9, 1, 1.1, 1.2]
    index_c = [0.6, 0.8, 1, 1.2, 1.4]
    bb = []
    MM = []
    
    results1 = {
        'd_value': [],
        'Ind_SP': [],
        'Ind_VCG': [],
        'SP_VCG': [],
        'Pct.': []
    }
    
    results2 = {
        'm_value': [],
        'Ind_SP': [],
        'Ind_VCG': [],
        'SP_VCG': [],
        'Pct.': []
    }
    
    results3 = {
        'c_value': [],
        'Ind_SP': [],
        'Ind_VCG': [],
        'SP_VCG': [],
        'Pct.': []
    }
    
    for i in range(len(index_b)):     
        bb = index_b[i] * b 
        MM = index_m[i] * np.array(Mtuda) 
        new_c = [
            [x * index_c[i] for x in c[0]],  # list_1
            *c[1:]
            ]
        
        
        SP_value1, VCG_value1, SP_VCG_sum1, Pct1 = task1_1(N, M, K, I, A, Btuda, Mtuda, bb, Q, c0, c)
        SP_value2, VCG_value2, SP_VCG_sum2, Pct2 = task1_1(N, M, K, I, A, Btuda, MM, b, Q, c0, c)
        SP_value3, VCG_value3, SP_VCG_sum3, Pct3 = task1_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, new_c)
    
        results1['d_value'].append(index_b[i])
        results1['Ind_SP'].append(SP_value1)
        results1['Ind_VCG'].append(VCG_value1)
        results1['SP_VCG'].append(SP_VCG_sum1)
        results1['Pct.'].append(Pct1)
        
        results2['m_value'].append(index_m[i])
        results2['Ind_SP'].append(SP_value2)
        results2['Ind_VCG'].append(VCG_value2)
        results2['SP_VCG'].append(SP_VCG_sum2)
        results2['Pct.'].append(Pct2)
        
        results3['c_value'].append(index_c[i])
        results3['Ind_SP'].append(SP_value3)
        results3['Ind_VCG'].append(VCG_value3)
        results3['SP_VCG'].append(SP_VCG_sum3)
        results3['Pct.'].append(Pct3)
        
    df1 = pd.DataFrame(results1)
    df2 = pd.DataFrame(results2)
    df3 = pd.DataFrame(results3)
    
    table_data1 = []
    table_data2 = []
    table_data3 = []
    
    for i, row in df1.iterrows():
        l_str = ', '.join(f'{x:.4f}' for x in row['Ind_SP'])
        v_str = ', '.join(f'{x:.4f}' for x in row['Ind_VCG'])
            
        table_data1.append([
            row['d_value'],
            f"[{l_str}]",
            f"[{v_str}]",
            row['SP_VCG'],
            row['Pct.']
        ])
    
    for i, row in df2.iterrows():
        l_str = ', '.join(f'{x:.4f}' for x in row['Ind_SP'])
        v_str = ', '.join(f'{x:.4f}' for x in row['Ind_VCG'])
            
        table_data2.append([
            row['m_value'],
            f"[{l_str}]",
            f"[{v_str}]",
            row['SP_VCG'],
            row['Pct.']
        ])
        
    for i, row in df3.iterrows():
        l_str = ', '.join(f'{x:.4f}' for x in row['Ind_SP'])
        v_str = ', '.join(f'{x:.4f}' for x in row['Ind_VCG'])
            
        table_data3.append([
            row['c_value'],
            f"[{l_str}]",
            f"[{v_str}]",
            row['SP_VCG'],
            row['Pct.']
        ])

    headers1 = ['Demand', 'SP_value', 'VCG_value', 'SP-VCG', 'Pct.']
    headers2 = ['Capcity', 'SP_value', 'VCG_value', 'SP-VCG', 'Pct.']
    headers3 = ['Parameter', 'SP_value', 'VCG_value', 'SP-VCG', 'Pct.']

    print(tabulate(table_data1, headers=headers1, tablefmt='grid'))
    print(tabulate(table_data2, headers=headers2, tablefmt='grid'))
    print(tabulate(table_data3, headers=headers3, tablefmt='grid'))
    
    return  results1, results2, results3



def task2_N0(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    unit = 0.05
    p = range(-100,101)
    
    check_c = p[0] * unit + np.array(c[0])
    
    if all(num >= 0 for num in check_c):
        print("T")
    else:
        print("F")
        sys.exit()
    
    
    result = {f'N{i}': [] for i in range(N)}
    result['Delta'] = []
    
    for i in range(len(p)):
        new_c = [  [x + p[i] * unit for x in c[0]],  
            c[1], c[2], c[3]
        ]
        
        Ind_utility = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, new_c)
        
        result['Delta'].append( np.round(p[i] * unit, 2) )
        
        for j in range(N):
            result[f'N{j}'].append(Ind_utility[j])
    
    headers = ['Delta'] + list(result.keys())  # ['Cycle', 'N0', 'N1', 'N2']
    table_data = []
     
    for i in range(len(p)): 
        row = [
        result['Delta'][i],
        *map(lambda j: result[f'N{j}'][i], range(N))
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return result


def task2_N1(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    unit = 0.05
    p = range(-100,101)
    
    check_c = p[0] * unit + np.array(c[1])
    
    if all(num >= 0 for num in check_c):
        print("T")
    else:
        print("F")
        sys.exit()
    
    
    result = {f'N{i}': [] for i in range(N)}
    result['Delta'] = []
    
    for i in range(len(p)):
        new_c = [ c[0],
            [x + p[i] * unit for x in c[1]],
            c[2], c[3]
        ]
        
        Ind_utility = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, new_c)
        
        result['Delta'].append( np.round(p[i] * unit, 2) )
        
        for j in range(N):
            result[f'N{j}'].append(Ind_utility[j])
    
    headers = ['Delta'] + list(result.keys())  # ['Cycle', 'N0', 'N1', 'N2']
    table_data = []
     
    for i in range(len(p)):
        row = [
        result['Delta'][i],
        *map(lambda j: result[f'N{j}'][i], range(N))
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return result
    

def task2_N2(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    unit = 0.05
    p = range(-100,101)
    
    check_c = p[0] * unit + np.array(c[2])
    
    if all(num >= 0 for num in check_c):
        print("T")
    else:
        print("F")
        sys.exit()
    
    
    result = {f'N{i}': [] for i in range(N)}
    result['Delta'] = []
    
    for i in range(len(p)):
        new_c = [ c[0], c[1], 
            [x + p[i] * unit for x in c[2]], # list_1
             c[3]
        ]
        
        Ind_utility = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, new_c)
        
        result['Delta'].append( np.round(p[i] * unit, 2) )
        
        for j in range(N):
            result[f'N{j}'].append(Ind_utility[j])
    
    headers = ['Delta'] + list(result.keys())  # ['Cycle', 'N0', 'N1', 'N2']
    table_data = []
     
    for i in range(len(p)):
        row = [
        result['Delta'][i],
        *map(lambda j: result[f'N{j}'][i], range(N))
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return result


def task2_N3(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
        
    unit = 0.05
    p = range(-100,101)
    
    check_c = p[0] * unit + np.array(c[3])
    
    if all(num >= 0 for num in check_c):
        print("T")
    else:
        print("F")
        sys.exit()
    
    
    result = {f'N{i}': [] for i in range(N)}
    result['Delta'] = []
    
    for i in range(len(p)):
        new_c = [ c[0], c[1], c[2],
            [x + p[i] * unit for x in c[3]]
        ]
        
        Ind_utility = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, new_c)
        
        result['Delta'].append( np.round(p[i] * unit, 2) )
        
        for j in range(N):
            result[f'N{j}'].append(Ind_utility[j])
    
    headers = ['Delta'] + list(result.keys())
    table_data = []
     
    for i in range(len(p)):
        row = [
        result['Delta'][i],
        *map(lambda j: result[f'N{j}'][i], range(N))
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return result



def task3(envpara):
    
    N, M, K, A, B, C, D, Q, W, L, St, I, c0, c, theta = envpara.out()
    
    b = np.zeros(M * K)
    for i in range(N):
        b = b + D[i]
    
    Btuda = np.vstack((B,C))
    Mtuda = []
    for i in range(N):
        Mtuda.append(np.concatenate((St[i],L[i])))
    
    
    optimal_base = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, c)
    
    case_num = 30
    p = 0.9
    case_c = generate_case(c, p, case_num, seed= 16)
    
    result = {f'N{i}': [] for i in range(N)}
    result['Case'] = []
    
    for i in range(case_num):
        
        Ind_utility = task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, case_c[i])
        
        result['Case'].append( f'C{i}' )
        
        for j in range(N):
            result[f'N{j}'].append(Ind_utility[j])
    
    headers = ['Case'] + list(result.keys())
    table_data = []
     
    for i in range(case_num):
        row = [
        result['Case'][i],
        *map(lambda j: result[f'N{j}'][i], range(N))
        ]
        table_data.append(row)
    
    opt_row = ['Optimal']
    
    for i in range(N):
        opt_row.append(optimal_base[i])
    
    table_data.append(opt_row)
    
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    return optimal_base, result


def task1_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c):
     
    x, d_lambda, d_mu = solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b)
    X = x.reshape(N, M * K * I)
    Ind_value = Ind_func_cost(N, X, Q, c0, c)
    SP_value = price_P(X, d_lambda, N, Q, A, c0)
    total_cost = funcg_P(N, X, Q, c0, c)
    VCG_f = []
    VCG_value = []
    
    for i in range(N):
        x2, d_lambda2 = solve_SP(i, N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b)
        X2 = x2.reshape(N, M * K * I)
        VCG_f.append (funcg_P(N, X2, Q, c0, c))
        VCG_value.append( VCG_f[i] - ( total_cost - Ind_value[i] ))
    
    SP_VCG_sum = sum(SP_value) - sum(VCG_value)
    Pct = abs(SP_VCG_sum)/max(sum(SP_value),sum(VCG_value))
    
    return np.round(SP_value,2), np.round(VCG_value,2), np.round(SP_VCG_sum,2), np.round(Pct,2)



def task2_1(N, M, K, I, A, Btuda, Mtuda, b, Q, c0, c, new_c):
     
    x, d_lambda, d_mu = solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, new_c, b)
    X = x.reshape(N, M * K * I)
    Ind_cost = Ind_func_cost(N, X, Q, c0, c)
    SP_value = price_P(X, d_lambda, N, Q, A, c0)
    
    Ind_utility = np.array(SP_value) - np.array(Ind_cost)
    
    return np.round(Ind_utility,2)



def generate_case(vectors, p, K, seed):
    """Add perturbations to a list of vectors and generate K perturbed results
    
    Args:
        vectors: Original list of vectors (contains 4 M-dimensional vectors)
        p: Perturbation ratio (0 <= p <= 1)
        K: Number of perturbations
    
    Returns:
        perturbed_results: A list containing K perturbed results, where each result is a list of 4 perturbed vectors
    """
    rng = np.random.RandomState(seed)
    
    M = vectors[0].shape[0]
    num_vectors = len(vectors)
    perturb = np.around(np.arange(-2,2, 0.05), decimals=2).tolist()
    perturbed_results = []
    
    for _ in range(K):  # K times perturbation
        perturbed_list = []
        
        for i in range(num_vectors):
            orig_vec = vectors[i].copy()
            n_perturb = max(1, int(round(p * M)))  # number of perturbation
            
            perturb_indices = rng.choice(M, size=n_perturb, replace=False)
            
            perturbations = rng.choice(perturb, size=n_perturb)
            
            orig_vec[perturb_indices] += perturbations
            perturbed_list.append(orig_vec)
        
        perturbed_results.append(perturbed_list)
    
    return perturbed_results






# def text2(envpara,index):
#     N, M, K, A, B, C, D, P, Q, W, L, St, I, path, c0, c, theta = envpara.out()
    
#     b = np.zeros(M * K)
#     for i in range(N):
#         b = b + D[i]
    
#     theta=np.concatenate(theta)
    
#     Btuda = np.vstack((B,C))
#     Mtuda = []
#     for i in range(N):
#         Mtuda.append(np.concatenate((St[i],L[i])))
    
#     x, d_lambda, d_mu = solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b)
#     X = x.reshape(N, M * K * I)
#     x2, d_lambda2 = solve_SP(index, N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b)
#     X2 = x2.reshape(N, M * K * I)
    
#     vec=[]
#     for i in range(N):
#         if i == index:
#             vec.append("*")
#         elif np.all(np.array(X2[i]) >= np.array(X[i])):
#             vec.append("T")
#         else:
#             vec.append("F")    
    
#     return X,X2,vec

#------------------------------------------------------------------------

def funcg_P(N, X, Q, c0, c):
    """Calculate objective function value."""
    f = c0 * sum(Q[j] @ X[j] for j in range(N)) @ sum(Q[k] @ X[k] for k in range(N))  + sum(c[i] @ Q[i] @ X[i] for i in range(N)) 
    return f


def Ind_func_cost(N, X, Q, c0, c):
    v= []
    temp = sum(Q[k] @ X[k] for k in range(N))
    for i in range(N):
        v.append( c0 * temp @ (Q[i] @ X[i]) + c[i] @ Q[i] @ X[i])
    
    return v



def price_P(X, d_lambda, N, Q, A, c0):
    """Calculate the price according to variable value and dual variable value."""
    v = []
    u = []
    temp = sum(Q[k] @ X[k] for k in range(N))
    for i in range(N):
        v.append(A.T @ d_lambda - c0 * Q[i].T @ (temp - Q[i] @ X[i]))
        u.append(v[i] @ X[i])
    return u
    

def solve_P(N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b):
    """Solve problems for N individuals, get the optimal value of lambda, and get the optimal price under the VCG mechanism."""
    model = gp.Model("std")
    # model.setParam(grb.GRB.Param.OutputFlag, 0)
    model.Params.MIPGap = 0
    X = model.addMVar((N, M * K * I), vtype = gp.GRB.CONTINUOUS, name = "X", lb = 0)
    #w = model.addMVar( M*K , vtype = gp.GRB.CONTINUOUS, name="w", lb = 0)
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
        return np.array([]), np.array([]), np.array([])
    
    
    Xval = X.X
    #wval = w.X
    
    dual_lambda=mu0.Pi
    dual_mu=[]
    for i in range(N):
        dual_mu.append(mu[i].Pi)
    
    return np.array(list(Xval)), np.array(list(dual_lambda)), np.array(dual_mu)



def solve_SP(i, N, M, K, I, A, Btuda, Mtuda, Q, c0, c, b):
    """Solve problems for N-1 individuals."""
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
    
    model.addConstr( X[i] == np.zeros( M*K*I ), name = "add_con_i")

    for i in range(N):
        model.addConstr(Btuda @ X[i] <= Mtuda[i], name = f"dual_{i}" )
            

    model.optimize()
    
    if model.status != GRB.OPTIMAL:
        print("Fail. Status: ", model.status)
        return np.array([]), np.array([])
    
    
    Xval = X.X
   
    dual_lambda=mu0.Pi
    
    return np.array(list(Xval)), np.array(list(dual_lambda))





