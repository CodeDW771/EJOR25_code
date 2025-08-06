import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
import pandas as pd
import random
from traffic import traffic_graph
from communicationnetwork import communication_graph


class Param():
    """Generate all parameters and network based on some input information.

    You can get a traffic network and communication network by setting supplier number, demander number and path number.
    And get all other parameters used in problem by setting lower bound, upper bound and some coefficients.
    We set 6 examples for numerical experiment. We can generate a set of parameters by input example = 1~6

    Attributes:
        N: # of supply nodes.
        M: # of demander nodes.
        K: # of kinds of goods.
        DN: Demand.
        c0: Coefficient of congestion cost.
        D: Demand after equal distribution to supply nodes.
        L: Transmission limit.
        St: Inventory.
        midpoint: # of nodes in traffic network except supplier and demander.
        allpoint: # of total nodes.
        point_pos: Position of all points.
        cost: Distance between two points.
        limit: Graph size, influence the max position of node.
        communication: Communication graph from communication_graph.
        traffic: Traffic graph and K paths from each supplier node to each demander node.
        DG: Communication graph.
        linknum: # of edges in communication graph.
        edgecnt: # of edges in traffic graph.
        pathnum: # of paths from each supplier to each demander.
        path: K shortest path from each supplier node to each demander node.
        c: Transportation cost of unit goods for each edge.
        A: Demand constraint matrix.
        B: Single inbound limit constraint matrix.
        C: Inventory constraint matrix.
        old_linkfrom: The begin node of this edge in communication graph.
        old_linkto: The end node of this edge in communication graph.
        Lap: Laplacian matrix of communication graph.
        W: Weight matrix.
        P: Whether edge t is in kth path from i to j.
        Q: The transpose of P.
        examplenum: Example ID.

    """
    def __init__(self, seed=0, N=20, M=5, K=10, Dlb=50, Dub=150, Llb=375, Lub=975, Stlb=200, Stub=350, midpoint=None, limit=None, communication_edge=50, traffic_edge=250, randomrate_communication = 0, pathnum = 3, c0 = 0.002, example = 0):
        """Initialize parameters, generate communication network and traffic network.

        Args:
            seed: Random seed.
            N: # of supplier nodes.
            M: # of demander nodes.
            K: # of kinds of goods.
            Dlb, Dub: Lower and upper bound of demand, then generate demand uniformly from [Dlb, Dub].
            Llb, Lub: Lower and upper bound of transmission, then generate transmission bound uniformly from [Llb, Lub].
            Stlb, Stub: Lower and upper bound of inventory, then generate inventory uniformly from [Stlb, Stub].
            midpoint: # of nodes in traffic network except supplier and demander.
            limit: Graph size, influence the max position of node.
            communication_edge: # of edges in communication network.
            traffic_edge: # of edges in traffic network.
            randomrate_communication: Randomness in topology in communication network.
            pathnum: I, # of paths from each supplier to each demander.
            c0: Coefficient of congestion cost.
            example: Example ID, 1 is for machanism comparison, 1/2/3 are for algorithm comparison, 
                4/5/6 are for acceleration comparison, representing small-scale, midium-scale, large-scale scenarios respectively.
        """
        self.examplenum = example
        with open("example.json") as json_file:
            examplelist = json.load(json_file)
        if self.examplenum in examplelist['exampleID']:
            N = examplelist['N'][self.examplenum - 1]
            M = examplelist['M'][self.examplenum - 1]
            K = examplelist['K'][self.examplenum - 1]
            pathnum = examplelist['pathnum'][self.examplenum - 1]
            c0 = examplelist['c0'][self.examplenum - 1]
            communication_edge = examplelist['communication_edge'][self.examplenum - 1]
            seed = examplelist['seed'][self.examplenum - 1]
            Dlb = examplelist['Dlb'][self.examplenum - 1]
            Dub = examplelist['Dub'][self.examplenum - 1]
            Llb = examplelist['Llb'][self.examplenum - 1]
            Lub = examplelist['Lub'][self.examplenum - 1]
            Stlb = examplelist['Stlb'][self.examplenum - 1]
            Stub = examplelist['Stub'][self.examplenum - 1]
        
        self.communication_edge = communication_edge
        
        random.seed(seed)
        np.random.seed(seed)
        self.Generate_paras(seed, N, M, K, Dlb, Dub, Llb, Lub, Stlb, Stub, c0)
        self.Generate_point(seed,midpoint,limit,pathnum, traffic_edge)
        #print("communication begin")
        self.communication = communication_graph(point_pos = self.point_pos, cost = self.cost, N = self.N, M = self.M, num_edge = self.communication_edge, randomrate = randomrate_communication,seed=seed)
        self.DG = self.communication.DG
        self.linknum = self.communication_edge * 2
        #print(self.communication.edgecnt)
        self.traffic = traffic_graph(point_pos = self.point_pos, cost = self.cost, N = self.N, M = self.M, num_edge = self.traffic_edge, pathnum=pathnum, midpoint = self.allpoint - self.N - self.M,seed=seed)
        self.edgecnt = self.traffic.edgecnt
        self.pathnum = self.traffic.pathnum
        self.path = self.traffic.path
        self.Generate_auxiliary_variables(seed)
        
        return
            
                
    def out(self):
        """Return useful parameters."""
        return self.N, self.M, self.K, self.A[0], self.B[0], self.C[0], self.D, self.Q, self.W, self.L, self.St, self.pathnum, self.c0, self.c, self.theta
    

    
    def Generate_paras(self, seed, N, M, K, Dlb, Dub, Llb, Lub, Stlb, Stub, c0):
        """Generate D, L and St from [Dlb, Dub], [Llb, Lub] and [Stlb, Stub]."""
        self.N = N
        self.M = M
        self.K = K
        self.DN = np.random.uniform(Dlb, Dub, self.M * self.K)
        self.c0 = c0
        self.D = []
        for i in range(self.N):
            self.D.append(self.DN / self.N)

        self.L = np.random.uniform(Llb, Lub, (self.N, self.M))
        
        self.St = np.random.uniform(Stlb, Stub, (self.N, self.K))

        
    
            

    def Generate_point(self,seed,midpoint,limit,pathnum,traffic_edge):
        """Generate position of supplier, demander and midpoints."""
        np.random.seed(seed)
        if midpoint == None:
            self.middlepoint = (self.N + self.M) * pathnum
        else:
            self.middlepoint = midpoint
        # calculate the number of total points
        self.usedpoint = self.N + self.M
        self.allpoint = self.N + self.M + self.middlepoint
        self.traffic_edge = int(self.allpoint * 2)
        
        # generate the position of points
        if limit == None:
            self.limit = round(np.sqrt(10 * self.allpoint))
        else:
            self.limit = limit
        
        self.space = self.limit / self.allpoint
        
        ypos = [i for i in range(self.allpoint)]
        xpos1 = [i for i in range(self.N*2)]
        xpos2 = [i for i in range(self.allpoint - self.M*2, self.allpoint)]
        xpos3 = [i for i in range(self.N*2, self.allpoint - self.M*2)]
        np.random.shuffle(xpos1)
        np.random.shuffle(xpos2)
        np.random.shuffle(xpos3)
        np.random.shuffle(ypos)
        #print(xpos1)
        #print(xpos2)
        xpos = xpos1[:self.N] + xpos2[:self.M] + xpos1[self.N:] + xpos2[self.M:] + xpos3
        #print(xpos)
        pos = [[0,0] for i in range(self.allpoint)]
        # 0~N-1: supply points， N~N+M-1: demand points
        for i in range(self.allpoint):
            pos[i][0] = self.space * xpos[i]
            pos[i][1] = self.space * ypos[i]
        self.point_pos = np.array(pos)
        self.cost = np.zeros((self.allpoint, self.allpoint))
        for i in range(self.allpoint):
            for j in range(self.allpoint):
                self.cost[i][j] = np.linalg.norm(x = self.point_pos[i] - self.point_pos[j], ord = 2)
        return

    def Generate_auxiliary_variables(self,seed):
        """Generate other useful parameter, including A, B, C, L, W, P, Q."""
        self.c = []
        self.theta = []
        for i in range(self.N):
            self.c.append(np.random.uniform(5,10,self.edgecnt))  
        
        for i in range(self.M):
            self.theta.append(np.random.uniform(150,200,self.K)) 
        
        
        self.A = np.zeros((self.N, self.M * self.K, self.M * self.K * self.pathnum)) 
        for t in range(self.N):
            for i in range(self.M):
                for j in range(self.K):
                    for k in range(self.pathnum):
                        self.A[t][i * self.K + j][(i * self.K + j) * self.pathnum + k] = 1
                
                
        self.B = np.zeros((self.N, self.K, self.M * self.K * self.pathnum))
        #print(self.B.shape)
        for t in range(self.N):
            for i in range(self.M):
                for j in range(self.K):
                    for k in range(self.pathnum):
                        #print(t,j,i * self.pathnum * self.K + j * self.pathnum + k)
                        self.B[t][j][i * self.pathnum * self.K + j * self.pathnum + k] = 1
                
        self.C = np.zeros((self.N, self.M, self.K * self.M * self.pathnum))
        for t in range(self.N):
            for i in range(self.M):
                for j in range(self.K * self.pathnum):
                    self.C[t][i][i * self.K * self.pathnum + j] = 1
        
        self.old_linkfrom = []
        self.old_linkto = []
        
        
        edgelist = list(self.DG.edges(data=True))
        #print(len(edgelist))
        #print(self.linknum)
        for i in range(self.linknum):
            edge = edgelist[i]
            self.old_linkfrom.append(edge[0])
            self.old_linkto.append(edge[1])
            
        self.old_deg = [0 for i in range(self.N)] 
        self.old_tonode = [[] for i in range(self.N)] 
        self.Lap = np.array([[0 for i in range(self.N)] for j in range(self.N)])
        for i in range(self.linknum):
            self.old_deg[self.old_linkfrom[i]] += 1
            self.old_tonode[self.old_linkfrom[i]].append(self.old_linkto[i])
            self.Lap[self.old_linkfrom[i]][self.old_linkto[i]] = -1
        for i in range(self.N):
            self.Lap[i][i] = self.old_deg[i]
            
        epsilon = 0.1
        self.W = np.zeros((self.N, self.N))
        for i in range(self.linknum):
            self.W[self.old_linkfrom[i]][self.old_linkto[i]] = 1 / (np.maximum(self.old_deg[self.old_linkfrom[i]], self.old_deg[self.old_linkto[i]]) + epsilon)


        for i in range(self.N):
            self.W[i][i] = 1 - self.W[i].sum()
            
        I = np.eye(self.N)
        self.W = (self.W + I) / 2
        
        self.P = [np.zeros((self.pathnum * self.M * self.K, self.edgecnt)) for i in range(self.N)]
        self.Q = [np.zeros((self.edgecnt, self.pathnum * self.M * self.K)) for i in range(self.N)]
        for i in range(self.N):
            for j in range(self.M):
                for k in range(self.K):
                    for t in range(self.pathnum):
                        for v in self.path[i][j][t]:
                            self.Q[i][v][j * self.K * self.pathnum + k * self.pathnum + t] = 1
                            self.P[i][j * self.K * self.pathnum + k * self.pathnum + t][v] = 1
        return

    def draw(self):
        """Draw the traffic network with used edges and nodes. For example 1, we add ID of nodes."""
        if self.examplenum == 1:
            linew = 4
            points = [600, 300]
        elif self.examplenum == 2:
            linew = 2
            points = [200, 100]
        else:
            linew = 1.5
            points = [100, 50]
            
        fig, ax = plt.subplots(figsize = (8,8))
        
        # plot edge
        for i in range(0, self.traffic.primeedgecnt, 2):
            if self.traffic.uselink[i] == 1 or self.traffic.uselink[i+1] == 1:
                #print(i, i+1, self.traffic.uselink[i], self.traffic.uselink[i+1])
                x1 = self.point_pos[self.traffic.linkfrom[i]][0]
                y1 = self.point_pos[self.traffic.linkfrom[i]][1]
                x2 = self.point_pos[self.traffic.linkto[i]][0]
                y2 = self.point_pos[self.traffic.linkto[i]][1]
            #print(self.traffic.linkfrom[i], self.traffic.linkto[i])
                ax.plot([x1, x2], [y1, y2], color='gray', lw = linew, zorder = 1)  
                '''
                plt.text((x1+x2)/2, (y1+y2)/2, f'{str(i)}',
                    ha='center', va='center',  
                    fontsize=12, color='red')
                '''

        # plot points
        midcounter = 0
        midmap = []
        midypos = []
        offset = [-0.6, 0.6, 0.6, -0.6]
        for i in range(self.allpoint):
            if self.traffic.usenode[i] == 1:
                if i < self.N:
                    ax.scatter(self.point_pos[i][0], self.point_pos[i][1], s=points[0], color='blue', zorder = 2) 
                    if self.examplenum == 1:
                        plt.text(self.point_pos[i][0] - 0.7, self.point_pos[i][1] + offset[i], f'{"N" + str(i)}', 
                            ha='center', va='center',  
                            fontfamily= 'Times New Roman', fontsize=35, color='black')
                    #ax.text(x, y, f'{i}', fontsize=12, ha='right', va='bottom', zorder=3)  
                elif i < self.N + self.M:
                    ax.scatter(self.point_pos[i][0], self.point_pos[i][1], s=points[0], color='red', zorder = 2)  
                    if self.examplenum == 1:
                        plt.text(self.point_pos[i][0] + 0.9, self.point_pos[i][1], f'{"M" + str(i - self.N)}',  
                            ha='center', va='center',  
                            fontfamily= 'Times New Roman', fontsize=35, color='black')
                else:
                    ax.scatter(self.point_pos[i][0], self.point_pos[i][1], s=points[1], color='gray', zorder = 2)
                    '''
                    plt.text(self.point_pos[i][0] - 0.1, self.point_pos[i][1] + 0.5, f'{str(midcounter)}',  
                        ha='center', va='center',  
                        fontfamily= 'Times New Roman', fontsize=17, color='gray')
                    '''
                    midmap.append(i)
                    midypos.append(self.point_pos[i][1])
                    midcounter += 1
        if self.examplenum == 1:
            sorted_indices = np.argsort(midypos)[::-1]
            for i in range(midcounter):
                point = midmap[sorted_indices[i]]
                plt.text(self.point_pos[point][0] - 0.1, self.point_pos[point][1] + 0.7, f'{str(i)}', 
                            ha='center', va='center', 
                            fontfamily= 'Times New Roman', fontsize=30, color='gray')
            
                

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        return fig
    