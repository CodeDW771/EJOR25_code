import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx

import random

class communication_graph:
    """Generate communication graph based on position, distance, # of nodes, # of edges and randomrate.

    Attributes:
        N: # of supplier nodes.
        M: # of demander nodes.
        point_pos: Position of all points.
        cost: Distance between two points.
        link: Whether (i,j) is an edge.
        linkfrom: The begin node of this edge.
        linkto: The end node of this edge.
        nodetoedge: The list of outgoing edges from this node.
        edgecnt: The counter of edges.
        deg: The degree of nodes.
        DG: Represent this communication graph.
    """
    def __init__(self, point_pos, cost, seed=None, N=20, M=5, num_edge = 50, randomrate = 0):
        """Initialize the number and position of points, then generate communication graph.

        Args:
            point_pos: Position of all points.
            cost: Distance between two points.
            seed: Random seed.
            N: # of supplier nodes.
            M: # of demander nodes.
            number_edge: # of edges in communication network.
            randomrate_communication: Randomness in topology in communication network.
        """
        self.N = N
        self.M = M

        self.point_pos = point_pos[:self.N]
        self.cost = cost

        self.Generate_graph(seed, num_edge, randomrate)
        return
    
    def addlink(self, fromid, toid):
        """Add undirected edge (fromid, toid) to communication graph."""
        self.link[fromid][toid] = 1
        self.link[toid][fromid] = 1
        
        self.linkfrom[self.edgecnt]=fromid
        self.linkto[self.edgecnt]=toid
        self.nodetoedge[fromid].append(self.edgecnt)
        self.edgecnt+=1
        
        self.linkfrom[self.edgecnt]=toid
        self.linkto[self.edgecnt]=fromid
        self.nodetoedge[toid].append(self.edgecnt)
        self.edgecnt+=1
        
        
        self.deg[fromid] += 1
        self.deg[toid] += 1
        #print(fromid, toid)
        self.DG.add_edge(fromid,toid)
        self.DG.add_edge(toid,fromid)
        #print(len(list(self.DG.edges)))

        return


    def Generate_graph(self, seed, num_edge, randomrate):
        """Generate traffic network based on seed, # of edges and randomrate."""
        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(range(0, self.N))
        self.linkfrom = []
        self.linkto = []
        self.edgecnt = 0
        inf = 1e6
        np.random.seed(seed)
        self.deg = np.zeros(self.N,dtype=int)
        self.link = np.zeros((self.N, self.N),dtype=int)
        self.linkfrom = np.zeros(num_edge*2,dtype=int)
        self.linkto = np.zeros(num_edge*2,dtype=int)
        self.edgecnt = 0
        self.nodetoedge = [[] for i in range(self.N)] 
        # generate a minimum spanning tree
        MST_dis = np.zeros(self.N)
        MST_dict = np.array([0 for i in range(self.N)])
        MST_linked = np.array([0 for i in range(self.N)])
        for i in range(1,self.N):
            MST_dis[i] = self.cost[i][0]
        MST_dis[0] = inf
        MST_linked[0] = 1
        for i in range(self.N - 1):
            fromid = np.argmin(MST_dis)
            toid = MST_dict[fromid]
            self.addlink(fromid, toid)
            
            MST_dis[fromid] = inf
            MST_linked[fromid] = 1
            for j in range(self.N):
                if MST_linked[j] == 0:
                    tempdis = self.cost[fromid][j]
                    if tempdis < MST_dis[j]:
                        MST_dis[j] = tempdis
                        MST_dict[j] = fromid

        # add remaining edges
        lastedge = num_edge - self.N + 1
        randomedge = int(lastedge * randomrate)
        fixedge = lastedge - randomedge

        # add shortest edges
        dis = [inf for i in range(self.N)]
        for _ in range(fixedge):
            fromid = np.argmin(self.deg)
            #print(self.edgecnt)

            for j in range(self.N):
                if self.link[fromid][j] == 1:
                    dis[j] = inf
                else:
                    dis[j] = self.cost[fromid][j]
            dis[fromid] = inf

            toid = np.argmin(dis)
            self.addlink(fromid, toid)

        # randomly add edges
        for _ in range(randomedge):
            fromid = np.random.randint(0, self.N)
            toid = np.random.randint(0, self.N)
            while fromid == toid or self.link[fromid][toid] == 1:
                fromid = np.random.randint(0, self.N)
                toid = np.random.randint(0, self.N)
            self.addlink(fromid,toid)

        #print("communication ", self.edgecnt)
        #print("communication DG ", len(list(self.DG.edges)))
        return