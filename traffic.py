import numpy as np
import matplotlib.pyplot as plt
import json
import networkx as nx
import pandas as pd
import random


class traffic_graph():
    """Generate traffic graph based on position, distance, # of nodes, # of edges and pathnum, then find K shortest paths.

    Attributes:
        N: # of supplier nodes.
        M: # of demander nodes.
        midpoint: # of nodes in traffic network except supplier and demander.
        allpoint: # of total nodes.
        point_pos: Position of all points.
        cost: Distance between two points.
        link: Whether (i,j) is an edge.
        linkfrom: The begin node of this edge.
        linkto: The end node of this edge.
        nodetoedge: The list of outgoing edges from this node.
        edgecnt: The counter of edges.
        deg: The degree of nodes.
        DG: Represent this traffic graph.
        pathnum: # of paths from each supplier to each demander.
        deleted: Whether this edge is feasible when finding K shortest path.
        path: Contain K shortest path from each supplier node to each demander node.
        uselink: Whether this edge is used after finding all paths.
        usenode: Whether this node is in one path after finding all paths.
        linkmap: The map of edges ID after removing useless edges.
        mapnum: The counter of used edges.
    """
    def __init__(self, point_pos, cost, seed=None, N=20, M=5, midpoint=80, num_edge = 200, pathnum = 3):
        """Initialize the number and position of points, then generate traffic graph.

        Args:
            point_pos: Position of all points.
            cost: Distance between two points.
            seed: Random seed.
            N: # of supplier nodes.
            M: # of demander nodes.
            midpoint: # of nodes in traffic network except supplier and demander.
            number_edge: # of edges in traffic network.
            pathnum: # of paths from each supplier to each demander.
        """
        np.random.seed(seed)
        self.N = N
        self.M = M
        self.midpoint = midpoint
        self.allpoint = self.N + self.M + self.midpoint


        self.point_pos = point_pos
        self.cost = cost
        self.Generate_graph(seed, num_edge, pathnum)
        return
    

    def addlink(self, fromid, toid):
        """Add undirected edge (fromid, toid) to traffic graph."""
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
        return
    
    def dijkstra(self, st, ed):
        """Use dijkstra algorithm to find shortest path from st to ed.

        Args:
            st: Starting node.
            ed: End node.
        Return:
            shpath: All edges in shortest path.
            dis[ed]: The length of shortest path.
        """
        inf = 1e8
        dis = [inf for i in range(self.allpoint)]
        dis[st] = 0
        used = [0 for i in range(self.allpoint)]
        notused_dis = [inf for i in range(self.allpoint)]
        notused_dis[st] = 0
        pathfrom = [-1 for i in range(self.allpoint)]
        for i in range(self.allpoint):
            fromid = np.argmin(notused_dis)
            if fromid == ed:
                break
            used[fromid] = 1
            notused_dis[fromid] = inf
            for v in self.nodetoedge[fromid]:
                if self.deleted[v] == 0:
                    toid = self.linkto[v]
                    if used[toid] == 0:
                        if dis[toid] > dis[fromid] + self.cost[fromid][toid]:
                            pathfrom[toid] = v
                            dis[toid] = dis[fromid] + self.cost[fromid][toid]
                            notused_dis[toid] = dis[toid]

        shpath = []
        v = pathfrom[ed]
        #print(dis)
        #print(ed)
        while v != -1:
            #print(self.linkfrom[v])
            shpath.append(v)
            v = pathfrom[self.linkfrom[v]]
        #print("")
        #print(ed, dis[ed])
        return shpath, dis[ed]


    def findpath(self, st, ed, K):
        """Find K shortest path from st to ed."""
        self.deleted = [0 for i in range(self.edgecnt)]
        for i in range(K):
            if i != 0:
                mincost = 1e8
                savedv = 0
                for changedv in self.path[st][ed-self.N][i-1]:
                    self.deleted[changedv] = 1
                    shpath, sumcost = self.dijkstra(st,ed)

                    if sumcost < mincost:
                        mincost = sumcost
                        savedv = changedv
                    self.deleted[changedv] = 0
                self.deleted[savedv] = 1
                #print(savedv)

            shpath, sumcost = self.dijkstra(st,ed)
            self.path[st][ed-self.N].append(shpath)
            for v in shpath:
                self.uselink[v] = 1
        return 
    

    def ccw(self, A, B, C):
        """Calculate cross product of (B-A) and (C-A)."""
        return (B[0] - A[0]) * (C[1] - A[1]) - (B[1] - A[1]) * (C[0] - A[0])

    def segments_intersect(self, A, B, C, D):
        """Test intersection between two line segments."""
        if max(A[0], B[0]) < min(C[0], D[0]) or max(C[0], D[0]) < min(A[0], B[0]):
            return False
        if max(A[1], B[1]) < min(C[1], D[1]) or max(C[1], D[1]) < min(A[1], B[1]):
            return False

        
        ccw1 = self.ccw(A, B, C)
        ccw2 = self.ccw(A, B, D)
        ccw3 = self.ccw(C, D, A)
        ccw4 = self.ccw(C, D, B)

        if ((ccw1 * ccw2) < 0) and ((ccw3 * ccw4) < 0):
            return True  

        return False
    
    def checkccw(self, fromid, toid):
        """Test whether the edge (fromid, toid) intersect with exsiting lines."""
        for i in range(0, self.edgecnt, 2):
            A = self.linkfrom[i]
            B = self.linkto[i]
            if self.segments_intersect(self.point_pos[fromid], self.point_pos[toid], self.point_pos[A], self.point_pos[B]) == True:
                return False
        #print(111)
        return True

    def Generate_graph(self, seed, num_edge, pathnum):
        """Generate traffic network based on seed and # of edges, then generate I (pathnum) paths from each supplier to demander."""

        self.DG = nx.DiGraph()
        self.DG.add_nodes_from(range(0, self.allpoint))
        self.linkfrom = []
        self.linkto = []
        inf = 1e8
        #self.edgecnt = 0
        np.random.seed(seed)
        self.deg = np.zeros(self.allpoint,dtype=int)
        self.link = np.zeros((self.allpoint, self.allpoint),dtype=int)
        self.linkfrom = np.zeros(num_edge*2,dtype=int)
        self.linkto = np.zeros(num_edge*2,dtype=int)
        self.edgecnt = 0
        self.nodetoedge = [[] for i in range(self.allpoint)]

        # generate a minimum spanning tree
        MST_dis = np.zeros(self.allpoint)
        MST_dict = np.array([0 for i in range(self.allpoint)])
        MST_linked = np.array([0 for i in range(self.allpoint)])
        for i in range(1,self.allpoint):
            MST_dis[i] = np.linalg.norm(x = self.point_pos[i] - self.point_pos[0], ord = 2)
        MST_dis[0] = 1e8
        MST_linked[0] = 1
        for i in range(self.allpoint - 1):
            fromid = np.argmin(MST_dis)
            toid = MST_dict[fromid]
            self.addlink(fromid, toid)
            
            MST_dis[fromid] = inf
            MST_linked[fromid] = 1
            for j in range(self.allpoint):
                if MST_linked[j] == 0:
                    tempdis = np.linalg.norm(x = self.point_pos[fromid] - self.point_pos[j], ord = 2)
                    if tempdis < MST_dis[j]:
                        MST_dis[j] = tempdis
                        MST_dict[j] = fromid

        # add remaining edges
        lastedge = num_edge - self.allpoint + 1
        # randomedge = int(lastedge * randomrate)
        fixedge = lastedge

        dis = [inf for i in range(self.allpoint)]
        for counter in range(fixedge):
            '''
            fromid = np.argmin(self.deg)

            for j in range(self.allpoint):
                if self.link[fromid][j] == 1:
                    dis[j] = inf
                else:
                    dis[j] = self.cost[fromid][j]
            dis[fromid] = inf

            toid = np.argmin(dis)
            self.addlink(fromid, toid)
            '''
            #print(counter, fixedge, num_edge)
            right = 0
            while right == 0:
                fromid = np.random.randint(0, self.allpoint)
                for j in range(self.allpoint):
                    if self.link[fromid][j] == 1 or fromid == j:
                        dis[j] = inf
                    else:
                        dis[j] = self.cost[fromid][j]
                #print(list(dis))
                #print(fixedge, self.edgecnt, fromid)
                #while(1):
                #    continue
                min_indices = np.argsort(dis)
                for toid in min_indices:
                    if dis[toid] + 1e-5 >= inf:
                        break
                    if self.checkccw(fromid, toid) == True:
                        right = 1
                        self.addlink(fromid, toid)
                        #print(fromid, toid)
                        break


        
        # generate path
        self.uselink = [0 for i in range(self.edgecnt)]
        self.path = [[[] for j in range(self.M)] for i in range(self.N)]
        self.pathnum = pathnum
        for i in range(0,self.N):
            for j in range(self.N, self.N+self.M):
                self.findpath(i,j,self.pathnum)
            

        self.usenode = [0 for i in range(self.allpoint)]
        for i in range(0, self.edgecnt, 2):
            if self.uselink[i] == 1 or self.uselink[i+1] == 1:
                self.usenode[self.linkfrom[i]] = 1
                self.usenode[self.linkto[i]] = 1
        
        self.linkmap = [-1 for i in range(self.edgecnt)]
        self.mapnum = 0
        for i in range(0, self.edgecnt):
            if self.uselink[i] == 1:
                self.linkmap[i] = self.mapnum
                self.mapnum += 1
        
        for i in range(0,self.N):
            for j in range(self.N, self.N+self.M):
                for l in range(self.pathnum):
                    for t in range(len(self.path[i][j-self.N][l])):
                        self.path[i][j-self.N][l][t] = self.linkmap[self.path[i][j-self.N][l][t]]
        # self.primeedgecnt = self.edgecnt
        self.edgecnt = self.mapnum
        
        # print(self.uselink)
        return