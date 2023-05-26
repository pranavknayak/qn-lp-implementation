#Import libraries
import pandas as pd
import itertools 
import numpy as np
import random
import networkx as nx
import json


import matplotlib as mpl
import matplotlib.pyplot as plt

from docplex.mp.model import Model
from docplex.mp.vartype import VarType


W_adj = np.array([[0, 30, 40, 10], [60, 0, 30, 0], [0, 24, 0, 30], [10, 0, 0, 0]])
# D = [(0, 1, 2, 2, 3), (2, 3, 5, 0, 5)]
D = [(0, 3, 2, 2, 3), (1, 3, 5, 0, 5)]


P = len(D)
n , e = len(W_adj), len(np.nonzero(W_adj)[0])
print(n, e)
T = 6

A = [W_adj]*T

m = Model(name='routing',log_output=False)
m.objective_sense = 'max'
m.parameters.threads.set(1)

def plot_graph(G):
    # Set up a plot
    pos = nx.spring_layout(G)
    plt.figure()

    # Draw nodes
    nx.draw_networkx_nodes(G, pos)

    # Draw edges
    nx.draw_networkx_edges(G, pos)

    # Draw edge labels with weights
    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Draw node labels
    nx.draw_networkx_labels(G, pos)
    # nx.draw_networkx(G)
    plt.show()

labels = [i for i in range(n)]
A2 = pd.DataFrame(W_adj, index=labels, columns=labels)
G_base = nx.from_pandas_adjacency(A2, create_using=nx.MultiDiGraph())
plot_graph(G_base)

####VARIABLES####
variables = m.integer_var_dict([(i, t, u, v) for i in range(P) for t in range(T) for u in range(n) for v in range(n) if W_adj[u][v] != 0], name='f')
print(variables)

####OBJECTIVE####
objective = 0
for i in range(P):
    for t in range(D[i][3], D[i][4]+1, 1):
        for v in range(n):
            if W_adj[D[i][0]][v] != 0:  #outgoing edge from src_i (D[i][0]) -> v
                print((i, t, D[i][0], v), variables[(i, t, D[i][0], v)])
                objective += variables[(i, t, D[i][0], v)]
print(objective)
m.maximize(objective)

####CONSTRAINTS####

#For a demand i, sum of all flows across all paths and all time-steps leaving the source si should be exactly equal to the demand size
for i in range(P):
    sum_allocated = 0
    for t in range(D[i][3], D[i][4]+1, 1):
        for v in range(n):
            if W_adj[D[i][0]][v] != 0:  #outgoing edge from src_i (D[i][0]) -> v
                sum_allocated += variables[(i, t, D[i][0], v)]
    m.add_constraint(sum_allocated == D[i][2], 'outflow_src_demand_'+str(i)+'_src_'+str(D[i][0]))
    print(m.get_constraint_by_name('outflow_src_demand_'+str(i)+'_src_'+str(D[i][0])))

#(Flow Conservation) At the intermediate nodes, incoming flow should be equal to the outgoing flow
for v in range(n):
    for t in range(T-1):
        inflow_v = 0
        outflow_v = 0
        for i in range(P):
            if v not in (D[i][0], D[i][1]):
                for u in range(n):
                    if W_adj[u][v] != 0: #incoming edge u->v
                        inflow_v += variables[(i, t, u, v)]
        
        for i in range(P):
            if v not in (D[i][0], D[i][1]):
                for w in range(n):
                    if W_adj[v][w] != 0: #outgoing edge v->w
                        outflow_v += variables[(i, t+1, v, w)]
        
        m.add_constraint(inflow_v == outflow_v, 'flow_cons_node_'+str(v)+'_t_'+str(t)+'_req_'+str(i))
        print(m.get_constraint_by_name('flow_cons_node_'+str(v)+'_t_'+str(t)+'_req_'+str(i)))

#(Capacity Constraints) At any time-step t, the sum of all allocations on the edge should be ≤ to the capacity of the edge:
for t in range(T):
    for u in range(n):
        for v in range(n):
            if W_adj[u][v] != 0:       #This is an edge
                sum_allocated = 0
                for i in range(P):
                    sum_allocated += variables[(i, t, u, v)]
                m.add_constraint(sum_allocated <= A[t][u][v], 'edge_capacity_'+str(u)+str(v)+'_t_'+str(t))
                print(m.get_constraint_by_name('edge_capacity_'+str(u)+str(v)+'_t_'+str(t)))
                print()

#For any time before STi or after ETi, no flow should be allocated to the demand i:
for i in range(P):
    sum_allocated = 0
    dont_allocate_time = [t for t in range(0, T) if t not in range(D[i][3],D[i][4])]
    print(f'dont allocate time for req: {D[i]} = {dont_allocate_time}')
    for t in dont_allocate_time:
        for u in range(n):
            for v in range(n):
                if W_adj[u][v] != 0: #edge u->v
                    sum_allocated += variables[(i, t, u, v)]
    m.add_constraint(sum_allocated == 0, 'no_allocation_bef_after_req_'+str(i))
    print(m.get_constraint_by_name('no_allocation_bef_after_req_'+str(i)))
    print()

#At the destination node ei, the sum of all the flows across all time-steps and all incoming paths for a request i should be equal to the demand size
for i in range(P):
    sum_allocated = 0
    for t in range(D[i][3], D[i][4]+1, 1):
        for u in range(n):
            if W_adj[u][D[i][1]] != 0:  #incoming edge from  u -> dest_i (D[i][1])
                sum_allocated += variables[(i, t, u, D[i][1])]
    m.add_constraint(sum_allocated == D[i][2], 'inflow_dest_demand_'+str(i)+'_dst_'+str(D[i][1]))
    print(m.get_constraint_by_name('inflow_dest_demand_'+str(i)+'_dst_'+str(D[i][1])))

#At any timestep t, the sum of all allocations for a demand should be <= |D_i|
for i in range(P):
    for t in range(D[i][3], D[i][4]+1, 1):
        sum_allocated=0
        for u in range(n):
            for v in range(n):
                if W_adj[u][v] != 0:  #edge from  u -> v
                    sum_allocated += variables[(i, t, u, v)]
        m.add_constraint(sum_allocated <= D[i][2], 'max_allocation_demand_'+str(i)+'_t_'+str(t))
        print(m.get_constraint_by_name('max_allocation_demand_'+str(i)+'_t_'+str(t)))


#Solver
solution = m.solve()
sol_json = solution.export_as_json_string()
# print('----------Problem Details: -----------')
# print(sol_json)
print('----------Solution Details: -----------')
vars = json.loads(sol_json)['CPLEXSolution']['variables']
for v in vars:
    print(v['name'], ':', v['value'])
