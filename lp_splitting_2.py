import pandas as pd
import itertools
import numpy as np
import random
import networkx as nx
import json

import matplotlib as mpl
import matplotlib.pyplot as plt

import docplex
from docplex.mp.model import Model
from docplex.mp.vartype import VarType

def generate_weight_matrix(nb_nodes: int, p:float, min_weight: int, max_weight: int) -> np.ndarray:
    assert nb_nodes >= 0, f"expected >=0 nodes, got {nb_nodes}"
    assert p >= 0, f"expected >=0 edge probability, got {p}"
    assert min_weight >= 0, f"expected >=0 min-weight, got {min_weight}"
    assert max_weight >= min_weight, f"expected max_weight>=min-weight, got max_weight: {max_weight} and min_weight: {min_weight}"

    er_graph = np.random.choice([0, 1], size=(nb_nodes,nb_nodes), p=[1-p, p])
    np.fill_diagonal(er_graph, 0)
    weights = np.random.randint(min_weight, max_weight, size=(nb_nodes, nb_nodes))
    return er_graph * weights


def generate_demands(nb_demands: int, nb_nodes: int, nb_timesteps: int, max_eprs: int) -> list[tuple]:
    assert nb_demands >= 0, f"expected >=0 demands, got {nb_demands}"
    assert nb_nodes >= 0, f"expected >=0 nodes, got {nb_nodes}"
    assert nb_timesteps >= 0, f"expected >=0 timesteps, got {nb_timesteps}"
    assert max_eprs >= 0, f"expected >=0 maximum EPRs, got {max_eprs}"


    required_pairs = np.random.randint(1, max_eprs, size=nb_demands)
    all_pairs = []
    for i in range(nb_nodes):
        for j in range(nb_nodes):
            if i != j: 
                all_pairs.append((i, j))
    demand_pairs = np.random.permutation(all_pairs)[:nb_demands].tolist()
    start_nodes, end_nodes = list(zip(*demand_pairs))

    all_intervals = []
    for i in range(nb_timesteps):
        for j in range(i+1, nb_timesteps):
            all_intervals.append((i, j))

    demand_intervals = np.random.permutation(all_intervals)[:nb_demands].tolist()
    start_times, end_times = list(zip(*demand_intervals))
    demand_list = list(zip(start_nodes, end_nodes, required_pairs, start_times, end_times))

    return demand_list

def create_digraph(W_adj: np.ndarray) -> nx.DiGraph:
    nb_nodes = W_adj.shape[0]
    labels = [i for i in range(nb_nodes)]
    W_df = pd.DataFrame(W_adj, index=labels, columns=labels)
    G_digraph = nx.from_pandas_adjacency(W_df, create_using=nx.DiGraph())
    return G_digraph


def plot_digraph(G: nx.DiGraph, filename: str) -> None:
    plt.figure()
    positions = nx.spring_layout(G)

    nx.draw_networkx_nodes(G, positions)
    nx.draw_networkx_edges(G, positions)

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)} 
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, positions)
    plt.savefig(filename)

def add_variables(model: Model, W_adj: np.ndarray, nb_demands: int, nb_timesteps: int, nb_nodes: int) -> dict:
    variables = model.integer_var_dict([(i, t, u, v) for i in range(nb_demands) for t in range(nb_timesteps) for u in range(nb_nodes) for v in range(nb_nodes) if W_adj[u][v] != 0], name='f')
    # DEBUG PRINTING
    for var_key in variables:
        print(variables[var_key].name)
    return variables

def define_objective(model: Model, W_adj: np.ndarray, nb_demands: int, nb_nodes:int, demand_list: list,  variables: dict) -> None:
    objective = 0
    for i in range(nb_demands):
        for t in range(demand_list[i][3], demand_list[i][4] + 1, 1):
            for v in range(nb_nodes):
                if W_adj[demand_list[i][0]][v] != 0:
                    # DEBUG PRINTING
                    print((i, t, demand_list[i][0], v), variables[(i, t, demand_list[i][0], v)])
                    objective += variables[(i, t, demand_list[i][0], v)]
    # DEGUB PRINTING
    # print(objective)
    model.maximize(objective)

def define_constraints(model: Model, W_adj: np.ndarray, nb_timesteps: int, nb_demands: int, nb_nodes: int, demand_list: list, variables: dict, nw_state: list):

    
    for i in range(nb_demands):
        sum_allocated = 0
        for t in range(demand_list[i][3], demand_list[i][4]+1, 1):
            for v in range(nb_nodes):
                if W_adj[demand_list[i][0]][v] != 0:
                    sum_allocated += variables[(i, t, demand_list[i][0], v)]
        model.add_constraint(sum_allocated == demand_list[i][2], f"outflow_src_demand_{i}_src_{demand_list[i][0]}")
        # DEBUG PRINTING
        print(model.get_constraint_by_name(f"outflow_src_demand_{i}_src_{demand_list[i][0]}"))
    
    # CONSTRAINT 2:
    for v in range(nb_nodes):
        for t in range(nb_timesteps):
            inflow_v = 0
            outflow_v = 0
            for i in range(nb_demands):
                if v not in (demand_list[i][0], demand_list[i][1]):
                    for u in range(nb_nodes):
                        if W_adj[u][v] != 0: #incoming edge u->v
                            inflow_v += variables[(i, t, u, v)]
    
            for i in range(nb_demands):
                if v not in (demand_list[i][0], demand_list[i][1]):
                    for w in range(nb_nodes):
                        if W_adj[v][w] != 0: #outgoing edge v->w
                            outflow_v += variables[(i, t, v, w)]
            # DEBUG PRINTING
            print(f"INFLOW_V_{v}_T_{t} = {inflow_v}")
            print(f"OUTFLOW_V_{v}_T_{t+1} = {outflow_v}")
            model.add_constraint(inflow_v == outflow_v, f"flow_cons_node_{v}_t_{t}")
            # DEBUG PRINTING
            print(model.get_constraint_by_name(f"flow_cons_node_{v}_t_{t}"))
    
    # CONSTRAINT 3:
    for t in range(nb_timesteps):
        for u in range(nb_nodes):
         for v in range(nb_nodes):
             if W_adj[u][v] != 0:       #This is an edge
                 sum_allocated = 0
                 for i in range(nb_demands):
                     sum_allocated += variables[(i, t, u, v)]
                 model.add_constraint(sum_allocated <= nw_state[t][u][v], f"edge_capacity_{u}{v}_t_{t}")
                 # DEBUG PRINTING
                 print(model.get_constraint_by_name(f"edge_capacity_{u}{v}_t_{t}"))
                 print()
    
    # CONSTRAINT 4:
    for i in range(nb_demands):
        sum_allocated = 0
        dont_allocate_time = [t for t in range(0, nb_timesteps) if t not in range(demand_list[i][3],demand_list[i][4])]
        print(f'dont allocate time for req: {demand_list[i]} = {dont_allocate_time}')
        for t in dont_allocate_time:
            for u in range(nb_nodes):
                for v in range(nb_nodes):
                    if W_adj[u][v] != 0: #edge u->v
                        sum_allocated += variables[(i, t, u, v)]
        model.add_constraint(sum_allocated == 0, 'no_allocation_bef_after_req_'+str(i))
        # DEBUG PRINTING
        print(model.get_constraint_by_name('no_allocation_bef_after_req_'+str(i)))
        print()
    
    # CONSTRAINT 5:

    for i in range(nb_demands):
        sum_allocated = 0
        for t in range(demand_list[i][3], demand_list[i][4]+1, 1):
            for u in range(nb_nodes):
                if W_adj[u][demand_list[i][1]] != 0:  #incoming edge from  u -> dest_i (D[i][1])
                    sum_allocated += variables[(i, t, u, demand_list[i][1])]
        model.add_constraint(sum_allocated == demand_list[i][2], 'inflow_dest_demand_'+str(i)+'_dst_'+str(demand_list[i][1]))
        # DEBUG PRINTING
        print(model.get_constraint_by_name('inflow_dest_demand_'+str(i)+'_dst_'+str(demand_list[i][1])))
    
    # CONSTRAINT 6
    for i in range(nb_demands):
        for t in range(demand_list[i][3], demand_list[i][4]+1, 1):
            sum_allocated=0
            for u in range(nb_nodes):
                for v in range(nb_nodes):
                    if W_adj[u][v] != 0:  #edge from  u -> v
                        sum_allocated += variables[(i, t, u, v)]
            model.add_constraint(sum_allocated <= demand_list[i][2], 'max_allocation_demand_'+str(i)+'_t_'+str(t))
            print(model.get_constraint_by_name('max_allocation_demand_'+str(i)+'_t_'+str(t)))


def lp_solve(nb_nodes: int, edge_probability: float, min_weight: int, max_weight: int, nb_demands: int, nb_timesteps:int, max_eprs: int) -> None:
    W_adj = generate_weight_matrix(nb_nodes=nb_nodes, p=edge_probability, min_weight=min_weight, max_weight=max_weight)
    D = generate_demands(nb_demands=nb_demands, nb_nodes=nb_nodes, nb_timesteps=nb_timesteps, max_eprs=max_eprs)
    network_state = [W_adj]*(nb_timesteps + 1)

    model = Model(name='routing', log_output=False)
    model.objective_sense = 'max'
    model.parameters.threads.set(1)

    G_base = create_digraph(W_adj)
    plot_digraph(G_base, "graph.png")

    variables = add_variables(model=model, W_adj=W_adj, nb_demands=nb_demands, nb_timesteps=nb_timesteps, nb_nodes=nb_nodes)
    define_objective(model=model, W_adj=W_adj, nb_demands=nb_demands, nb_nodes=nb_nodes, demand_list=D, variables=variables)
    define_constraints(model=model, W_adj=W_adj, nb_timesteps=nb_timesteps, nb_demands=nb_demands, nb_nodes=nb_nodes, demand_list=D, variables=variables, nw_state=network_state)

    solution = model.solve()
    if model is None:
        print("ERR: Model returned None")
        return
    sol_json = solution.export_as_json_string()

    print('----------Solution Details: -----------')
    vars = json.loads(sol_json)['CPLEXSolution']['variables']
    for v in vars:
        print(v['name'], ':', v['value'])


if __name__ == "__main__":
    lp_solve(nb_nodes=4, edge_probability=0.5, min_weight=10, max_weight=20, nb_demands=3, nb_timesteps=6, max_eprs=5)
