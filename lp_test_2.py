# TODO: add code to check how many requests were satisfied
# TODO: test for small instances:
#   1. Check if all demands are satisfied
#   2. Check for instances that solver cannot solve
#   3. Check why the constraint generation error comes
# TODO: stress test LP for large instances, plot results

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

np.random.seed(42)
random.seed(42)
def generate_weight_matrix(nb_nodes: int, p:float, min_weight: int, max_weight: int, debug: bool = False) -> np.ndarray:
    assert nb_nodes >= 0, f"expected >=0 nodes, got {nb_nodes}"
    assert p >= 0, f"expected >=0 edge probability, got {p}"
    assert min_weight >= 0, f"expected >=0 min-weight, got {min_weight}"
    assert max_weight >= min_weight, f"expected max_weight>=min-weight, got max_weight: {max_weight} and min_weight: {min_weight}"

    er_graph = np.random.choice([0, 1], size=(nb_nodes,nb_nodes), p=[1-p, p])
    np.fill_diagonal(er_graph, 0)
    weights = np.random.randint(min_weight, max_weight, size=(nb_nodes, nb_nodes))
    W_adj = er_graph * weights
    W_adj = (W_adj + W_adj.T) // 2
    if debug:
        print(f"WEIGHT MATRIX:\n{W_adj}")
    return W_adj


def generate_demands(nb_demands: int, nb_nodes: int, nb_timesteps: int, max_eprs: int, debug: bool = False) -> list[tuple]:
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
    demand_pairs = random.choices(all_pairs, k=nb_demands)
    start_nodes, end_nodes = list(zip(*demand_pairs))

    all_intervals = []
    for i in range(nb_timesteps):
        for j in range(i+1, nb_timesteps):
            all_intervals.append((i, j))


    demand_intervals = random.choices(all_intervals, k=nb_demands)
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

    nx.draw_networkx_edges(G, pos=positions, width=1.0)

    edge_labels = {(u, v): d["weight"] for u, v, d in G.edges(data=True)} 
    nx.draw_networkx_edge_labels(G, positions, edge_labels=edge_labels)
    nx.draw_networkx_labels(G, positions)
    plt.savefig(filename)

def add_variables(model: Model, W_adj: np.ndarray, nb_demands: int, nb_timesteps: int, nb_nodes: int, debug: bool = False) -> dict:
    variables = model.integer_var_dict([(i, t, u, v) for i in range(nb_demands) for t in range(nb_timesteps) for u in range(nb_nodes) for v in range(nb_nodes) if W_adj[u][v] != 0], name='f')
    # DEBUG PRINTING
    if debug: 
        for var_key in variables:
            print(variables[var_key].name)
    return variables

def define_objective(model: Model, W_adj: np.ndarray, nb_demands: int, nb_nodes:int, demand_list: list,  variables: dict, debug: bool = False) -> None:
    objective = 0
    for i in range(nb_demands):
        for t in range(demand_list[i][3], demand_list[i][4] + 1, 1):
            for v in range(nb_nodes):
                if W_adj[demand_list[i][0]][v] != 0:
                    # DEBUG PRINTING
                    if debug:
                        print((i, t, demand_list[i][0], v), variables[(i, t, demand_list[i][0], v)])
                    objective += variables[(i, t, demand_list[i][0], v)]
    # DEGUB PRINTING
    # print(objective)
    model.maximize(objective)

def define_constraints(model: Model, W_adj: np.ndarray, nb_timesteps: int, nb_demands: int, nb_nodes: int, demand_list: list, variables: dict, nw_state: list, debug: bool = False):
    source_outflow = []
    for i in range(nb_demands):
        sum_allocated = 0
        for t in range(demand_list[i][3], demand_list[i][4]+1, 1):
            for v in range(nb_nodes):
                if W_adj[demand_list[i][0]][v] != 0:
                    sum_allocated += variables[(i, t, demand_list[i][0], v)]
        model.add_constraint(sum_allocated <= demand_list[i][2], f"outflow_src_demand_{i}_src_{demand_list[i][0]}")
        source_outflow.append(sum_allocated)
        # DEBUG PRINTING
        if debug:
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
            if debug:
                print(f"INFLOW_V_{v}_T_{t} = {inflow_v}")
                print(f"OUTFLOW_V_{v}_T_{t} = {outflow_v}")
            model.add_constraint(inflow_v == outflow_v, f"flow_cons_node_{v}_t_{t}")
            # DEBUG PRINTING
            if debug: print(model.get_constraint_by_name(f"flow_cons_node_{v}_t_{t}"))

    # for i in range(nb_demands):
    #     for t in range(nb_timesteps):
    #         for v in range(nb_nodes):
    #             if v not in (demand_list[i][0], demand_list[i][1]):
    #                 inflow_v = 0
    #                 outflow_v = 0
    #                 for u in range(nb_nodes):
    #                     if W_adj[u][v] != 0:
    #                         inflow_v += variables[(i, t, u, v)]
    #                         outflow_v += variables[(i, t, v, u)]
    #                 if debug:
    #                     print(f"INFLOW_I_{i}_T_{t}_V_{v} = {inflow_v}")
    #                     print(f"OUTFLOW_I_{i}_T_{t}_V_{v} = {outflow_v}")
    #                 model.add_constraint(inflow_v == outflow_v, f"flow_cons_node_i_{i}_t_{t}_v_{v}")
    #                 if debug:
    #                     print(model.get_constraint_by_name(f"flow_cons_node_i_{i}_t_{t}_v_{v}"))
    
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
                 if debug:
                    print(model.get_constraint_by_name(f"edge_capacity_{u}{v}_t_{t}"))
                    print()
    
    # CONSTRAINT 4:
    for i in range(nb_demands):
        sum_allocated = 0
        dont_allocate_time = [t for t in range(0, nb_timesteps) if t not in range(demand_list[i][3],demand_list[i][4])]
        if debug:
            print(f'dont allocate time for req: {demand_list[i]} = {dont_allocate_time}')
        for t in dont_allocate_time:
            for u in range(nb_nodes):
                for v in range(nb_nodes):
                    if W_adj[u][v] != 0: #edge u->v
                        sum_allocated += variables[(i, t, u, v)]
        model.add_constraint(sum_allocated == 0, 'no_allocation_bef_after_req_'+str(i))
        # DEBUG PRINTING
        if debug:
            print(model.get_constraint_by_name('no_allocation_bef_after_req_'+str(i)))
            print()
    
    # CONSTRAINT 5:

    for i in range(nb_demands):
        sum_allocated = 0
        for t in range(demand_list[i][3], demand_list[i][4]+1, 1):
            for u in range(nb_nodes):
                if W_adj[u][demand_list[i][1]] != 0:  #incoming edge from  u -> dest_i (D[i][1])
                    sum_allocated += variables[(i, t, u, demand_list[i][1])]
        model.add_constraint(sum_allocated <= source_outflow[i], 'inflow_dest_demand_'+str(i)+'_dst_'+str(demand_list[i][1]))
        # DEBUG PRINTING
        if debug:
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
            if debug:
                print(model.get_constraint_by_name('max_allocation_demand_'+str(i)+'_t_'+str(t)))


def generate_flow_matrix(vars: list[dict], nb_demands: int, nb_timesteps: int, nb_nodes: int, D: list, debug: bool = False) -> np.ndarray:
    print("DEMANDS")
    print("------------------------------------------------------------------")
    print("Di | (start_node, end_node, requested_pairs, start_time, end_time)")
    print("------------------------------------------------------------------")
    for i, demand in enumerate(D):
        print(f"D{i} | {demand}")
    print("------------------------------------------------------------------")
    F = np.zeros(shape=(nb_demands, nb_timesteps, nb_nodes, nb_nodes))
    print("")
    print("")
    print("FLOWS")
    print("-------------------")
    print("[i, t, u, v] | flow")
    print("-------------------")
    for v in vars:
        if v['value'] != 0:
            id = v['name']
            id = id.split('_')[1:]
            id = list(map(int, id))
            F[id[0], id[1], id[2], id[3]] = float(v['value'])
            print(f"{id} | {v['value']}")
    print("-------------------")
    return F



def compute_served(F: np.ndarray, demand_list: list) -> int:
    nb_demands = len(demand_list)
    served_count = 0
    for i in range(nb_demands):
        eprs_served = 0 
        for t in range(F.shape[1]):
            for v in range(F.shape[2]):
                eprs_served += F[i][t][demand_list[i][0]][v]
        if eprs_served == demand_list[i][2]:
            served_count += 1 
            print(f"Served demand {i}")

    print(f"Served {served_count} out of {nb_demands} demands.")
    return served_count


def lp_solve(nb_nodes: int, edge_probability: float, min_weight: int, max_weight: int, nb_demands: int, nb_timesteps:int, max_eprs: int, debug: bool = False) -> None:
    # TODO: uncomment the lines below
    # W_adj = generate_weight_matrix(nb_nodes=nb_nodes, p=edge_probability, min_weight=min_weight, max_weight=max_weight, debug=debug)
    # D = generate_demands(nb_demands=nb_demands, nb_nodes=nb_nodes, nb_timesteps=nb_timesteps, max_eprs=max_eprs, debug=debug)
    
    W_adj = [[0, 0, 0, 0, 5, 5, 5],
             [0, 0, 0, 0, 0, 5, 5],
             [0, 0, 0, 0, 0, 5, 0],
             [0, 0, 0, 0, 5, 0, 0],
             [5, 0, 0, 5, 0, 0, 0],
             [5, 5, 5, 0, 0, 0, 0],
             [5, 5, 0, 0, 0, 0, 0]]
    W_adj = np.array(W_adj)
    D = [(0, 1, 10, 0, 1), (0, 2, 2, 0, 1), (0, 3, 2, 0, 1)]
    network_state = [W_adj]*(nb_timesteps + 1)

    model = Model(name='routing', log_output=False)
    model.objective_sense = 'max'
    model.parameters.threads.set(1)

    G_base = create_digraph(W_adj)
    plot_digraph(G_base, "graph.png")

    variables = add_variables(model=model, W_adj=W_adj, nb_demands=3, nb_timesteps=2, nb_nodes=7, debug=debug)
    define_objective(model=model, W_adj=W_adj, nb_demands=3, nb_nodes=7, demand_list=D, variables=variables, debug=debug)
    define_constraints(model=model, W_adj=W_adj, nb_timesteps=2, nb_demands=3, nb_nodes=7, demand_list=D, variables=variables, nw_state=network_state, debug=debug)

    solution = model.solve()
    if solution is None:
        print("ERR: Model returned None")
        return
    sol_json = solution.export_as_json_string()

    print('----------Solution Details: -----------')
    vars = json.loads(sol_json)['CPLEXSolution']['variables']
    F = generate_flow_matrix(vars, nb_demands=3, nb_timesteps=2, nb_nodes=7, D=D, debug=debug)
    served_count = compute_served(F, D)




if __name__ == "__main__":
    lp_solve(nb_nodes=3, edge_probability=0.3, min_weight=1, max_weight=5, nb_demands=5, nb_timesteps=10, max_eprs=8, debug=False)
