import gurobipy as gp
from gurobipy import *
from classes import Parameters, InstanceGenerator


def build_model(params: Parameters, of : str = "MAX_PROFIT") -> gp.Model:
    """
    Build a Gurobi model for the given Parameters instance.

    Parameters
    ----------
    params : Parameters
        The parameters of the model.

    Returns
    -------
    model : Model
        The Gurobi model.
    """
    model = gp.Model()

    # VARIABLES
    indices_ISJ = []
    indices_IS = []
    indices_JS = []
    for i in params.I:
        for s in params.S_i_p[i]:
            if (i, s) not in indices_IS:
                indices_IS.append((i, s))

            for j in params.J_is[i, s]:
                indices_ISJ.append((i, s, j))

                if (j, s) not in indices_JS:
                    indices_JS.append((j, s))

    X = model.addVars(indices_ISJ, vtype=GRB.BINARY, name="X") # 9
    Y = model.addVars(indices_IS, vtype=GRB.CONTINUOUS, name="Y", lb=0) # 10

    # OBJECTIVE
    # MAX-PROFIT
    # 1 -> checked
    if of == "MAX_PROFIT":
        model.setObjective((gp.quicksum(params.p[j] * X[i, params.alpha[j], j] 
                                        for j in params.J 
                                        for i in params.I_j_1[j] if (i, params.alpha[j], j) in X.keys()) # Adjusted

                                - gp.quicksum(params.f * Y[i, s] 
                                            for i in params.I 
                                            for s in params.S_i_p[i])), GRB.MAXIMIZE)
        
    elif of == "MAX_PARCELS":
        model.setObjective(gp.quicksum(X[i, params.alpha[j], j] 
                                        for j in params.J
                                        for i in params.I_j_1[j] if (i, params.alpha[j], j) in X.keys()), GRB.MAXIMIZE)

    else:
        raise ValueError("Objective function not recognized.")

    # CONSTRAINTS
    # 2 -> checked
    model.addConstrs((gp.quicksum(X[i, s, j] for j in params.J_is[i, s]) <= 1
                      for i in params.I 
                      for s in params.S_i_p[i]), "Constraint_2")

    # 3 -> checked
    model.addConstrs((gp.quicksum(X[i, s, j] 
                                  for i in params.I 
                                  if ((i,s) in params.J_is.keys() and j in params.J_is[i, s]
                                      and (i, s, j) in X.keys())) <= 1 

                            for j in params.J for s in params.S), "Constraint_3")
    
    # 3.5 Every parcel can only be moved to a single station once
    model.addConstrs((gp.quicksum(X[i, s, j] 
                                  for i in params.I 
                                  for s in params.S_i[i] 
                                  if (i, s) in params.s_is_p.keys() and params.s_is_p[i, s] == next_station
                                  if (i, s, j) in X.keys()) <= 1
                      for next_station in params.S 
                      for j in params.J), "Constraint_3.5")

    # 4 -> checked
    model.addConstrs((X[i, s, j] - gp.quicksum(X[i_p, params.s_is_p[i, s], j] 
                                    for i_p in params.I_is_p[i, params.s_is_p[i, s]]
                                    if (i_p, params.s_is_p[i, s], j) in X.keys()) <= 0 # Adjusted

                        for (i, s, j) in indices_ISJ if params.s_is_p[i, s] != params.omega[j]), "Constraint_4")

    # 5 -> checked
    model.addConstrs(((gp.quicksum(X[i, params.alpha[j], j] 
                                  for i in params.I_j_1[j] if (i, params.alpha[j], j) in X.keys()) # Adjusted
                        - gp.quicksum(X[i, params.s_is_m[i, params.omega[j]], j] 
                                      for i in params.I_j_2[j] if (i, params.s_is_m[i, params.omega[j]], j) in X.keys())) == 0 # Adjusted
                        for j in params.J), "Constraint_5")

    # 6 -> checked
    model.addConstrs((((gp.quicksum(X[i_p, params.alpha[j], j] 
                                  for j in params.J 
                                  for i_p in params.I_j_1[j]
                                  if (params.alpha[j] == s and params.t[i, s] >= params.r[j])
                                  and (i_p, params.alpha[j], j) in X.keys()) # Adjusted

                        + gp.quicksum(X[i_p, params.s_is_m[i_p, s], j] 
                                        for i_p in (set(params.I_is_m[i, s]) & set(params.I_s_p[s])) 
                                        for j in params.J_is[i_p, params.s_is_m[i_p, s]] 
                                        if params.omega[j] != s) 

                        + gp.quicksum(X[i_p, params.s_is_m[i_p, s], j] 
                                        for i_p in (set(params.I_is_m[i,s]) & set(params.I_s_p[s]))
                                        for j in params.J_is[i_p, params.s_is_m[i_p, s]]
                                        if (params.omega[j] == s 
                                            and params.d[j] >= params.t[i, s])) 

                        - gp.quicksum(X[i_p, s, j] 
                                      for i_p in params.I_is_m[i, s] 
                                      for j in params.J_is[i_p, s] if ((i_p, s, j) in X.keys()))) <= params.l[s]) # Adjusted

                                      for i in params.I for s in params.S_i[i]), "Constraint_6")

    # 7 -> checked
    model.addConstrs((X[i, s, j] - X[i, params.s_is_m[i, s], j] <= Y[i, s] 
                        for (i, s, j) in indices_ISJ if i in params.I_s_p[s]
                        and (i, params.s_is_m[i, s], j) in X.keys()), "Constraint_7") # Adjusted

    # 8 -> checked
    model.addConstrs(X[i, s, j] <= Y[i, s] 
                        for i in params.I for s in params.S_i_p[i] 
                        for j in params.J_is[i, s] 
                        if i not in params.I_s_p[s]), "Constraint_8"
    
    # 11
    model.addConstrs((gp.quicksum(X[i, params.s_is_m[i, params.alpha[j]], j]
                                 for i in params.I_s_p[params.alpha[j]] if j in params.J_is[i, params.s_is_m[i, params.alpha[j]]]) <= 0
                                 for j in params.J), "Constraint_11")
    
    # 12 
    model.addConstrs((gp.quicksum(X[i, params.omega[j], j] 
                                  for i in params.I_s_p[params.omega[j]] 
                                  if j in params.J_is[i, params.omega[j]] and (i, params.omega[j], j) in X.keys()) <= 0
                        for j in params.J), "Constraint_12")
    
    # Save model to lp file
    # model.write("model.lp")
    
    return model


def sort_by_minutes(data):
    # Sort by extracting the minutes after "@" and converting to an integer
    sorted_data = dict(sorted(data.items(), key=lambda item: int(item[1].split('@')[1].replace('min', ''))))
    return sorted_data


def sort_parcels(parcels):
    for j in parcels.keys():
        parcels[j] = sort_by_minutes(parcels[j])


def print_res(model, params):
    if model.status == GRB.OPTIMAL:
        print("\nFound an optimal solution:\n")
        parcels = {}
        for variable in model.getVars():
            if variable.x > 0 and "X" in variable.Varname:
                vars = variable.Varname.split("[")[1].split("]")[0].split(",")
                i, j = vars[0], vars[-1]
                s = ",".join(vars[1:-1])

                if j not in parcels.keys():
                    parcels[j] = {}

                parcels[j][(s, params.s_is_p[i, s])] = f"{i}@{params.t[i, s]}min"

        # Sort parcels
        sort_parcels(parcels)

        for j in parcels.keys():
            print(f"--- Parcel {j} ---")
            print(f"Origin station: {params.alpha[j]} @{params.r[j]}min\nTarget station: {params.omega[j]} @{params.d[j]}min\n")
            print(parcels[j], "\n")

    elif model.status == GRB.INFEASIBLE:
        print("\nThe model is infeasible. \n")

    elif model.status == GRB.UNBOUNDED:
        print("\nThe model is unbounded. \n")

    else:
        print("\nUnknown model status. Error code:", model.status, "\n")


if __name__ == "__main__":
    # params = Parameters.load("data/minimalinstanz.pkl")
    num_crowdshippers = 150
    num_parcels = 50
    entrainment_fee = 5
    generator = InstanceGenerator(num_crowdshippers, 
                                  num_parcels, 
                                  entrainment_fee)
    
    C, S, P = ('C44', 'Appelstrasse', 'P14')

    print(f"\nPath of {C}: \n", generator.find_path(generator.alpha_crowd[C], generator.omega_crowd[C])[0], "\n") 
    print(f"{C} is at {S} @{generator.t[C, S]}min\n") 
    print(f"Start of {P}: {generator.alpha[P]} @{generator.r[P]}min\nEnd of {P}: {generator.omega[P]} @{generator.d[P]}min", "\n")

    params = Parameters(**generator.return_kwargs())

    # generator.plot_graph()

    # MODEL
    model = build_model(params, of="MAX_PARCELS")

    # OPTIMIZATION
    model.optimize()

    # PRINT
    print_res(model, params)
