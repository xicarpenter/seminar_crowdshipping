import gurobipy as gp
from gurobipy import *
from classes import Parameters, InstanceGenerator


def build_model(params: Parameters) -> gp.Model:
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
    # 1
    model.setObjective((gp.quicksum(params.p[j] * X[i, params.alpha[j], j] 
                                    for j in params.J 
                                    for i in params.I_j_1[j] if (i,params.alpha[j],j) in X.keys())

                            - gp.quicksum(params.f * Y[i, s] 
                                          for i in params.I 
                                          for s in params.S_i_p[i])), GRB.MAXIMIZE)

    # CONSTRAINTS
    # 2
    model.addConstrs((gp.quicksum(X[i, s, j] for j in params.J_is[i, s]) <= 1
                      for i in params.I 
                      for s in params.S_i_p[i]), "Constraint_2")

    # 3
    model.addConstrs((gp.quicksum(X[i, s, j] 
                                  for i in params.I 
                                  if ((i, s) in params.J_is.keys() 
                                      and j in params.J_is[i, s]
                                      and (i, s, j) in X.keys())) <= 1 

                            for (j, s) in indices_JS), "Constraint_3")

    # 4
    model.addConstrs((X[i, s, j] - gp.quicksum(X[i_p, params.s_is_p[i, s], j] 
                                    for i_p in params.I_is_p[i, params.s_is_p[i, s]] 
                                    if (i_p, params.s_is_p[i, s], j) in X.keys()) <= 0

                        for (i, s, j) in indices_ISJ 
                        if params.s_is_p[i, s] != params.omega[j]), "Constraint_4")

    # 5
    model.addConstrs((gp.quicksum(X[i, params.alpha[j], j] for i in params.I_j_1[j] if (i, params.alpha[j], j) in X.keys()) 
                        - gp.quicksum(X[i, params.s_is_m[i, params.omega[j]], j] 
                        for i in params.I_j_2[j] if (i, params.omega[j]) in params.s_is_m.keys() and (i, params.s_is_m[i, params.omega[j]], j) in X.keys()) == 0 
                        for j in params.J), "Constraint_5")

    # 6
    model.addConstrs((gp.quicksum(X[i_p, params.alpha[j], j] 
                                  for j in params.J 
                                  for i_p in params.I_j_1[j]
                                  if (params.alpha[j] == s and params.t[i, s] >= params.r[j])
                                  and (i_p, params.alpha[j], j) in X.keys()) 

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
                                      for j in params.J_is[i_p, s] 
                                      if (i_p, s, j) in X.keys()) <= params.l[s] 

                                      for i in params.I for s in params.S_i[i]), "Constraint_6")

    # 7
    model.addConstrs((X[i, s, j] - X[i, params.s_is_m[i, s], j] <= Y[i, s] 
                        for (i, s, j) in indices_ISJ if i in params.I_s_p[s]
                        and (i, params.s_is_m[i, s], j) in X.keys()), "Constraint_7")

    # 8
    model.addConstrs(X[i, s, j] <= Y[i, s] 
                        for i in params.I for s in params.S_i_p[i] 
                        for j in params.J_is[i, s] 
                        if i not in params.I_s_p[s]), "Constraint_8"
    
    return model


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

                parcels[j][(s, params.s_is_p[i, s])] = i

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

    params = Parameters(**generator.return_kwargs())

    generator.plot_graph()

    # # MODEL
    # model = build_model(params)

    # # OPTIMIZATION
    # model.optimize()

    # # PRINT
    # print_res(model, params)
