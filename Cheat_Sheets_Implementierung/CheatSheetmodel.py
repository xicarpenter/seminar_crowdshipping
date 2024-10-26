import gurobipy as gp
from gurobipy import *

def Ablaufplanung_model(J, R, S, T, SJ, a, b, c, d, ls, model):
    model = gp.Model()
    TF = model.addVars(J,S, vtype=GRB.CONTINUOUS, lb=0.0, name="TF")
    X = model.addVars(J,S,T, vtype=GRB.BINARY, name="X")

    obj = c * quicksum(TF[j,SJ[j][-1]] for j in J)

    model.setObjective(obj, GRB.MINIMIZE)

    model.addConstrs((quicksum(X[j,s,t] for t in T) == 1 for j in J for s in SJ[j]), "Einmal")
    model.addConstrs((TF[j,s] >= TF[j,S[index_s-1]] + d[j,s] for j in J for index_s, s in enumerate(SJ[j]) if index_s > 1 ), "Schritte")
    model.addConstrs((quicksum((t_index+1) * X[j,s,t] for t_index, t in enumerate(T)) == TF[j,s] for j in J for s in SJ[j]), "Zeitpunkte")
    model.addConstrs((quicksum(  a[j, s, r] * X[j, s, tau] for j in J for s in SJ[j] for index_tau, tau in enumerate(T) if index_tau >= index_t if index_tau <= index_t + d[j, s] - 1) <= b[r, t] for r in R for index_t, t in enumerate(T)), "capacity")

    return model
