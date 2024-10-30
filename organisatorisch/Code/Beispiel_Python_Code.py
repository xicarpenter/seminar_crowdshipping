#import gurobipy as gp
from gurobipy import *

def Ablaufplanung_model(J, R, S, T, SJ, a, b, c, d, ls):
    model = Model()
    
    TF = model.addVars(J, S, vtype=GRB.CONTINUOUS, lb=0.0, name='TF')
    
    X = model.addVars(J, S, T, vtype=GRB.BINARY, name='X')

    #quicksum ist eine Funktion des package "gurobipy"
    obj = c * quicksum(TF[j,ls[j]] for j in J)

    model.setObjective(obj, GRB.MINIMIZE)
    

    model.addConstrs((quicksum (X[j,s,t] for t in T) == 1 for j in J for s in SJ[j]), 'Einmal')
    
    for j in J:
        for indes, s in enumerate(SJ[j]):
            if indes >= 1:
                model.addConstr((TF[j,s] >= TF[j,S[indes-1]] + d[j,s]), 'Schritte')
    
    model.addConstrs((quicksum((t index+1) * X[j,s,t] for tindex, t in enumerate(T)) == TF[j,s] for j in J for s in SJ[j]), 'Zeitpunkte')
    
    model.addConstrs((quicksum(a[j,s,r] * X[j,s,tau] for j in J for s in SJ[j] for indetau, tau in enumerate(T) if indet <= indetau <= indet + d[j,s] - 1) <= b[r,t] for r in R for indet, t in enumerate(T)), 'Ressourcen')
   
    return model