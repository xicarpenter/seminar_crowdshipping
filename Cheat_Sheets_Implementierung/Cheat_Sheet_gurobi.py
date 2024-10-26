import CheatSheetmodel
import gurobipy as gp
from gurobipy import *


Jobs = 4
Ressourcen = 3
Schritte = 2
Perioden = 20

J = [f"j{j}" for j in range(1, Jobs + 1)]

#print(J)

R = ["rA", "rB", "rC"]

S = [f"s{s}" for s in range(1, Schritte + 1)]

T = [f"t{t}" for t in range(1, Perioden + 1)]


letzte_Schritte = {"j1": 2, "j2": 2, "j3": 2, "j4": 1}
ls = {}
for job, letzter_Schritt in letzte_Schritte.items():
    ls[job] = S[letzter_Schritt-1]



d = tupledict({("j1", "s1"): 3,
               ("j1", "s2"): 2,
               ("j2", "s1"): 1,
               ("j2", "s2"): 3,
               ("j3", "s1"): 3,
               ("j3", "s2"): 2,
               ("j4", "s1"): 4,
               ("j4", "s2"): 0})
SJ = {}
for j in J:
    S_help = []
    for index_i, i in enumerate(S):
        if index_i+1 <= letzte_Schritte[j]:
            S_help.append(i)
    SJ[j] = S_help
print(SJ)




b = {(r, t): 1 for r in R for t in T}

c = 1

a = {(j,s,r): 0 for j in J for s in S for r in R}


a["j1", "s1", "rA"] = 1
a["j1", "s2", "rB"] = 1

a["j2", "s1", "rA"] = 1
a["j2", "s2", "rC"] = 1

a["j3", "s1", "rB"] = 1
a["j3", "s2", "rA"] = 1

a["j4", "s1", "rC"] = 1



Ablauf = CheatSheetmodel.Ablaufplanung_model(J, R, S, T, SJ, a, b, c, d, ls)
Ablauf.optimize()

def PrintVars(model):
    Vars = {}
    if model.status == GRB.OPTIMAL:
        for variable in model.getVars():
            if variable.x > 0:
                Vars[variable.Varname] = variable.x
        print('Obj: %g' % model.ObjVal)
        print(Vars)
    elif model.status == GRB.INFEASIBLE:
        print('is infeasible')

PrintVars(Ablauf)