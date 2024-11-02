import gurobipy as gp
from gurobipy import *
import random
import pickle
from copy import deepcopy


class Parameters:
    def __init__(self, **kwargs):
        # setting a seed
        self.seed = kwargs["seed"]
        random.seed(self.seed)

        # set of crowdshippers
        self.I = kwargs["I"]

        # set of parcels
        self.J = kwargs["J"]

        # set of stations
        self.S = kwargs["S"]

        # set of parcel starting and destination stations
        self.alpha = kwargs["alpha"]
        self.omega = kwargs["omega"]

        # set of release and deadline times
        self.r = kwargs["r"]
        self.d = kwargs["d"]

        # postal charge of parcels
        self.p = kwargs["p"]

        # point in time when crowdshipper i is at station s
        self.t = kwargs["t"]

        self.sorted_stations = self.sort_stations()

        # fixed entrainment fee
        self.f = kwargs["f"]

        # locker capacity
        if "l" in kwargs:
            self.l = kwargs["l"]
        else:
            self.l = {s: random.randint(1, 4) for s in self.S}

        # generate subsets based on sets and parameters
        self.generate_subsets()


    def sort_stations(self):
        """Return the stations based on the time they are visited in sorted order per crowdshipper."""
        sorted_stations = {}

        for (i, s), _ in sorted(self.t.items(), key=lambda item: item[1]):
            if i not in sorted_stations:
                sorted_stations[i] = []

            sorted_stations[i].append(s)

        return sorted_stations
    

    def generate_subsets(self):
        """
        Generate subsets based on sets and parameters of the instance.

        This function initializes the following subsets based on the parameters of the instance:
        - S_i: set of stations crowdshipper i visits
        - S_i_p: set of stations crowdshipper i visits before his/her last visit
        - I_s: set of crowdshippers visiting station s
        - I_s_p: set of crowdshippers visiting station s after their first visit
        - I_j_1, I_j_2: set of crowdshippers that can pick up or drop off parcel j
        - I_js_m, I_js_p: set of crowdshippers that can pick up or drop off parcel j
        - max_times, min_times: maximum and minimum times of visits of crowdshippers

        Note that the subsets are initialized as empty lists/dictionaries and are filled in the loop below.
        """
        # set of stations visited by crowdshipper i
        self.S_i = {i: list() for i in self.I}

        # set of stations visited by crowdshipper i before their last visit
        self.S_i_p = deepcopy(self.S_i)

        # set of crowdshippers visiting station s
        self.I_s = {s: list() for s in self.S}

        # set of crowdshippers visiting station s but not starting there
        self.I_s_p = deepcopy(self.I_s)

        # set of crowdshippers that can pick up parcel j 
        self.I_j_1 = {j: list() for j in self.J}

        # set of crowdshippers that can deliver parcel j
        self.I_j_2 = deepcopy(self.I_j_1)

        # set of crowdshippers who can deliver 
        # a non finished entrainment with parcel j from station s
        self.I_is_m = {}

        # set of crowdshippers who can pick up 
        # a non finished entrainment with parcel j from station s
        self.I_is_p = {}

        # set of parcels that can be picked up 
        # by crowdshipper i at station s
        self.J_is = {}

        # maximum and minimum times of visits of crowdshippers
        self.max_times = {i: 0 for i in self.I}
        self.min_times = {i: max(self.t.values()) for i in self.I}

        # generate max and min times
        for (i, s), t in self.t.items():
            if t > self.max_times[i]:
                self.max_times[i] = t

            if t < self.min_times[i]:
                self.min_times[i] = t

        # generate subsets for stations and crowdshippers
        for (i, s), t in self.t.items():
            self.S_i[i].append(s)
            self.I_s[s].append(i)

            if t < self.max_times[i]:
                self.S_i_p[i].append(s)

            if t > self.min_times[i]:
                self.I_s_p[s].append(i)

            for j in self.J:
                if self.alpha[j] == s:
                    if t >= self.r[j]:
                        self.I_j_1[j].append(i)

                if self.omega[j] == s:
                    if t <= self.d[j]:
                        self.I_j_2[j].append(i)

        # generate subsets for non finished entrainments
        for s in self.S:
            for i in self.I_s[s]:
                self.I_is_m[i, s] = [c for c in self.I_s[s] 
                                     if self.t[c, s] <= self.t[i, s]]
                
                self.I_is_p[i, s] = [c for c in self.I_s[s] 
                                     if self.t[c, s] >= self.t[i, s]]
                
        # generate subsets of parcels
        for i in self.I:
            for s in self.S_i[i]:
                self.J_is[i, s] = [j for j in self.J 
                                   if self.r[j] <= self.t[i, s] <= self.d[j]]
  
        # station visited by crowdshipper i in I_s_p immediately before/after station s
        self.s_is_m = {(i, s) : self.get_last_station(i, s) 
                        for s in self.S for i in self.I_s_p[s]}
        
        self.s_is_p = {(i, s) : self.get_next_station(i, s) 
                       for i in self.I for s in self.S_i_p[i]}
        

    def get_last_station(self, i: int, s: str) -> str:
        """
        Return the station visited by crowdshipper i immediately before station s.

        Parameters
        ----------
        i : int
            crowdshipper identifier
        s : str
            station identifier

        Returns
        -------
        str
            station identifier
        """
        stations = self.sorted_stations[i]
        last_index = stations.index(s)-1

        return stations[last_index]
    

    def get_next_station(self, i: int, s: str) -> str:
        """
        Return the station visited by crowdshipper i immediately after station s.

        Parameters
        ----------
        i : int or str
            Identifier for the crowdshipper.
        s : str
            The current station for which the next station is to be found.

        Returns
        -------
        str
            The next station visited by the crowdshipper.
        """
        stations = self.sorted_stations[i]
        next_index = stations.index(s)+1

        return stations[next_index]


    def __repr__(self):
        """
        Return a string representation of the Parameters instance.

        The string includes the number of crowdshippers, parcels, and stations
        associated with the instance.
        """
        return f"Instance of Parameters with {len(self.I)} crowdshippers, {len(self.J)} parcels and {len(self.S)} stations"
    

    def save(self, filename: str) -> None:
        """
        Saves the instance of Parameters to a file.

        The file is saved in the binary pickle format.

        Parameters
        ----------
        filename : str
            The name of the file to save the instance to. The file extension
            (.pkl) is added automatically.

        Returns
        -------
        None
        """
        with open(f"{filename}.pkl", "wb") as f:
            pickle.dump(self, f)


    @staticmethod
    def load(filename: str):
        """
        Load a Parameters instance from a file.

        The file is expected to be in the binary pickle format.

        Parameters
        ----------
        filename : str
            The name of the file to load the instance from. The file extension
            (.pkl) is added automatically.

        Returns
        -------
        Parameters
            The loaded Parameters instance.
        """
        with open(filename, "rb") as f:
            params = pickle.load(f)

        return params
    

    def plot(self):
        """
        Plot a visualization of the model parameters.

        This could be a map with nodes for stations and arcs for crowdshippers
        and parcels. The map could be colored according to the type of node/arc.
        Alternatively, a Gantt chart could be used to visualize the time
        windows of the crowdshippers and the parcels.

        :return: None
        """
        pass


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
                                    for i in params.I_j_1[j])

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
                                      and j in params.J_is[i, s])) <= 1 

                            for (j, s) in indices_JS), "Constraint_3")

    # 4
    model.addConstrs((X[i, s, j] - gp.quicksum(X[i_p, params.s_is_p[i, s], j] 
                                    for i_p in params.I_is_p[i, params.s_is_p[i, s]] 
                                    if (i_p, params.s_is_p[i, s], j) in X.keys()) <= 0

                        for (i, s, j) in indices_ISJ 
                        if params.s_is_p[i, s] != params.omega[j]), "Constraint_4")

    # 5
    model.addConstrs((gp.quicksum(X[i, params.alpha[j], j] for i in params.I_j_1[j]) 
                        - gp.quicksum(X[i, params.s_is_m[i, params.omega[j]], j] 
                        for i in params.I_j_2[j]) == 0 
                        for j in params.J), "Constraint_5")

    # 6
    model.addConstrs((gp.quicksum(X[i_p, params.alpha[j], j] 
                                  for j in params.J 
                                  for i_p in params.I_j_1[j]
                                  if (params.alpha[j] == s and params.t[i, s] >= params.r[j])) 

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
                        for (i, s, j) in indices_ISJ 
                        if (i, s) in params.s_is_m.keys()), "Constraint_7")

    # 8
    model.addConstrs(X[i, s, j] <= Y[i, s] 
                        for i in params.I for s in params.S_i_p[i] 
                        for j in params.J_is[i, s] 
                        if i not in params.I_s_p[s]), "Constraint_8"
    
    return model


def print_res(model):
    if model.status == GRB.OPTIMAL:
        print("\nFound an optimal solution:\n")

        for variable in model.getVars():
            if variable.x > 0:
                print(variable.Varname, variable.x)

    elif model.status == GRB.INFEASIBLE:
        print("\nThe model is infeasible. \n")

    elif model.status == GRB.UNBOUNDED:
        print("\nThe model is unbounded. \n")

    else:
        print("\nUnknown model status. Error code:", model.status, "\n")


if __name__ == "__main__":
    params = Parameters.load("data/minimalinstanz.pkl")
    model = build_model(params)

    # OPTIMIZATION
    model.optimize()

    # PRINT
    print_res(model)
