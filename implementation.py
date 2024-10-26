import gurobipy as gp
from gurobipy import *
import random
import pickle
from copy import deepcopy


class Parameters:
    def __init__(self, **kwargs):
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

        # fixed entrainment fee
        self.f = kwargs["f"]

        # locker capacity
        self.l = {s: random.randint(1, 4) for s in self.S}

        # generate subsets based on sets and parameters
        self.generate_subsets()


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
        self.S_i = dict.fromkeys(self.I, list())
        self.S_i_p = deepcopy(self.S_i)

        self.I_s = dict.fromkeys(self.S, list())
        self.I_s_p = deepcopy(self.I_s)
        self.I_j_1 = deepcopy(self.I_s)
        self.I_j_2 = deepcopy(self.I_s)

        self.I_js_m = {(i, j): list() for i in self.I for j in self.J}
        self.I_js_p = deepcopy(self.I_js_m)

        self.max_times = {i: 0 for i in self.I}
        self.min_times = {i: max(self.t.values()) for i in self.I}

        for (i, s), t in self.t.items():
            if t > self.max_times[i]:
                self.max_times[i] = t

            if t < self.min_times[i]:
                self.min_times[i] = t

        for (i, s), t in self.t.items():
            self.S_i[i].append(s)
            self.I_s[s].append(i)

            if t < self.max_times[i]:
                self.S_i_p[i].append(s)

            if t > self.min_times[i]:
                self.I_s_p[s].append(i)
    
    def __repr__(self):
        """
        Return a string representation of the Parameters instance.

        The string includes the number of crowdshippers, parcels, and stations
        associated with the instance.
        """
        return f"Instance of Parameters with {len(self.I)} crowdshippers, {len(self.J)} parcels and {len(self.S)} stations"
    

    def save(self, filename):
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
    def load(filename):
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
        with open(f"{filename}.pkl", "rb") as f:
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


def build_model(params):
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
    X = model.addVars(params.I, params.S, params.J, vtype=GRB.BINARY, name="X")
    Y = model.addVars(params.I, params.S, vtype=GRB.CONTINUOUS, name="Y")

    # OBJECTIVE
    ...

    # CONSTRAINTS
    ...


def print_res(model):
    if model.status == GRB.OPTIMAL:
        print("is optimal")

        for variable in model.getVars():
            if variable.x > 0:
                print(variable.Varname, variable.x)

    elif model.status == GRB.INFEASIBLE:
        print("is infeasible")

    elif model.status == GRB.INF_OR_UNBD:
        print("is infeasible or unbounded")

    elif model.status == GRB.UNBOUNDED:
        print("is unbounded")

    else:
        print("unknown")


if __name__ == "__main__":
    min_inst = {
                    "I": ["a", "b", "c"],
                    "J": ["A", "B"],
                    "S": ["s1","s2","s3","s4","s5","s6","s7","s8","s9","s10","s11"],
                    "alpha": {"A": 7, "B": 2},
                    "omega": {"A": 9, "B": 11},
                    "r": {"A": 1, "B": 2},
                    "d": {"A": 7, "B": 9},
                    "p": {"A": 5, "B": 5},
                    "f": 1,
                    "t": {
                        ("a", "s10") : 2,
                        ("a", "s7") : 3,
                        ("a", "s5") : 4,
                        ("a", "s4") : 5,
                        ("a", "s3") : 6,
                        ("b", "s1") : 5,
                        ("b", "s4") : 6,
                        ("b", "s9") : 7,
                        ("b", "s11") : 8,
                        ("c", "s2"): 3,
                        ("c", "s5"): 4,
                        ("c", "s6"): 5,
                        ("c", "s9"): 6,
                        ("c", "s11"): 7
                    }
                }
    
    params = Parameters(**min_inst)

    Parameters.save(params, "minimalinstanz")

    print(params.S_i_p)

    # model = build_model(params)

    # # OPTIMIZATION
    # model.optimize()

    # # PRINT
    # print_res(model)