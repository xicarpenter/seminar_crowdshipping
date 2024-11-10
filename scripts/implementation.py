import gurobipy as gp
from gurobipy import *
from classes import Parameters, InstanceGenerator
import random



class CrowdshipperModel(gp.Model):
    def __init__(self, 
                 params: Parameters,
                 of: str = "MAX_PROFIT"):
        super().__init__()
        self.params = params
        self.of = of

    
    @property
    def of(self):
        return self._of
    
    @of.setter
    def of(self, of):
        self._of = of
        self.build(params=self.params, of=self.of)


    def set_params(self, params: Parameters, of: str = None):
        self.params = params
        if of is not None:
            self.of = of
        
        else:
            self.build(params=self.params, of=self.of)


    def build(self) -> gp.Model:
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
        # VARIABLES
        self.indices_ISJ = []
        self.indices_IS = []
        self.indices_JS = []
        for i in self.params.I:
            for s in self.params.S_i_p[i]:
                if (i, s) not in self.indices_IS:
                    self.indices_IS.append((i, s))

                for j in self.params.J_is_p[i, s]:
                    self.indices_ISJ.append((i, s, j))

                    if (j, s) not in self.indices_JS:
                        self.indices_JS.append((j, s))

        self.X = self.addVars(self.indices_ISJ, vtype=GRB.BINARY, name="X") # 9
        self.Y = self.addVars(self.indices_IS, vtype=GRB.CONTINUOUS, name="Y", lb=0) # 10

        # OBJECTIVE
        # MAX-PROFIT
        # 1 -> checked
        if self.of == "MAX_PROFIT":
            self.setObjective((gp.quicksum(self.params.p[j] * self.X[i, self.params.alpha[j], j] 
                                            for j in self.params.J 
                                            for i in self.params.I_j_1[j] if (i, self.params.alpha[j], j) in self.X.keys()) # Adjusted

                                    - gp.quicksum(self.params.f * self.Y[i, s] 
                                                for i in self.params.I 
                                                for s in self.params.S_i_p[i])), GRB.MAXIMIZE)
            
        elif self.of == "MAX_PARCELS":
            self.setObjective((gp.quicksum(self.X[i, self.params.alpha[j], j] 
                                            for j in self.params.J
                                            for i in self.params.I_j_1[j] if (i, self.params.alpha[j], j) in self.X.keys())), GRB.MAXIMIZE)

        else:
            raise ValueError("Objective function not recognized.")

        # CONSTRAINTS
        # 2 -> checked
        self.addConstrs((gp.quicksum(self.X[i, s, j] for j in self.params.J_is[i, s]  if (i, s, j) in self.X.keys()) <= 1
                        for i in self.params.I 
                        for s in self.params.S_i_p[i]), "Constraint_2")

        # 3 -> checked
        self.addConstrs((gp.quicksum(self.X[i, s, j] 
                                    for i in self.params.I 
                                    if ((i,s) in self.params.J_is.keys() and j in self.params.J_is[i, s]
                                        and (i, s, j) in self.X.keys())) <= 1 

                                for j in self.params.J for s in self.params.S), "Constraint_3")
        
        # 3.5 Every parcel can only be moved to a single station once
        if self.of in ["MAX_PARCELS", "MAX_PROFIT"]:
            self.addConstrs((gp.quicksum(self.X[i, s, j] 
                                        for i in self.params.I 
                                        for s in self.params.S_i[i] 
                                        if (i, s) in self.params.s_is_p.keys() and self.params.s_is_p[i, s] == next_station
                                        and (i, s, j) in self.X.keys()) <= 1
                            for next_station in self.params.S 
                            for j in self.params.J), "Constraint_3.5")

        # 4 -> checked
        self.addConstrs((self.X[i, s, j] - gp.quicksum(self.X[i_p, self.params.s_is_p[i, s], j] 
                                        for i_p in self.params.I_is_p[i, self.params.s_is_p[i, s]]
                                        if (i_p, self.params.s_is_p[i, s], j) in self.X.keys()) <= 0 # Adjusted

                            for (i, s, j) in self.indices_ISJ if self.params.s_is_p[i, s] != self.params.omega[j]), "Constraint_4")

        # 5 -> checked
        self.addConstrs(((gp.quicksum(self.X[i, self.params.alpha[j], j] 
                                    for i in self.params.I_j_1[j] if (i, self.params.alpha[j], j) in self.X.keys()) # Adjusted
                            - gp.quicksum(self.X[i, self.params.s_is_m[i, self.params.omega[j]], j] 
                                        for i in self.params.I_j_2[j] if (i, self.params.s_is_m[i, self.params.omega[j]], j) in self.X.keys())) == 0 # Adjusted
                            for j in self.params.J), "Constraint_5")

        # 6 -> checked
        self.addConstrs((((gp.quicksum(self.X[i_p, self.params.alpha[j], j] 
                                    for j in self.params.J 
                                    for i_p in self.params.I_j_1[j]
                                    if (self.params.alpha[j] == s and self.params.t[i, s] >= self.params.r[j])
                                    and (i_p, self.params.alpha[j], j) in self.X.keys()) # Adjusted

                            + gp.quicksum(self.X[i_p, self.params.s_is_m[i_p, s], j] 
                                            for i_p in (set(self.params.I_is_m[i, s]) & set(self.params.I_s_p[s])) 
                                            for j in self.params.J_is[i_p, self.params.s_is_m[i_p, s]] 
                                            if self.params.omega[j] != s and (i, s, j) in self.X.keys())

                            + gp.quicksum(self.X[i_p, self.params.s_is_m[i_p, s], j] 
                                            for i_p in (set(self.params.I_is_m[i,s]) & set(self.params.I_s_p[s]))
                                            for j in self.params.J_is[i_p, self.params.s_is_m[i_p, s]]
                                            if (self.params.omega[j] == s 
                                                and self.params.d[j] >= self.params.t[i, s]) and (i_p, self.params.omega[j], j) in self.X.keys()) # Adjusted

                            - gp.quicksum(self.X[i_p, s, j] 
                                        for i_p in self.params.I_is_m[i, s] 
                                        for j in self.params.J_is[i_p, s] if ((i_p, s, j) in self.X.keys()))) <= self.params.l[s]) # Adjusted

                                        for i in self.params.I for s in self.params.S_i[i]), "Constraint_6")

        # 7 -> checked
        self.addConstrs((self.X[i, s, j] - self.X[i, self.params.s_is_m[i, s], j] <= self.Y[i, s] 
                            for (i, s, j) in self.indices_ISJ if i in self.params.I_s_p[s]
                            and (i, self.params.s_is_m[i, s], j) in self.X.keys()), "Constraint_7") # Adjusted

        # 8 -> checked
        self.addConstrs(self.X[i, s, j] <= self.Y[i, s] 
                            for i in self.params.I for s in self.params.S_i_p[i] 
                            for j in self.params.J_is[i, s] 
                            if i not in self.params.I_s_p[s] and (i, s, j) in self.X.keys()), "Constraint_8"
        
        # 11
        self.addConstrs((gp.quicksum(self.X[i, self.params.s_is_m[i, self.params.alpha[j]], j]
                                    for i in self.params.I_s_p[self.params.alpha[j]] 
                                    if j in self.params.J_is_p[i, self.params.s_is_m[i, self.params.alpha[j]]]) <= 0
                                    for j in self.params.J), "Constraint_11")
        
        # 12 
        self.addConstrs((gp.quicksum(self.X[i, self.params.omega[j], j] 
                                    for i in self.params.I_s_p[self.params.omega[j]] 
                                    if j in self.params.J_is[i, self.params.omega[j]] and (i, self.params.omega[j], j) in self.X.keys()) <= 0
                            for j in self.params.J), "Constraint_12")
        
        # Save model to lp file
        # self.write("model.lp")


    @staticmethod
    def sort_by_minutes(data):
        # Sort by extracting the minutes after "@" and converting to an integer
        """
        Sort a dictionary by the numeric minutes extracted from the values.

        Parameters
        ----------
        data : dict
            A dictionary where each value is a string containing a number 
            followed by "min" after an "@" symbol.

        Returns
        -------
        dict
            A dictionary sorted by the integer value of minutes in ascending order.
        """
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1][2]))
        return sorted_data


    def sort_parcels(self, 
                     parcels: dict):
        """
        Sort parcels by minutes after "@" in ascending order

        Parameters
        ----------
        parcels : dict
            Dictionary with keys as parcel IDs and values as dictionaries
            with start and end times as strings in the format "HH:MM@mm"

        Returns
        -------
        dict
            Sorted dictionary of parcels
        """
        for j in parcels.keys():
            parcels[j] = self.sort_by_minutes(parcels[j])

        self.parcels = dict(sorted(parcels.items(), key=lambda item: int(item[0][1:])))


    def calc_max_parcels(self):
        """
        Calculate the maximum number of parcels that can be moved.

        Parameters
        ----------
        X : dict
            Dictionary of decision variables X[i, s, j] for i in I, s in S_i and j in J_is[i, s]
        params : Parameters
            Instance of Parameters class

        Returns
        -------
        int
            Maximum number of parcels that can be moved
        """
        max_parcels = 0
        for j in self.params.J:
            for i in self.params.I_j_1[j]:
                if (i, self.params.alpha[j], j) in self.X.keys():
                    max_parcels += self.X[i, self.params.alpha[j], j]

        return max_parcels


    def calc_max_profit(self, long_print: bool = True):
        """
        Calculate the maximum profit that can be obtained given the
        decision variables X[i, s, j] and Y[i, s].

        Parameters
        ----------
        X : dict
            Dictionary of decision variables X[i, s, j] for i in I, s in S_i and j in J_is[i, s]
        Y : dict
            Dictionary of decision variables Y[i, s] for i in I and s in S_i_p[i]
        params : Parameters
            Instance of Parameters class

        Returns
        -------
        int
            Maximum profit that can be obtained
        """
        max_profit = 0
        for j in self.params.J:
            for i in self.params.I_j_1[j]:
                if (i, self.params.alpha[j], j) in self.X.keys() and self.X[i, self.params.alpha[j], j] > 0:
                    if long_print:
                        print(f"Using X[{i, self.params.alpha[j], j}]")
                    max_profit += self.params.p[j] * self.X[i, self.params.alpha[j], j]

        for i in self.params.I:
            for s in self.params.S_i_p[i]:
                if self.Y[i, s] > 0:
                    if long_print:
                        print(f"Using Y[{i, s}]")
                    max_profit -= self.params.f * self.Y[i, s]

        return max_profit


    def check_results(self, long_print: bool = True):
        """
        Print the results of the Gurobi model.

        Parameters
        ----------
        model : Model
            The Gurobi model
        params : Parameters
            Instance of Parameters class

        Returns
        -------
        None

        Notes
        -----
        If the model is optimal, it prints the number of parcels and the profit.
        If the model is infeasible or unbounded, it prints an error message.
        """
        if self.status == GRB.OPTIMAL:
            print("\nFound an optimal solution:\n")
            self.parcels = {}
            
            for (i, s, j), val in self.X.items():
                if val > 0:
                    if j not in self.parcels.keys():
                        self.parcels[j] = {}
                    try:
                        self.parcels[j][(s, self.params.s_is_p[i, s])] = [
                            i, self.params.t[i, s], self.params.t[i, self.params.s_is_p[i, s]]] # i, current time, next time
                    
                    except KeyError:
                        print(f"X[{i, s, j}] is invalid!")
                        
            max_parcels = self.calc_max_parcels()
            max_profit = self.calc_max_profit(long_print)

            print(f"Number of parcels: {max_parcels}")
            print(f"Profit: {max_profit}\n")

            # Sort parcels
            self.sort_parcels(self.parcels)

            if long_print:
                self.print_parcels()

        elif self.status == GRB.INFEASIBLE:
            print("\nThe model is infeasible. \n")

        elif self.status == GRB.UNBOUNDED:
            print("\nThe model is unbounded. \n")

        else:
            print("\nUnknown model status. Error code:", self.status, "\n")


    def check_validity(self, parcel_id: str, 
                       parcel_dict: dict):
        last_end = ""
        parcel_start = self.params.r[parcel_id]
        parcel_deadline = self.params.d[parcel_id]

        for idx, ((start, end), info) in enumerate(parcel_dict.items()):
            time_next = info[2]
            time_curr = info[1]

            if idx == 0:
                if start != self.params.alpha[parcel_id]:
                    return False

                last_end = end
                continue
            
            elif idx == len(parcel_dict) - 1:
                if end != self.params.omega[parcel_id]:
                    return False

            if start != last_end or parcel_deadline < time_next or time_curr < parcel_start:
                return False

            last_end = end
        
        return True
    


    def print_parcels(self):
        for j in self.parcels.keys():
            print(f"--- Parcel {j} ---")
            print(f"Origin station: {self.params.alpha[j]}"\
                    f" @{self.params.r[j]}min\nTarget station: {self.params.omega[j]} @{self.params.d[j]}min\n")
            print(self.parcels[j], "\n")

            if self.check_validity(j, self.parcels[j], self.params):
                print("Valid!\n")
            
            else:
                print("Invalid!\n")


    def check_parcels(self, seed):
        count = 0
        for j, stations in self.parcels.items():
            if not self.check_validity(j, stations, self.params):
                print(f"Parcel {j} is invalid!")
                print("Seed:", seed)
                print(f"Origin station: {self.params.alpha[j]} @{self.params.r[j]}min\nTarget station: {self.params.omega[j]} @{self.params.d[j]}min\n")
                print(stations, "\n")
                count += 1

        return count


def test_seeds(num_crowdshippers, num_parcels, entrainment_fee, output: bool = False, of = "MAX_PARCELS"):
    """
    Test 10 different seeds of the given parameters and print the results.
    """
    random.seed()
    random_seeds = [random.randint(0, 1e5) for _ in range(20)]
    seed = random_seeds[0]
    try:
        generator = InstanceGenerator(num_crowdshippers, 
                                    num_parcels, 
                                    entrainment_fee,
                                    seed=seed)

    except ValueError:
        print(f"Seed {seed} is invalid!\n")

    params = Parameters(**generator.return_kwargs())

    # MODEL
    model = CrowdshipperModel(params, of=of)

    if not output:
        model.setParam(GRB.Param.OutputFlag, 0)
    
    for seed in random_seeds[1:]:
        try:
            generator = InstanceGenerator(num_crowdshippers, 
                                        num_parcels, 
                                        entrainment_fee,
                                        seed=seed)

        except ValueError:
            print(f"Seed {seed} is invalid!\n")
            continue

        params = Parameters(**generator.return_kwargs())

        model.set_params(params)

        # OPTIMIZATION
        model.optimize()

        # PRINT
        model.check_results()
        
        count = model.check_parcels(seed)

        print("Done with seed:", seed)
        print(f"Invalid parcels: {count}\n")
        

if __name__ == "__main__":
    # params = Parameters.load("data/minmalinstanz.pkl")
    num_crowdshippers = 150
    num_parcels = 50
    entrainment_fee = 5
    of = "MAX_PROFIT"

    test_seeds(num_crowdshippers, 
               num_parcels,
               entrainment_fee,
               of=of)
