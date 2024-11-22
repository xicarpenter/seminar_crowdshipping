import gurobipy as gp
from gurobipy import GRB
from classes import Parameters, InstanceGenerator, update_minimalinstanz
import random
import pickle



class CrowdshippingModel(gp.Model):
    def __init__(self, 
                 params: Parameters,
                 of: str = "MAX_PROFIT",
                 use_3_5: bool = True,
                 save_lp: bool = False):
        super().__init__()
        self._use_3_5 = use_3_5
        self._save_lp = save_lp
        self._params = params
        self._of = of

    
    @property
    def _of(self):
        return self.__of
    
    @_of.setter
    def _of(self, of):
        self.__of = of
        self.build()


    def set_params(self, params: Parameters, of: str = None):
        self._params = params
        if of is not None:
            self._of = of
        
        else:
            self.build()


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
        indices_ISJ = []
        indices_IS = []
        indices_JS = []
        for i in self._params.I:
            for s in self._params.S_i_p[i]:
                if (i, s) not in indices_IS:
                    indices_IS.append((i, s))

                for j in self._params.J_is[i, s]:
                    indices_ISJ.append((i, s, j))

                    if (j, s) not in indices_JS:
                        indices_JS.append((j, s))                

        self._X = self.addVars(indices_ISJ, vtype=GRB.BINARY, name="X") # 9
        self._Y = self.addVars(indices_IS, vtype=GRB.CONTINUOUS, name="Y", lb=0) # 10

        # OBJECTIVE
        # MAX-PROFIT
        # 1 -> checked
        if self._of == "MAX_PROFIT":
            self.setObjective((gp.quicksum(self._params.p[j] * self._X[i, self._params.alpha[j], j] 
                                            for j in self._params.J 
                                            for i in self._params.I_j_1[j])

                                    - gp.quicksum(self._params.f * self._Y[i, s] 
                                                for i in self._params.I 
                                                for s in self._params.S_i_p[i])), GRB.MAXIMIZE)
            
        elif self._of == "MAX_PARCELS":
            self.setObjective(gp.quicksum(self._X[i, self._params.alpha[j], j] 
                                            for j in self._params.J
                                            for i in self._params.I_j_1[j]), GRB.MAXIMIZE)

        else:
            raise ValueError("Objective function not recognized.")

        # CONSTRAINTS
        # 2 -> checked
        (i_check, s_check, j_check) = ("C138", 'Altwarmb., Ernst-Grote-Strasse', 'P43')

        self.addConstrs((gp.quicksum(self._X[i, s, j] 
                                     for j in self._params.J_is[i, s]) <= 1
                        for (i, s) in indices_IS), "Constraint_2")

        # 3 -> checked
        self.addConstrs((gp.quicksum(self._X[i, s, j] 
                                    for i in self._params.I 
                                    if j in self._params.J_is[i, s]) <= 1 
                                for j in self._params.J 
                                for s in self._params.S), "Constraint_3")
        
        # 3.5 Every parcel can only be moved to a single station once
        if self._use_3_5:
            self.addConstrs((gp.quicksum(self._X[i, s, j]  
                                        for (i, s) in self._params.s_is_p.keys() 
                                        if self._params.s_is_p[i, s] == next_station
                                        and (i, s, j) in self._X.keys()) <= 1
                            for next_station in self._params.S 
                            for j in self._params.J), "Constraint_3.5") # if j != "P43"

        # 4 -> checked
        self.addConstrs((self._X[i, s, j] 
                         - gp.quicksum(self._X[i_p, self._params.s_is_p[i, s], j] 
                                        for i_p in self._params.I_is_p[i, self._params.s_is_p[i, s]]
                                        if (i_p, self._params.s_is_p[i, s], j) in self._X.keys()) <= 0 # Adjusted
                            for (i, s, j) in indices_ISJ 
                            if self._params.s_is_p[i, s] != self._params.omega[j]), "Constraint_4")

        # 5 -> checked
        self.addConstrs(((gp.quicksum(self._X[i, self._params.alpha[j], j] 
                                    for i in self._params.I_j_1[j])
                            - gp.quicksum(self._X[i, self._params.s_is_m[i, self._params.omega[j]], j] 
                                        for i in self._params.I_j_2[j])) == 0
                            for j in self._params.J), "Constraint_5")

        # 6 -> checked Fehler irgendwo hier
        self.addConstrs((((gp.quicksum(self._X[i_p, self._params.alpha[j], j] 
                                    for j in self._params.J 
                                    for i_p in self._params.I_j_1[j]
                                    if (self._params.alpha[j] == s and self._params.t[i, s] >= self._params.r[j]))

                            + gp.quicksum(self._X[i_p, self._params.s_is_m[i_p, s], j] 
                                            for i_p in (set(self._params.I_is_m[i, s]) & set(self._params.I_s_p[s])) 
                                            for j in self._params.J_is[i_p, self._params.s_is_m[i_p, s]] 
                                            if self._params.omega[j] != s)

                            + gp.quicksum(self._X[i_p, self._params.s_is_m[i_p, s], j] 
                                            for i_p in (set(self._params.I_is_m[i,s]) & set(self._params.I_s_p[s]))
                                            for j in self._params.J_is[i_p, self._params.s_is_m[i_p, s]]
                                            if (self._params.omega[j] == s 
                                                and self._params.d[j] >= self._params.t[i, s]))

                            - gp.quicksum(self._X[i_p, s, j] 
                                        for i_p in self._params.I_is_m[i, s] 
                                        for j in self._params.J_is[i_p, s])) <= 100)

                                        for i in self._params.I for s in self._params.S_i[i]), "Constraint_6")

        # 7 and 8
        for (i, s, j) in indices_ISJ:
            if i in self._params.I_s_p[s] and (i, self._params.s_is_m[i, s], j) in self._X.keys():
                self.addConstr((self._X[i, s, j] 
                                - self._X[i, self._params.s_is_m[i, s], j] 
                                <= self._Y[i, s]), "Constraint_7")

            else:
                self.addConstr((self._X[i, s, j] 
                                <= self._Y[i, s]), "Constraint_8")
        
        # 11
        self.addConstrs((gp.quicksum(self._X[i, self._params.s_is_m[i, self._params.alpha[j]], j]
                                    for i in self._params.I_s_p[self._params.alpha[j]] 
                                    if j in self._params.J_is[i, self._params.s_is_m[i, self._params.alpha[j]]]) <= 0
                                    for j in self._params.J), "Constraint_11")
        
        # 12 
        self.addConstrs((gp.quicksum(self._X[i, self._params.omega[j], j] 
                                    for i in self._params.I_s_p[self._params.omega[j]] 
                                    if j in self._params.J_is[i, self._params.omega[j]]) <= 0
                            for j in self._params.J), "Constraint_12")
        
        # Save model to lp file
        if self._save_lp:
            self.write(f"output/model_{'3_5' if self._use_3_5 else 'no_3_5'}.lp")


    @staticmethod
    def sort_by_minutes(data):
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1][2]))
        return sorted_data


    def sort_parcels(self):
        for j in self._parcels.keys():
            self._parcels[j] = self.sort_by_minutes(self._parcels[j])

        try:
            self._parcels = dict(sorted(self._parcels.items(), key=lambda item: int(item[0][1:])))

        except ValueError:
            print("Sorting failed due to invalid parcel name. \n")


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
        for j in self._params.J:
            for i in self._params.I_j_1[j]:
                if (i, self._params.alpha[j], j) in self._X.keys():
                    max_parcels += self._X[i, self._params.alpha[j], j].x

        return max_parcels


    def calc_max_profit(self, 
                        print_level: int = 0):
        max_profit = 0

        if print_level > 0:
            print("Using ", end="")

        parcels = []
        for j in self._params.J:
            for i in self._params.I_j_1[j]:
                if (i, self._params.alpha[j], j) in self._X.keys() and self._X[i, self._params.alpha[j], j].x > 0:
                    if print_level > 0:
                        print(f"X[{i, self._params.alpha[j], j}]", end=", ")
                    parcels.append(j)
                    max_profit += self._params.p[j] * self._X[i, self._params.alpha[j], j].x

        for i in self._params.I:
            for s in self._params.S_i_p[i]:
                if self._Y[i, s].x > 0:
                    if print_level > 0:
                        print(f"Y[{i, s}]", end=", ")
                    max_profit -= self._params.f * self._Y[i, s].x

        if print_level > 0:
            print(f"Parcels: {parcels}")

        return max_profit


    def check_results(self, print_level: int = 0):
        if self.status == GRB.OPTIMAL:
            if print_level > 0:
                print("\nFound an optimal solution:")
            self._parcels = {}
            
            for (i, s, j), val in self._X.items():
                if val.x > 0:
                    if j not in self._parcels.keys():
                        self._parcels[j] = {}
                    try:
                        self._parcels[j][(s, self._params.s_is_p[i, s])] = [
                            i, self._params.t[i, s], self._params.t[i, self._params.s_is_p[i, s]]] # i, current time, next time
                    
                    except KeyError:
                        if print_level > 0:
                            print(f"X[{i, s, j}] is invalid!")
                        
            max_parcels = self.calc_max_parcels()
            max_profit = self.calc_max_profit(print_level=print_level)

            if print_level > 0:
                print(f"Number of parcels: {max_parcels}")
                print(f"Profit: {max_profit}\n")

            # Sort parcels
            self.sort_parcels()

            if print_level > 1:
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
        parcel_start = self._params.r[parcel_id]
        parcel_deadline = self._params.d[parcel_id]

        for idx, ((start, end), info) in enumerate(parcel_dict.items()):
            time_next = info[2]
            time_curr = info[1]

            if parcel_deadline < time_next or time_curr < parcel_start:
                return False
            
            if idx == 0:
                if start != self._params.alpha[parcel_id]:
                    return False

                last_end = end
                continue
            
            elif idx == len(parcel_dict) - 1:
                if end != self._params.omega[parcel_id]:
                    return False

            if start != last_end:
                return False

            last_end = end
        
        return True


    def print_parcels(self):
        for j in self._parcels.keys():
            print(f"--- Parcel {j} ---")
            print(f"Origin station: {self._params.alpha[j]}"\
                    f" @{self._params.r[j]}min\nTarget station: {self._params.omega[j]} @{self._params.d[j]}min\n")
            print(self._parcels[j], "\n")

            if self.check_validity(j, self._parcels[j]):
                print("Valid!\n")
            
            else:
                print("Invalid!\n")


    def check_parcels(self, seed, print_level: int = 0):
        count = 0
        for j, stations in self._parcels.items():
            if not self.check_validity(j, stations):
                if print_level == 1:
                    print(f"Parcel {j} is invalid!")
                    print("Seed:", seed)
                    print(f"Origin station: {self._params.alpha[j]} @{self._params.r[j]}min\n"\
                        f"Target station: {self._params.omega[j]} @{self._params.d[j]}min\n")
                    print(stations, "\n")
                count += 1

        return count


def test_seeds(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               print_level: int = 0,
               of: str = "MAX_PARCELS",
               number_of_seeds: int = 5,
               use_3_5: bool = True,
               seed: int = None,
               used_seeds: list = None):
    """
    Test 10 different seeds of the given parameters and print the results.
    """
    models = []

    if used_seeds is None:
        if seed is not None:
            random.seed(seed) # Make reproducible by setting with some seed if needed

        used_seeds = []
    
    while len(models) < number_of_seeds:
        if len(used_seeds) <= len(models):
            seed = random.randint(0, 1e5)

            while seed in used_seeds:
                seed = random.randint(0, 1e5)

            try:
                generator = InstanceGenerator(num_crowdshippers, 
                                            num_parcels, 
                                            entrainment_fee,
                                            seed=seed)
            # Seed is invalid
            except ValueError:
                continue

        else:
            seed = used_seeds[len(models)]
            generator = InstanceGenerator(num_crowdshippers, 
                                        num_parcels, 
                                        entrainment_fee,
                                        seed=seed)

        params = Parameters(**generator.return_kwargs())

        # MODEL
        model = CrowdshippingModel(params, of=of, use_3_5=use_3_5)

        if print_level < 3:
            model.setParam(GRB.Param.OutputFlag, 0)

        # OPTIMIZATION
        model.optimize()

        # PRINT
        model.check_results(print_level=print_level)

        count = model.check_parcels(seed, print_level)

        if print_level > 0:
            print("Done with seed:", seed)
            print(f"Invalid parcels: {count}\n")

        used_seeds.append(seed)
        models.append(model)

    return used_seeds, models


def test_seed(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               print_level: int = 0,
               of: str = "MAX_PARCELS",
               seed: int = 42,
               use_3_5: bool = True):
    """
    Test 10 different seeds of the given parameters and print the results.
    """
    generator = InstanceGenerator(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                seed=seed)

    params = Parameters(**generator.return_kwargs())

    # MODEL
    model = CrowdshippingModel(params, of=of, use_3_5=use_3_5)

    if print_level < 3:
        model.setParam(GRB.Param.OutputFlag, 0)

    # OPTIMIZATION
    model.optimize()

    # PRINT
    model.check_results(print_level=print_level)
    
    count = model.check_parcels(seed, print_level)

    print("Done with seed:", seed)
    print(f"Invalid parcels: {count}\n")

    return model
    

def check_minimalinstanz(path: str = "data/minimalinstanz.pkl"):
    """
    Load parameters from a specified file, update the minimal instance,
    and optimize the crowdshipper model using these parameters.

    Parameters
    ----------
    path : str, optional
        The path to the pickle file containing the parameters (default is "data/minimalinstanz.pkl").

    Executes
    --------
    1. Updates the minimal instance with the given parameters.
    2. Loads parameters from the specified file.
    3. Builds and optimizes the CrowdshipperModel using the loaded parameters.
    4. Prints detailed results of the optimization process.
    5. Prints the number of invalid parcels in the model.
    """
    update_minimalinstanz(path)

    params = Parameters.load(path)
    model = CrowdshippingModel(params)
    model.optimize()
    model.check_results(print_level=1)
    print("Invalid parcels:", model.check_parcels(seed=None))


def compare_3_5(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               of: str = "MAX_PARCELS",
               seed: int = None,
               number_of_seeds: int = 1,
               print_level: int = 0):
    if number_of_seeds == 1:
        if seed is None:
            seeds, model_3_5 = test_seeds(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=True,
                                number_of_seeds=number_of_seeds)
            
            seed = seeds[0]
            model_3_5 = model_3_5[0]
            
            model_no_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=False)
            
            print(f"--- Seed: {seed} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")

        else:
            model_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=True)
            
            model_no_3_5 = test_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                use_3_5=False)
            
            # with open(f"output/models_{seed}.pkl", "wb") as f:
            #     pickle.dump(model_3_5._params, f)
            
            print(f"--- Seed: {seed} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print()
                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")
            
            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")    
    
    else:
        used_seeds, models_3_5 = test_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              use_3_5=True)
        
        _, models_no_3_5 = test_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              use_3_5=False,
                              used_seeds=used_seeds)
        
        for model_3_5, model_no_3_5 in zip(models_3_5, models_no_3_5):
            print(f"--- Seed: {used_seeds[models_3_5.index(model_3_5)]} ---")
            if of == "MAX_PROFIT":
                of_3_5 = model_3_5.calc_max_profit(print_level=0)
                of_no_3_5 = model_no_3_5.calc_max_profit(print_level=0)

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")
            
            elif of == "MAX_PARCELS":
                of_3_5 = model_3_5.calc_max_parcels()
                of_no_3_5 = model_no_3_5.calc_max_parcels()

                print(f"3.5: {of_3_5}, no 3.5: {of_no_3_5}\n")

            else:
                raise ValueError("of must be MAX_PROFIT or MAX_PARCELS")


if __name__ == "__main__":
    # check_minimalinstanz()

    num_crowdshippers = 150
    num_parcels = 50
    entrainment_fee = 1
    of = "MAX_PROFIT"
    print_level = 0
    seed = None # 10751 # Seed to none for test_seeds if unproucable behaviour is needed 90016
    number_of_seeds = 20

    compare_3_5(num_crowdshippers, 
                num_parcels, 
                entrainment_fee,
                of=of,
                seed=seed,  
                number_of_seeds=number_of_seeds,
                print_level=print_level)
