import gurobipy as gp
from gurobipy import GRB
from classes import Parameters, InstanceGenerator, update_minimalinstanz
import random
import pickle


class CrowdshippingModel(gp.Model):
    def __init__(self, 
                 params: Parameters,
                 of: str = "MAX_PROFIT",
                 save_lp: bool = False,
                 use_ten: bool = False) -> None:
        """
        Initialize a CrowdshippingModel instance.

        Parameters
        ----------
        params : Parameters
            The parameters instance containing data for the model.
        of : str, optional
            The objective function to be used, either "MAX_PROFIT" or another valid option 
            (default is "MAX_PROFIT").
        save_lp : bool, optional
            If True, save the linear programming model (default is False).
        use_ten : bool, optional
            If True, utilize the 'ten' feature in the model (default is False).
        """
        super().__init__()
        self._use_ten = use_ten
        self._save_lp = save_lp
        self._params = params
        self._of = of

    
    @property
    def _of(self) -> str:
        """
        Get the objective function being used.

        Returns
        -------
        str
            The objective function, either "MAX_PROFIT" or another valid option.
        """
        return self.__of
    
    @_of.setter
    def _of(self, of: str) -> None:
        """
        Set the objective function to be used.

        Parameters
        ----------
        of : str
            The objective function, either "MAX_PROFIT" or another valid option.
        """
        self.__of = of
        self.build()


    def set_params(self, params: Parameters, 
                   of: str = None) -> None:
        """
        Set the parameters of the model.

        Parameters
        ----------
        params : Parameters
            The new parameters of the model.
        of : str, optional
            The new objective function, if different from the current one (default is None).
        """
        self._params = params
        if of is not None:
            self._of = of
        
        else:
            self.build()


    def build(self) -> None:
        # VARIABLES
        """
        Build the optimization model for the crowdshipping problem.

        This method constructs the decision variables, objective function, and constraints
        for the optimization model based on the provided parameters. It supports multiple
        objective functions, such as maximizing profit or the number of parcels delivered.
        Various constraints are added to ensure the validity of the problem, including 
        limitations on parcel pick-up and delivery, crowdshipper availability, and time 
        windows. Optionally, the model can include an additional constraint set and save 
        the model to a file.

        Raises
        ------
        ValueError
            If the objective function specified is not recognized.
        """
        indices_ISJ = []
        indices_IS = []
        for i in self._params.I:
            for s in self._params.S_i_p[i]:
                if (i, s) not in indices_IS:
                    indices_IS.append((i, s))

                for j in self._params.J_is[i, s]:
                    indices_ISJ.append((i, s, j))               

        self._X = self.addVars(indices_ISJ, vtype=GRB.BINARY, name="X") # 11
        self._Y = self.addVars(indices_IS, vtype=GRB.CONTINUOUS, name="Y", lb=0) # 12

        # OBJECTIVE
        # MAX-PROFIT
        # 1 
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
        # 1
        self.addConstrs((gp.quicksum(self._X[i, s, j] 
                                     for j in self._params.J_is[i, s]) <= 1
                        for (i, s) in indices_IS), "Constraint_1")

        # 2 
        self.addConstrs((gp.quicksum(self._X[i, s, j] 
                                    for i in self._params.I 
                                    if j in self._params.J_is[i, s]) <= 1 
                                for j in self._params.J 
                                for s in self._params.S), "Constraint_2")

        # 3 
        self.addConstrs((self._X[i, s, j] 
                         - gp.quicksum(self._X[i_p, self._params.s_is_p[i, s], j] 
                                        for i_p in self._params.I_is_p[i, self._params.s_is_p[i, s]]
                                        if self._params.s_is_p[i, s] in self._params.S_i_p[i_p]
                                        and j in self._params.J_is[i_p, self._params.s_is_p[i, s]]) <= 0
                            for (i, s, j) in indices_ISJ 
                            if self._params.s_is_p[i, s] != self._params.omega[j]), "Constraint_3")

        # 4 
        self.addConstrs(((gp.quicksum(self._X[i, self._params.alpha[j], j] 
                                    for i in self._params.I_j_1[j])
                            - gp.quicksum(self._X[i, self._params.s_is_m[i, self._params.omega[j]], j] 
                                        for i in self._params.I_j_2[j])) == 0
                            for j in self._params.J), "Constraint_4")
        
        # 5 
        self.addConstrs((((gp.quicksum(self._X[i_p, s, j] 
                                    for j in self._params.J 
                                    for i_p in self._params.I_j_1[j]
                                    if (self._params.alpha[j] == s and self._params.t[i, s] >= self._params.r[j]))

                            + gp.quicksum(self._X[i_p, self._params.s_is_m[i_p, s], j] 
                                            for i_p in (set(self._params.I_is_m[i,s]) & set(self._params.I_s_p[s]))
                                            for j in self._params.J_is[i_p, self._params.s_is_m[i_p, s]]
                                            if (self._params.omega[j] != s 
                                                or self._params.d[j] >= self._params.t[i, s]))

                            - gp.quicksum(self._X[i_p, s, j] 
                                        for i_p in self._params.I_is_m[i, s] 
                                        for j in self._params.J_is[i_p, s])) <= self._params.l[s])

                                        for i in self._params.I for s in self._params.S_i[i]), "Constraint_5")
        # 6, 7
        for (i, s, j) in indices_ISJ:
            if (i in self._params.I_s_p[s] 
                and self._params.t[i, self._params.s_is_m[i, s]] >= self._params.r[j]):
                self.addConstr((self._X[i, s, j] 
                                - self._X[i, self._params.s_is_m[i, s], j] 
                                <= self._Y[i, s]), "Constraint_6")

            else:
                self.addConstr((self._X[i, s, j] 
                                <= self._Y[i, s]), "Constraint_7")
        
        # 8
        self.addConstrs((gp.quicksum(self._X[i, self._params.s_is_m[i, self._params.alpha[j]], j]
                                    for i in self._params.I_s_p[self._params.alpha[j]] 
                                    if j in self._params.J_is[i, self._params.s_is_m[i, self._params.alpha[j]]]) <= 0
                                    for j in self._params.J), "Constraint_8")
        
        # 9 
        self.addConstrs((gp.quicksum(self._X[i, self._params.omega[j], j] 
                                    for i in self._params.I_s_p[self._params.omega[j]] 
                                    if j in self._params.J_is[i, self._params.omega[j]]) <= 0
                            for j in self._params.J), "Constraint_9")
        
        # 10
        # Pakets can only be transported if they are at the station
        # they are transported from at the right time
        if self._use_ten:
            self.addConstrs((self._X[i, s, j] <= (gp.quicksum(self._X[i_p, self._params.s_is_m[i_p, s], j] 
                                for i_p in self._params.I_s_p[s]
                                if self._params.t[i_p, self._params.s_is_m[i_p, s]] >= self._params.r[j]
                                and self._params.t[i_p, s] <= self._params.t[i, s])) 
                            for i in self._params.I 
                            for s in self._params.S_i[i] 
                            for j in self._params.J_is[i, s] 
                            if s != self._params.alpha[j]), "Constraint_10")
        
        # Save model to lp file
        if self._save_lp:
            self.write(f"output/model.lp")


    @staticmethod
    def sort_by_minutes(data) -> dict:
        """
        Sort a dictionary by the third element of its values.

        This static method takes a dictionary where each value is a 
        sequence (e.g., a tuple or list) and sorts the dictionary by 
        the third element of each value.

        Parameters
        ----------
        data : dict
            A dictionary where each value is a sequence with at least 
            three elements.

        Returns
        -------
        dict
            A dictionary sorted by the third element of the value sequences.
        """
        sorted_data = dict(sorted(data.items(), key=lambda item: item[1][2]))
        return sorted_data


    def sort_parcels(self) -> None:
        """
        Sorts the parcels dictionary by the parcel number.

        This method sorts the parcels dictionary (self._parcels) by the parcel number.
        The parcels dictionary is a dictionary where each key is a parcel and each value is a dictionary of stations
        along with the corresponding time and crowdshipper for each station.

        The sorting is done in two steps. First, the stations of each parcel are sorted by the time they are visited.
        Then, the parcels are sorted by their parcel number.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for j in self._parcels.keys():
            self._parcels[j] = self.sort_by_minutes(self._parcels[j])

        try:
            self._parcels = dict(sorted(self._parcels.items(), key=lambda item: int(item[0][1:])))

        except ValueError:
            print("Sorting failed due to invalid parcel name. \n")


    def calc_max_parcels(self, eps: float = 1e-3) -> int:
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
                if self._params.check_time(i, self._params.alpha[j], j) and self._X[i, self._params.alpha[j], j].x > eps:
                    max_parcels += 1

        return max_parcels


    def calc_max_profit(self, 
                        print_level: int = 0,
                        eps: float = 1e-3) -> float:
        """
        Calculate the maximum profit from the crowdshipping model.

        This function computes the maximum profit by evaluating the decision variables
        for parcels and crowdshippers. It adds the profit for each parcel that can be 
        moved (i.e., when the decision variable X[i, alpha[j], j] is greater than eps) 
        and subtracts the fixed costs for each crowdshipper's visit to a station 
        (i.e., when the decision variable Y[i, s] is greater than eps).

        Parameters
        ----------
        print_level : int, optional
            Determines the level of detail in the printed output. If greater than 0, 
            prints the decision variable being used.
        eps : float, optional
            The threshold for considering a decision variable as active (greater 
            than this value).

        Returns
        -------
        float
            The maximum profit calculated from the decision variables.
        """
        max_profit = 0

        if print_level > 0:
            print("Using ", end="")

        parcels = []
        for j in self._params.J:
            for i in self._params.I_j_1[j]:
                if self._X[i, self._params.alpha[j], j].x > eps:
                    if print_level > 0:
                        print(f"X[{i, self._params.alpha[j], j}]", end=", ")
                    parcels.append(j)
                    max_profit += self._params.p[j]             

        for i in self._params.I:
            for s in self._params.S_i_p[i]:
                if self._Y[i, s].x > eps:
                    if print_level > 0:
                        print(f"Y[{i, s}]", end=", ")
                    max_profit -= self._params.f

        if print_level > 0:
            print(f"\n\nParcels: {parcels}")

        return max_profit


    def check_results(self, 
                      print_level: int = 1, 
                      eps: float = 1e-3) -> None:
        """
        Checks the results of the model and prints a summary of the solution.

        Parameters
        ----------
        print_level : int, optional
            Determines the level of detail in the printed output. If greater than 0, 
            prints the number of parcels and the profit of the solution. If greater 
            than 1, also prints all the parcels and their respective stations and 
            times.
        eps : float, optional
            The threshold for considering a decision variable as active (greater 
            than this value).

        Returns
        -------
        None
        """
        if self.status == GRB.OPTIMAL:
            if print_level > 0:
                print("\nFound an optimal solution:")
            self._parcels = {}
            
            for (i, s, j), val in self._X.items():
                if val.x > eps:
                    if j not in self._parcels.keys():
                        self._parcels[j] = {}
                    
                    self._parcels[j][(s, self._params.s_is_p[i, s])] = [
                        i, self._params.t[i, s], self._params.t[i, self._params.s_is_p[i, s]]]
                        
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
                       parcel_dict: dict) -> bool:
        """
        Checks if the given parcel is valid.

        Parameters
        ----------
        parcel_id : str
            ID of the parcel
        parcel_dict : dict
            Dictionary of the parcel, where each key is a tuple (start, end) 
            representing the start and end stations of a part of the parcel, 
            and each value is a list [crowdshipper_id, start_time, end_time] 
            representing the crowdshipper's id, start time and end time of the part of the parcel.

        Returns
        -------
        bool
            True if the parcel is valid, False otherwise.
        """
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


    def print_parcels(self) -> None:
        """
        Prints the parcels and their corresponding stations and times.

        This method prints out the information of each parcel, including the origin station, target station,
        and the stations and times of each part of the parcel. It also checks if the parcel is valid
        and prints out "Valid!" if it is, or "Invalid!" if it is not.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for j in self._parcels.keys():
            print(f"--- Parcel {j} ---")
            print(f"Origin station: {self._params.alpha[j]}"\
                    f" @{self._params.r[j]}min\nTarget station: {self._params.omega[j]} @{self._params.d[j]}min\n")
            print(self._parcels[j], "\n")

            if self.check_validity(j, self._parcels[j]):
                print("Valid!\n")
            
            else:
                print("Invalid!\n")


    def check_parcels(self, seed: int, 
                      print_level: int = 0) -> int:
        """
        Checks all the parcels and counts the number of invalid ones.

        Parameters
        ----------
        seed : int
            The seed used to generate the instance.
        print_level : int, optional
            The level of detail in the printed output. If greater than 0,
            prints the information of the invalid parcels. The default is 0.

        Returns
        -------
        int
            The number of invalid parcels.

        Notes
        -----
        This method checks if each parcel is valid by calling the method
        `check_validity` and increments a counter if it is not valid. If
        `print_level` is greater than 0, it also prints the information of
        the invalid parcel.
        """
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
    

    def return_x_y(self, 
                   eps: float = 1e-3) -> tuple:
        """
        Return the decision variables X and Y as dictionaries.

        Parameters
        ----------
        eps : float, optional
            The threshold for considering a decision variable as active (greater than this value).
            The default is 1e-3.

        Returns
        -------
        tuple
            A tuple of two dictionaries. The first dictionary contains the values of X[i, s, j]
            for i in I, s in S_i and j in J_is[i, s]. The second dictionary contains the values of
            Y[i, s] for i in I and s in S_i_p[i].
        """
        X = {}
        Y = {}

        for (i, s, j), val in self._X.items():
            if val.x > eps:
                X[(i, s, j)] = 1

            else:
                X[(i, s, j)] = 0

        for (i, s), val in self._Y.items():
            if val.x > eps:
                Y[(i, s)] = 1

            else:
                Y[(i, s)] = 0

        return X, Y


def run_seed(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               print_level: int = 0,
               of: str = "MAX_PARCELS",
               seed: int = 42,
               save_lp: bool = False,
               use_ten: bool = True,
               load_from_file: str = None,
               return_model: bool = True,
               mode = "of") -> CrowdshippingModel | dict:
    """
    Run the model with the given parameters and options.

    Parameters
    ----------
    num_crowdshippers : int
        The number of crowdshippers.
    num_parcels : int
        The number of parcels.
    entrainment_fee : int
        The entrainment fee.
    print_level : int, optional
        The print level. 0: nothing, 1: invalid parcels, 2: all variables, 3: full log.
        The default is 0.
    of : str, optional
        The objective function. "MAX_PARCELS" or "MAX_PROFIT".
        The default is "MAX_PARCELS".
    seed : int, optional
        The seed. The default is 42.
    save_lp : bool, optional
        Whether to save the LP file. The default is False.
    use_ten : bool, optional
        Whether to use the 10% rule. The default is True.
    load_from_file : str, optional
        The file to load the parameters from. The default is None.
    return_model : bool, optional
        Whether to return the model. The default is True.
    mode : str, optional
        The mode. "of" or "10". The default is "of".

    Returns
    -------
    model : CrowdshippingModel
        The model.
    results : dict
        The results.
    """
    if not return_model:
        if mode == "of":
            results = {of: {}}

        elif mode == "10":
            results = {"10" if use_ten else "no_10": {}}

    if load_from_file is not None:
        params = Parameters.load(load_from_file)
        
    else:
        generator = InstanceGenerator(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                seed=seed)
        params = Parameters(**generator.return_kwargs())

    # MODEL
    model = CrowdshippingModel(params, of=of, save_lp=save_lp, use_ten=use_ten)

    if print_level < 3:
        model.setParam(GRB.Param.OutputFlag, 0)

    model.setParam(GRB.Param.TimeLimit, 1200)

    # OPTIMIZATION
    model.optimize()

    # PRINT
    model.check_results(print_level=print_level)
    
    count = model.check_parcels(seed, print_level)

    print("Done with seed:", seed)
    print(f"Invalid parcels: {count}\n")

    # with open(f"output/params_{seed}.pkl", "wb") as f:
    #     pickle.dump(model._params, f)

    # with open(f"problem_instance/results_{seed}.pkl", "wb") as f:
    #     pickle.dump(model.return_x_y(), f)

    if return_model:
        return model

    else:
        results = add_results(results=results,
                                  seed=seed,
                                  model=model,
                                  of=of,
                                  use_ten=use_ten,
                                  mode=mode)
        return results


def run_seeds(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int, 
               print_level: int = 0,
               of: str = "MAX_PARCELS",
               number_of_seeds: int = 5,
               seed: int = None,
               used_seeds: list = None,
               use_ten: bool = True,
               load_from_file: str = None,
               return_model: bool = False,
               mode: str = "of") -> list[CrowdshippingModel] | CrowdshippingModel | dict:
    """
    Runs a specified number of seeds and returns either a list of models or
    a dictionary of results.

    Parameters
    ----------
    num_crowdshippers : int
        Number of crowdshippers
    num_parcels : int
        Number of parcels
    entrainment_fee : int
        Fee for entrainment
    print_level : int, optional
        Print level, by default 0
    of : str, optional
        Objective function, by default "MAX_PARCELS"
    number_of_seeds : int, optional
        Number of seeds, by default 5
    seed : int, optional
        Seed, by default None
    used_seeds : list, optional
        List of used seeds, by default None
    use_ten : bool, optional
        Whether to use 10, by default True
    load_from_file : str, optional
        File to load parameters from, by default None
    return_model : bool, optional
        Whether to return a model, by default False
    mode : str, optional
        Mode, either "of" or "10", by default "of"

    Returns
    -------
    list[CrowdshippingModel] | CrowdshippingModel | dict
        Either a list of models or a dictionary of results
    """
    if return_model:
        models = []

    else:
        if mode == "of":
            results = {of: {}}

        elif mode == "10":
            results = {"10" if use_ten else "no_10": {}}

    if number_of_seeds == 1 and seed is not None:
        model = run_seed(num_crowdshippers,
                    num_parcels,
                    entrainment_fee,
                    print_level=print_level,
                    of=of,
                    seed=seed,
                    use_ten=use_ten,
                    load_from_file=load_from_file)
        
        return model

    if used_seeds is None:
        if seed is not None:
            random.seed(seed) # Make reproducible by setting with some seed if needed

        used_seeds = []
    
    count = 0
    while count < number_of_seeds:
        if len(used_seeds) <= count:
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
            seed = used_seeds[count]
            generator = InstanceGenerator(num_crowdshippers, 
                                        num_parcels, 
                                        entrainment_fee,
                                        seed=seed)

        params = Parameters(**generator.return_kwargs())

        # MODEL
        model = CrowdshippingModel(params, of=of, use_ten=use_ten)

        if print_level < 3:
            model.setParam(GRB.Param.OutputFlag, 0)

        model.setParam(GRB.Param.TimeLimit, 1200)

        # OPTIMIZATION
        model.optimize()

        # PRINT
        model.check_results(print_level=print_level)

        parcel_count = model.check_parcels(seed, print_level)

        if print_level > 0:
            print("Done with seed:", seed)
            print(f"Invalid parcels: {parcel_count}\n")

        used_seeds.append(seed)

        if return_model:
            models.append(model)

        else:
            results = add_results(results=results,
                                  seed=seed,
                                  model=model,
                                  of=of,
                                  use_ten=use_ten,
                                  mode=mode)
            
        # with open(f"problem_instance/results_{seed}.pkl", "wb") as f:
        #     pickle.dump(model.return_x_y(), f)

        count += 1

    if return_model:
        return used_seeds, models

    else:
        return used_seeds, results

    

def check_minimalinstanz(path: str = "data/minimalinstanz.pkl", 
                         print_level: int = 0) -> None:
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
    model.check_results(print_level=print_level)
    print("Invalid parcels:", model.check_parcels(seed=None))
            

def compare_of(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int,
               print_level: int = 0,
               number_of_seeds: int = 1,
               seed: int = None,
               load_from_file: str = None) -> tuple:
    """
    Compare the performance of the crowdshipping model using different objectives.

    This function runs the crowdshipping model with two different objectives
    ("MAX_PROFIT" and "MAX_PARCELS") and compares their results. It can handle
    either a single seed or multiple seeds and prints detailed results for each
    seed used.

    Parameters
    ----------
    num_crowdshippers : int
        The number of crowdshippers.
    num_parcels : int
        The number of parcels.
    entrainment_fee : int
        The entrainment fee.
    print_level : int, optional
        The level of detail to print. Default is 0.
    number_of_seeds : int, optional
        The number of seeds to use. Default is 1.
    seed : int, optional
        The specific seed to use. If None, seeds are generated. Default is None.
    load_from_file : str, optional
        Path to the file to load the model parameters from. Default is None.

    Returns
    -------
    results : dict
        A dictionary containing the results for each objective and seed.
    seed or return_seeds : int or list
        The seed(s) used in the simulation.
    """
    if number_of_seeds == 1:
        if seed is None:
            seeds, results = run_seeds(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of="MAX_PROFIT",
                                return_model=False,
                                seed=seed,
                                number_of_seeds=number_of_seeds)
            
            seed = seeds[0]
            results_ext = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of="MAX_PARCELS",
                                seed=seed,
                                return_model=False,)

            results.update(results_ext)
            
            result_max_profit = results["MAX_PROFIT"][seed]
            result_max_parcels = results["MAX_PARCELS"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("MAX_PROFIT", result_max_profit)
            print("MAX_PARCELS", result_max_parcels)
            print("\n")

            return results, seed

        else:
            results = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of="MAX_PROFIT",
                                seed=seed,
                                load_from_file=load_from_file,
                                return_model=False,)
            
            results_ext = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of="MAX_PARCELS",
                                seed=seed,
                                load_from_file=load_from_file,
                                return_model=False,)

            results.update(results_ext)
            
            result_max_profit = results["MAX_PROFIT"][seed]
            result_max_parcels = results["MAX_PARCELS"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("MAX_PROFIT", result_max_profit)
            print("MAX_PARCELS", result_max_parcels)
            print("\n")

            return results    
    
    else:
        used_seeds, results = run_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of="MAX_PROFIT",
                              number_of_seeds=number_of_seeds)
        
        _, results_ext = run_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of="MAX_PARCELS",
                              number_of_seeds=number_of_seeds,
                              used_seeds=used_seeds)

        results.update(results_ext)
        
        return_seeds = used_seeds[:len(used_seeds)//2]

        for seed in return_seeds:
            result_max_profit = results["MAX_PROFIT"][seed]
            result_max_parcels = results["MAX_PARCELS"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("MAX_PROFIT", result_max_profit)
            print("MAX_PARCELS", result_max_parcels)
            print("\n")

        print("--- Summary ---")
        print(results, return_seeds)

        with open(f"output/results_of.pkl", "wb") as f:
            pickle.dump((results, return_seeds), f)

        return results, return_seeds
    

def add_results(results: dict, seed: int, 
                model: CrowdshippingModel, 
                of: str, use_ten: bool,
                  mode = "10") -> dict:
    """
    Adds results of a model run to a dictionary.

    Parameters
    ----------
    results : dict
        The dictionary to add the results to.
    seed : int
        The seed used for the model run.
    model : CrowdshippingModel
        The model to get the results from.
    of : str
        The objective function used. "MAX_PROFIT" or "MAX_PARCELS".
    use_ten : bool
        Whether the 10% rule was used.
    mode : str, optional
        The mode of the experiment. "10" or "of". The default is "10".

    Returns
    -------
    dict
        The updated dictionary with the results added.
    """
    if mode == "10":
        model_type = "10" if use_ten else "no_10"

        results[model_type][seed] = {}
        results[model_type][seed]["profit"] = model.calc_max_profit(print_level=0)
        results[model_type][seed]["parcels"] = model.calc_max_parcels()
        results[model_type][seed]["runtime"] = round(model.runtime, 3)

    elif mode == "of":
        results[of][seed] = {}

        if of == "MAX_PROFIT":
            results["MAX_PROFIT"][seed]["profit"] = model.calc_max_profit(print_level=0)
            results["MAX_PROFIT"][seed]["parcels"] = model.calc_max_parcels()

        elif of == "MAX_PARCELS":
            results["MAX_PARCELS"][seed]["profit"] = model.calc_max_profit(print_level=0)
            results["MAX_PARCELS"][seed]["parcels"] = model.calc_max_parcels()

        results[of][seed]["runtime"] = round(model.runtime, 3)

    else:
        raise ValueError("Mode not in available modes. ")

    return results
    

def compare_10(num_crowdshippers: int, 
               num_parcels: int, 
               entrainment_fee: int,
               print_level: int = 0,
               of: str = "MAX_PARCELS",
               number_of_seeds: int = 1,
               seed: int = None,
               load_from_file: str = None) -> tuple:
    """
    Compares the performance of the crowdshipping model with and without the 10% rule.

    Parameters
    ----------
    num_crowdshippers : int
        The number of crowdshippers.
    num_parcels : int
        The number of parcels.
    entrainment_fee : int
        The entrainment fee.
    print_level : int, optional
        The level of detail to print. Default is 0.
    of : str, optional
        The objective function. "MAX_PROFIT" or "MAX_PARCELS". Default is "MAX_PARCELS".
    number_of_seeds : int, optional
        The number of seeds to use. Default is 1.
    seed : int, optional
        The specific seed to use. If None, seeds are generated. Default is None.
    load_from_file : str, optional
        Path to the file to load the model parameters from. Default is None.

    Returns
    -------
    results : dict
        A dictionary containing the results for each seed and mode.
    seed or return_seeds : int or list
        The seed(s) used in the simulation.
    """
    if number_of_seeds == 1:
        if seed is None:
            seeds, results = run_seeds(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                number_of_seeds=number_of_seeds,
                                return_model=False,
                                mode="10",
                                use_ten=True)
            
            seed = seeds[0]
            results_ext = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                return_model=False,
                                mode="10",
                                use_ten=False)

            results.update(results_ext)
            results["objective"] = of

            result_max_profit = results["10"][seed]
            result_max_parcels = results["no_10"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("10", result_max_profit)
            print("no_10", result_max_parcels)
            print("\n")

            return results, seed

        else:
            results = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                load_from_file=load_from_file,
                                return_model=False,
                                mode="10",
                                use_ten=True)
            
            results_ext = run_seed(num_crowdshippers, 
                                num_parcels, 
                                entrainment_fee,
                                print_level=print_level,
                                of=of,
                                seed=seed,
                                load_from_file=load_from_file,
                                return_model=False,
                                mode="10",
                                use_ten=False)

            results.update(results_ext)
            results["objective"] = of
            
            result_max_profit = results["10"][seed]
            result_max_parcels = results["no_10"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("10", result_max_profit)
            print("no_10", result_max_parcels)
            print("\n")

            return results 
    
    else:
        used_seeds, results = run_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              use_ten=True,
                              mode="10")
        
        _, results_ext = run_seeds(num_crowdshippers, 
                              num_parcels, 
                              entrainment_fee,  
                              print_level=print_level,
                              of=of,
                              number_of_seeds=number_of_seeds,
                              used_seeds=used_seeds,
                              use_ten=False,
                              mode="10")

        results.update(results_ext)
        results["objective"] = of
        
        return_seeds = used_seeds[:len(used_seeds)//2]

        for seed in return_seeds:
            result_max_profit = results["10"][seed]
            result_max_parcels = results["no_10"][seed]
            
            print(f"--- Seed: {seed} ---")
            print("10", result_max_profit)
            print("no_10", result_max_parcels)
            print("\n")

        print("--- Summary ---")
        print(results, return_seeds)

        with open(f"output/results_10.pkl", "wb") as f:
            pickle.dump((results, return_seeds), f)

        return results, return_seeds
            

if __name__ == "__main__":
    num_crowdshippers = 100
    num_parcels = 40
    entrainment_fee = 1
    of = "MAX_PROFIT"
    print_level = 1
    seed = None # Seed to none for test_seeds if unproucable behaviour is needed
    number_of_seeds = 1
    load_from_file = None

    run_seeds(num_crowdshippers, 
                num_parcels, 
                entrainment_fee,
                print_level=print_level,
                of=of,
                number_of_seeds=number_of_seeds,
                seed=seed,
                load_from_file=load_from_file)
    