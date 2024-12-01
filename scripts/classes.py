from data_retrieval import generate
import random
import random
import pickle
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class Parameters:
    def __init__(self, **kwargs) -> None:
        """
        Initialize Parameters with provided keyword arguments.

        This constructor initializes various attributes related to crowdshipping,
        including sets of crowdshippers, parcels, stations, and their associated
        parameters such as release times, deadlines, and postal charges. It also 
        sorts stations by visit time and generates subsets based on the provided
        data.

        Parameters
        ----------
        kwargs : dict
            A dictionary of keyword arguments containing:
            - seed : int
                Random seed for reproducibility.
            - I : list
                List of crowdshippers.
            - J : list
                List of parcels.
            - S : list
                List of stations.
            - alpha : dict
                Dictionary of starting stations for each parcel.
            - omega : dict
                Dictionary of destination stations for each parcel.
            - r : dict
                Dictionary of release times for each parcel.
            - d : dict
                Dictionary of deadline times for each parcel.
            - p : dict
                Dictionary of postal charges for each parcel.
            - t : dict
                Dictionary of times when each crowdshipper is at each station.
            - f : float
                Fixed entrainment fee.
            - l : dict, optional
                Dictionary of locker capacities, default is 20 for each station.
        """
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
            self.l = {s: 20 for s in self.S}

        # generate subsets based on sets and parameters
        self.generate_subsets()


    def sort_stations(self) -> dict:
        """
        Sort stations for each crowdshipper by the time they are visited.

        This function processes the `t` dictionary, which contains the times at which 
        each crowdshipper `i` visits each station `s`. It sorts the stations for each 
        crowdshipper based on the visit time and returns a dictionary where each key 
        is a crowdshipper and the value is a list of stations in the order they are visited.

        Returns
        -------
        dict
            A dictionary mapping each crowdshipper to a list of stations sorted by visit time.
        """
        sorted_stations = {}

        for (i, s), _ in sorted(self.t.items(), key=lambda item: item[1]):
            if i not in sorted_stations:
                sorted_stations[i] = []

            sorted_stations[i].append(s)

        return sorted_stations
    

    def generate_subsets(self) -> None:
        # set of stations visited by crowdshipper i
        """
        Generate all the subsets of the model, such as the set of stations visited by crowdshipper i, 
        the set of crowdshippers visiting station s, the set of crowdshippers who can pick up/deliver parcel j 
        at station s, the set of parcels that can be picked up/delivered by crowdshipper i at station s, 
        and the set of crowdshippers who can deliver/pick up a non finished entrainment with parcel j from station s.

        These subsets are used to generate the constraints of the model and to generate the decision variables X[i, s, j].
        """
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
        self.I_j_2 = {j: list() for j in self.J}

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

        # station visited by crowdshipper i in I_s_p immediately before/after station s
        self.s_is_m = {(i, s) : self.get_last_station(i, s) 
                        for s in self.S for i in self.I_s_p[s]}
        
        self.s_is_p = {(i, s) : self.get_next_station(i, s) 
                       for i in self.I for s in self.S_i_p[i]}
        
        for (i, s), t in self.t.items():
            for j in self.J:
                if self.alpha[j] == s:
                    if (s != self.sorted_stations[i][-1] 
                        and self.t[i, s] >= self.r[j]# ): 
                        and self.t[i, self.s_is_p[i, s]] <= self.d[j]):
                        if i not in self.I_j_1[j]:
                            self.I_j_1[j].append(i)

                if self.omega[j] == s:
                    if (s != self.sorted_stations[i][0] 
                        and self.t[i, s] <= self.d[j] # ): 
                        and self.t[i, self.s_is_m[i, s]] >= self.r[j]):
                        if i not in self.I_j_2[j]:
                            self.I_j_2[j].append(i)

        # J_is is a subset of J which contains the parcels for a combination of crowdshipper i and station s
        # which can be delivered to the next station (being at stations at r or later and at s+1 at d or earlier) 
        # thus generating valid X
        for i in self.I:
            for s in self.S:
                if s in self.S_i_p[i]:
                    self.J_is[i, s] = [j for j in self.J 
                                    if self.check_time(i, s, j)]
                else:
                    self.J_is[i, s] = []

        # generate subsets for non finished entrainments
        for s in self.S:
            for i in self.I_s[s]:
                self.I_is_m[i, s] = [c for c in self.I_s[s] 
                                     if self.t[c, s] <= self.t[i, s]]
                
                self.I_is_p[i, s] = [c for c in self.I_s[s] 
                                     if self.t[c, s] >= self.t[i, s]]
    

    def check_time(self, i: str, s: str, j: str) -> bool:
        """
        Check if the crowdshipper can deliver the parcel within the specified time window.

        This function checks whether the crowdshipper `i` can deliver parcel `j` 
        at station `s` by verifying if the visit time at station `s` is within the 
        release time and deadline of parcel `j`.

        Parameters
        ----------
        i : str
            Identifier of the crowdshipper.
        s : str
            Station identifier where the crowdshipper is supposed to deliver the parcel.
        j : str
            Identifier of the parcel.

        Returns
        -------
        bool
            True if the crowdshipper can deliver the parcel on time, False otherwise.
        """
        return self.t[i, s] >= self.r[j] and self.t[i, self.s_is_p[i, s]] <= self.d[j]


    def get_last_station(self, i: int, s: str) -> str:
        """
        Return the station visited by crowdshipper i immediately before station s.

        Parameters
        ----------
        i : int
            Identifier of the crowdshipper.
        s : str
            Station identifier.

        Returns
        -------
        str
            Station identifier visited by the crowdshipper before station s.
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


    def __repr__(self) -> str:
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


class InstanceGenerator:
    def __init__(self, 
                 num_crowdshippers: int, 
                 num_parcels: int, 
                 entrainment_fee: int, seed: int = 42) -> None:
        """
        Initialize an instance of the InstanceGenerator class.

        Parameters
        ----------
        num_crowdshippers : int
            The number of crowdshippers in the instance.
        num_parcels : int
            The number of parcels in the instance.
        entrainment_fee : int
            The entrainment fee.
        seed : int, optional
            The random seed to use for generating the instance. The default is 42.

        Notes
        -----
        The instance is generated by creating a graph with the given number of
        crowdshippers, parcels and stations. The edges of the graph are the
        possible routes between the stations. The travel times are loaded from
        a file and the entrainment fee is set to the given value. The parcels
        are randomly assigned to the crowdshippers and the stations are randomly
        assigned as start and end stations for the crowdshippers.

        The instance is represented as a dictionary with the following keys:
        - I: The set of crowdshippers.
        - J: The set of parcels.
        - S: The set of stations.
        - l: A dictionary with the number of parcels each station can hold.
        - p: A dictionary with the profit of each parcel.
        - f: The entrainment fee.
        - travel_times: A dictionary with the travel times between each pair of
        stations.
        - start: The start station of each crowdshipper.
        - end: The end station of each crowdshipper.
        - parcels: A dictionary with the parcels assigned to each crowdshipper.
        """
        self.num_crowdshippers = num_crowdshippers
        self.num_parcels = num_parcels
        self.seed = seed
        self.max_time = 600

        # Setting random seeds
        np.random.seed(self.seed)
        random.seed(self.seed)

        with open("data/stations_data/lines.pkl", "rb") as f:
            self.lines, self.stations = pickle.load(f)

        # self.lines, self.stations = generate()
        self.travel_times = self.read_travel_times()

        self.I = [f"C{i}" for i in range(1, self.num_crowdshippers + 1)]
        self.J = [f"P{j}" for j in range(1, self.num_parcels + 1)]
        self.S = list(self.stations.keys())
        
        self.generate_graph()
        
        self.l = {}
        for s in self.S:
            if s in ["Kroepcke", "Aegidientorplatz", "Hauptbahnhof"]:
                self.l[s] = np.random.binomial(len(self.J)//2, 0.5)
            else:
                self.l[s] = np.random.binomial(len(self.J)//4, 0.5)

            if self.l[s] == 0:
                self.l[s] = 1

        # self.l = {s: random.randint(1, 4) for s in self.S}
        # self.p = {p: random.randint(1, 10) for p in self.J} 

        self.p = {p: 5 for p in self.J} 
        self.f = entrainment_fee

        self.init_parcels()
        self.init_crowdshippers()


    def read_travel_times(self) -> dict:
        """
        Generate travel times between stations.

        The travel times are randomly generated for each pair of directly
        connected stations. The distribution of travel times is given by
        `choices`.

        Returns
        -------
        connections : dict
            A dictionary mapping each line to a dictionary mapping each pair of
            stations to the travel time between the two stations.
        """
        with open("data/stations_data/connections_full.pkl", "rb") as f:
            connections = pickle.load(f)
                
        return connections    


    def init_crowdshippers(self) -> None:
        """
        Initialize the crowdshippers by randomly selecting a start and end
        station for each of them. The start time, end time, and time of visit
        for each station in the path between the start and end stations are then
        randomly generated. Time is given in minutes from start of day (0 -> 00:00) to end of day (1439 -> 23:59); thus in [0, 1440).

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self.alpha_crowd = {c: random.choice(self.S) for c in self.I}
        self.omega_crowd = {c: random.choice([s for s in self.S if s != self.alpha_crowd[c]]) for c in self.I}

        choices = {1: 0.4, # early movers
                   2: 0.2, # working hours
                   3: 0.4} # late hours
        self.r_crowd = {}
        self.d_crowd = {}
        self.t = {}

        for i in self.I:
            starting_station = self.alpha_crowd[i]
            target_station = self.omega_crowd[i]
            path, time_taken = self.find_path(starting_station, target_station)

            draw = int(np.random.choice(list(choices.keys()), 1, p=list(choices.values()))[0])

            if draw == 1:
                start = random.randint(1, 0.3*self.max_time)

            elif draw == 2:
                start = random.randint(0.3*self.max_time, 0.6*self.max_time)

            elif draw == 3:
                start = random.randint(0.6*self.max_time, 0.9*self.max_time)

            self.r_crowd[i] = start
            self.d_crowd[i] = self.r_crowd[i] + time_taken

            for idx, station in enumerate(path):
                if idx == 0:
                    self.t[(i, station)] = self.r_crowd[i]
                    last_t = self.t[(i, station)]
                    last_station = station 

                else:
                    travel_time = self.station_graph.edges()[(last_station, station)]["weight"]

                    self.t[(i, station)] = last_t + travel_time
                    last_t = self.t[(i, station)]
                    last_station = station


    def init_parcels(self) -> None:
        """
        Assigns a release and deadline for each parcel at random.

        The release and deadline are times represented as integers in the range [0, 1440), which corresponds
        to the number of minutes since midnight (00:00). This function should ensure that each parcel
        is assigned a release and deadline, assuming a uniform distribution of times.
        """
        self.alpha = {j: random.choice(self.S) for j in self.J}
        self.omega = {j: random.choice(self.S) for j in self.J if j != self.alpha[j]}

        self.r = {j: random.randint(1, 0.1*self.max_time) for j in self.J}
        self.d = {j: random.randint(0.9*self.max_time, self.max_time) for j in self.J}


    def find_path(self, i: str, j: str) -> tuple:
        """
        Finds the shortest path and the time taken between two stations in the station network.

        Parameters
        ----------
        i : str
            The starting station.
        j : str
            The destination station.

        Returns
        -------
        tuple
            A tuple containing the shortest path as a list of stations and the time taken to travel
            between the two stations as an integer, in minutes.
        """
        path = nx.dijkstra_path(self.station_graph, i, j)
        time_taken = nx.dijkstra_path_length(self.station_graph, i, j)

        return path, time_taken


    def generate_graph(self) -> None:
        """
        Generates the graph of the station network.

        The graph has nodes representing each station S. Two nodes are connected
        by an edge if there is a direct connection between the two stations. The
        weight of the edge is the travel time between the two stations.

        A copy of the graph is also saved without weights, for better plotting of the graph.
        """
        self.station_graph = nx.Graph()
        self.station_graph.add_nodes_from(self.S)

        self.station_graph_const = self.station_graph.copy()

        for line in self.lines.keys():
            for connection, value in self.travel_times[line].items():
                self.station_graph.add_edge(connection[0], connection[1], weight=value)
                self.station_graph_const.add_edge(connection[0], connection[1], weight=1)


    def plot_graph(self) -> None:
        """
        Plots the graph of the station network with nodes and edges.

        This uses the Kamada-Kawai algorithm to generate node positions with increased distance between nodes.
        The graph is then drawn with node labels and node colors.
        # labels = nx.get_edge_attributes(self.station_graph, "weight")
        # nx.draw_networkx_edge_labels(self.station_graph, pos, edge_labels=labels)
        # Uncomment the above lines to add edge labels with weights.
        """
        pos = nx.kamada_kawai_layout(self.station_graph_const)  # Increase the scale as needed

        # Draw the graph
        nx.draw(self.station_graph_const, pos, with_labels=True, node_color="skyblue", node_size=150, font_size=7)
        # labels = nx.get_edge_attributes(self.station_graph, "weight")
        # nx.draw_networkx_edge_labels(self.station_graph, pos, edge_labels=labels)
        plt.show()


    def __repr__(self) -> str:
        """
        Return a string representation of the instance parameters.

        This is used to provide a better representation of the instance when
        printing it. The string includes the sets of crowdshippers, parcels and
        stations, as well as the dictionaries of starting and target stations,
        travel times and locker capacities.

        Returns
        -------
        str
            A string representation of the instance parameters.
        """
        return f'''--- Set I ---\n{self.I}\n\n--- Set J ---\
            \n{self.J}\n\n--- Set S ---\n{self.S}\n\n--- Alpha ---\
            \n{self.alpha}\n\n--- Omega ---\n{self.omega}\n\n--- Times ---\
            \n{self.travel_times}\n\n--- L ---\n{self.l}'''

    
    def return_kwargs(self) -> dict:
        """
        Return the keyword arguments representing the instance parameters.

        This method returns a dictionary containing the attributes of the instance,
        including crowdshippers, parcels, stations, and various parameters related 
        to their operations such as travel times, starting and ending stations, 
        release and deadline times, postal charges, locker capacities, 
        and the random seed used for reproducibility.

        Returns
        -------
        dict
            A dictionary with keys:
            - "I": list of crowdshippers.
            - "J": list of parcels.
            - "S": list of stations.
            - "alpha": dict of starting stations for each parcel.
            - "omega": dict of destination stations for each parcel.
            - "r": dict of release times for each parcel.
            - "d": dict of deadline times for each parcel.
            - "p": dict of postal charges for each parcel.
            - "t": dict of times when each crowdshipper is at each station.
            - "f": float representing the fixed entrainment fee.
            - "l": dict of locker capacities for each station.
            - "seed": int representing the random seed.
        """
        return {
            "I": self.I,
            "J": self.J,
            "S": self.S,
            "alpha": self.alpha,
            "omega": self.omega,
            "r": self.r,
            "d": self.d,
            "p": self.p,
            "t": self.t,
            "f": self.f,
            "l": self.l,
            "seed": self.seed
        }
    

def update_minimalinstanz(path: str) -> None:
    """
    Updates the minimal instance parameters with the ones loaded from the given file.

    Parameters
    ----------
    path : str
        The path to the pickle file containing the parameters.

    Notes
    -----
    This function updates the minimal instance parameters by loading the parameters 
    from the file specified by the path and then saving them to the file 
    "data/minimalinstanz.pkl".
    """
    with open(path, "rb") as f:
        params = pickle.load(f)

    kwargs = {
            "I": params.I,
            "J": params.J,
            "S": params.S,
            "alpha": params.alpha,
            "omega": params.omega,
            "r": params.r,
            "d": params.d,
            "p": params.p,
            "t": params.t,
            "f": params.f,
            "l": params.l,
            "seed": params.seed
        }

    params = Parameters(**kwargs)
    
    with open("data/minimalinstanz.pkl", "wb") as f:
        pickle.dump(params, f)


if __name__ == "__main__":
    num_crowdshippers = 50
    num_parcels = 10
    entrainment_fee = 5
    generator = InstanceGenerator(num_crowdshippers, 
                                  num_parcels, 
                                  entrainment_fee)

    params = Parameters(**generator.return_kwargs())

    print(len(params.S))
