import pickle
import numpy as np



def get_probability(connections: dict) -> tuple:
    """
    Calculate the probability of each duration in the connections.

    Parameters
    ----------
    connections : dict
        Dictionary of connections, mapping each line to a dictionary mapping each pair of stations
        to the travel time between the two stations.

    Returns
    -------
    total : int
        The total number of valid connections.
    prob_dict : dict
        Dictionary mapping each travel time to its probability.
    """
    
    conn_unpacked = {}
    for line, value in connections.items():
        for station_pair, duration in value.items():
            if duration is not None:
                conn_unpacked[station_pair] = duration

    for line, value in connections.items():
        for station_pair, duration in value.items():
            if duration is None:
                if station_pair in conn_unpacked.keys():
                    connections[line][station_pair] = conn_unpacked[station_pair]

    prob_dict = {"Invalid": 0}
    for line, value in connections.items():
        for station_pair, duration in value.items():
            if duration is not None:
                if duration not in prob_dict.keys():
                    prob_dict[duration] = 1
                else:
                    prob_dict[duration] += 1

            else:
                prob_dict["Invalid"] += 1

    total = sum([value for key, value in prob_dict.items() 
                 if key != "Invalid"])
    
    for key, value in prob_dict.items():
        if key != "Invalid":
            prob_dict[key] = value/total

    return total, prob_dict


def extend_connections(connections: dict, prob_dict: dict) -> None:
    """
    Extend connections with known durations based on the given probabilities.

    Parameters
    ----------
    connections : dict
        Dictionary of connections, mapping each line to a dictionary mapping each pair of stations
        to the travel time between the two stations.
    prob_dict : dict
        Dictionary mapping each travel time to its probability.

    Returns
    -------
    None
    """
    choices = {1: prob_dict[1], 2: prob_dict[2], 3: prob_dict[3]}

    for line, value in connections.items():
        for station_pair, duration in value.items():
            if duration is None:
                connections[line][station_pair] = int(np.random.choice(list(choices.keys()), 
                                                                       1, 
                                                                       p=list(choices.values()))[0])

            else:
                connections[line][station_pair] = int(connections[line][station_pair])


if __name__ == "__main__":
    # Set save to True if you want to save the connections
    save = False

    # Load connections
    with open("../data/stations_data/connections_saved.pkl", "rb") as f:
        connections = pickle.load(f)

    # Calculate probabilities
    total, prob_dict = get_probability(connections)
    print(prob_dict)

    # Print the probabilites of the different durations
    print("Time from s to s+1")
    for key, value in prob_dict.items():
        if key != "Invalid":
            print(f"{int(key)} min[s]: {round(value, 3)*100}%")

    # Print the percentage of valid connections
    print("Percentage of valid:", 
          round(total / (total + prob_dict["Invalid"]), 3)*100, "%")
    
    # Extend the connections with known durations
    extend_connections(connections, prob_dict)

    # Save connections if desired
    if save:
        with open("data/stations_data/connections_full.pkl", "wb") as f:
            pickle.dump(connections, f)
