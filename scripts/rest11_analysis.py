from classes import Parameters
import pickle


def generate_visits(params: Parameters, 
                    X: dict, s_l: str) -> dict:
    """
    Generate a dictionary of visits, where each key is a tuple (i, s, j) for crowdshipper i visiting station s with parcel j, 
    or a string "Start j" or "End j" for the start or end of parcel j, respectively. 
    The value for each key is the time at which the visit or start/end of parcel occurs.

    Parameters
    ----------
    params : Parameters
        Instance of Parameters class
    X : dict
        Dictionary of decision variables X[i, s, j] for i in I, s in S_i and j in J_is[i, s]
    s_l : str
        Locker location

    Returns
    -------
    dict
        Dictionary of visits, sorted by time
    """
    visits = {}

    for (i,s,j), val in X.items():
        if val == 1 and s == s_l:
            visits[(i, s, j)] = params.t[i, s]

        if val == 1 and (i,s) in params.s_is_m.keys() and params.s_is_p[i,s] == s_l:
            visits[(i, s, j)] = params.t[i, params.s_is_p[i, s]]

    for j in params.J:
        if params.alpha[j] == s_l:
            visits[f"Start {j}"] = params.r[j]

        if params.omega[j] == s_l:
            visits[f"End {j}"] = params.d[j]

    sorted_visits = dict(sorted(visits.items(), key=lambda item: item[1]))

    return sorted_visits


def print_visits(sorted_visits: dict, s_l: str) -> None:
    """
    Prints the visits of all crowdshippers at the given station s_l, 
    as well as the start and end times of all jobs.

    Parameters
    ----------
    sorted_visits : dict
        Dictionary with the visits of all crowdshippers at the given station s_l, 
        as well as the start and end times of all jobs.
    s_l : str
        The station for which the visits should be printed.
    """
    for key, val in sorted_visits.items():
        if isinstance(key, str):
            print(key)
            print(val, "\n")

        else:
            i, s, j = key
            
            if s == s_l:
                print(i, s, j)
                print(val, "\n")

            else:
                print(i, s, j)
                print(val, "\n")


if __name__ == "__main__":
    s_l = "Steintor"

    params = Parameters.load("problem_instance/params.pkl")

    with open("problem_instance/results.pkl", "rb") as f:
        X, Y = pickle.load(f)

    print(f"Capacity of {s_l}:", params.l[s_l])

    sorted_visits = generate_visits(params, X, s_l)
    print_visits(sorted_visits, s_l)
