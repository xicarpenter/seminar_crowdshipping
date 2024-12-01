import requests
from datetime import datetime, timedelta
import pickle
from pypdf import PdfReader
import re
import os
import pickle
from copy import deepcopy


URL_JOURNEY = '''https://gvh.hafas.de/hamm?requestId=undefined&hciMethod=StationBoard&hciVersion=
1.62&hciClientType=WEB&hciClientVersion=10107&aid=IKSEvZ1SsVdfIRSK&rnd=1730290996611'''

URL_LID = '''https://gvh.hafas.de/hamm?requestId=undefined&hciMethod=
LocMatch&hciVersion=1.62&hciClientType=WEB&hciClientVersion=10107&aid=IKSEvZ1SsVdfIRSK&rnd=1730291024259'''

DATA_JOURNEY = {"ver":"1.62",
                "lang":"deu",
                "auth":{
                    "type":"AID",
                    "aid":"IKSEvZ1SsVdfIRSK"},
                "client":
                    {
                        "id":"HAFAS",
                        "type":"WEB",
                        "name":"webapp",
                        "l":"vs_webapp",
                        "v":10107
                    },
                "formatted":'false',
                "svcReqL":[
                    {
                        "req":
                        {
                            "jnyFltrL":
                            [
                                {"type":"PROD",
                                "mode":"INC",
                                "value":831}
                                ],
                            "stbLoc":
                                {
                                    "name":"xxx",
                                    "lid":"de:03241:121",
                                    "extId":"xxx",
                                    "eteId":"xxx"
                                },
                            "type":"DEP",
                            "sort":"PT",
                            "maxJny":100},
                        "meth":"StationBoard",
                        "id":"1|5|"}
                    ]
                }

DATA_LID = {"ver":"1.62",
            "lang":"deu",
            "auth":
                {"type":"AID",
                 "aid":"IKSEvZ1SsVdfIRSK"},
            "client":{"id":"HAFAS",
                      "type":"WEB",
                      "name":"webapp",
                      "l":"vs_webapp",
                      "v":10107},
            "formatted":'false',
            "svcReqL":[
                {"req":
                    {"input":
                        {"field":"S",
                         "loc":{
                             "type":"S",
                             "dist":1000,
                             "name":"Hannover, Kröpcke"},
                         "maxLoc":7}
                        },
                    "meth":"LocMatch",
                    "id":"1|8|"}
                ]
            }


def get_lid(name: str = None,
            url: str = URL_LID,
            data: dict = DATA_LID) -> str:
    """
    Get the LID (location id) for a given name.

    If the name is not found, it will be tried again with "Hannover, " as a prefix.
    
    Parameters
    ----------
    name : str
        The name of the location.
    url : str
        The URL for the LocMatch request.
    data : dict
        The data for the LocMatch request.
        
    Returns
    -------
    str
        The LID of the location.
    """
    if name is not None:
        data["svcReqL"][0]["req"]["input"]["loc"]["name"] = name
        
    response = requests.post(url, json=data)
    contents = response.json()
    
    try:
        lid = contents["svcResL"][0]["res"]["match"]["locL"][0]["lid"]
        
    except IndexError:
        print(f"LID not found for name: {name}. Trying again with Hannover prefix.")
        
        data["svcReqL"][0]["req"]["input"]["loc"]["name"] = "Hannover, " + name
        
        response = requests.post(url, json=data)
        contents = response.json()
        
        lid = contents["svcResL"][0]["res"]["match"]["locL"][0]["lid"]
    
    return lid


def get_journeys(name: str, 
                 url: str = URL_JOURNEY,
                 data: dict = DATA_JOURNEY,
                 num_journeys: int = 200,
                 journey_type: str = "DEP",
                 lid: str = None) -> dict:
    
    """
    Retrieves the journeys for a given location.
    
    Parameters
    ----------
    name : str
        The name of the location.
    url : str, optional
        The URL for the StationBoard request. Default is URL_JOURNEY.
    data : dict, optional
        The data for the StationBoard request. Default is DATA_JOURNEY.
    num_journeys : int, optional
        The number of journeys to retrieve. Default is 200.
    journey_type : str, optional
        The type of journey to retrieve. Default is "DEP" for departures.
    lid : str, optional
        The LID of the location. If None, the LID is retrieved using the get_lid function.
    
    Returns
    -------
    dict
        The journeys for the given location.
    """
    if lid is not None:
        data["svcReqL"][0]["req"]["stbLoc"]["lid"] = lid

    else:
        data["svcReqL"][0]["req"]["stbLoc"]["lid"] = get_lid(name)
        
    data["svcReqL"][0]["req"]["maxJny"] = num_journeys
    data["svcReqL"][0]["req"]["type"] = journey_type

    response = requests.post(url, json=data)
    contents = response.json()
    journeys = contents["svcResL"][0]["res"]["jnyL"]
    
    return journeys


def get_delay(journeys: dict, 
              print_delay: bool = False, 
              journey_type: str = "DEP") -> dict:
    """
    Retrieves the delays for a given journey.
    
    Parameters
    ----------
    journeys : dict
        The journeys for which to retrieve the delays.
    print_delay : bool, optional
        Whether to print the delays. Default is False.
    journey_type : str, optional
        The type of journey to retrieve the delays for. Default is "DEP" for departures.
    
    Returns
    -------
    dict
        The delays for the given journey, with each key being the train name and each value being the delay in seconds.
    """
    delays = {}
    for journey in journeys:
        train_name = journey["dirTxt"]
        
        if "dTimeR" in journey["stbStop"].keys() or "aTimeR" in journey["stbStop"].keys():
            if journey_type == "DEP":
                timeS = journey["stbStop"]["dTimeS"]
                timeR = journey["stbStop"]["dTimeR"]
            
            elif journey_type == "ARR":
                timeS = journey["stbStop"]["aTimeS"]
                timeR = journey["stbStop"]["aTimeR"]
                
            else:
                raise ValueError("journey_type must be DEP or ARR")
            
            delay = int(timeR) - int(timeS)
            
            delays[train_name] = delay
            
            if print_delay:
                print(f"Train to {train_name} is delayed by {delay} second[s].")
            
        else:
            delays[train_name] = None
            
            if print_delay:
                print(f"No delay given for the train to {train_name}.")
            
    return delays


def get_standard_times(journeys: list, 
                       journey_type: str = "DEP") -> dict:
    """
    Retrieves the standard times for a given journey.
    
    Parameters
    ----------
    journeys : list
        The journeys for which to retrieve the standard times.
    journey_type : str, optional
        The type of journey to retrieve the standard times for. Default is "DEP" for departures.
    
    Returns
    -------
    dict
        The standard times for the given journey, with each key being the train name and each value being the standard time in seconds.
    """
    ret_dict = {}
    for idx, journey in enumerate(journeys):
        if journey_type == "DEP":
            ret_dict[f'{journey["dirTxt"]}_{idx}'] = journey["stbStop"]["dTimeS"]
            
        elif journey_type == "ARR":
            ret_dict[f'{journey["dirTxt"]}_{idx}'] = journey["stbStop"]["aTimeS"]
            
        else:
            raise ValueError("Neither ""DEP"" nor ""ARR"" given")
            
    return ret_dict
            

def filter_by_target(journeys: list, 
                     target: str) -> list:
    """
    Filter a list of journeys by a target station.
    
    Parameters
    ----------
    journeys : list
        The list of journeys to filter.
    target : str
        The target station to filter by.
    
    Returns
    -------
    list
        The filtered list of journeys.
    """
    return [journey for journey in journeys if target in journey["dirTxt"]]


def get_x_along_line(lines_dict: dict, 
                     line_nr: str, 
                     print_times: bool = False, 
                     func: callable = get_standard_times,
                     filter: bool = False,
                     num_journeys: int = 100) -> dict:
    """
    Retrieves a dictionary of values along a given line.
    
    Parameters
    ----------
    lines_dict : dict
        The dictionary containing the lines, with each key being the line number and 
        each value being a dictionary containing the line's stations and target station.
    line_nr : str
        The number of the line to retrieve the dictionary for.
    print_times : bool, optional
        Whether to print the times for each station. Default is False.
    func : callable, optional
        The function to use to retrieve the values for each station. Default is get_standard_times.
    filter : bool, optional
        Whether to filter the journeys by the target station. Default is False.
    num_journeys : int, optional
        The number of journeys to retrieve for each station. Default is 100.
    
    Returns
    -------
    dict
        The dictionary of values for each station along the given line.
    """
    func_ret_dict = {}
    for station in lines_dict[line_nr]["stations"]:
        if station == lines_dict[line_nr]["target_station"]:
            if "lid" in lines_dict[line_nr].keys():
                journeys = get_journeys(name=lines_dict[line_nr]["start_station"], 
                                        num_journeys=num_journeys, journey_type="ARR", 
                                        lid=lines_dict[line_nr]["lid"])
            else:
                journeys = get_journeys(name=lines_dict[line_nr]["start_station"], 
                                        num_journeys=num_journeys, 
                                        journey_type="ARR")
            journey_type = "ARR"
            
        else:
            journeys = get_journeys(name=station, 
                                    num_journeys=num_journeys)
            journey_type = "DEP"
            
        if filter:
            journeys = filter_by_target(journeys, 
                                        lines_dict[line_nr]["target_station"])
            
        func_ret = func(journeys, 
                        journey_type=journey_type)
        
        func_ret_dict[station] = func_ret
        
        if print_times:
            print("--- " + station + " ---")
            print(func_ret)
            print("\n")
            
    return func_ret_dict


def convert_hh_mm_ss_to_timedelta(time_string: str) -> timedelta:
    # Define the format
    """
    Convert a time string in format HHMMSS to a timedelta object.

    Parameters
    ----------
    time_string : str
        The time string in format HHMMSS.

    Returns
    -------
    timedelta
        The corresponding timedelta object.
    """
    time_format = "%H%M%S"
    
    # Convert the string to a datetime object
    time_object = datetime.strptime(time_string, time_format)
    
    # Create a timedelta based on hours, minutes, and seconds
    return timedelta(hours=time_object.hour, 
                     minutes=time_object.minute, 
                     seconds=time_object.second)


def get_times_along_line(standard_times: dict) -> dict:
    """
    Calculate the travel times between stations in a line given a dictionary of standard times.

    Parameters
    ----------
    standard_times : dict
        A dictionary mapping each station to a dictionary mapping each journey to its start time.

    Returns
    -------
    connections : dict
        A dictionary mapping each pair of directly connected stations to the travel time between the two stations.
    """
    connections = {}
    
    for idx, (station, rides) in enumerate(standard_times.items()):
        if idx == 0:
            start_time = rides[list(rides.keys())[0]]
            last_station = station
            
        else:
            try:
                min_time = min([time for time in rides.values() if time > start_time])
                
                connections[last_station, station] = (convert_hh_mm_ss_to_timedelta(min_time) 
                                                    - convert_hh_mm_ss_to_timedelta(start_time)).total_seconds()/60
                connections[last_station, station] %= 10

                if connections[last_station, station] > 3 or connections[last_station, station] < 1:
                    print(f"Time is out of bounds for station {station}. Will be set at random.")
                    connections[last_station, station] = None
                
                start_time = min_time
                last_station = station

            except (ValueError, TypeError):
                print(f"Error: Station {station} not found in dictionary. Will be set at random. ")
                connections[last_station, station] = None
            
    return connections


def get_duration(lines_dict: dict, 
                 line_nr: str) -> dict:
    """
    Calculate the duration between each station along a specified line.

    This function retrieves the standard times for a given line and calculates
    the travel duration between each pair of consecutive stations. If the standard
    times cannot be retrieved due to an error, it attempts to modify the start
    station name and retrieve the times again. If the error persists, it returns
    a dictionary with None values for the duration.

    Parameters
    ----------
    lines_dict : dict
        A dictionary where each key is a line number and each value contains
        station data including the start and target stations.
    line_nr : str
        The line number for which the durations are to be calculated.

    Returns
    -------
    dict
        A dictionary mapping each pair of consecutive stations to the travel
        duration between them in minutes. If the duration cannot be determined,
        the value is None.
    """
    standard_times = get_x_along_line(lines_dict, line_nr, filter=True)

    try:
        duration = get_times_along_line(standard_times)

    except IndexError:
        try:
            lines_dict_new = deepcopy(lines_dict)
            lines_dict_new[line_nr]["start_station"] = f"Hannover, {lines_dict_new[line_nr]['start_station']}"
            print("Attention!!!!")
            standard_times = get_x_along_line(lines_dict_new, line_nr, filter=True, print_times=True)
            duration = get_times_along_line(standard_times)

        except IndexError:
            duration = {}
    
            for idx, (station, rides) in enumerate(standard_times.items()):
                if idx == 0:
                    last_station = station
                    
                else:
                    duration[last_station, station] = None
                    last_station = station

    return duration


def load_lines(file_path: str = "data/lines.pkl") -> dict:
    """
    Load a pickled dictionary of lines data from a file.

    Parameters
    ----------
    file_path : str, optional
        The path to the file containing the pickled dictionary. Defaults to
        "data/lines.pkl".

    Returns
    -------
    dict
        A dictionary where each key is a line number and each value contains
        station data including the start and target stations.
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)
    

def remove_only_multiple_spaces_and_trim(text: str):
    """
    Remove multiple spaces from a given text and trim leading and trailing single spaces.

    Parameters
    ----------
    text : str
        The string from which multiple spaces are to be removed and leading and
        trailing single spaces are to be trimmed.

    Returns
    -------
    str
        The string with multiple spaces removed and leading and trailing single
        spaces trimmed.
    """
    cleaned_text = re.sub(r'\s{2,}', '', text)  # Remove multiple spaces
    return cleaned_text.strip()  # Trim leading and trailing spaces


def check_substrings(string: str, substrings) -> bool:
    """
    Check if any of the substrings are present in the given string.

    Parameters
    ----------
    string : str
        The string to check for the presence of substrings.
    substrings : list[str]
        The list of substrings to check in the given string.

    Returns
    -------
    bool
        True if any of the substrings is present in the string, False otherwise.
    """
    for substring in substrings:
        if substring in string:
            return True
    return False


def remove_non_stations(string_list: list, 
                        forbidden_substrings: list = [" - ", 
                                                      "Verkehrsmittel", 
                                                      "Samstag", 
                                                      "=",
                                                      "- ",
                                                      "Min"],
                        forbidden_strings = ["alle"]) -> list:
    """
    Removes non-station entries from a list of strings based on certain criteria.

    Parameters
    ----------
    string_list : list
        A list of strings potentially containing station names.
    forbidden_substrings : list, optional
        A list of substrings that, if present in a string, disqualify it from being a station.
        Default includes common non-station markers such as " - ", "Verkehrsmittel", "Samstag", etc.
    forbidden_strings : list, optional
        A list of strings that, if exactly matched, disqualify it from being a station.
        Default is ["alle"].

    Returns
    -------
    list
        A cleaned list of strings that are considered to be valid station names, with duplicates
        removed and slashes replaced by commas.
    """
    ret = []
    for string in string_list:
        string = string.replace(" ab", "")
        string = re.sub(r" an.*", "", string)
        string = remove_only_multiple_spaces_and_trim(string)

        if (not any(char.isdigit() for char in string) 
            and (3 <= len(string) <= 30) 
            and not check_substrings(string, forbidden_substrings)):
            ret.append(string)

    out = list(dict.fromkeys(ret))
    ext = extend_slashes(out)

    for string in ext:
        for forbidden_string in forbidden_strings:
            if forbidden_string == string:
                ext.remove(string)

    return [station.replace("/ ", ", ").replace("/", ", ") for station in ext]


def read_pdf(pdf_path: str) -> list:
    # Only read the first page of the pdf
    """
    Reads the first page of a PDF file and splits the text into lines.

    Parameters
    ----------
    pdf_path : str
        The path to the PDF file to read.

    Returns
    -------
    list
        A list of strings, where each string is a line from the first page of the PDF.
    """
    reader = PdfReader(pdf_path)
    
    page = reader.pages[0]
    text = page.extract_text()

    return text.split("\n")


def extend_slashes(input_list: list) -> list:
    """
    Expands entries that start with a slash by prefixing them with the previous entry.

    This function iterates over the input list and checks each item. If an item starts
    with a "/", it is concatenated with the last prefix (the part of the previous item
    before any "/") and added to the output list. If an item does not start with a "/",
    it updates the last prefix using the part of the item before any "/".

    Parameters
    ----------
    input_list : list
        A list of strings where some items may start with a slash.

    Returns
    -------
    list
        A list of strings where items that started with a slash have been expanded
        with the last known prefix.
    """
    output_list = []
    last_prefix = input_list[0]
    
    for item in input_list:
        if item.startswith("/"):
            # Replace with the last prefix if it exists
            output_list.append((last_prefix + item))
        else:
            # Update the last prefix if the current item has a "/"
            last_prefix = item.split("/")[0]
            output_list.append(item)
    
    return output_list


def generate(base_path: str = "data/gvh_linien") -> tuple:
    """
    Generates dictionaries of lines and stations from PDF files in the specified directory.

    This function reads PDF files from the given base path, extracts station names,
    and generates two dictionaries: one mapping line names to their stations, start, 
    and target station, and another mapping stations to the lines they belong to.

    Parameters
    ----------
    base_path : str, optional
        The directory containing PDF files for each line. Defaults to "data/gvh_linien".

    Returns
    -------
    tuple
        A tuple containing two dictionaries:
        - lines_dict: Maps line names to their details, including stations.
        - stations_dict: Maps station names to the list of lines they are part of.
    """
    lines_dict = {}
    stations_dict = {}
    
    for file_path in os.listdir(base_path):
        # Get the line name from the file path
        line = f"U{file_path.split('.')[0]}"
        
        lines_dict[line] = {}

        # Read the pdf
        split_text = read_pdf(os.path.join(base_path, file_path))

        # Get the stations
        stations = remove_non_stations(split_text)

        # Replace german umlaute and ß
        stations_replaced = [station.replace("ö", "oe").replace("ü", "ue").replace("ß", "ss").replace("ä", "ae") 
                             for station in stations]

        lines_dict[line]["stations"] = stations_replaced
        lines_dict[line]["target_station"] = stations_replaced[-1]
        lines_dict[line]["start_station"] = stations_replaced[0]
        
        for station in stations_replaced:
            if station not in stations_dict:
                stations_dict[station] = []
            stations_dict[station].append(line)

    # Print the dictionary
    return lines_dict, stations_dict


def save_lines(file_path: str = "data/lines.pkl") -> None:
    """
    Saves the lines and stations dictionaries to a pickle file.

    Parameters
    ----------
    file_path : str, optional
        The path to the file to save the data to. Defaults to "data/lines.pkl".

    Notes
    -----
    This function currently sets custom lids for some lines. These lids should
    be hardcoded into the lines dictionary as soon as they are known.
    """
    lines_dict, stations_dict = generate()

    # Set custom lids
    # Sarstedt
    lines_dict["U1"]["lid"] = "de:03241:1731"
    
    with open(file_path, "wb") as f:
        pickle.dump((lines_dict, stations_dict), f)


def test_line(folder: str = "data/gvh_linien", 
              line_nr: int = 1) -> None:
    """
    Test function to read and print station names from a PDF file for a specified line.

    This function reads a PDF file corresponding to a given line number from a specified
    folder, removes non-station entries from the extracted text, and prints the list of
    station names.

    Parameters
    ----------
    folder : str, optional
        The directory containing PDF files for each line. Defaults to "data/gvh_linien".
    line_nr : int, optional
        The line number for which the stations are to be extracted. Defaults to 1.
    """
    split_text = read_pdf(os.path.join(folder, f"{line_nr}.pdf"))
    stations = remove_non_stations(split_text)
    print(stations)


if __name__ == "__main__":
    save = False
    folder = "data/gvh_linien"

    with open("data/stations_data/lines.pkl", "rb") as f:
        lines_dict, stations_dict = pickle.load(f)

    print(len(stations_dict.keys()))

    connections = {}
    for i in os.listdir(folder):
        line = f"U{i.split('.')[0]}"
        dur = get_duration(lines_dict, line)
        connections[line] = dur

        print(line)
        print(dur)

    if save:
        with open("data/connections.pkl", "wb") as f:
            pickle.dump(connections, f)
