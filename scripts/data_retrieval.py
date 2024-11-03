import requests
from datetime import datetime, timedelta
import pickle
from pypdf import PdfReader
import re
import os
import pickle
import osmnx as ox
import json


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
    if name is not None:
        data["svcReqL"][0]["req"]["input"]["loc"]["name"] = name
        
    response = requests.post(url, json=data)
    contents = response.json()
    
    try:
        lid = contents["svcResL"][0]["res"]["match"]["locL"][0]["lid"]
        
    except IndexError:
        print(f"LID not found for name: {name}. Trying again with Hannover suffix.")
        
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


def get_delay(journeys: dict, print_delay: bool = False, journey_type: str = "DEP") -> dict:
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


def get_standard_times(journeys: list, journey_type: str = "DEP") -> dict:
    ret_dict = {}
    for idx, journey in enumerate(journeys):
        if journey_type == "DEP":
            ret_dict[f'{journey["dirTxt"]}_{idx}'] = journey["stbStop"]["dTimeS"]
            
        elif journey_type == "ARR":
            ret_dict[f'{journey["dirTxt"]}_{idx}'] = journey["stbStop"]["aTimeS"]
            
        else:
            raise ValueError("Neither ""DEP"" nor ""ARR"" given")
            
    return ret_dict
            

def filter_by_target(journeys: list, target: str) -> list:
    return [journey for journey in journeys if target in journey["dirTxt"]]


def get_x_along_line(lines_dict: dict, 
                     line_nr: str, 
                     print_times: bool = False, 
                     func: callable = get_standard_times,
                     filter: bool = False,
                     num_journeys: int = 100,):
    func_ret_dict = {}
    for station in lines_dict[line_nr]["stations"]:
        if station == lines_dict[line_nr]["target_station"]:
            if "lid" in lines_dict[line_nr].keys():
                journeys = get_journeys(name=lines_dict[line_nr]["start_station"], num_journeys=num_journeys, journey_type="ARR", lid=lines_dict[line_nr]["lid"])
            else:
                journeys = get_journeys(name=lines_dict[line_nr]["start_station"], num_journeys=num_journeys, journey_type="ARR")
            journey_type = "ARR"
            
        else:
            journeys = get_journeys(name=station, num_journeys=num_journeys)
            journey_type = "DEP"
            
        if filter:
            journeys = filter_by_target(journeys, lines_dict[line_nr]["target_station"])
            
        func_ret = func(journeys, journey_type=journey_type)
        func_ret_dict[station] = func_ret
        
        if print_times:
            print("--- " + station + " ---")
            print(func_ret)
            print("\n")
            
    return func_ret_dict


def convert_hh_mm_ss_to_timedelta(time_string):
    # Define the format
    time_format = "%H%M%S"
    # Convert the string to a datetime object
    time_object = datetime.strptime(time_string, time_format)
    
    # Create a timedelta based on hours, minutes, and seconds
    return timedelta(hours=time_object.hour, minutes=time_object.minute, seconds=time_object.second)


def get_times_along_line(standard_times: dict):
    connections = {}
    print(standard_times)
    for idx, (station, rides) in enumerate(standard_times.items()):
        if idx == 0:
            try:
                start_time = rides[list(rides.keys())[0]]
                last_station = station

            except IndexError:
                start_time = 0
                last_station = station
                print(f"IndexError: Station {station} not found in dictionary.")
            
        else:
            try:
                min_time = min([time for time in rides.values() if time > start_time])
                
                connections[last_station, station] = (convert_hh_mm_ss_to_timedelta(min_time) 
                                                    - convert_hh_mm_ss_to_timedelta(start_time)).total_seconds()/60
                start_time = min_time
                last_station = station

            except ValueError:
                print(f"ValueError: Station {station} not found in dictionary.")
            
    return connections


def get_duration(lines_dict: dict, line_nr: str) -> dict:
    standard_times = get_x_along_line(lines_dict, line_nr, filter=True)
    duration = get_times_along_line(standard_times)
    
    return duration


def load_lines(file_path: str = "data/lines.pkl"):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    

def remove_only_multiple_spaces_and_trim(text: str):
    # Remove multiple spaces, and trim leading and trailing single spaces
    cleaned_text = re.sub(r'\s{2,}', '', text)  # Remove multiple spaces
    return cleaned_text.strip()  # Trim leading and trailing spaces


def check_substrings(string: str, substrings):
    for substring in substrings:
        if substring in string and string != "Wallensteinstraße" and "Markthalle" not in string:
            return True
    return False


def remove_non_stations(string_list, forbidden_substrings = [" - ", 
                                                             "Verkehrsmittel", 
                                                             "Samstag", 
                                                             "=",
                                                             "- ",
                                                             "alle",
                                                             "Min"]):
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

    return [station.replace("/ ", ", ").replace("/", ", ") for station in ext]


def read_pdf(pdf_path: str):
    # Only read the first page of the pdf
    reader = PdfReader(pdf_path)
    
    page = reader.pages[0]
    text = page.extract_text()

    return text.split("\n")


def extend_slashes(input_list):
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


def generate(base_path: str = "data/gvh_linien",):
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

        lines_dict[line]["stations"] = stations
        lines_dict[line]["target_station"] = stations[-1]
        lines_dict[line]["start_station"] = stations[0]
        
        for station in stations:
            if station not in stations_dict:
                stations_dict[station] = []
            stations_dict[station].append(line)

    # Print the dictionary
    return lines_dict, stations_dict


def save(file_path: str = "data/lines.pkl"):
    lines_dict, stations_dict = generate()

    # Set custom lids
    # Sarstedt
    lines_dict["U1"]["lid"] = "de:03241:1731"
    
    with open(file_path, "wb") as f:
        pickle.dump((lines_dict, stations_dict), f)


def test_line(folder: str = "data/gvh_linien", line_nr: int = 1):
    split_text = read_pdf(os.path.join(folder, f"{line_nr}.pdf"))
    stations = remove_non_stations(split_text)

    print(stations)


def get_geo_location(stations: list):
        # Dictionary to store station names and their coordinates
        station_coords = {}

        # Query OSM for each station to get coordinates
        for station_name in stations:
            try:
                location = ox.geocode(f"{station_name}, Hannover, Germany")
                station_coords[station_name] = location
                print(f"Found coordinates for {station_name}: {location}")

            except Exception as e:
                print(f"Could not find coordinates for {station_name}: {e}\n Trying again with different format...")

                try:
                    location = ox.geocode(f"{station_name}, Germany")
                    station_coords[station_name] = list(location)
                    print(f"Found coordinates for {station_name}: {location}")

                except Exception as e:
                    print(f"Could not find coordinates for {station_name}: {e}")
                    station_coords[station_name] = [0, 0]

        # Check coordinates
        print("Station coordinates:", station_coords)

        # Saving to json
        with open("data/station_coords.json", "w") as f:
            json.dump(station_coords, f, indent=4)


if __name__ == "__main__":
    lines_dict, stations_dict = generate()
    stations = stations_dict.keys()
    get_geo_location(stations)