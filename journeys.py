import json
import requests
from collect_lines import generate
from datetime import datetime, timedelta
import pickle


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


if __name__ == "__main__":
    lines = load_lines()[0]
    for i in range(1, 14):
        durations = get_duration(lines, f"U{i}")
        print(f"U{i}:")
        print(durations, "\n")
    
    print("U17:")
    durations = get_duration(lines, "U17")
    print(durations, "\n")