from pypdf import PdfReader
import re
import os
import pickle
import numpy.random as random


def remove_only_multiple_spaces_and_trim(text: str):
    # Remove multiple spaces, and trim leading and trailing single spaces
    cleaned_text = re.sub(r'\s{2,}', '', text)  # Remove multiple spaces
    return cleaned_text.strip()  # Trim leading and trailing spaces


def check_substrings(string: str, substrings):
    for substring in substrings:
        if substring in string and string != "Wallensteinstraße":
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


def generate_times(lines_dict: dict):
    random.seed(43)
    connections = {line: dict() for line in lines_dict.keys()}

    for line in lines_dict.keys():
        for idx, station in enumerate(lines_dict[line]["stations"]):
            if idx == 0:
                last_station = station
                
            else:
                connections[line][last_station, station] = random.choice([i for i in range(1,5)], 1,
                                p=[0.5, 0.3, 0.15, 0.05])[0]
                last_station = station
            
    return connections


if __name__ == "__main__":
    lines_dict, stations_dict = generate()
    print(stations_dict)
