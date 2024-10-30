from pypdf import PdfReader
import re
import os


def remove_only_multiple_spaces_and_trim(text: str):
    # Remove multiple spaces, and trim leading and trailing single spaces
    cleaned_text = re.sub(r'\s{2,}', '', text)  # Remove multiple spaces
    return cleaned_text.strip()  # Trim leading and trailing spaces


def check_substrings(string: str, substrings):
    for substring in substrings:
        if substring in string:
            return True
    return False


def remove_non_stations(string_list, forbidden_substrings = [" - ", 
                                                                                   "Verkehrsmittel", 
                                                                                   "Samstag", 
                                                                                   "=",
                                                                                   "- "]):
    out = [remove_only_multiple_spaces_and_trim(
                s.replace(" ab", "").replace(" an", "")) for s in string_list 
                if not any(char.isdigit() for char in s) 
                and 5 <= len(s) <= 30 
                and not check_substrings(s, forbidden_substrings)]
    
    return list(dict.fromkeys(out))


def read_pdf(pdf_path: str):
    # Only read the first page of the pdf
    reader = PdfReader(pdf_path)
    
    page = reader.pages[0]
    text = page.extract_text()

    return text.split("\n")


if __name__ == "__main__":
    base_path = "gvh_linien"
    lines_dict = {}
    stations_dict = {}
    for file_path in os.listdir(base_path):
        # Get the line name from the file path
        line = f"U{file_path.split('.')[0]}"

        # Read the pdf
        split_text = read_pdf(os.path.join(base_path, file_path))

        # Get the stations
        stations = remove_non_stations(split_text)
        lines_dict[line] = stations
        
        for station in stations:
            if station not in stations_dict:
                stations_dict[station] = []
            stations_dict[station].append(line)

        # Print the stations
        # print(f"{line} mit {len(stations)} Stationen:")
        # print(stations, "\n")

# Print the dictionary
#cprint(stations_dict["Kröpcke"])
print(lines_dict)