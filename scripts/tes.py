import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
import json

station_coords = json.load(open("data/station_coords.json", "r"))


# Create an empty graph
G = nx.Graph()

# Add nodes with positions based on coordinates
for station_name, coord in station_coords.items():
    G.add_node(station_name, pos=coord)

# Extract positions for NetworkX to use
pos = nx.get_node_attributes(G, 'pos')

# Plot the graph
plt.figure(figsize=(10, 10))
nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=500, font_size=8)
plt.show()

