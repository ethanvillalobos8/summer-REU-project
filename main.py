import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
import numpy as np


# Load the dataset
dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\REU GITHUB\summer-REU-project\Highway-Rail_Grade_Crossing_Accident_Data.csv'
data_field = pd.read_csv(dataset_path, low_memory=False)

# Group by "Railroad Code", "Report Year", and "County Code" and count the number of occurrences
grouped_data = data_field.groupby(['Railroad Code', 'Report Year', 'County Code']).size().reset_index(name='Accident Count')

# For the bar graph
grouped_data_year = grouped_data.groupby(['Report Year'])['Accident Count'].sum().reset_index()

plt.figure(figsize=(10,6))
plt.bar(grouped_data_year['Report Year'], grouped_data_year['Accident Count'])
plt.xticks(grouped_data_year['Report Year'], rotation=46)  # This line sets the x-ticks to be every year and rotates them to prevent overlap
plt.xlabel('Year')
plt.ylabel('Accident Count')
plt.title('Accidents Count by Year')
plt.tight_layout()  # This line can help with spacing issues
plt.show()

# For the network graph
# Create an empty undirected graph
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(grouped_data['Railroad Code'].unique())

# Group the data by 'Report Year' and 'County Code', and create a list of 'Railroad Code' for each group
grouped_by_year_county = grouped_data.groupby(['Report Year', 'County Code'])['Railroad Code'].apply(list).reset_index()

# Add edges to the graph
for railroad_codes in grouped_by_year_county['Railroad Code']:
    if len(railroad_codes) > 1:
        G.add_edges_from([(railroad_codes[i], railroad_codes[j]) for i in range(len(railroad_codes)) for j in range(i+1, len(railroad_codes))])

# Print the number of nodes and edges
print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

# Remove unconnected nodes
isolates = list(nx.isolates(G))
G.remove_nodes_from(isolates)

# Print the number of nodes and edges after removing unconnected nodes
print(f"Number of nodes after removing isolates: {G.number_of_nodes()}")
print(f"Number of edges after removing isolates: {G.number_of_edges()}")

# Draw the graph using a spring layout
pos = nx.spring_layout(G, iterations=100)  # positions for all nodes

# Apply Spectral Clustering
adjacency_matrix = nx.to_scipy_sparse_matrix(G)
sc = SpectralClustering(2, affinity='precomputed', n_init=100)
sc.fit(adjacency_matrix)
labels = sc.labels_
