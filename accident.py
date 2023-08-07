import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
from tqdm import tqdm
import numpy as np


# Load the dataset
dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\REU GITHUB\summer-REU-project\Highway-Rail_Grade_Crossing_Accident_Data.csv'
data_field = pd.read_csv(dataset_path, low_memory=False)


# Count the occurrences of each unique value in the 'Weather Condition' column
weather_counts = data_field['Weather Condition'].value_counts()

# Plot the counts
plt.figure(figsize=(10,6))
weather_counts.plot(kind='bar')
plt.xlabel('Weather Condition')
plt.ylabel('Count')
plt.title('Count of Accidents by Weather Condition')
plt.show()

# Group by "Railroad Code", "Incident Year", and "Weather Condition Code" and count the number of occurrences
grouped_data = data_field.groupby(['Railroad Code', 'Incident Year', 'Weather Condition Code']).size().reset_index(name='Accident Count')

# For the network graph
# Create an empty undirected graph
G = nx.Graph()

# Add nodes to the graph
G.add_nodes_from(grouped_data['Railroad Code'].unique())

# Group the data by 'Incident Year' and 'Weather Condition Code', and create a list of 'Railroad Code' for each group
grouped_by_year_weather = grouped_data.groupby(['Incident Year', 'Weather Condition Code'])['Railroad Code'].apply(list).reset_index()

# Add edges to the graph
for railroad_codes in tqdm(grouped_by_year_weather['Railroad Code']):
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

# Convert the graph into a scipy sparse matrix
A = nx.convert_matrix.to_scipy_sparse_matrix(G)

# Apply Spectral Clustering
sc = SpectralClustering(2, affinity='precomputed', n_init=100, assign_labels='discretize')
sc.fit(A)
labels = sc.labels_

# Draw the graph using a spring layout
pos = nx.spring_layout(G, iterations=100)  # positions for all nodes

# Draw the graph with node color indicating cluster
nx.draw(G, pos, node_color=labels, with_labels=True)
plt.show()
