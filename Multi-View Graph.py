import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D


# Load the dataset
dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\UCR\Highway-Rail_Grade_Crossing_Accident_Data.csv'
data_field = pd.read_csv(dataset_path, low_memory=False)
# Remove missing values
data_field = data_field.dropna(subset=['Railroad Code', 'County Code'])

# Encode "Railroad Code" to numeric values
data_field['Railroad Code'] = data_field['Railroad Code'].astype('category').cat.codes

# Create graph from pandas DataFrame
G = nx.from_pandas_edgelist(data_field, 'Railroad Code', 'County Code')

# Create a dictionary of positions
pos = {node: (node, data_field.loc[node, 'County Code']) for node in G.nodes()}

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw nodes
for node in G.nodes():
    ax.scatter(*pos[node])

# Draw edges
for edge in G.edges():
    x = [pos[edge[0]][0], pos[edge[1]][0]]
    z = [pos[edge[0]][1], pos[edge[1]][1]]
    ax.plot([x[0], x[1]], [0, 0], [z[0], z[1]], color='grey')

ax.set_xlabel('Railroad Code')
ax.set_zlabel('County Code')

plt.show()
