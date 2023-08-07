import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# Load the dataset
dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\UCR\Highway-Rail_Grade_Crossing_Accident_Data.csv'
data_field = pd.read_csv(dataset_path, low_memory=False)

# Create an Accident ID combining 'Report Year' and 'County Code'
data_field['Accident_ID'] = data_field['Report Year'].astype(str) + "_" + data_field['County Code'].astype(str)

# Create a dictionary to hold the number of accidents per company
accidents_dict = {}

# Create an empty directed graph
G = nx.DiGraph()

# Add edges to the graph and count the number of accidents per company
for _, row in data_field.iterrows():
    G.add_edge(row['Accident_ID'], row['Railroad Code'])
    if row['Railroad Code'] in accidents_dict:
        accidents_dict[row['Railroad Code']] += 1
    else:
        accidents_dict[row['Railroad Code']] = 1

# Compute PageRank
pagerank = nx.pagerank(G)

# Compute the average PageRank for each company
avg_pagerank = {company: np.mean([pagerank[node] for node in G.nodes() if node.startswith(company)]) for company in accidents_dict.keys()}

for company in accidents_dict.keys():
    print(f"Company: {company}")
    print(f"Number of Accidents: {accidents_dict[company]}")
    print(f"Average PageRank: {avg_pagerank[company]}")
    print()

# Group data by 'Report Year' and 'Railroad Code' and count the number of accidents for each group
accidents_by_year = data_field.groupby(['Report Year', 'Railroad Code']).size().reset_index(name='Accident Count')

# Get unique company codes
company_codes = data_field['Railroad Code'].unique()

# Plot number of accidents over time for each company
plt.figure(figsize=(10, 6))
for company in company_codes:
    company_data = accidents_by_year[accidents_by_year['Railroad Code'] == company]
    plt.plot(company_data['Report Year'], company_data['Accident Count'], label=company)
plt.legend()
plt.xlabel('Year')
plt.ylabel('Number of Accidents')
plt.title('Number of Accidents Over Time')
plt.show()
