import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations


# Load the dataset
dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\UCR\Highway-Rail_Grade_Crossing_Accident_Data.csv'
data_field = pd.read_csv(dataset_path, low_memory=False)
# Combine the "Killed" and "Injured" columns to get total casualties
data_field['Total Casualties'] = data_field['Total Killed Form 57'] + data_field['Total Killed Form 55A'] + data_field['Total Injured Form 57'] + data_field['Total Injured Form 55A']

# Group by County Code and railroad code
grouped_data_county = data_field.groupby(['County Code', 'Railroad Code'])['Total Casualties'].sum().reset_index()

print(grouped_data_county)  # Print the grouped data

# Initialize a graph
G_county = nx.Graph()

# Add nodes to the graph
for _, row in grouped_data_county.iterrows():
    G_county.add_node(f"{row['Railroad Code']}({row['Total Casualties']})", casualties=row['Total Casualties'])

# Add edges to the graph for companies in the same county
for county in grouped_data_county['County Code'].unique():
    same_county_companies = grouped_data_county[grouped_data_county['County Code'] == county]['Railroad Code']
    print(f"Companies in county {county}: {same_county_companies}")  # Print the companies in each county
    for company1, company2 in combinations(same_county_companies, 2):
        print(f"Adding edge between {company1} and {company2}")  # Print the pairs of companies that will be connected
        G_county.add_edge(f"{company1}({grouped_data_county.loc[grouped_data_county['Railroad Code'] == company1, 'Total Casualties'].values[0]})", f"{company2}({grouped_data_county.loc[grouped_data_county['Railroad Code'] == company2, 'Total Casualties'].values[0]})")

# Visualize the graph
nx.draw(G_county, with_labels=True)
plt.show()
