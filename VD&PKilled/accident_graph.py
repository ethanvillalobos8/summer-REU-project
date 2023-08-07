import pandas as pd
import networkx as nx
from matplotlib import pyplot as plt
from tqdm import tqdm


def generate_graph():
    # Load the dataset
    dataset_path = r'C:\Users\hlugo\OneDrive - The University of Texas-Rio Grande Valley\Documents\UCR\Highway-Rail_Grade_Crossing_Accident_Data.csv'
    data_field = pd.read_csv(dataset_path, low_memory=False)

    grouped_data = data_field.groupby(['County Code', 'Railroad Code'])['Vehicle Damage Cost'].sum().reset_index()

    G = nx.Graph()
    G.add_nodes_from(grouped_data['Railroad Code'].unique())

    # Experiment with different node attributes (categories of attributes, wetather related, maintincence, etc.)
    for _, row in tqdm(grouped_data.iterrows(), total=grouped_data.shape[0]):
        railroad_code = row['Railroad Code']
        incident_year = row['County Code']

        accident_count = row['Vehicle Damage Cost']

        other_companies = grouped_data[
            (grouped_data['County Code'] == incident_year)  &
            (grouped_data['Railroad Code'] != railroad_code)
            ]

        for j, other_row in other_companies.iterrows():
            other_railroad_code = other_row['Railroad Code']
            other_accident_count = other_row['Vehicle Damage Cost']

            if not G.has_edge(railroad_code, other_railroad_code):
                G.add_edge(railroad_code, other_railroad_code, weight=0)

            G[railroad_code][other_railroad_code]['weight'] += accident_count + other_accident_count

    # Remove nodes with degree 0 (disconnected nodes)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


G = generate_graph()

pos = nx.spring_layout(G, seed=42)
plt.figure(figsize=(12, 8))
nx.draw(G, pos, with_labels=True, node_size=500, node_color='skyblue', font_size=10, font_weight='bold')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title('Shared Vehicle Damage Cost Between Companies (Fully Connected)')
plt.show()