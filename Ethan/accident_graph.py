import pandas as pd
import networkx as nx
from tqdm import tqdm


def generate_graph():
    dataset_path = '../Datasets/Highway-Rail_Grade_Crossing_Accident_Data.csv'
    data_field = pd.read_csv(dataset_path, low_memory=False)

    grouped_data = data_field.groupby(['Railroad Code', 'Incident Year', 'State Name']).size().reset_index(
        name='Accident Count')

    G = nx.Graph()
    G.add_nodes_from(grouped_data['Railroad Code'].unique())

    # Experiment with different node attributes (categories of attributes, weather related, maintenance, etc.)
    for _, row in tqdm(grouped_data.iterrows(), total=grouped_data.shape[0]):
        railroad_code = row['Railroad Code']
        incident_year = row['Incident Year']
        state_code = row['State Name']
        accident_count = row['Accident Count']

        other_companies = grouped_data[
            (grouped_data['Incident Year'] == incident_year) &
            (grouped_data['State Name'] == state_code) &
            (grouped_data['Railroad Code'] != railroad_code)
            ]

        for j, other_row in other_companies.iterrows():
            other_railroad_code = other_row['Railroad Code']
            other_accident_count = other_row['Accident Count']

            if not G.has_edge(railroad_code, other_railroad_code):
                G.add_edge(railroad_code, other_railroad_code, weight=0)

            G[railroad_code][other_railroad_code]['weight'] += accident_count + other_accident_count

    # Remove nodes with degree 0 (disconnected nodes)
    G.remove_nodes_from(list(nx.isolates(G)))

    return G


G = generate_graph()
