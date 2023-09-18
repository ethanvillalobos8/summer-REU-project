import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load cluster results
df_clusters = pd.read_csv('cluster_results.csv')

# Load original data
df_data = pd.read_csv('../Datasets/Highway-Rail_Grade_Crossing_Accident_Data.csv')

# Merge cluster results with original data
df = pd.merge(df_clusters, df_data, left_on='Node', right_on='Railroad Code', how='left')

# Analyze the distribution of clusters
cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)

# Plot the distribution of clusters
sns.countplot(x='Cluster', data=df)
plt.title('Cluster Distribution')
plt.show()

# Temporal Analysis
temporal_df = df.groupby(['Incident Year', 'Cluster']).size().unstack().fillna(0)
plt.figure(figsize=(10, 8))
sns.heatmap(temporal_df, cmap='YlGnBu')
plt.title('Temporal Cluster Distribution')
plt.show()

# Geographical Analysis
geographical_df = df.groupby(['State Name', 'Cluster']).size().unstack().fillna(0)
plt.figure(figsize=(10, 8))
sns.heatmap(geographical_df, cmap='YlGnBu')
plt.title('Geographical Cluster Distribution')
plt.show()

# Use subsets of features to analyze clusters (Equipment 1, Weather 2, etc)
# Analyze patterns within each cluster based on various factors
factors = ['Highway User Position', 'Equipment Involved', 'Equipment Type', 'Equipment Struck', 'Visibility',
           'Weather Condition', 'Date', 'Time']

# Add other factors you are interested in
extended_factors = ['State Name', 'Railroad Code', 'Incident Year']

for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]

    # Identify the Railroad Code with the most accidents in the cluster
    top_railroad_code = cluster_data['Railroad Code'].value_counts().idxmax()
    print(f"Railroad Code with most accidents: {top_railroad_code}")

    # Analyze the distribution of each factor within the cluster
    for factor in factors:
        most_common = cluster_data[factor].mode()[0]
        print(f"Most common {factor}: {most_common}")

    # Analyze extended factors
    for factor in extended_factors:
        top_value = cluster_data[cluster_data['Railroad Code'] == top_railroad_code][factor].value_counts().idxmax()
        print(f"Most common {factor} for Railroad Code {top_railroad_code}: {top_value}")

    # Analyze common accident characteristics in depth
    common_accidents = cluster_data[
        (cluster_data['Highway User Position'] == 'Moving over crossing') &
        (cluster_data['Equipment Involved'] == 'Train (units pulling)') &
        (cluster_data['Visibility'] == 'Day') &
        (cluster_data['Weather Condition'] == 'Clear') &
        (cluster_data['Equipment Struck'] == 'Rail equipment struck highway user')
    ]

    for factor in extended_factors:
        top_value = common_accidents[factor].value_counts().idxmax()
        print(f"Most common {factor} for common accident characteristics: {top_value}")
