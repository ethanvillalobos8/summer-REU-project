import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import OneHotEncoder

# Load original data
df_data = pd.read_csv('../Datasets/Highway-Rail_Grade_Crossing_Accident_Data.csv', dtype='str', low_memory=False)


# Custom function to calculate Jaccard similarity
def jaccard_similarity(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0.0


# Load cluster results from a CSV file
df_clusters = pd.read_csv('data/cluster_results.csv', index_col=False)

# Load original data from another CSV file in chunks
chunk_size = 10000
data_chunks = []
for chunk in pd.read_csv('../Datasets/Highway-Rail_Grade_Crossing_Accident_Data.csv', low_memory=False,
                         chunksize=chunk_size):
    data_chunks.append(chunk)

df_data = pd.concat(data_chunks, ignore_index=True)

grouped_data = df_data.groupby(['Railroad Code', 'Incident Year', 'State Name']).size().reset_index(
    name='Accident Count')

# Merge cluster results with the original data based on a common column
df = pd.merge(df_clusters, df_data, left_on='Node', right_on='Railroad Code', how='left')

# Merge the 'Accident Count' column into the existing DataFrame
df = pd.merge(df, grouped_data[['Railroad Code', 'Incident Year', 'State Name', 'Accident Count']],
              left_on=['Railroad Code', 'Incident Year', 'State Name'],
              right_on=['Railroad Code', 'Incident Year', 'State Name'], how='left')

# Analyze the distribution of clusters
cluster_counts = df['Cluster'].value_counts()
print(cluster_counts)

# Plot the distribution of clusters using seaborn
# sns.countplot(x='Cluster', data=df)
# plt.title('Cluster Distribution')

# Temporal Analysis
# temporal_df = df.groupby(['Incident Year', 'Cluster']).size().unstack().fillna(0)
# plt.figure(figsize=(10, 8))
# sns.heatmap(temporal_df, cmap='YlGnBu')
# plt.title('Temporal Cluster Distribution')

# Geographical Analysis
# geographical_df = df.groupby(['State Name', 'Cluster']).size().unstack().fillna(0)
# plt.figure(figsize=(10, 8))
# sns.heatmap(geographical_df, cmap='YlGnBu')
# plt.title('Geographical Cluster Distribution')

# Define factors for analysis
factors = ['Highway User Position', 'Equipment Involved', 'Equipment Type', 'Equipment Struck', 'Visibility',
           'Weather Condition', 'Date', 'Time']

# Ensure that all characteristics are treated as strings (categorical)
df[factors] = df[factors].astype('category')  # Using 'category' data type for categorical columns

# Using label encoding to convert categorical data to numeric values
label_encoders = {}
for factor in factors:
    label_encoders[factor] = LabelEncoder()
    df[factor] = label_encoders[factor].fit_transform(df[factor])

# Calculate Jaccard similarity for each pair of clusters (all factors)
num_clusters = len(df['Cluster'].unique())
similarity_matrix = np.zeros((num_clusters, num_clusters))

for i in range(num_clusters):
    for j in range(i, num_clusters):
        similarities = []
        for factor in factors:
            cluster_i = set(df[df['Cluster'] == i][factor])
            cluster_j = set(df[df['Cluster'] == j][factor])

            similarity = jaccard_similarity(cluster_i, cluster_j)
            similarities.append(similarity)

        avg_similarity = np.mean(similarities)
        similarity_matrix[i, j] = avg_similarity
        similarity_matrix[j, i] = avg_similarity

# Create a heatmap to visualize cluster similarity
# plt.figure(figsize=(10, 8))
# sns.heatmap(similarity_matrix, cmap='YlGnBu', annot=True, fmt=".2f",
#             xticklabels=df['Cluster'].unique(), yticklabels=df['Cluster'].unique())
# plt.title('Cluster Similarity Based on Characteristics')
# plt.xlabel('Cluster')
# plt.ylabel('Cluster')

# Create a dictionary to store similarity matrices for each factor
similarity_matrices = {}

# Calculate Jaccard similarity for each pair of clusters for each factor
for factor in factors:
    num_clusters = len(df['Cluster'].unique())
    similarity_matrix = np.zeros((num_clusters, num_clusters))

    for i in range(num_clusters):
        for j in range(i, num_clusters):
            cluster_i = set(df[df['Cluster'] == i][factor])
            cluster_j = set(df[df['Cluster'] == j][factor])

            similarity = jaccard_similarity(cluster_i, cluster_j)

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    similarity_matrices[factor] = similarity_matrix

# Create a subdirectory for saving PNG files
if not os.path.exists("factor_heatmaps"):
    os.mkdir("factor_heatmaps")

# Create an empty dictionary to store results
factor_influence = {}

# Calculate the influence of each factor individually for all iterations
# for num_factors in range(1, len(factors) + 1):
#     selected_factors = factors[:num_factors]
#
#     num_clusters = len(df['Cluster'].unique())
#     similarity_matrix = np.zeros((num_clusters, num_clusters))
#
#     for i in range(num_clusters):
#         for j in range(i, num_clusters):
#             similarities = []
#             for factor in selected_factors:
#                 cluster_i_vals = set(df[df['Cluster'] == i][factor])
#                 cluster_j_vals = set(df[df['Cluster'] == j][factor])
#
#                 similarity = jaccard_similarity(cluster_i_vals, cluster_j_vals)
#                 similarities.append(similarity)
#
#             avg_similarity = np.mean(similarities)
#             similarity_matrix[i, j] = avg_similarity
#             similarity_matrix[j, i] = avg_similarity
#
#     factor_influence[num_factors] = similarity_matrix
#
#     # Generate and save individual factor heatmaps
#     for factor in selected_factors:
#         plt.figure(figsize=(6, 4))
#         sns.heatmap(factor_influence[num_factors], cmap='YlGnBu', annot=True, fmt=".2f",
#                     xticklabels=df['Cluster'].unique(), yticklabels=df['Cluster'].unique())
#         plt.title(f'Influence of {factor}')
#         plt.xlabel('Cluster')
#         plt.ylabel('Cluster')
#
#         # Save PNG files to the subdirectory
#         filename = os.path.join("factor_heatmaps", f"{factor}_{num_factors}_heatmap.png")
#         plt.savefig(filename)
#         plt.close()
#
#     # Print the similarity matrix to console for each factor
#     for factor in selected_factors:
#         print(f"Similarity matrix for {factor} ({num_factors}/{len(factors)} Factors):")
#         print(factor_influence[num_factors])
#
# # Visualize the influence of each factor across all iterations
# plt.figure(figsize=(12, 10))
# for num_factors, similarity_matrix in factor_influence.items():
#     plt.subplot(3, 3, num_factors)
#     sns.heatmap(similarity_matrix, cmap='YlGnBu', annot=True, fmt=".2f",
#                 xticklabels=df['Cluster'].unique(), yticklabels=df['Cluster'].unique())
#     plt.title(f'Influence of {num_factors}/{len(factors)} Factors')
#     plt.xlabel('Cluster')
#     plt.ylabel('Cluster')
#
# plt.tight_layout()
# plt.show()

# Add other factors interested in
extended_factors = ['State Name', 'Railroad Code', 'Incident Year']

# Loop through clusters and analyze various characteristics
for cluster in df['Cluster'].unique():
    print(f"\nCluster {cluster}:")
    cluster_data = df[df['Cluster'] == cluster]

    if cluster_data.empty:
        print("Cluster is empty.")
        continue

    # Identify the Railroad Code with the most accidents in the cluster
    top_railroad_code = cluster_data['Railroad Code'].value_counts().idxmax()
    print(f"Railroad Code with most accidents: {top_railroad_code}")

    # Analyze the distribution of each factor within the cluster
    for factor in factors:
        most_common = cluster_data[factor].mode()[0] if not cluster_data[factor].empty else "N/A"
        print(f"Most common {factor}: {most_common}")

    # Analyze extended factors
    for factor in extended_factors:
        if factor in cluster_data:
            top_value = cluster_data[cluster_data['Railroad Code'] == top_railroad_code][factor].value_counts().idxmax()
            print(f"Top {factor} for Railroad {top_railroad_code}: {top_value}")

# Machine Learning Model

# Using a random subset of the data to improve runtime
subset_fraction = 0.10  # 10% of dataset
df = df.sample(frac=subset_fraction, random_state=42)

# Define features and target variable
categorical_features = ['Highway User Position', 'Equipment Involved']
numeric_features = ['Incident Year']
target = 'Accident Count'

# Drop rows with missing values in the target variable
df.dropna(subset=[target], inplace=True)

# Convert 'Incident Year' to full year format
df['Incident Year'] = df['Incident Year'].apply(lambda year: 2000 + year if year <= 21 else 1900 + year)

# Split the dataset
X = df[numeric_features + categorical_features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the target variable
y_scaler = MinMaxScaler()
y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()

# One-hot encode categorical features
encoder = OneHotEncoder(sparse=False, drop='first', handle_unknown='ignore')
encoder.fit(df[categorical_features])
X_train_encoded = encoder.transform(X_train[categorical_features])
X_test_encoded = encoder.transform(X_test[categorical_features])

# Combine numeric and encoded categorical features
X_train_final = np.hstack((X_train[numeric_features].values, X_train_encoded))
X_test_final = np.hstack((X_test[numeric_features].values, X_test_encoded))

# Standardize features
scaler = StandardScaler()
X_train_final = scaler.fit_transform(X_train_final)
X_test_final = scaler.transform(X_test_final)

# Train the global model
global_model = KernelRidge(kernel='rbf')
global_model.fit(X_train_final, y_train_scaled)

# Predict and inverse transform the predictions
y_pred_scaled = global_model.predict(X_test_final)
y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Calculate RMSE for the global model
global_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
print(f"Global Model RMSE: {global_rmse:.2f}")

# Process and evaluate models for each cluster
clusters = df['Cluster'].unique()
cluster_models = {}
cluster_scalers = {}
y_cluster_scalers = {}

for cluster in clusters:
    df_cluster = df[df['Cluster'] == cluster]
    X_cluster = df_cluster[numeric_features + categorical_features]
    y_cluster = df_cluster[target]
    X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
        X_cluster, y_cluster, test_size=0.2, random_state=42
    )

    # Scale target variable for cluster
    y_cluster_scaler = MinMaxScaler()
    y_train_cluster_scaled = y_cluster_scaler.fit_transform(y_train_cluster.values.reshape(-1, 1)).ravel()
    y_test_cluster_scaled = y_cluster_scaler.transform(y_test_cluster.values.reshape(-1, 1)).ravel()
    y_cluster_scalers[cluster] = y_cluster_scaler

    # Preprocess cluster data
    X_train_encoded_cluster = encoder.transform(X_train_cluster[categorical_features])
    X_test_encoded_cluster = encoder.transform(X_test_cluster[categorical_features])

    # Combine numeric and encoded features
    X_train_final_cluster = np.hstack((X_train_cluster[numeric_features].values, X_train_encoded_cluster))
    X_test_final_cluster = np.hstack((X_test_cluster[numeric_features].values, X_test_encoded_cluster))

    # Standardize cluster features
    cluster_scaler = StandardScaler()
    X_train_final_cluster = cluster_scaler.fit_transform(X_train_final_cluster)
    X_test_final_cluster = cluster_scaler.transform(X_test_final_cluster)
    cluster_scalers[cluster] = cluster_scaler

    # Train the cluster-specific model
    model_cluster = KernelRidge(kernel='rbf')
    model_cluster.fit(X_train_final_cluster, y_train_cluster_scaled)
    cluster_models[cluster] = model_cluster

    # Predict and inverse transform the cluster-specific predictions
    y_pred_cluster_scaled = model_cluster.predict(X_test_final_cluster)
    y_pred_cluster = y_cluster_scalers[cluster].inverse_transform(y_pred_cluster_scaled.reshape(-1, 1)).ravel()

    # Calculate and print RMSE for the cluster
    rmse_cluster = np.sqrt(mean_squared_error(y_test_cluster, y_pred_cluster))
    print(f"Cluster {cluster}: RMSE: {rmse_cluster:.2f}")

# Visualization for global and cluster-specific models
# Global model visualization
actual_test_accidents = y_test.groupby(X_test['Incident Year']).sum()
predicted_test_accidents = pd.DataFrame({'Incident Year': X_test['Incident Year'], 'Predicted': y_pred}).groupby(
    'Incident Year').sum()

plt.figure(figsize=(15, 7))
plt.plot(actual_test_accidents.index, actual_test_accidents.values, label='Actual (Test)', marker='o', linestyle='--',
         color='green')
plt.plot(predicted_test_accidents.index, predicted_test_accidents['Predicted'], label='Predicted (Test)', marker='x',
         color='red')
plt.title('Yearly Accident Counts for Global Model: Actual vs Predicted')
plt.xlabel('Year')
plt.ylabel('Accident Count')
plt.legend()
plt.grid(True)

# For each cluster-specific model
for cluster in clusters:
    df_cluster = df[df['Cluster'] == cluster]
    X_cluster = df_cluster[numeric_features + categorical_features]
    y_cluster = df_cluster[target]
    _, X_test_cluster, _, y_test_cluster = train_test_split(X_cluster, y_cluster, test_size=0.2, random_state=42)

    # Preprocess the test data using the global encoder
    X_test_encoded_cluster = encoder.transform(X_test_cluster[categorical_features])
    X_test_final_cluster = np.hstack((X_test_cluster[numeric_features].values, X_test_encoded_cluster))

    # Standardize the features for the test set using the cluster-specific scaler
    X_test_final_cluster = cluster_scalers[cluster].transform(X_test_final_cluster)

    # Make predictions for this cluster
    y_pred_cluster_scaled = cluster_models[cluster].predict(X_test_final_cluster)
    y_pred_cluster = y_cluster_scalers[cluster].inverse_transform(y_pred_cluster_scaled.reshape(-1, 1)).ravel()

    # Visualization for this cluster
    actual_test_accidents_cluster = y_test_cluster.groupby(X_test_cluster['Incident Year']).sum()
    predicted_test_accidents_cluster = pd.DataFrame({'Incident Year': X_test_cluster['Incident Year'], 'Predicted': y_pred_cluster}).groupby('Incident Year').sum()

    plt.figure(figsize=(15, 7))
    plt.plot(actual_test_accidents_cluster.index, actual_test_accidents_cluster.values, label='Actual (Test)', marker='o', linestyle='--', color='green')
    plt.plot(predicted_test_accidents_cluster.index, predicted_test_accidents_cluster['Predicted'], label='Predicted (Test)', marker='x', color='red')
    plt.title(f'Yearly Accident Counts for Cluster {cluster}: Actual vs Predicted')
    plt.xlabel('Year')
    plt.ylabel('Accident Count')
    plt.legend()
    plt.grid(True)
    plt.show()

# Machine Learning Portion Using RBF Sampler

# # Using a random subset of the data to improve runtime
# subset_fraction = 0.10  # 10% of dataset
# df = df.sample(frac=subset_fraction, random_state=42)
#
# # Define features and target variable based on the identified important features
# categorical_features = ['Highway User Position', 'Equipment Involved']
# numeric_features = ['Incident Year']
# target = 'Accident Count'
#
# # Drop rows with missing values in the target variable
# df.dropna(subset=[target], inplace=True)
#
#
# # Convert 'Incident Year' to full year format
# def convert_year(year):
#     if year <= 21:
#         return 2000 + year
#     else:
#         return 1900 + year
#
#
# df['Incident Year'] = df['Incident Year'].apply(convert_year)
#
# # Training and evaluating the global model
# X = df[numeric_features + categorical_features]
# y = df[target]
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Use MinMaxScaler for the target variable
# y_scaler = MinMaxScaler()
# y_train_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
# y_test_scaled = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
#
# # Preprocess data for global model using a single encoder
# encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
# encoder.fit(df[categorical_features])
#
# X_train_encoded = encoder.transform(X_train[categorical_features])
# X_test_encoded = encoder.transform(X_test[categorical_features])
#
# X_train_final = np.hstack((X_train[numeric_features].values, X_train_encoded))
# X_test_final = np.hstack((X_test[numeric_features].values, X_test_encoded))
#
# # Standardize the features using a single scaler
# scaler = StandardScaler()
# X_train_final = scaler.fit_transform(X_train_final)
# X_test_final = scaler.transform(X_test_final)
#
# # Preprocess data for global model with RBFSampler
# rbf_sampler = RBFSampler(n_components=18, random_state=42)
# X_train_final = rbf_sampler.fit_transform(X_train_final)
# X_test_final = rbf_sampler.transform(X_test_final)
#
# # Train a KernelRidge model for the global model
# global_model = KernelRidge(kernel='rbf')
# global_model.fit(X_train_final, y_train_scaled)
#
# # Make predictions for global model
# y_pred_scaled = global_model.predict(X_test_final)
# y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
#
# # Calculate RMSE for global model
# global_rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
# print(f"Global Model RMSE: {global_rmse:.2f}")
#
# clusters = df['Cluster'].unique()
#
# cluster_performances = {}  # Store each cluster's performance
# cluster_models = {}  # Dictionary to store the models for each cluster
# cluster_scalers = {}  # Store scalers for each cluster
# y_cluster_scalers = {}  # Dictionary to store scalers for y for each cluster
# cluster_rbfsamplers = {}  # Initialize a dictionary to store the RBFSampler for each cluster
#
# for cluster in clusters:
#     df_cluster = df[df['Cluster'] == cluster]
#
#     # Split the data for this cluster
#     X_cluster = df_cluster[numeric_features + categorical_features]
#     y_cluster = df_cluster[target]
#     X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
#         X_cluster, y_cluster, test_size=0.2, random_state=42
#     )
#
#     # Scale y for this cluster
#     y_cluster_scaler = MinMaxScaler()
#     y_train_cluster_scaled = y_cluster_scaler.fit_transform(y_train_cluster.values.reshape(-1, 1)).ravel()
#     y_test_cluster_scaled = y_cluster_scaler.transform(y_test_cluster.values.reshape(-1, 1)).ravel()
#     y_cluster_scalers[cluster] = y_cluster_scaler  # Store the y scaler for this cluster
#
#     # Preprocess data as before using the global encoder
#     X_train_encoded_cluster = encoder.transform(X_train_cluster[categorical_features])
#     X_train_final_cluster = np.hstack((X_train_cluster[numeric_features].values, X_train_encoded_cluster))
#
#     # Standardize the features for the training set using a cluster-specific scaler
#     cluster_scaler = StandardScaler()
#     X_train_final_cluster = cluster_scaler.fit_transform(X_train_final_cluster)
#     cluster_scalers[cluster] = cluster_scaler  # Store the scaler for this cluster
#
#     # Preprocess the test data using the global encoder and cluster-specific scaler
#     X_test_encoded_cluster = encoder.transform(X_test_cluster[categorical_features])
#     X_test_final_cluster = np.hstack((X_test_cluster[numeric_features].values, X_test_encoded_cluster))
#     X_test_final_cluster = cluster_scaler.transform(X_test_final_cluster)
#
#     # Ensure that the test data has the same number of features as the training data
#     if isinstance(X_train_final_cluster, np.ndarray):
#         X_train_final_cluster = pd.DataFrame(X_train_final_cluster)
#         X_test_final_cluster = pd.DataFrame(X_test_final_cluster)
#
#     missing_cols = set(X_train_final_cluster.columns) - set(X_test_final_cluster.columns)
#     for c in missing_cols:
#         X_test_final_cluster[c] = 0
#
#     # Ensure the order of column in the test set is in the same order than in train set
#     X_test_final_cluster = X_test_final_cluster[X_train_final_cluster.columns]
#
#     print(
#         f"Before RBFSampler for Cluster {cluster} - Training: {X_train_final_cluster.shape[1]}, Test: {X_test_final_cluster.shape[1]}")
#
#     # Applying RBFSampler for this cluster
#     cluster_rbf_sampler = RBFSampler(n_components=18, random_state=42)
#     X_train_final_cluster = cluster_rbf_sampler.fit_transform(X_train_final_cluster)
#     X_test_final_cluster = cluster_rbf_sampler.transform(X_test_final_cluster)
#
#     print(
#         f"After RBFSampler for Cluster {cluster} - Training: {X_train_final_cluster.shape[1]}, Test: {X_test_final_cluster.shape[1]}")
#
#     # Check for feature mismatch
#     if X_train_final_cluster.shape[1] != X_test_final_cluster.shape[1]:
#         print(
#             f"Feature mismatch in Cluster {cluster}: Training has {X_train_final_cluster.shape[1]} features, Test has {X_test_final_cluster.shape[1]} features.")
#         continue
#
#     # Train a KernelRidge model and measure its training time
#     model_cluster = KernelRidge(kernel='rbf')
#     model_cluster.fit(X_train_final_cluster, y_train_cluster_scaled)
#
#     cluster_models[cluster] = model_cluster  # Store the trained model
#
#     # Make predictions for this cluster
#     y_pred_cluster_scaled = model_cluster.predict(X_test_final_cluster)
#     y_pred_cluster = y_cluster_scalers[cluster].inverse_transform(y_pred_cluster_scaled.reshape(-1, 1)).ravel()
#
#     # Calculate RMSE for this cluster and store
#     rmse_cluster = np.sqrt(mean_squared_error(y_test_cluster, y_pred_cluster))
#     cluster_performances[cluster] = rmse_cluster
#
#     # Print results for this cluster
#     print()
#     print(f"Cluster {cluster}:")
#     print(f"Root Mean Squared Error (RMSE): {rmse_cluster:.2f}")
#     print("-" * 50)
#
# # Compare the RMSE of each cluster-specific model with the RMSE of the global model
# print(f"Global Model RMSE (repeated for clarity): {global_rmse:.2f}")
#
# # Visualization
# # For the global model
# actual_values_global = y_test_scaled
# predicted_values_global = y_pred_scaled
#
# # Sorting and grouping by 'Incident Year'
# actual_test_accidents = y_test.groupby(X_test['Incident Year']).sum()
# predicted_test_accidents = pd.DataFrame({'Incident Year': X_test['Incident Year'], 'Predicted': y_pred}).groupby(
#     'Incident Year').sum()
#
# plt.figure(figsize=(15, 7))
# plt.plot(actual_test_accidents.index, actual_test_accidents.values, label='Actual (Test)', marker='o', linestyle='--',
#          color='green')
# plt.plot(predicted_test_accidents.index, predicted_test_accidents['Predicted'], label='Predicted (Test)', marker='x',
#          color='red')
# plt.title('Yearly Accident Counts for Global Model (with RBFSampler): Actual vs Predicted')
# plt.xlabel('Year')
# plt.ylabel('Accident Count')
# plt.legend()
# plt.grid(True)
#
# # For each cluster-specific model
# for cluster in clusters:
#     df_cluster = df[df['Cluster'] == cluster]
#     X_cluster = df_cluster[numeric_features + categorical_features]
#     y_cluster = df_cluster[target]
#     X_train_cluster, X_test_cluster, y_train_cluster, y_test_cluster = train_test_split(
#         X_cluster, y_cluster, test_size=0.2, random_state=42
#     )
#
#     # Scale y for this cluster
#     y_cluster_scaler = MinMaxScaler()
#     y_train_cluster_scaled = y_cluster_scaler.fit_transform(y_train_cluster.values.reshape(-1, 1)).ravel()
#     y_test_cluster_scaled = y_cluster_scaler.transform(y_test_cluster.values.reshape(-1, 1)).ravel()
#     y_cluster_scalers[cluster] = y_cluster_scaler  # Store the y scaler for this cluster
#
#     # Preprocess data as before using the global encoder
#     X_train_encoded_cluster = encoder.transform(X_train_cluster[categorical_features])
#     X_test_encoded_cluster = encoder.transform(X_test_cluster[categorical_features])
#
#     # Construct the full feature set for the training data
#     X_train_final_cluster = np.hstack((X_train_cluster[numeric_features].values, X_train_encoded_cluster))
#
#     # Construct the full feature set for the test data
#     X_test_final_cluster = np.hstack((X_test_cluster[numeric_features].values, X_test_encoded_cluster))
#
#     # Standardize the features for both training and test set using a cluster-specific scaler
#     cluster_scaler = StandardScaler()
#     X_train_final_cluster = cluster_scaler.fit_transform(X_train_final_cluster)
#     X_test_final_cluster = cluster_scaler.transform(X_test_final_cluster)
#     cluster_scalers[cluster] = cluster_scaler  # Store the scaler for this cluster
#
#     # Applying RBFSampler for this cluster to the training data
#     cluster_rbf_sampler = RBFSampler(n_components=18, random_state=42)
#     X_train_final_cluster = cluster_rbf_sampler.fit_transform(X_train_final_cluster)
#     # Store the RBFSampler for this cluster
#     cluster_rbfsamplers[cluster] = cluster_rbf_sampler
#
#     # Applying RBFSampler for this cluster to the test data
#     X_test_final_cluster = cluster_rbfsamplers[cluster].transform(X_test_final_cluster)
#
#     # Train a KernelRidge model and measure its training time
#     model_cluster = KernelRidge(kernel='rbf')
#     model_cluster.fit(X_train_final_cluster, y_train_cluster_scaled)
#
#     cluster_models[cluster] = model_cluster  # Store the trained model
#
#     # Make predictions for this cluster
#     y_pred_cluster_scaled = model_cluster.predict(X_test_final_cluster)
#     y_pred_cluster = y_cluster_scalers[cluster].inverse_transform(y_pred_cluster_scaled.reshape(-1, 1)).ravel()
#
#     # Calculate RMSE for this cluster and store
#     rmse_cluster = np.sqrt(mean_squared_error(y_test_cluster_scaled, y_pred_cluster_scaled))
#     cluster_performances[cluster] = rmse_cluster
#
#     # Visualization for each cluster
#     actual_test_accidents_cluster = y_test_cluster.groupby(X_test_cluster['Incident Year']).sum()
#     predicted_test_accidents_cluster = pd.DataFrame(
#         {'Incident Year': X_test_cluster['Incident Year'], 'Predicted': y_pred_cluster}
#     ).groupby('Incident Year').sum()
#
#     plt.figure(figsize=(15, 7))
#     plt.plot(actual_test_accidents_cluster.index, actual_test_accidents_cluster.values, label='Actual (Test)',
#              marker='o', linestyle='--', color='green')
#     plt.plot(predicted_test_accidents_cluster.index, predicted_test_accidents_cluster['Predicted'],
#              label='Predicted (Test)', marker='x', color='red')
#     plt.title(f'Yearly Accident Counts for Cluster {cluster} (with RBFSampler): Actual vs Predicted')
#     plt.xlabel('Year')
#     plt.ylabel('Accident Count')
#     plt.legend()
#     plt.grid(True)
#     plt.show()
