from accident_graph import G, nx, pd
from scipy.sparse import csgraph
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import numpy as np
import matplotlib.pyplot as plt

# Convert the graph to an adjacency matrix
adj_matrix = nx.adjacency_matrix(G).toarray()

# Step 1: Apply RBF Kernel on the adjacency matrix
gamma = 0.1
adj_matrix_rbf = np.exp(-gamma * adj_matrix ** 2)

# Step 2: Perform t-SNE on the entire dataset
tsne = TSNE(n_components=2, method='exact')
tsne_result = tsne.fit_transform(adj_matrix_rbf)

# Step 3: Apply PCA on the t-SNE result
pca_tsne = PCA(n_components=2)  # Reduce to 2 dimensions
pca_tsne_result = pca_tsne.fit_transform(tsne_result)

# Get first row, first column, and last column of adjacency matrix
first_row = adj_matrix[0]
first_col = adj_matrix[:, 0]
last_col = adj_matrix[:, -1]
print(f"First row: {first_row}", f"First column: {first_col}", f"Last column: {last_col}", sep="\n")

# Compute the normalized Laplacian
laplacian = csgraph.laplacian(adj_matrix, normed=True)

# Step 3: Compute eigenvalues and eigenvectors
num_eigenvectors = 3  # Experimented with different values here, found 3 to be the best since 2 lost too much information
eigenvalues, eigenvectors = eigsh(laplacian, k=num_eigenvectors, which='SM')

# Normalize the eigenvectors
eigenvectors = normalize(eigenvectors)

best_score = -1
best_k = 2  # minimum number of clusters

# Trying different numbers of clusters and find the best one based on silhouette score
for k in range(2, 11):  # testing different numbers of clusters
    kmeans = KMeans(n_clusters=k, n_init=10)
    labels = kmeans.fit_predict(eigenvectors[:, :2])  # use first 2 eigenvectors

    score = silhouette_score(eigenvectors, labels)

    if score > best_score:
        best_score = score
        best_k = k

    print(f"For k={k}, silhouette score: {score}")

print(f"\nBest number of clusters (k): {best_k}")
print(f"Best silhouette score: {best_score}")

# Use best_k as the number of clusters for KMeans
kmeans = KMeans(n_clusters=best_k, n_init=10)
labels = kmeans.fit_predict(eigenvectors[:, :2])  # use first 2 eigenvectors

# Apply PCA to reduce dimensions to 2
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(eigenvectors)

node_labels = dict(zip(G.nodes, labels))
node_coordinates = dict(zip(G.nodes, principalComponents))

# Create DataFrame to hold results
results = pd.DataFrame(columns=["Cluster", "Node", "Coordinates"])

for label in set(labels):
    for node, node_label in node_labels.items():
        if node_label == label:
            # Append result to DataFrame
            results = results._append({
                "Cluster": label,
                "Node": node,
                "Coordinates": node_coordinates[node]
            }, ignore_index=True)

# Save DataFrame to CSV
results.to_csv('cluster_results.csv', index=False)

# Create a scatter plot to visualize the results
tick_label_fontsize = 14

plt.figure(figsize=(8, 8))
for i in range(len(principalComponents)):
    plt.scatter(principalComponents[i][0], principalComponents[i][1],
                color=plt.cm.nipy_spectral(labels[i] / 10.), s=7)

plt.xticks(fontsize=tick_label_fontsize)
plt.yticks(fontsize=tick_label_fontsize)

plt.show()
