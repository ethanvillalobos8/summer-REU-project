from spectral_clustering import adj_matrix_rbf
import matplotlib.pyplot as plt
import seaborn as sns

# Define the number of data points to display in the heatmap
subset_size = 100

# Subsample the similarity matrix to select a subset of data points
subset_similarity_matrix = adj_matrix_rbf[:subset_size, :subset_size]

# Visualize the similarity matrix as a heatmap
plt.figure(figsize=(8, 8))
sns.heatmap(subset_similarity_matrix, cmap='viridis', annot=False)
plt.title("Subset of Similarity Matrix")
plt.xlabel("Data Points")
plt.ylabel("Data Points")
plt.show()
