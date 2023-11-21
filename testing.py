import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Generate some sample data
np.random.seed(42)
data = np.random.rand(10, 2)

# Perform hierarchical clustering
# You can use different linkage methods (e.g., 'single', 'complete', 'average')
# and distance metrics (e.g., 'euclidean', 'cityblock', 'correlation')
linked = linkage(data, 'single')

# Plot the dendrogram
plt.figure(figsize=(10, 5))
dendrogram(linked, orientation='top', labels=None, distance_sort='descending', show_leaf_counts=True)
plt.title('Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Cluster Distance')
plt.show()
