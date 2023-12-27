# Clustering-with-K-Means-Algorithm
Example using the K-Means algorithm for data clustering.
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Generate random data for clustering
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Apply K-Means algorithm
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Get cluster labels
labels = kmeans.labels_

# Visualize results
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.title('K-Means Clustering')
plt.show()
