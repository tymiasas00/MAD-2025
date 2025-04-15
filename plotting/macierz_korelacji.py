import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from clustering.kmeans_clustering import movies_data_KMeans
from clustering.hierarchal_cluster import movies_data_hierarchy
from clustering.dbscan_clustering import movies_data_DBScan

# Wczytanie plików z etykietami klastrów
kmeans_df = movies_data_KMeans
dbscan_df = movies_data_hierarchy
hierarchical_df = movies_data_DBScan

# Połączenie danych po ID filmu
combined = kmeans_df[['id', 'kmeans_cluster']].merge(
    dbscan_df[['id', 'dbscan_cluster']], on='id', how='inner'
).merge(
    hierarchical_df[['id', 'hierarchical_cluster']], on='id', how='inner'
)

# Obliczenie macierzy korelacji
correlation_matrix = combined[['kmeans_cluster', 'dbscan_cluster', 'hierarchical_cluster']].corr(method='pearson')

# Rysowanie macierzy korelacji
plt.figure(figsize=(6, 4))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Macierz korelacji między metodami klasteryzacji")
plt.tight_layout()
plt.show()


