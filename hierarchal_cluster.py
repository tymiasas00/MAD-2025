# Importowanie niezbędnych bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from data_analysis import movies_data  # Import przetworzonego DataFrame

# Wczytanie danych bezpośrednio z data_analysis.py
movies_with_reviews = movies_data

# Przygotowanie danych: wybór tylko potrzebnych kolumn
cluster_data = movies_with_reviews[['audienceScore', 'tomatoMeter', 'runtimeMinutes']].dropna()

# Skalowanie danych (normalizacja)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(cluster_data)

# Klasteryzacja hierarchiczna: tworzymy dendrogram
plt.figure(figsize=(10, 7))
sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Filmy')
plt.ylabel('Odległość')
plt.show()

# Na podstawie dendrogramu wybieramy liczbę klastrów, np. 3
hierarchical = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
hierarchical_labels = hierarchical.fit_predict(scaled_data)

# Dodanie etykiet klastrów do danych
movies_with_reviews['hierarchical_cluster'] = hierarchical_labels

# Wizualizacja wyników klasteryzacji hierarchicznej
plt.figure(figsize=(10, 7))
sns.scatterplot(x=movies_with_reviews['audienceScore'], y=movies_with_reviews['tomatoMeter'],
                hue=movies_with_reviews['hierarchical_cluster'], palette='tab10', s=100)
plt.title('Klasteryzacja Hierarchiczna')
plt.xlabel('Audience Score')
plt.ylabel('Tomato Meter')
plt.legend(title='Klaster')
plt.show()
