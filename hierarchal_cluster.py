# Importowanie bibliotek
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from itertools import combinations
from data_analysis import movies_data  # Zakładamy, że to gotowy DataFrame

# Parametr: liczba klastrów wybrana na podstawie dendrogramu
n_clusters = 3

# Lista cech do analizy
features = [
    'audienceScore', 'tomatoMeter', 'runtimeMinutes', 'scoreSentiment', 'topCriticRatio',
    'min_originalScore', 'max_originalScore', 'mean_originalScore', 'median_originalScore'
]

# Oczyszczenie danych (usuwamy wiersze z brakującymi wartościami)
data_clean = movies_data.dropna(subset=features).copy()

# Standaryzacja cech
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean[features])

# Dendrogram – do oceny liczby klastrów
plt.figure(figsize=(10, 7))
sch.dendrogram(sch.linkage(scaled_data, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Filmy')
plt.ylabel('Odległość')
plt.tight_layout()
plt.show()

# Klasteryzacja hierarchiczna z wybraną liczbą klastrów
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
data_clean['hierarchical_cluster'] = hierarchical.fit_predict(scaled_data)

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
