import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import numpy as np
from data_analysis import movies_data  # Zakładamy gotowy DataFrame

# Parametry DBSCAN
eps_value = 1  # wartość do zmiany po analizie wykresu k-Distance
min_samples_value = 5

# Lista cech
features = [
    'audienceScore', 'tomatoMeter', 'runtimeMinutes', 'scoreSentiment', 'topCriticRatio',
    'min_originalScore', 'max_originalScore', 'mean_originalScore', 'median_originalScore'
]

# Oczyszczanie danych
data_clean = movies_data.dropna(subset=features).copy()

# Standaryzacja
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_clean[features])

# Wykres k-Distance do wyboru eps
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_data)
distances, indices = neighbors_fit.kneighbors(scaled_data)
distances = np.sort(distances[:, 4])

plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title('k-Distance Graph (do wyboru eps)')
plt.xlabel('Punkty posortowane')
plt.ylabel('Odległość do 5-tego sąsiada')
plt.grid(True)
plt.tight_layout()
plt.show()

# Dopasowanie DBSCAN
dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
data_clean['dbscan_cluster'] = dbscan.fit_predict(scaled_data)

# 6. Dodaj klaster do danych
data_clean['dbscan_cluster'] = clusters

# 7. Zapisz dane z klastrami
data_clean.to_csv('movies_dbscan_clustered.csv', index=False)

# 8. Wizualizacja wyników DBSCAN (2 cechy)
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=data_clean,
    x='audienceScore',
    y='tomatoMeter',
    hue='dbscan_cluster',
    palette='Set2',
    legend='full'
)
plt.title('DBSCAN: audienceScore vs tomatoMeter')
plt.xlabel('Audience Score')
plt.ylabel('Tomato Meter')
plt.legend(title='Klaster')
plt.tight_layout()
plt.show()
