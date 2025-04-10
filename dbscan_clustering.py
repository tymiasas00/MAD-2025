import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import numpy as np
from data_analysis import movies_data  # Import przetworzonego DataFrame

# 1. Wczytaj oczyszczony DataFrame z data_analysis.py
data = movies_data

# 2. Wybierz tylko kolumny numeryczne do klasteryzacji
features = ['audienceScore', 'tomatoMeter', 'runtimeMinutes']
data_clean = data[features].dropna().copy()

# 3. Standaryzacja danych
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_clean)

# 4. Wybór dobrego eps (odległość sąsiedztwa) – wykres k-Distance
neighbors = NearestNeighbors(n_neighbors=5)
neighbors_fit = neighbors.fit(scaled_features)
distances, indices = neighbors_fit.kneighbors(scaled_features)

# Sortuj i narysuj – pomoże dobrać eps
distances = np.sort(distances[:, 4])
plt.figure(figsize=(8, 5))
plt.plot(distances)
plt.title('k-Distance Graph (do wyboru eps)')
plt.xlabel('Punkty posortowane')
plt.ylabel('Odległość do 5-tego sąsiada')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. Dopasuj DBSCAN – ustal eps na podstawie wykresu powyżej
# UWAGA: warto najpierw poeksperymentować z wartością eps!
dbscan = DBSCAN(eps=0.35, min_samples=5)
clusters = dbscan.fit_predict(scaled_features)

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
