import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import numpy as np
from data.data_analysis import movies_data  # Import przetworzonego DataFrame

# Parametry DBSCAN
eps_value = 1  # wartość do zmiany po analizie wykresu k-Distance
min_samples_value = 7

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

# Dodanie etykiet do oryginalnego DataFrame
movies_data.loc[data_clean.index, 'dbscan_cluster'] = data_clean['dbscan_cluster']
# movies_data.to_csv('movies_data_DBScan.csv', index=False)
movies_data_DBScan = movies_data.copy()
# Wizualizacja wszystkich możliwych par
for x_feature, y_feature in combinations(features, 2):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=data_clean,
        x=x_feature,
        y=y_feature,
        hue='dbscan_cluster',
        palette='tab10',
        alpha=0.7
    )
    plt.title(f'DBSCAN: {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Klaster')
    plt.tight_layout()
    # plt.savefig(f'dbscan_{x_feature}_vs_{y_feature}.png')  # opcjonalnie
    plt.show()
