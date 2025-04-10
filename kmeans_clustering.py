import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from data_analysis import movies_data

# 2. Wybierz tylko kolumny numeryczne do klasteryzacji
features = [
    'audienceScore', 'tomatoMeter', 'runtimeMinutes', 'scoreSentiment', 'topCriticRatio',
    'min_originalScore', 'max_originalScore', 'mean_originalScore', 'median_originalScore'
]

# 3. Zachowaj tylko wiersze bez braków danych
data_clean = movies_data.dropna(subset=features).copy()

# 4. Standaryzacja danych
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_clean[features])

# 5. Metoda łokcia – wybór liczby klastrów
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

# 6. Rysuj wykres łokcia
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.title('Metoda łokcia – wybór liczby klastrów')
plt.xlabel('Liczba klastrów (K)')
plt.ylabel('Inercja (suma kwadratów odległości)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 7. Trenuj KMeans z wybraną liczbą klastrów (np. 3)
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
data_clean['kmeans_cluster'] = kmeans.fit_predict(scaled_features)

# 8. Dołącz etykiety klastrów do oryginalnego dataframe
movies_data.loc[data_clean.index, 'kmeans_cluster'] = data_clean['kmeans_cluster']

# # 9. Zapisz dane z klastrami
# data.to_csv('movies_clustered.csv', index=False)

# 10. Wizualizacja
for x_feature, y_feature in combinations(features, 2):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=data_clean,
        x=x_feature,
        y=y_feature,
        hue='kmeans_cluster',
        palette='Set2',
        alpha=0.7
    )
    plt.title(f'KMeans: {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Klaster')
    plt.tight_layout()
    #plt.savefig(f'kmeans_{x_feature}_vs_{y_feature}.png')  # Zapisz jako plik
    plt.show()  # Pokaż interaktywnie
