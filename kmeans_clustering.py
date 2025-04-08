import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Wczytaj dane
data = pd.read_csv('movies_grouped.csv')

# 2. Wybierz tylko kolumny numeryczne do klasteryzacji
features = ['audienceScore', 'tomatoMeter', 'runtimeMinutes']

# 3. Zachowaj tylko wiersze bez braków danych
data_clean = data[features].dropna()

# 4. Standaryzacja danych
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data_clean)

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
optimal_k = 3
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42)
data_clean['kmeans_cluster'] = kmeans_final.fit_predict(scaled_features)

# 8. Dołącz etykiety klastrów do oryginalnego dataframe
data.loc[data_clean.index, 'kmeans_cluster'] = data_clean['kmeans_cluster']

# 9. Zapisz dane z klastrami
data.to_csv('movies_clustered.csv', index=False)

# 10. Wizualizacja
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x=data['audienceScore'],
    y=data['tomatoMeter'],
    hue=data['kmeans_cluster'],
    palette='Set2'
)
plt.title('K-means: audienceScore vs tomatoMeter')
plt.xlabel('Audience Score')
plt.ylabel('Tomato Meter')
plt.legend(title='Klaster')
plt.tight_layout()
plt.show()
