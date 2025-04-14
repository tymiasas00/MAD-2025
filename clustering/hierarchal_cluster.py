# Importowanie niezbędnych bibliotek
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import scipy.cluster.hierarchy as sch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from data.data_analysis import movies_data  # Import przetworzonego DataFrame

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
plt.title('Dendrogram – wszystkie cechy')
plt.xlabel('Filmy')
plt.ylabel('Odległość')
plt.tight_layout()
plt.show()

# Klasteryzacja hierarchiczna z wybraną liczbą klastrów
hierarchical = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
data_clean['hierarchical_cluster'] = hierarchical.fit_predict(scaled_data)

# Dołączenie etykiet klastrów do oryginalnego DataFrame
movies_data.loc[data_clean.index, 'hierarchical_cluster'] = data_clean['hierarchical_cluster']

# Wizualizacja: wszystkie możliwe pary cech
for x_feature, y_feature in combinations(features, 2):
    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        data=data_clean,
        x=x_feature,
        y=y_feature,
        hue='hierarchical_cluster',
        palette='Set1',
        alpha=0.7
    )
    plt.title(f'Hierarchiczna klasteryzacja: {x_feature} vs {y_feature}')
    plt.xlabel(x_feature)
    plt.ylabel(y_feature)
    plt.legend(title='Klaster')
    plt.tight_layout()
    # plt.savefig(f'hierarchical_{x_feature}_vs_{y_feature}.png')  # opcjonalnie zapis
    plt.show()
