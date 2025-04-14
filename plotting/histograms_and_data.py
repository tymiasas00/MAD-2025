from data.data_analysis import movies_data  # Import przetworzonego DataFrame
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import zscore

# Przypisanie danych do zmiennej
data = movies_data

# podstawowe informacje o połączonym zbiorze po usunięciu kolumn
print(data.info())
print('-' * 20)
print(data.head())
print('-' * 20)
print(data.describe())
print('-' * 20)
print(data.columns)
print('-' * 20)
print(data.isnull().sum())
print('-' * 20)
print(data.dtypes.value_counts())

numeric_features = ['audienceScore', 'tomatoMeter', 'runtimeMinutes']
# Obliczanie podstawowych wartości
min_values = data[numeric_features].min()  # Minimum
max_values = data[numeric_features].max()  # Maksimum
mean_values = data[numeric_features].mean()  # Średnia
median_values = data[numeric_features].median()  # Mediana
std_values = data[numeric_features].std()  # Odchylenie standardowe
skewness_values = data[numeric_features].skew()  # Skośność

# Wyświetlanie wyników
print("Minimum:")
print(min_values)
print('-' * 20)

print("Maksimum:")
print(max_values)
print('-' * 20)

print("Średnia:")
print(mean_values)
print('-' * 20)

print("Mediana:")
print(median_values)
print('-' * 20)

print("Odchylenie standardowe:")
print(std_values)
print('-' * 20)

print("Skośność:")
print(skewness_values)
print('-' * 20)

# Histogramy dla cech numerycznych
for column in numeric_features:
    if column != 'runtimeMinutes':
        plt.figure(figsize=(8, 6))

        values = data[column].dropna()
        counts, bins = np.histogram(values, bins=30)
        percentages = (counts / counts.sum()) * 100

        plt.bar(bins[:-1], percentages, width=(bins[1] - bins[0]), edgecolor='black', align='edge')
        plt.title(f"Histogram of {column}", fontsize=16)
        plt.xlabel(column, fontsize=14)
        plt.ylabel('Procent', fontsize=14)
        plt.tight_layout()
        plt.show()

plt.figure(figsize=(8, 6))

info = data['runtimeMinutes']
counts, bins, patches = plt.hist(info, bins=50, edgecolor='black')
total = counts.sum()
percentages = (counts / total) * 100

plt.clf()  # czyścimy poprzedni wykres

# Rysujemy wykres z procentami
plt.bar(bins[:-1], percentages, width=(bins[1] - bins[0]), align='edge', edgecolor='black')
plt.xlim(0, 220)  # <- przenieśliśmy tutaj!
plt.title("Histogram of runtime", fontsize=16)
plt.xlabel("Runtime in minutes", fontsize=14)
plt.ylabel("Frequency (%)", fontsize=14)
plt.tight_layout()
plt.show()

# Przeliczenie gatunków na procenty
genre_counts = data['genre'].value_counts(normalize=True) * 100

# Wykres słupkowy dla top 10
plt.figure(figsize=(12, 8))
genre_counts.head(10).plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Most Common Genres (percentage)", fontsize=16)
plt.xlabel("Genre", fontsize=14)
plt.ylabel("Percentage of Movies", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Procent braków
missing_percent = data.isnull().mean() * 100
print("Braki danych (%):")
print(missing_percent[missing_percent > 0])
print('-' * 20)

# Wstępna detekcja obserwacji odstających
z_scores = data[numeric_features].apply(zscore)
outliers = (abs(z_scores) > 3).sum()

print("Liczba obserwacji odstających dla każdej zmiennej:")
print(outliers)
print('-' * 20)
