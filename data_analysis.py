import kagglehub
import matplotlib.pyplot as plt
import pandas as pd

# Pobieram dane z Kaggle
path = kagglehub.dataset_download(
    "andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews")
print("Ścieżka do plików zestawu danych:", path)

# Wczytanie danych
movies = pd.read_csv(f'{path}/rotten_tomatoes_movies.csv')
reviews = pd.read_csv(f'{path}/rotten_tomatoes_movie_reviews.csv')

# Złączenie dwóch plików
data = pd.merge(movies, reviews, on='id', how='inner')

# podstawowe informacje o połączonym zbiorze
print(data.info())
print('-' * 20)
print(data.head())
print('-' * 20)
print(data.describe())
print('-' * 20)
print(data.columns)
print('-' * 20)
print(data.isnull().sum())


# Usunięcie niepotrzebnych kolumn z movies
columns_to_drop = [
    'rating', 'ratingContents',
    'boxOffice', 'writer', 'distributor', 'soundMix', 'reviewState', 'reviewUrl'
]
data = data.drop(columns=columns_to_drop)

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

numeric_features = ['audienceScore', 'tomatoMeter',
                    'runtimeMinutes',]
# Obliczanie podstawowych wartości
min_values = data[numeric_features].min()  # Minimum
max_values = data[numeric_features].max()  # Maksimum
mean_values = data[numeric_features].mean()  # Średnia
median_values = data[numeric_features].median()  # Mediana

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

# Histogramy dla cech numerycznych
for column in numeric_features:
    plt.figure(figsize=(8, 6))  # Size of each histogram plot
    data[column].hist(bins=30, edgecolor='black')  # Plot histogram
    plt.title(f"Histogram of {column}", fontsize=16)
    plt.xlabel(column, fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.show()

# wykres słupkowy dla najczęstszych gatunków
genre_counts = data['genre'].value_counts()
plt.figure(figsize=(12, 8))  # Size of the plot
genre_counts.head(10).plot(kind='bar', color='skyblue',
                           edgecolor='black')  # Top 10 genres
plt.title("Most Common Genres", fontsize=16)
plt.xlabel("Genre", fontsize=14)
plt.ylabel("Number of Movies", fontsize=14)
plt.xticks(rotation=45, ha='right')  # Rotate genre names for readability
plt.show()
