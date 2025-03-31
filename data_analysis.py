import pandas as pd

# Wczytanie przesłanych plików próbki
movies_df = pd.read_csv('rotten_tomatoes_movies.csv')
reviews_df = pd.read_csv('rotten_tomatoes_movie_reviews.csv')

# Usunięcie niepotrzebnych kolumn z movies_df
columns_to_drop_movies = [
    'rating', 'ratingContents', 'releaseDateTheaters',
    'boxOffice', 'writer', 'distributor', 'soundMix'
]
movies_df.drop(columns=[col for col in columns_to_drop_movies if col in movies_df.columns], inplace=True)

# Usunięcie niepotrzebnych kolumn z reviews_df
reviews_df = reviews_df.rename(columns={'id': 'movie_id'})
columns_to_drop_reviews = ['reviewState', 'publicationName']
reviews_df.drop(columns=[col for col in columns_to_drop_reviews if col in reviews_df.columns], inplace=True)

# Mapowanie 'scoreSentiment' na wartości liczbowe
sentiment_map = {'POSITIVE': 1, 'NEUTRAL': 0, 'NEGATIVE': -1}
reviews_df['scoreSentiment'] = reviews_df['scoreSentiment'].map(sentiment_map)

# Ponowna agregacja recenzji
review_agg = reviews_df.groupby('movie_id').agg({
    'scoreSentiment': 'mean',
    'reviewId': 'count'
}).rename(columns={'scoreSentiment': 'avg_sentiment', 'reviewId': 'review_count'}).reset_index()

# Ponowne połączenie z danymi o filmach
movies_with_reviews = pd.merge(movies_df, review_agg, left_on='id', right_on='movie_id', how='left')

# Wybór kolumn numerycznych do analizy
num_cols = ['audienceScore', 'tomatoMeter', 'avg_sentiment', 'review_count', 'runtimeMinutes']
num_cols = [col for col in num_cols if col in movies_with_reviews.columns]

# Opis statystyczny z dodatkiem skośności
stats = movies_with_reviews[num_cols].describe().T
stats['skewness'] = movies_with_reviews[num_cols].skew()

# Obsługa braków, uzupełnienie medianą
for col in num_cols:
    if movies_with_reviews[col].isna().sum() > 0:
        median_val = movies_with_reviews[col].median()
        movies_with_reviews[col] = movies_with_reviews[col].fillna(median_val)

# Wykrywanie wartości odstających - 1.5 * IQR
def detect_outliers(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[col] < lower_bound) | (df[col] > upper_bound)]

# Usunięcie odstających wartości z każdej kolumny numerycznej
for col in num_cols:
    outliers = detect_outliers(movies_with_reviews, col)
    movies_with_reviews = movies_with_reviews[~movies_with_reviews.index.isin(outliers.index)]

import ace_tools_open as tools; tools.display_dataframe_to_user(name="Statystyki opisowe po oczyszczeniu", dataframe=stats)

import matplotlib.pyplot as plt
import seaborn as sns

print(movies_with_reviews['audienceScore'].nunique())
print(movies_with_reviews['audienceScore'].dtype)
print(movies_with_reviews['audienceScore'].describe())


# Histogram ocen widzów z zaznaczoną min i max
plt.figure(figsize=(10, 5))
sns.histplot(data=movies_with_reviews, x='originalScore', bins=30, kde=False)
plt.axvline(movies_with_reviews['originalScore'].min(), color='red', linestyle='--', linewidth=2, label='Najniższa ocena')
plt.axvline(movies_with_reviews['originalScore'].max(), color='green', linestyle='--', linewidth=2, label='Najwyższa ocena')
plt.axvline(movies_with_reviews['originalScore'].median(), color='blue', linestyle='-', linewidth=3,label='Mediana')
plt.title('Rozkład ocen widzów')
plt.xlabel('Ocena widzów (originalScore)')
plt.ylabel('Liczba filmów')
plt.legend()
plt.tight_layout()
plt.show()

# Histogram ocen widzów z zaznaczoną min i max
plt.figure(figsize=(10, 5))
sns.histplot(data=movies_with_reviews, x='audienceScore', bins=30, kde=False)
plt.axvline(movies_with_reviews['audienceScore'].min(), color='red', linestyle='--', linewidth=2, label='Najniższa ocena')
plt.axvline(movies_with_reviews['audienceScore'].max(), color='green', linestyle='--', linewidth=2, label='Najwyższa ocena')
plt.axvline(movies_with_reviews['audienceScore'].median(), color='blue', linestyle='-', linewidth=3,label='Mediana')
plt.title('Rozkład ocen widzów')
plt.xlabel('Ocena widzów (audienceScore)')
plt.ylabel('Liczba filmów')
plt.legend()
plt.tight_layout()
plt.show()