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

# Usunięcie dodatkowych kolumn z reviews_df
columns_to_drop_reviews = ['reviewState', 'reviewUrl']
reviews_df.drop(columns=[col for col in columns_to_drop_reviews if col in reviews_df.columns], inplace=True)

# Zmiana nazwy kolumny 'id' w reviews_df
reviews_df = reviews_df.rename(columns={'id': 'movie_id'})

# Ponowne połączenie z danymi o filmach
movies_with_reviews = pd.merge(movies_df, reviews_df, left_on='id', right_on='movie_id', how='outer')
# Usuwam duplikat kolumny
movies_with_reviews.drop(columns=['movie_id'], inplace=True)


# Zapisuję do pliku, by nie odczytywać, za każdym razem
movies_with_reviews.to_csv('movies_with_reviews.csv', index=False)

