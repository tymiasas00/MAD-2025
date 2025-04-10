import kagglehub
import pandas as pd
from  changing_original_score import parse_original_score
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
    'boxOffice', 'writer', 'distributor', 'soundMix', 'reviewState', 'reviewUrl',
    'reviewText','title','reviewId','creationDate','releaseDateStreaming','releaseDateTheaters',
    'publicatioName',
]
data = data.drop(columns=columns_to_drop)

# oScore = data['originalScore']
# oScore.to_csv('original_score.csv', index=False)
data['originalScore'] = data['originalScore'].apply(parse_original_score)
data = data.dropna()

# Usuwanie wartości odstających za pomocą IQR
Q1 = data[['tomatoMeter', 'audienceScore', 'runtimeMinutes', 'originalScore']].quantile(0.25)
Q3 = data[['tomatoMeter', 'audienceScore', 'runtimeMinutes', 'originalScore']].quantile(0.75)
IQR = Q3 - Q1

# Filtruj dane, usuwając wartości odstające
data = data[~((data[['tomatoMeter', 'audienceScore', 'runtimeMinutes', 'originalScore']] < (Q1 - 1.5 * IQR)) | 
              (data[['tomatoMeter', 'audienceScore', 'runtimeMinutes', 'originalScore']] > (Q3 + 1.5 * IQR))).any(axis=1)]

sentiment_map = {'POSITIVE': 1, 'NEGATIVE': 0}
isTopCritic_map = {True: 1, False: 0}
data['scoreSentiment'] = data['scoreSentiment'].map(sentiment_map)
data['isTopCritic'] = data['isTopCritic'].map(isTopCritic_map)
data.to_csv('reviews_with_movie_info.csv', index=False)

movies_data = {
    'id': [],
    'tomatoMeter':[],
    'audienceScore':[],
    'runtimeMinutes':[],
    'scoreSentiment':[],
    'genre':[],
    'originalLanguage':[],
    'director':[],
    'topCriticRatio':[],
    'min_originalScore':[],
    'max_originalScore':[],
    'mean_originalScore':[],
    'median_originalScore':[],
}

movie_id = 'adrift_2018' # Autorytarnie wybieram pierwsze id

movie_sentiment_score = []
movie_isTopCritic = []
movie_originalScore = []
for index, row in data.iterrows():
    if row['id'] == movie_id:
        movie_tomato_meter=row['tomatoMeter']
        movie_audience_score=row['audienceScore']
        movie_sentiment_score.append(row['scoreSentiment'])
        movie_originalScore.append(row['originalScore'])
        runtime_minutes=row['runtimeMinutes']
        movie_isTopCritic.append(row['isTopCritic'])
        original_language=row['originalLanguage']
        director=row['director']
        genre=row['genre']
    else:
        movies_data['id'].append(movie_id)
        movies_data['tomatoMeter'].append(movie_tomato_meter)
        movies_data['audienceScore'].append(movie_audience_score)
        movies_data['scoreSentiment'].append(sum(movie_sentiment_score)/len(movie_sentiment_score))
        movies_data['runtimeMinutes'].append(runtime_minutes)
        movies_data['genre'].append(genre)
        movies_data['originalLanguage'].append(original_language)
        movies_data['director'].append(director)
        movies_data['topCriticRatio'].append(sum(movie_isTopCritic) / len(movie_isTopCritic))
        movies_data['min_originalScore'].append(min(movie_originalScore))
        movies_data['max_originalScore'].append(max(movie_originalScore))
        movies_data['mean_originalScore'].append(sum(movie_originalScore) / len(movie_originalScore))
        movies_data['median_originalScore'].append(pd.Series(movie_originalScore).median())

        # Czyszczę tabelki i zmieniam id, które będę badał
        movie_id = row['id']
        movie_tomato_meter = row['tomatoMeter']
        movie_audience_score = row['audienceScore']
        movie_sentiment_score = [row['scoreSentiment']]
        runtime_minutes = row['runtimeMinutes']
        genre = row['genre']
        director = row['director']
        original_language = row['originalLanguage']
        movie_isTopCritic = [row['isTopCritic']]
        movie_originalScore = [row['originalScore']]

movies_data = pd.DataFrame(movies_data)

