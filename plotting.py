import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

movies_with_reviews = pd.read_csv('movies_grouped.csv')


# Agregacja: jedna wartość na film
agg_funcs = {
    'audienceScore': 'mean',
    'tomatoMeter': 'mean',
    'runtimeMinutes': 'mean',
    'genre': 'first'
}

# Wybór kolumn numerycznych do analizy
movies_grouped = movies_with_reviews.groupby('id').agg(agg_funcs).reset_index()

# movies_grouped.to_csv('movies_grouped.csv', index=False)

# Histogramy dla cech liczbowych
numeric_cols = ['audienceScore', 'tomatoMeter']
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=movies_grouped, x=col, bins=30)
    plt.title(f'Histogram: {col}')
    plt.xlabel(col)
    plt.ylabel('Liczba filmów')
    plt.tight_layout()
    plt.show()

plt.figure(figsize=(8, 4))
sns.histplot(data=movies_grouped, x='runtimeMinutes', bins=50)
plt.xlim(0, 300)  # ogranicz oś X do 5 godzin
plt.title('Histogram: runtimeMinutes')
plt.xlabel('runtimeMinutes')
plt.ylabel('Liczba filmów')
plt.tight_layout()
plt.show()




# Wykres słupkowy dla najczęstszych gatunków
plt.figure(figsize=(10, 5))
movies_grouped['genre'] = movies_grouped['genre'].str.split(r'[|,]').str[0]
top_genres = movies_grouped['genre'].value_counts().head(10)
sns.barplot(x=top_genres.values, y=top_genres.index, palette='muted')
plt.title('Top 10 najczęstszych gatunków filmowych (na podstawie unikalnych tytułów)')
plt.xlabel('Liczba filmów')
plt.ylabel('Gatunek')
plt.tight_layout()
plt.show()

# Wykres słupkowy: średnia wartość tomatoMeter dla każdego gatunku
plt.figure(figsize=(10, 5))
avg_tomato_by_genre = movies_grouped.groupby('genre')['tomatoMeter'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=avg_tomato_by_genre.values, y=avg_tomato_by_genre.index, palette='Paired')
plt.title('Średnia wartość tomatoMeter dla 10 najpopularniejszych gatunków')
plt.xlabel('Średnia wartość tomatoMeter')
plt.ylabel('Gatunek')
plt.tight_layout()
plt.show()