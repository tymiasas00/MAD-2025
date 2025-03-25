import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Download latest version
path = kagglehub.dataset_download("andrezaza/clapper-massive-rotten-tomatoes-movies-and-reviews")

print("Path to dataset files:", path)

# Load the datasets
movies_path = path + "/rotten_tomatoes_movies.csv"  # Adjust the filename if necessary
reviews_path = path + "/rotten_tomatoes_movie_reviews.csv"  # Adjust the filename if necessary

movies_data = pd.read_csv(movies_path)
reviews_data = pd.read_csv(reviews_path)

# Display basic information about the datasets
print("Movies Data Info:")
print(movies_data.info())
print(movies_data.head())

print("Reviews Data Info:")
print(reviews_data.info())
print(reviews_data.head())

# Merge datasets if necessary (assuming they can be merged on a common column, e.g., 'movie_id')
merged_data = pd.merge(movies_data, reviews_data, on='id', how='inner')
print(merged_data.info())
print(merged_data.head())

# Plotting
# Example 1: Distribution of movie ratings
plt.figure(figsize=(10, 6))
sns.histplot(movies_data['rating'], bins=20, stat='percent')
plt.title('Distribution of Movie Ratings')
plt.xlabel('Rating')
plt.ylabel('Frequency %')
plt.show()

# Example 2: Average rating by genre
plt.figure(figsize=(12, 8))
avg_rating_by_genre = movies_data.groupby('genre')['rating'].mean().sort_values()
sns.barplot(x=avg_rating_by_genre.values, y=avg_rating_by_genre.index)
plt.title('Average Rating by Genre')
plt.xlabel('Average Rating')
plt.ylabel('Genre')
plt.show()

# Example 3: Number of movies released per year
plt.figure(figsize=(14, 7))
movies_per_year = movies_data['release_year'].value_counts().sort_index()
sns.lineplot(x=movies_per_year.index, y=movies_per_year.values)
plt.title('Number of Movies Released Per Year')
plt.xlabel('Year')
plt.ylabel('Number of Movies')
plt.show()
