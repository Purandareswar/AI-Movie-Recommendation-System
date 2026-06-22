import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample movie dataset
movies = pd.DataFrame({
    'title': [
        'Avatar',
        'Titanic',
        'Avengers',
        'Iron Man',
        'The Dark Knight'
    ],
    'genre': [
        'Action Adventure Sci-Fi',
        'Romance Drama',
        'Action Superhero',
        'Action Superhero',
        'Action Crime Drama'
    ]
})

# Convert text into vectors
cv = CountVectorizer()
matrix = cv.fit_transform(movies['genre'])

# Calculate similarity
similarity = cosine_similarity(matrix)

def recommend(movie_name):
    movie_name = movie_name.lower()

    for i in range(len(movies)):
        if movies.iloc[i]['title'].lower() == movie_name:

            scores = list(enumerate(similarity[i]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            print("\nRecommended Movies:\n")

            for movie in scores[1:4]:
                print(movies.iloc[movie[0]]['title'])

            return

    print("Movie not found!")

# Test
movie = input("Enter movie name: ")
recommend(movie)