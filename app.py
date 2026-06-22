import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import base64

def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img = get_base64("movie background picture.jpg")

st.markdown(
    f"""
    <style>
    .stApp {{
        background:
            linear-gradient(
                rgba(0,0,0,0.60),
                rgba(0,0,0,0.60)
            ),
            url("data:image/jpg;base64,{img}");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

st.set_page_config(
    page_title="AI Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Read movie dataset
movies = pd.read_csv("Movies-Dataset.csv", encoding="latin1")
movies = movies[['title', 'genre']]
movies.dropna(subset=['title', 'genre'], inplace=True)
movies = movies.head(5000)

# Create similarity matrix
movies['genre'] = movies['genre'].str.replace('|', ' ', regex=False)
cv = CountVectorizer()
@st.cache_resource
def create_similarity():
    cv = CountVectorizer()
    matrix = cv.fit_transform(movies['genre'].astype(str))
    return cosine_similarity(matrix)

similarity = create_similarity()

st.sidebar.title("📊 Project Statistics")
st.sidebar.write(f"🎬 Movies: {len(movies):,}")
st.sidebar.write("🤖 Algorithm: Cosine Similarity")
st.sidebar.write("🧠 Genre-Based Filtering")

# Recommendation function
def recommend(movie_name):
    movie_name = movie_name.lower()

    for i in range(len(movies)):
        if movies.iloc[i]['title'].lower() == movie_name:

            scores = list(enumerate(similarity[i]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)

            recommendations = []

            for movie in scores[1:11]:
                recommendations.append(
                    movies.iloc[movie[0]]['title']
                )

            return recommendations

    return []

# Streamlit UI
st.markdown(
    """
    <h1 style='text-align:center; color:#E50914;'>
    🎬 Content-Based AI Movie Recommendation System
    </h1>
    """,
    unsafe_allow_html=True
)

st.markdown(
    "<h4 style='text-align:center;'>Discover personalized movie recommendations 🍿 you'd love ❤️ to watch 🎥</h4>",
    unsafe_allow_html=True
)
st.divider()
st.write("Get movie recommendations based on your favorite movies.")

movie = st.selectbox(
    "Select a Movie",
    movies['title']
)

if st.button("Recommend"):

    result = recommend(movie)

    st.subheader("Recommended Movies")

    if result:

        cols = st.columns(2)

        for idx, movie in enumerate(result):

            genre = movies[movies['title'] == movie]['genre'].values[0]

            with cols[idx % 2]:

                st.markdown(
                    f"""
                    <div style="
                        border:1px solid #444;
                        border-radius:10px;
                        padding:15px;
                        margin:10px;
                        background-color:#1e1e1e;
                    ">
                        <h4>
                        <a href="https://www.google.com/search?q={movie}+movie"
                            target="_blank"
                            style="color:white; text-decoration:none;">
                            🎥 {movie}
                        </a>
                        </h4>
                        <p><b>Genre:</b> {genre}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.write("No recommendations found.")