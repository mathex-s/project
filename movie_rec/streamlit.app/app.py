import streamlit as st
import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(layout="wide")
st.title("🎬 Movie Recommendation App")

# ---------- User Input ----------
OMDB_API_KEY = "1b456bd2"  # Replace with your OMDb API key
DATA_PATH = "muhammadnaumank/tmdb-5000-movies-dataset"  # Replace with your Kaggle path

# ---------- Load Dataset ----------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[['title', 'overview', 'id']].dropna(subset=['overview']).reset_index(drop=True)
    return df

df = load_data()

# ---------- Build similarity matrix ----------
@st.cache_data
def build_similarity():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(df['overview'])
    return cosine_similarity(tfidf_matrix)

similarity = build_similarity()

# ---------- OMDb API Functions ----------
def fetch_movie_details(title):
    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"
    data = requests.get(url).json()
    if data['Response'] == 'True':
        return {
            'Title': data.get('Title'),
            'Year': data.get('Year'),
            'Genre': data.get('Genre'),
            'Director': data.get('Director'),
            'Actors': data.get('Actors'),
            'Plot': data.get('Plot'),
            'Poster': data.get('Poster')
        }
    else:
        return None

# ---------- Recommendation Function ----------
def recommend_movies(movie_name, top_n=5):
    if movie_name not in df['title'].values:
        return []
    idx = df[df.title == movie_name].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    recs = scores[1:top_n+1]
    rec_movies = [df.iloc[i[0]].title for i in recs]
    return rec_movies

# ---------- Streamlit UI ----------
movie_name = st.text_input("Enter a movie name:")

if st.button("Recommend"):
    if movie_name:
        details = fetch_movie_details(movie_name)
        if details:
            st.subheader("🎬 Movie Details")
            col1, col2 = st.columns([1,2])
            with col1:
                if details['Poster'] != "N/A":
                    st.image(details['Poster'])
            with col2:
                st.write(f"**Title:** {details['Title']}")
                st.write(f"**Year:** {details['Year']}")
                st.write(f"**Genre:** {details['Genre']}")
                st.write(f"**Director:** {details['Director']}")
                st.write(f"**Actors:** {details['Actors']}")
                st.write(f"**Plot:** {details['Plot']}")

            recs = recommend_movies(movie_name, top_n=5)
            if recs:
                st.subheader("⭐ Recommended Movies")
                cols = st.columns(5)
                for idx, rec in enumerate(recs):
                    rec_details = fetch_movie_details(rec)
                    if rec_details:
                        with cols[idx % 5]:
                            if rec_details['Poster'] != "N/A":
                                st.image(rec_details['Poster'], use_column_width=True)
                            st.caption(f"{rec_details['Title']} ({rec_details['Year']})")
            else:
                st.warning("No recommendations found.")
        else:
            st.error("Movie not found in OMDb!")
    else:
        st.warning("Please enter a movie name.")
