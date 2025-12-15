import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import hstack

# LOAD AND PREPARE DATA

anime = pd.read_csv("anime.csv")

# Combine text fields
anime['genre'] = anime['genre'].fillna("")
anime['type'] = anime['type'].fillna("")
anime['text_content'] = anime['genre'] + " " + anime['type']

# TF-IDF for text
tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
tfidf_matrix = tfidf.fit_transform(anime['text_content'])

# Scale numeric features
anime[['rating', 'members']] = anime[['rating','members']].fillna(0)
scaler = MinMaxScaler()
numeric = scaler.fit_transform(anime[['rating','members']])

# Combine all features
content_matrix = hstack([tfidf_matrix, numeric])
cosine_sim = cosine_similarity(content_matrix)

# RECOMMENDATION FUNCTION

def recommend_anime(title):
    # Case-insensitive matching
    matches = anime[anime['name'].str.lower() == title.lower()]
    if matches.empty:
        return None
    
    idx = matches.index[0]

    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:11]

    anime_indices = [i[0] for i in sim_scores]

    # Return ONLY NAMES
    return anime.iloc[anime_indices]["name"].reset_index(drop=True)

# STREAMLIT UI

st.title("Anime Recommender System")

# User input
user_input = st.text_input("Enter an anime name:")

# Button
if st.button("Recommend"):
    if user_input.strip() == "":
        st.warning("Please enter an anime name.")
    else:
        recommendations = recommend_anime(user_input)

        if recommendations is None:
            st.error("Anime not found. Please type the exact name.")
        else:
            st.subheader("Recommended Anime:")
            for name in recommendations:
                st.write("• " + name)






