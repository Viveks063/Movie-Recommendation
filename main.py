import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

movies = pd.read_csv('movie_dataset.csv')

print(movies.head())

print(movies.info())

movies['tags'] = movies['genres'] + movies['overview']

new_df = movies[['id', 'title', 'genres', 'overview', 'tags']]

print(new_df.head())

new_df.drop(columns=['genres', 'overview'])

print(new_df.head())

cv = CountVectorizer(max_features= 10000, stop_words='english')

vec = cv.fit_transform(new_df['tags'].values.astype('U')).toarray()

print(vec)

print(vec.shape)

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vec)

print(similarity)

def recommend_movies(movies):
    index = new_df[new_df['title'] == movies].index[0]
    dist = sorted(list(enumerate(similarity[index])), reverse=True, key = lambda value:value[1])

    recommendations = []
    for i in dist[1:6]:
        recommendations.append(new_df.iloc[i[0]].title)

    return recommendations

st.title("Movie Recommendation")
st.subheader("Find movies similar to your favorite!")

movies = st.text_input("Enter movies here : ")

if st.button("Recommend"):
    recommendations = recommend_movies(movies)

    if recommendations:
        st.write("Movies you might like:")
        for rec in recommendations:
            st.write(f"{rec}")

