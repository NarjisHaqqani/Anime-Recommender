# Anime-Recommender
A simple content-based recommender app that suggests similar anime based on genre, type, rating, and popularity. Enter an anime name, and the system finds the top 10 most similar shows using TF-IDF + cosine similarity.
# Anime Recommender System

## Overview

This project is a content-based anime recommender system built using Python. The main goal of this project is to recommend similar anime based on their genres and type using similarity techniques.

## What I Did

* Loaded and cleaned the anime dataset
* Handled missing values in important columns
* Combined genre and type information to create a content feature
* Used **TF-IDF Vectorizer** to convert text data into numerical form
* Calculated similarity between anime using **Cosine Similarity**
* Built an interactive **Streamlit app** to display recommendation

## How It Works

* The user selects an anime name from the dropdown
* The system finds anime with similar content features
* Top similar anime are shown as recommendations

## Tools Used

* Python
* Pandas, NumPy
* Scikit-learn
* Streamlit

## Deployment

The project is deployed using **Streamlit Cloud** with the code hosted on **GitHub**.

## Future Scope

* Add anime descriptions for better recommendations
* Improve recommendation quality using collaborative filtering
* Enhance the UI by adding posters and more details

## Author:Narjiz Faroza Haqqani

