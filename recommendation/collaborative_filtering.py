import pandas as pd 
import numpy as np
from liked_songs.models import FavoriteSong, DislikedSongs
from sklearn.metrics.pairwise import cosine_similarity

# implement collaborative filtering - get user item matrix + cosine similarity
def collaborative_filtering(username):
    data = []

    # Collect all ratings from FavoriteSong
    for entry in FavoriteSong.objects.all():
        try:
            rating = int(entry.rating)
        except (ValueError, TypeError):
            rating = np.nan

        data.append({
            "song_id": entry.id,                 
            "username": entry.user.username,
            "artist": entry.artist_name,
            "track_name": entry.track_name,
            "genre": entry.genre,
            "valence": entry.valence,
            "rating": rating,
        })

    # Collect all ratings from DislikedSongs
    for entry in DislikedSongs.objects.all():
        try:
            rating = int(entry.rating)
        except (ValueError, TypeError):
            rating = np.nan

        data.append({
            "song_id": entry.id,                  
            "username": entry.user.username,
            "artist": entry.artist_name,
            "track_name": entry.track_name,
            "genre": entry.genre,
            "valence": entry.valence,
            "rating": rating,
        })

    # Convert to DataFrame
    user_song_ratings = pd.DataFrame(data)

    
    #user_song_ratings.to_csv("user_song_ratings.csv", index=False)   
    # form the user item matrix using pivot 
    user_song_ratings_pivot = user_song_ratings.pivot_table(index='username', columns='track_name', values='rating', fill_value=0)  # fill the missing ratings with 0

    #similarity matrix
    similarity_matrix = cosine_similarity(user_song_ratings_pivot)
    similarity_matrix_df = pd.DataFrame(similarity_matrix, index=user_song_ratings_pivot.index, columns = user_song_ratings_pivot.index)

    #similarity_matrix_df.to_csv("similarity_matrix.csv", index=False)
    user_name = username
    similarities = similarity_matrix_df[user_name].drop(user_name)
    weights = similarities/similarities.sum()
    n= 10 # number of similar users 
    user_similarity_threshold = 0.1 #change according to size of user database 

    # get the top similar users
    most_similar_user = similarity_matrix_df[similarity_matrix_df[user_name]>user_similarity_threshold][user_name].sort_values(ascending=False)[:n]
    if most_similar_user.index[0] == user_name: 
        most_similar_user = most_similar_user[1:] # remove the first element to inhibit same username being returned
    
    no_sim_user = "There are no similar users."
    if most_similar_user.empty: 
        return no_sim_user
    else: 
        return most_similar_user.index[0]
  