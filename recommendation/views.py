import os
import json
import re
from spellchecker import SpellChecker
from django.http import JsonResponse
import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder
from kneed import KneeLocator

from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.templatetags.static import static

from liked_songs.models import FavoriteSong, DislikedSongs
from .model_loader import tokenizer, model
from django.shortcuts import render, redirect 
from django.contrib import messages

from django.conf import settings

SONG_DATASET = pd.read_csv('song_dataset_16_aug.csv')


# access spotify client from settings
sp = getattr(settings, 'SPOTIFY_CLIENT', None)
if not sp:
  sp = spotipy.Spotify(
      auth_manager=SpotifyClientCredentials(
          client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
          client_secret=os.environ.get('SPOTIFY_CLIENT_SECRET'),
      )
  )


# function to check for meaningful text input before classifying 
def is_meaningful_input(text): 
    spell = SpellChecker()
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip()
    if len(cleaned)<3: 
        return False
    words = cleaned.split() 

    has_valid_word = False
    for w in words: 
        if len(w)>=2: 
            has_valid_word = True 
            break
    if not has_valid_word: 
        return False

    real_candidates = [w for w in words if len(w)>=2]
    valid_words = spell.known(real_candidates)
    if not real_candidates: 
        return False

    # additionally check for 5 repeated characters in a row 
    if re.search(r'(.)\1{4,}', text):
        return False

    return True

    

# HANDLING THE RATINGS
@login_required 
def rating(request):
    if request.method =="POST":
        try: 
            data = json.loads(request.body)
        except (json.JSONDecodeError, TypeError): 
            return JsonResponse({"status": "error", "message": "Invalid JSON"}, status=400)


        
        title = data.get("title")
        artist = data.get("artist")
        value = data.get("value")



        if artist: 
            artist_clean = str(artist).strip().lower() 
        else: 
            artist_clean = ""

        if title: 
            title_clean =str(title).strip().lower() 
        else: 
            title_clean =""
        
        # add the songs to either the favorite songs or disliked songs, perform search of dataset to record attributes too, similar to the adding of liked songs to create a  favourite song object
        row = SONG_DATASET[
                (SONG_DATASET['artist_name'].str.lower() == artist_clean) &
                (SONG_DATASET['track_name'].str.lower() == title_clean)
        ]

        if row.empty: 
            return JsonResponse({"status": "error", "message": "Song not found in dataset."}, status=400)
        song_data = row.iloc[0]
        if value > 3 :  # if rated 4 or 5, it becomes a liked song 
            FavoriteSong.objects.create(
                user=request.user,
                rating= value,
                artist_name=song_data['artist_name'],
                track_name=song_data['track_name'],
                popularity=song_data.get('popularity', "unknown"),
                year=song_data.get('year', "unknown"),
                genre=song_data.get('genre', "unknown"),
                danceability=song_data.get('danceability', "unknown"),
                energy=song_data.get('energy', "unknown"),
                loudness=song_data.get('loudness', "unknown"),
                speechiness=song_data.get('speechiness', "unknown"),
                acousticness=song_data.get('acousticness', "unknown"),
                instrumentalness=song_data.get('instrumentalness', "unknown"),
                liveness=song_data.get('liveness', "unknown"),
                valence=song_data.get('valence', "unknown"),
                tempo=song_data.get('tempo', "unknown"),
            )
            
        elif value < 3: # if the value is  one or two it becomes a disliked song
            DislikedSongs.objects.create(
                user=request.user,
                rating= value,
                artist_name=song_data['artist_name'],
                track_name=song_data['track_name'],
                popularity=song_data.get('popularity', "unknown"),
                year=song_data.get('year', "unknown"),
                genre=song_data.get('genre', "unknown"),
                danceability=song_data.get('danceability', "unknown"),
                energy=song_data.get('energy', "unknown"),
                loudness=song_data.get('loudness', "unknown"),
                speechiness=song_data.get('speechiness', "unknown"),
                acousticness=song_data.get('acousticness', "unknown"),
                instrumentalness=song_data.get('instrumentalness', "unknown"),
                liveness=song_data.get('liveness', "unknown"),
                valence=song_data.get('valence', "unknown"),
                tempo=song_data.get('tempo', "unknown"),
            )
        
            # if value is 3 it will show up again in recommendations until rated
            
        return JsonResponse({"status": "ok", "received": value})




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
  


@login_required
def get_recommendation(request):
    songs_qs = FavoriteSong.objects.filter(user=request.user)
    if not songs_qs: 
     
        return redirect('liked_songs:liked_songs')
    # to handle hiding elements for cleaner design
    hide_intro = False
    # HANDLE EMOTIONS
    emotion_result = None
    recommendations = []
    songs_with_cover_art = []


    if request.method == "POST":
        short_text = request.POST.get("shortText", "").strip()
        if not short_text or not is_meaningful_input(short_text): 
            messages.error(request,"Please enter a valid message.")
            return render(request,"recommendation/get_recommendation.html", {
                "hide_intro": False,
                "short_text": short_text,
            })


        text = request.POST.get("shortText", "")  #get user text

        # Tokenize
        inputs = tokenizer(   # tokenize the text
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128,
        )

        # Predict the emotoin
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1).squeeze().tolist()

        # Get predicted label
        predicted_idx = torch.argmax(logits, dim=1).item()
        label_map = {
            "0": "sadness",
            "1": "joy",
            "2": "love",
            "3": "anger",
            "4": "fear",
          
        }

        print("DEBUG id2label_mapping:",label_map)
        print("DEBUG predicted_idx:", predicted_idx)
        print("DEBUG str(predicted_idx):", str(predicted_idx))
        predicted_label = label_map[str(predicted_idx)]
        emotion_probabilities = []
        for i, prob in enumerate(probs):
            label = label_map[str(i)]
            emotion_probabilities.append((label,prob))

        emotion_probabilities  = sorted(emotion_probabilities, key=lambda x: x[1], reverse=True)
        #for being sent to the template
        emotion_result = {
            "emotion": predicted_label,
            "probabilities": emotion_probabilities,
        }


        # deal with the song recommendations :
     
        # setting the emotion to the actual emotion and not the number + lowercase

        emotion_string = emotion_result["emotion"].lower()

        # build the vectors
    
        
        
      

        # get the users disliked songs
        disliked_songs = DislikedSongs.objects.filter(user=request.user)
        user_vectors= []
        for song in songs_qs:
            vector =[
                song.artist_name,
                song.track_name,
                song.popularity,
                song.year,
                song.genre,
                song.danceability,
                song.energy,
                song.loudness,
                song.speechiness,
                song.acousticness,
                song.instrumentalness,
                song.liveness,
                song.valence,
                song.tempo,
            ]
            user_vectors.append(vector)
        # create a df for the user:
        user_df = pd.DataFrame(user_vectors, columns =['artist_name','track_name','popularity','year','genre','danceability','energy','loudness','speechiness','acousticness','instrumentalness','liveness','valence','tempo'])
        

        # one hot encoding for the genre since it is the only non-numerical vector
        encoder =  OneHotEncoder(sparse_output = False, handle_unknown='ignore')
        encoded = encoder.fit_transform(user_df[['genre']])
        encoded_df = pd.DataFrame(
            encoded,
            columns=encoder.get_feature_names_out(['genre']),
            index=user_df.index
        )
        user_df_encoded = pd.concat([user_df.drop('genre', axis=1), encoded_df], axis=1)

        # for the main dataset encoding
        dataset_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        dataset_encoded = dataset_encoder.fit_transform(SONG_DATASET[['genre']])
        encoded_song_dataset = pd.DataFrame(
            dataset_encoded,
            columns = dataset_encoder.get_feature_names_out(['genre']),
            index = SONG_DATASET.index
        )
        song_dataset_encoded = pd.concat([SONG_DATASET.drop('genre', axis=1), encoded_song_dataset], axis=1)
        # normalizing main vector values:
        song_dataset_encoded['loudness'] = (
            (song_dataset_encoded['loudness']-song_dataset_encoded['loudness'].min())/(song_dataset_encoded['loudness'].max()-song_dataset_encoded['loudness'].min())
        )
        song_dataset_encoded['year'] = (
            (song_dataset_encoded['year']-song_dataset_encoded['year'].min())/(song_dataset_encoded['year'].max()-song_dataset_encoded['year'].min())
        )

        song_dataset_encoded['tempo'] = (
            (song_dataset_encoded['tempo']-song_dataset_encoded['tempo'].min())/(song_dataset_encoded['tempo'].max()-song_dataset_encoded['tempo'].min())
        )
        # reindex the dummy encoded to match the main song dataset encded columns
        user_df_encoded_aligned = user_df_encoded.reindex(columns=song_dataset_encoded.columns, fill_value=0)

        # define the actual vector columns - exclude certain columns
        vector_columns = [c for c in song_dataset_encoded.columns if c not in ['artist_name','track_name','emotion','instrumentalness','liveness','time_signature']]  # year, time_signature
        # normalizing
        user_df_encoded_aligned['loudness'] = pd.to_numeric(
            user_df_encoded_aligned['loudness'], errors='coerce'
        )
        user_df_encoded_aligned['loudness'] = (
            (user_df_encoded_aligned['loudness']-user_df_encoded_aligned['loudness'].min())/(user_df_encoded_aligned['loudness'].max()-user_df_encoded_aligned['loudness'].min())
        )
        user_df_encoded_aligned['year'] = pd.to_numeric(
            user_df_encoded_aligned['year'], errors='coerce'
        )

        user_df_encoded_aligned['year'] = (
            (user_df_encoded_aligned['year']-user_df_encoded_aligned['year'].min())/(user_df_encoded_aligned['year'].max()-user_df_encoded_aligned['year'].min())
        )
        user_df_encoded_aligned['tempo'] = pd.to_numeric(
            user_df_encoded_aligned['tempo'], errors='coerce'
        )
        user_df_encoded_aligned['tempo'] = (
            (user_df_encoded_aligned['tempo']-user_df_encoded_aligned['tempo'].min())/(user_df_encoded_aligned['tempo'].max()-user_df_encoded_aligned['tempo'].min())
        )
        
      


            
        
        
        # if more than 10 songs, use k means clustering to find 5 most similar songs, to keep recommendation numbers at 10
        if len(user_df_encoded_aligned)  > 5: 
            #user_df_encoded_aligned.to_csv("user_df_encoded_aligned.csv", index=False)
            cols =[

                'popularity','year','danceability','energy','loudness','speechiness',
                'acousticness','instrumentalness','liveness','valence','tempo',
                'genre_acoustic','genre_afrobeat','genre_altrock','genre_ambient',
                'genre_blackmetal','genre_blues','genre_breakbeat','genre_cantopop',
                'genre_chicagohouse','genre_chill','genre_classical','genre_club',
                'genre_comedy','genre_country','genre_dance','genre_dancehall',
                'genre_deathmetal','genre_deephouse','genre_detroittechno','genre_disco',
                'genre_drumandbass','genre_dub','genre_dubstep','genre_edm','genre_electro',
                'genre_electronic','genre_emo','genre_folk','genre_forro','genre_french',
                'genre_funk','genre_garage','genre_german','genre_gospel','genre_goth',
                'genre_grindcore','genre_groove','genre_guitar','genre_hardcore',
                'genre_hardrock','genre_hardstyle','genre_heavymetal','genre_hiphop',
                'genre_house','genre_indian','genre_indiepop','genre_industrial','genre_jazz',
                'genre_kpop','genre_metal','genre_metalcore','genre_minimaltechno','genre_newage',
                'genre_opera','genre_party','genre_piano','genre_pop','genre_popfilm',
                'genre_powerpop','genre_progressivehouse','genre_psychrock','genre_punk',
                'genre_punkrock','genre_rock','genre_rocknroll','genre_romance','genre_sad',
                'genre_salsa','genre_samba','genre_sertanejo','genre_showtunes',
                'genre_singersongwriter','genre_ska','genre_sleep','genre_songwriter',
                'genre_soul','genre_spanish','genre_swedish','genre_tango','genre_techno',
                'genre_trance','genre_triphop'
            ]
            X = user_df_encoded_aligned[cols].apply(pd.to_numeric, errors='coerce').dropna()
            X_array = X.to_numpy()
            print("Length before clustering:", len(X))
            
            # set the max clusters 
            max_clusters = len(X)+ 1
            wcss = []
            for i in range(1, max_clusters):
                kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
                kmeans.fit(X)
                wcss.append(kmeans.inertia_)
            n_clusters = KneeLocator([i for i in range(1,max_clusters)], wcss, curve='convex', direction='decreasing').knee
            # Fallback if KneeLocator fails
            if n_clusters is None or n_clusters < 1:
                n_clusters = 2
            print("Optimal number of clusters:", n_clusters)
            # fit KMeans with optimal clusters
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
            y_kmeans = kmeans.fit_predict(X)
            songs_with_clusters = user_df_encoded_aligned.copy()
            songs_with_clusters['cluster'] = y_kmeans
            # get distances and select representatives
            representatives = []
            song_distances = []
            for cluster_idx in range(n_clusters):
                cluster_songs_idx = np.where(y_kmeans == cluster_idx)[0]
                distances = np.linalg.norm(X_array[cluster_songs_idx] - kmeans.cluster_centers_[cluster_idx], axis=1)
                closest_idx = cluster_songs_idx[np.argmin(distances)]
                representatives.append(closest_idx)
                song_distances.extend(list(zip(distances, cluster_songs_idx)))  # store all distances

            remaining_slots = 5 - len(representatives)
            if remaining_slots > 0: 
                temp_distances = song_distances.copy()
                remaining = []
                while len(remaining) < remaining_slots and temp_distances:
                    # Find song with min dist
                    min_dist, min_idx = min(temp_distances, key=lambda x: x[0])
                    if min_idx not in representatives and min_idx not in remaining:
                        remaining.append(min_idx)
                    temp_distances = [item for item in temp_distances if item[1] != min_idx] # remove this song from temp distances

                representatives.extend(remaining)
            #get top fve songs
            top_5_songs = songs_with_clusters.iloc[representatives][['artist_name', 'track_name'] + cols]
            user_df_encoded_aligned = top_5_songs.reset_index(drop=True)

            print("Length of df after clustering:", len(user_df_encoded_aligned))
            print(user_df_encoded_aligned)


            print("length of df after clustering=", len(user_df_encoded_aligned))
            
            
  
        # numpy arrays
        user_final_vectors = user_df_encoded_aligned[vector_columns].values

        main_vectors = song_dataset_encoded[vector_columns].values

        # if the length of the user vectors is < 5 then attempt the collaborative filtering method do decrease sparseness
      

        user_final_vectors = [np.array(vec,dtype=float) for vec in user_final_vectors]
        main_vectors = [np.array(vec, dtype=float) for vec in main_vectors]
        print(user_df_encoded_aligned.isna().sum().sum())  # should be 0


        # CALCULATING THE SIMILARITY SCORES USING EUCLIDEAN DISTANCE:
        def get_euclidean_distance(a,b):
            return np.linalg.norm(a - b)

        #distances = []
        #for user_vec in user_final_vectors:
            #dist_scores = [get_euclidean_distance(user_vec, main_vec) for main_vec in main_vectors]
            #distances.append(dist_scores)

        distances = cdist(user_final_vectors, main_vectors, metric='euclidean')
        recommendations = []
        
        top_n = 2  # top two recommendations per song // ensure 10 recs
        liked_pairs = set(
            (row['track_name'], row['artist_name']) for _, row in user_df.iterrows()
        )
        disliked_pairs = set(
            (song.track_name, song.artist_name ) for song in disliked_songs
        )
        for i, scores in enumerate(distances):
            sorted_indices = np.argsort(scores)
            filtered_indices = []
            print(f"\nRecommendations for: {user_df.iloc[i]['track_name']}")
            for idx in sorted_indices:
               song_emotion = SONG_DATASET.iloc[idx]['emotion']
               song_track = SONG_DATASET.iloc[idx]['track_name']
               song_artist = SONG_DATASET.iloc[idx]['artist_name']
               match = False

               # if the song is the same as what the user liked already, skip it to avoid redundancy
               if (song_track, song_artist) in liked_pairs:
                   continue
               # avoid disliked songs popping up in recommendations again
               if (song_track, song_artist) in disliked_pairs: 
                   continue
                
                # matching algorithm according to final set of emotions, matching the datset
                   
               if emotion_string == "joy" and song_emotion in ["joy", "joy, love","joy, fear"]:
                   match = True
               elif emotion_string == "sadness" and song_emotion in ["sadness, fear", "sadness, anger, fear"]:
                   match = True
               elif emotion_string == "anger" and song_emotion == "sadness, anger, fear":
                   match = True
               elif emotion_string == "fear" and song_emotion in ["sadness, fear", "sadness, anger, fear"]:
                   match = True
               elif emotion_string == "love" and song_emotion in ["love", "joy, love"]:
                   match = True

             # fill the filtered indices until the top n is reached
               if match:
                   filtered_indices.append(idx)
               if len(filtered_indices) == top_n:
                   break

            rec_pair = {(rec['track'], rec['artist']) for rec in recommendations}

            for idx in filtered_indices:
                #handle duplicates
                song = (SONG_DATASET.iloc[idx]['track_name'], SONG_DATASET.iloc[idx]['artist_name'])
                if song in rec_pair:
                    continue

                # append final recommendations array
                row = SONG_DATASET.iloc[idx]
                recommendations.append({
                #"user_song": user_df.iloc[i]['track_name'],
                "track": row['track_name'],
                "artist": row['artist_name'],
                #"emotion": row['emotion']
                })
            

        

        # returning the album cover, name and title using the spotify api, similar to liked songs
        
        print(recommendations)

        # if len(recommendations)<10, implement  collaborative filtering, will return 10-x songs
        
        if len(recommendations) <10: 
            num_tracks_left = 10 - len(recommendations)
            user = request.user
            
            most_similar_user = collaborative_filtering(user.username)
           
            if not most_similar_user or  most_similar_user == "There are no similar users.":
                print("no similar users to append for now.")
            else: 
                 print("there is a similar user")
                 user_obj = User.objects.get(username=most_similar_user)
                 
                # get the most similar user's liked songs
                 similar_user_fav_songs = FavoriteSong.objects.filter(user=user_obj).only('track_name', 'artist_name')[:num_tracks_left]
                 existing_songs = {(rec["track"], rec["artist"]) for rec in recommendations}
                 new_songs = [song for song in similar_user_fav_songs if (song.track_name, song.artist_name) not in existing_songs]
                # get the most similar user's liked songs
                 for song in new_songs[:num_tracks_left]:
                     recommendations.append({"track": song.track_name, "artist": song.artist_name})
                     existing_songs.add((song.track_name, song.artist_name))

            
            

        print("length of recs", len(recommendations)) # for debugging

        for x in recommendations:
            track = x["track"]
            artist = x["artist"]
            cover_url = None
            query = f"track:{track} artist:{artist}"
            try:
                results = sp.search(q=query, type="track", limit=1)
                items = results.get('tracks', {}).get('items',[])
                if items and items[0]['album']['images']:
                    cover_url = items[0]['album']['images'][0]['url']
                else:
                    cover_url = static("users/alt_cover.png")

            except Exception as e:
                print(f"Error fetching cover art for '{track}': {e}")

            songs_with_cover_art.append({
                'title': track,
                'artist': artist,
                'cover_url': cover_url,
                
            })

            hide_intro = True
      
   
    return render(request, "recommendation/get_recommendation.html", {
        "emotion_result": emotion_result,
        "recommendations": songs_with_cover_art,
        
        "hide_intro" : hide_intro
    })
 