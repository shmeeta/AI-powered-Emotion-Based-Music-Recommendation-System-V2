import os
import json
import re
import math 
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from kneed import KneeLocator
from concurrent.futures import ThreadPoolExecutor
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
from sklearn.preprocessing import OneHotEncoder,MinMaxScaler

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

# helper function to fetch cover art 
def fetch_song_cover(x):
    track = x["track"]
    artist = x["artist"]
    cover_url = None 
    query = f"track:{track} artist:{artist}"

    try: 
        results = sp.search(q=query, type="track", limit=1)
        items = results.get('tracks',{}).get('items', [])
        if items and items[0]['album']['images']:
            cover_url= items[0]['album']['images'][0]['url']
        else:
            cover_url = static("users/alt_cover.png")
    except Exception as e: 
        print(f"Error fetching the cover art for '{track}: {e}")
        cover_url = static("users/alt_cover.png")

    return{
        'title': track, 
        'artist': artist, 
        'cover_url': cover_url, 

    }







# helper function to convert nan values from pandas to defaults 
def clean_val(val, default="unknown"):
    if pd.isna(val): 
        return default
    else: 
        return val

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

        # get the cover url 
        
        cover_url = data.get("cover_url") or data.get("image_url")



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

        if not cover_url: 
            cover_url = clean_val(song_data.get('cover_url'), default=None)

        # if still not then query spotify for the api cover
        if not cover_url: 
            query = f"track:{song_data['track_name']} artist:{song_data['artist_name']}"
            try:
                results = sp.search(q=query, type="track", limit=1)
                items = results.get('tracks', {}).get('items')
                if items:
                    cover_url = items[0]['album']['images'][0]['url']
            except Exception as e:
                print(f"Error fetching Spotify cover art for '{song_data['track_name']}': {e}")

        if not cover_url: 
            cover_url = "/static/users/alt_cover.png"


        
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
                cover_url=cover_url,
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
        genre_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        genre_encoder.fit(SONG_DATASET[['genre']])

        def encode_genres(df, encoder):
            encoded = encoder.transform(df[['genre']])
            encoded_df = pd.DataFrame(
                encoded, 
                columns = encoder.get_feature_names_out(['genre']),
                index=df.index
            )
            return pd.concat([df.drop('genre', axis=1), encoded_df], axis=1)

        song_dataset_encoded = encode_genres(SONG_DATASET, genre_encoder)
        user_df_encoded = encode_genres(user_df, genre_encoder)

        # align the user columns to match the dataset structure 
        user_df_encoded_aligned = user_df_encoded.reindex(columns=song_dataset_encoded.columns, fill_value=0)

        # fit and transform the minmax scaler to prevent division by zero
        norm_cols = ['loudness', 'year', 'tempo']

        for col in norm_cols: 
            song_dataset_encoded[col] = pd.to_numeric(song_dataset_encoded[col], errors='coerce')
            user_df_encoded_aligned[col] = pd.to_numeric(user_df_encoded_aligned[col], errors='coerce')

        scaler = MinMaxScaler() 
        song_dataset_encoded[norm_cols] = scaler.fit_transform(song_dataset_encoded[norm_cols])
        user_df_encoded_aligned[norm_cols] = scaler.transform(user_df_encoded_aligned[norm_cols])

        exclude_cols = ['artist_name', 'track_name', 'emotion', 'instrumentalness', 'liveness', 'time_signature']
        vector_columns = [c for c in song_dataset_encoded.columns if c not in exclude_cols]

            
        
      
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

        num_songs= len(user_df_encoded_aligned)

        if num_songs>5: 
            X_df = user_df_encoded_aligned[cols].apply(pd.to_numeric, errors='coerce').dropna()
            valid_indices = X_df.index # keep track of the indices that survived dropna()

            scaler = StandardScaler() # scale the features so the tempo and popularity dont overpower 0/1 genre flags
            X_scaled = scaler.fit_transform(X_df)
            print("Length before clustering:", len(X_df))

            # replaced with a dynamic cap for clusters based on N (max 10) to improve efficiency 
            max_clusters = min(math.ceil(math.sqrt(len(X_df) / 2)) + 1, 10)

            if max_clusters <=2: 
                n_clusters = min(2, len(X_df))
            else: 
                # faster elbow search with minibatchkmeans 
                wcss = [
                    MiniBatchKMeans(n_clusters = i, init='k-means++', batch_size=256, random_state=42).fit(X_scaled).inertia_
                    for i in range (1,max_clusters)
                ]
                n_clusters = KneeLocator(
                    range(1, max_clusters), wcss, curve='convex', direction='decreasing'
                ).knee

                if n_clusters is None or n_clusters<1: 
                    n_clusters = min(3,len(X_df))


            print("optimal number of clusters is ", n_clusters)


            kmeans = KMeans(n_clusters = n_clusters, init ='k-means++', random_state=42)
            y_means = kmeans.fit_predict(X_scaled)


            representatives = []
            song_distances = []

            for cluster_idx in range(n_clusters): 
                cluster_songs_idx = np.where(y_means == cluster_idx)[0]
                if len(cluster_songs_idx)==0:
                    continue
                distances = np.linalg.norm(X_scaled[cluster_songs_idx] - kmeans.cluster_centers_[cluster_idx], axis=1)
                closest_in_cluster = cluster_songs_idx[np.argmin(distances)]
                representatives.append(closest_in_cluster)
                song_distances.extend(zip(distances, cluster_songs_idx))


            if len(representatives) <5 and song_distances: 
                dists, indices = zip(*song_distances)
                sorted_indices = np.array(indices)[np.argsort(dists)]

                for idx in sorted_indices:
                    if len(representatives) >= 5:
                        break
                    if idx not in representatives: 
                        representatives.append(int(idx))

            selected_df_indices = valid_indices[representatives]

            user_df_encoded_aligned =(
                user_df_encoded_aligned.loc[selected_df_indices, ['artist_name', 'track_name'] + cols].reset_index(drop=True)
            )

            print("Length of df after clustering:", len(user_df_encoded_aligned))


        
            
        

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


            dataset_emotions = SONG_DATASET['emotion'].values
            dataset_tracks = SONG_DATASET['track_name'].values
            dataset_artists = SONG_DATASET['artist_name'].values




            for idx in sorted_indices:
               song_emotion = dataset_emotions[idx]
               song_track = dataset_tracks[idx]
               song_artist = dataset_artists[idx]
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
                track = dataset_tracks[idx]
                artist = dataset_artists[idx]
                
                song = (track, artist)


                if song in rec_pair:
                    continue

                # append final recommendations array
                row = SONG_DATASET.iloc[idx]
                recommendations.append({
                "track":track, 
                "artist": artist,
                })
                rec_pair.add(song)
            

        

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

        with ThreadPoolExecutor(max_workers = 5) as executor: 
            songs_with_cover_art = list(executor.map(fetch_song_cover, recommendations))
        hide_intro = True
      
   
    return render(request, "recommendation/get_recommendation.html", {
        "emotion_result": emotion_result,
        "recommendations": songs_with_cover_art,
        
        "hide_intro" : hide_intro
    })