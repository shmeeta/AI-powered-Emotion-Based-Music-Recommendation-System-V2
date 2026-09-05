import json
import os
import re
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.shortcuts import redirect, render
import pandas as pd
from rapidfuzz import fuzz, process
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

from .models import FavoriteSong, DislikedSongs

# access spotify client from settings
sp = getattr(settings, 'SPOTIFY_CLIENT', None)
if not sp:
  sp = spotipy.Spotify(
      auth_manager=SpotifyClientCredentials(
          client_id=os.environ.get('SPOTIFY_CLIENT_ID'),
          client_secret=os.environ.get('SPOTIFY_CLIENT_SECRET'),
      )
  )


SONG_DATASET = pd.read_csv('song_dataset_16_aug.csv')


# handling entering the 5 favourite songs
@login_required
def enter_favorites(request):
    prev_data = {}
    errors = {}
    songs_added = []
    songs_qs = FavoriteSong.objects.filter(user=request.user)
    # get songs that are stored in liked if they exist yet
    liked_pairs = set(
            (song.track_name, song.artist_name) for song in  songs_qs 
        )


    # for cleaning the user text (to increase chances of matching the dataset - same function used in cleaning dataset)
    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9 ]+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()


    if request.method== 'POST':
        titles = [k for k in request.POST.keys() if k.startswith("title")] # find all the fields that look like the titleX
        for t_field in titles: 
            idx = t_field.replace("title","")
            title = clean_text(request.POST.get(f"title{idx}")) 
            artist = clean_text(request.POST.get(f"artist{idx}"))
            prev_data[f"title{idx}"] = title
            prev_data[f"artist{idx}"] = artist
            if not (title and artist):
                continue # to skip empty entries
            # if found but already exists in liked songs then alert user that it already exists
         
            if (title, artist) in liked_pairs: 
                    errors[idx] = f"'{title}' by {artist} is already in your liked songs."
                    continue    
               
               
            # start with exacct matches
            row = SONG_DATASET[
                (SONG_DATASET['artist_name'].str.lower() == artist.lower()) &
                (SONG_DATASET['track_name'].str.lower() == title.lower())
            ]

            # if not found resort to fuzzy match in the case of typos etc
            if row.empty:
                search_string = f"{artist} {title}".lower()
                dataset_strings = (
                    SONG_DATASET['artist_name'].str.lower() + " " + SONG_DATASET['track_name'].str.lower()
                ).tolist()

                best_match_result = process.extractOne( #retrn tuple with 3 items
                    search_string,
                    dataset_strings,
                    scorer=fuzz.token_sort_ratio,
                    score_cutoff=80  #if no match meets the cutoff then return none so no match
                )
                if best_match_result:
                    _, score, match_idx = best_match_result
                    row = SONG_DATASET.iloc[[match_idx]]
            # if still not found :
            if row.empty:
                errors[idx] = f"'{title}' by {artist} is not in our dataset, please try another."
                continue
            song_data = row.iloc[0]
            cover_url = None
            query = f"track:{song_data['track_name']} artist:{song_data['artist_name']}"
            try:
                results = sp.search(q=query, type="track", limit=1)
                items = results.get('tracks', {}).get('items')
                if items: 
                    cover_url = items[0]['album']['images'][0]['url']
            except Exception as e: 
                print(f"Error fetching cover art for '{song_data['track_name']}': {e}")
            
            
            FavoriteSong.objects.create(
                user=request.user,
                rating=5,
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
                cover_url=cover_url
            )
            songs_added.append(f"{song_data['track_name']} by {song_data['artist_name']}")


            # if a liked song was added that was previously in disliked songs, delete that song to prevent issues 
            DislikedSongs.objects.filter(
                user = request.user, 
                track_name=song_data['track_name'], 
                artist_name = song_data['artist_name']
            ).delete()

        # user feedback: 
        if songs_added:
            messages.success(request, f"Successfully added: {', '.join(songs_added)}")
        if errors:
            messages.warning(request, "Some songs were not found in our dataset or are already added to your liked songs. Please check and try again.")

    return render(request, 'liked_songs/enter_favorites.html', {"prev_data": prev_data, "errors": errors})



# HANDLING THE GENERAL DISPLAY OF LIKED SONGS IN THE LIKED SONGS TAB

@login_required
def liked_songs(request):
    if request.method == "POST":
        return redirect('liked_songs:enter_favorites')

    # allows for an instant query wihtout the api calls for faster loading
    songs = FavoriteSong.objects.filter(user=request.user)

    return render(request, 'liked_songs/liked_songs.html', {'songs': songs})

@login_required
def delete_liked(request):
    
    if request.method =="POST":
        data = json.loads(request.body)
        title = data.get("title")
        artist = data.get("artist")

        # delete from fav song model (button click on template )
        deleted_count, _ = FavoriteSong.objects.filter(
            user=request.user,
            track_name=title,
            artist_name=artist
        ).delete()

        if deleted_count:
            status = f"Deleted {title} by {artist}"
        else:
            status = f"No matching song found for {title} by {artist}"
        return JsonResponse({"status": "ok", "deleted": status})
    return JsonResponse({"status": "error", "message": "POST required"})
       
   
        