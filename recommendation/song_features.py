# feature engineering for the song recommender 
# everything that depends on the song dataset is built once and cached 

from dataclasses import dataclass
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

USER_COLUMNS = [
    'artist_name', 'track_name', 'popularity', 'year', 'genre', 'danceability', 'energy', 'loudness', 
    'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',

]

NORM_COLS = ['loudness', 'year', 'tempo']

EXCLUDE_COLS = ['artist_name', 'track_name', 'emotion', 'instrumentalness', 'liveness', 'time_signature']


@dataclass 
class SongFeatures: 
    encoded: pd.DataFrame 
    genre_encoder: OneHotEncoder 
    scaler: MinMaxScaler
    vector_columns: list 
    matrix: np.ndarray

def _encode_genres(df: pd.DataFrame, encoder: OneHotEncoder) -> pd.DataFrame: 
    arr = encoder.transform(df[['genre']])
    enc_df = pd.DataFrame(
        arr, 
        columns = encoder.get_feature_names_out(['genre']), 
        index = df.index,
    )
    return pd.concat([df.drop('genre', axis = 1), enc_df], axis = 1)

def build_features(song_dataset: pd.DataFrame) -> SongFeatures: 
    # expensive- run once not per request 
    genre_encoder = OneHotEncoder(
        sparse_output = False, handle_unknown='ignore', dtype=np.float32
    )
    genre_encoder.fit(song_dataset[['genre']])
    encoded = _encode_genres(song_dataset, genre_encoder)

    for col in NORM_COLS: 
        encoded[col] = pd.to_numeric(encoded[col], errors = 'coerce')

    scaler = MinMaxScaler()
    encoded[NORM_COLS] = scaler.fit_transform(encoded[NORM_COLS])

    vector_columns = [c for c in encoded.columns if c not in EXCLUDE_COLS]

    matrix = encoded[vector_columns].to_numpy(dtype=np.float32)

    return SongFeatures(encoded, genre_encoder, scaler, vector_columns, matrix)

def build_user_df(songs_qs) -> pd.DataFrame: 
    # turn the users favorite song queryset into a dataframe
    rows = [[getattr(song, col) for col in USER_COLUMNS] for song in songs_qs]
    return pd.DataFrame(rows, columns = USER_COLUMNS)


def encode_user_df(user_df: pd.DataFrame, features: SongFeatures) -> pd.DataFrame: 
    # encode and scale the users songs using the already fitted encoder and scaler 
    encoded = _encode_genres(user_df, features.genre_encoder)
    encoded = encoded.reindex(columns=features.encoded.columns, fill_value=0)
    for col in NORM_COLS: 
        encoded[col] = pd.to_numeric(encoded[col], errors='coerce')

    encoded[NORM_COLS] = features.scaler.transform(encoded[NORM_COLS])
    return encoded 


# cached access
_FEATURES = None 

def get_song_features(song_dataset: pd.DataFrame) -> SongFeatures: 
    # build on the first call and reuse afterwards
    global _FEATURES
    if _FEATURES is None: 
        _FEATURES = build_features(song_dataset)
    return _FEATURES
