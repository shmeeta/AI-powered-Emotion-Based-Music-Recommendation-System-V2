from kneed import KneeLocator 
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.preprocessing import StandardScaler 
import math 
import numpy as np 
import pandas as pd 


# pick a set of representative songs from a users favorite songs library using kmeans
# allows for the similarity search to run on a handful of vectors instead of every liked song

MIN_SONGS_FOR_CLUSTERING = 5
MAX_CLUSTERS_CAP = 10 
TARGET_REPRESENTATIVES = 5


def choose_n_clusters(X_scaled: np.ndarray) -> int: 
    n = len(X_scaled)
    max_clusters = min(math.ceil(math.sqrt(n/2))+1, MAX_CLUSTERS_CAP)

    if max_clusters<=2: 
        return min(2,n)

    wcss = [
        MiniBatchKMeans(
            n_clusters = k, init = 'k-means++', batch_size=256, random_state=42
        ).fit(X_scaled).inertia_
        for k in range(1,max_clusters)
    ]

    knee = KneeLocator(
        range(1,max_clusters), wcss, curve ="convex", direction='decreasing'
    ).knee

    if knee is None or knee<1: 
        return min(3,n)
    return min(int(knee), n)

def select_representative_songs(
        user_encoded: pd.DataFrame, 
        feature_columns: list, 
        target: int = TARGET_REPRESENTATIVES, 

) -> pd.DataFrame: 
    keep_cols = ['artist_name', 'track_name'] + list(feature_columns)

    if len(user_encoded) <= MIN_SONGS_FOR_CLUSTERING: 
        return user_encoded[keep_cols].reset_index(drop=True)

    X_df = (
        user_encoded[feature_columns].apply(pd.to_numeric, errors='coerce').dropna()
    )
    valid_indices = X_df.index


    X_scaled = StandardScaler().fit_transform(X_df)

    n_clusters = choose_n_clusters(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    representatives = [] #positions of the song closest to each centroi
    song_distances = [] #(distance, position) for eevry song for topu p 

    for cluster_idx in range(n_clusters): 
        members = np.where(labels == cluster_idx)[0]
        if len(members) == 0: 
            continue
        dists = np.linalg.norm(X_scaled[members]-kmeans.cluster_centers_[cluster_idx], axis = 1)
        representatives.append(int(members[np.argmin(dists)]))
        song_distances.extend(zip(dists,members ))


    # top up with the next closest songs if we have fewer than target: 
    if len(representatives) < target and song_distances: 
        dists, positions = zip(*song_distances)
        for pos in np.array(positions)[np.argsort(dists)]: 
            if len(representatives) >= target:
                break 
            if int(pos) not in representatives: 
                representatives.append(int(pos))

    selected = valid_indices[representatives]
    return user_encoded.loc[selected, keep_cols].reset_index(drop=True)




    
    

