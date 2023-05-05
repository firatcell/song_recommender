from bs4 import BeautifulSoup
import requests
import pandas as pd
from time import sleep
from random import randint
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
from config import *
import numpy as np
# import libraries
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
from IPython.display import HTML

import sys
from termcolor import colored, cprint

pd.set_option('display.width', 50000)
pd.set_option('display.max_colwidth',None)
#pd.set_option('display.large_repr','truncate')

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=client_id,
                                                           client_secret=client_secret_id))

def search_song(title, artist, limit=5):
    id_song = []
    # Search for song on Spotify
    query = "artist: " + artist + " track: " + title
    
    # Get ID if found, else add NaN
    try: # if everything goes well
        results = sp.search(q=query, limit=limit)
        song_id = results["tracks"]["items"][0]["id"]
        id_song.append(song_id)
    
    
    except:
        song_id = np.nan
        id_song.append(song_id)
        print("ID not found for {} and {}".format(artist, title))
 
    return id_song

def get_audio_features(id_song):
    '''
    This function gets a dataframe with a column with songs' ids and
    returns a dataframe with ALL audio features
    '''
    # create empty dictionary
    audio_features_dict = {}

    # get the audio features for the songs in this chunk
    audio_features_list = sp.audio_features(id_song)


    # iterate through the audio features and add them to the dictionary
    for features in audio_features_list:
        if features:
            audio_features_dict[features['id']] = features
            

    # cast dictionary into dataframe and reset index
    features_df = pd.DataFrame.from_dict(audio_features_dict, orient='index').reset_index()
    

    return features_df

def load(filename = "filename.pickle"): 
    try: 
        with open(filename, "rb") as file: 
            return pickle.load(file) 
    except FileNotFoundError: # specific python error message
        print("File not found!") # it will print this error ONLY for the error specify.

def song_hotness(df_user, df_all_songs): 
    '''
    This function checks in which dataset is the user song and it adds a new column to the df_user with the dataset.
    '''
    if (df_user['id'].values[0] in list(df_all_songs[df_all_songs['dataset']=="hot"]['ids'].values)):
            df_user['dataset'] = 'hot'
    else:
            df_user['dataset'] = 'not_hot'
    return df_user

def get_url(df):
    urls = []
    for song_id in df['id']:
        try:
            track = sp.track(song_id)
            url = track["external_urls"]["spotify"]
            urls.append(url)
        except:
            pass
    return urls

def recommend_song(df_user, df_all_songs):
    songs_df = df_all_songs[(df_all_songs['dataset']==df_user['dataset'].values[0]) & (df_all_songs['KMeans']==df_user['KMeans'].values[0])]
    if (songs_df.shape[0] < 5):
        recommended_songs = songs_df
    else:
        recommended_songs = songs_df.sample(5)
    return recommended_songs

def make_clickable(val):
    """
    Function to convert a URL string to a clickable HTML link.
    """
    return f'<a href="{val}" target="_blank">{val}</a>'


def best_recommender():
    answer = ""
    while answer != "no":
        title = input("What song title do you have in mind?: ")
        artist = input("What's the artist of the song?: ")
        title = title.title()
        artist = artist.title()

        songs_database_df = pd.read_csv('all_songs_clusters.csv')

        song_id = search_song(title, artist)

        df_audio_features = get_audio_features(song_id)

        X = df_audio_features[['danceability', 'energy', 'acousticness', 'instrumentalness', 'valence', 'tempo', 'time_signature']]

        scaler2 = load("scaler.pickle")
        best_model = load("kmeans_13.pickle")
        X_scaled = scaler2.transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

        cluster = best_model.predict(X_scaled_df)

        user_df = song_hotness(df_audio_features, songs_database_df)
        user_df['KMeans'] = cluster

        recommended_songs_df = recommend_song(user_df, songs_database_df)
        recommended_songs_df['Public URL'] = recommended_songs_df['ids'].apply(lambda x: "https://open.spotify.com/track/" + x)
        recommended_songs_df.rename(columns={'titles': 'Song titles', 'artists': 'Artists'}, inplace=True)
        recommended_songs_df = recommended_songs_df[['Song titles', 'Artists', 'Public URL']]

        print(colored(f"\nHere are some songs similar to {title} by {artist}:\n", 'cyan'))
        recommended_songs_df = recommended_songs_df.style.format({'Public URL': make_clickable})
        display(recommended_songs_df)
        
        answer = input("\nDo you want another song recommendation? (yes/no) ")
    print("   ")
    cprint('**************************************','green',attrs=['bold'])
    print(colored("Thanks for using our song recommender!", "green", attrs=['bold']))
    cprint('**************************************','green',attrs=['bold'])
