import spotipy
import pandas as pd
import time


# given csv file of uri,artist columns, return dictionary of format {"artist": "artistURI"}
# you have to manually make the csv file of artists inputte here
def load_uri_data(uri_data_path):
    df = pd.read_csv(uri_data_path)  # NOTE columns header "artist,uri" defined in the csv already
    return dict((row["artist"], row["uri"]) for _, row in df.iterrows())


# given spotipy object, artists dictionary of form {"artist": "uri"}, and list of track features to collect, get track data
# for all tracks of each artist and write this data as a csv to out_path in the given out_mode (should be 'w' or 'a')
def get_spotify_data(sp, artists_dict, col_features, out_path, out_mode='w'):
    # pd data frame for all of the data to collect
    final_df = pd.DataFrame(columns=col_features)

    # get data from all tracks in all albums of each artist_uri 
    for artist_name, artist_uri in artists_dict.items():
        artist_start = time.time()
        album_uris = [album["uri"] for album in sp.artist_albums(artist_uri)["items"]]  # get all album uris of artist
        
        artist_track_count = 0
        for album_uri in album_uris:
            album_data = sp.album(album_uri)  # get data on album

            for track_data in album_data["tracks"]["items"]:
                found_artist_name = track_data["artists"][0]["name"]  # get the artist name (class label for data)
                
                # only collect the track if the found artist is the expected artist of the uri
                if found_artist_name == artist_name:
                    # get track uri and its audio features
                    track_uri = track_data["uri"]
                    track_features = sp.audio_features(track_uri)[0]  

                    # get dictionary of the features defined in data_cols
                    track_dict = dict((key, track_features[key]) for key in track_features if key in col_features)  
                    track_dict["artist"] = artist_name  # add class label to dict of data

                    # make the data a pd df row and concatenate it to the entire df
                    track_df = pd.DataFrame(track_dict, index=[0])  
                    final_df = pd.concat([final_df, track_df], ignore_index=True)
                    artist_track_count += 1

        artist_elapsed = time.time() - artist_start
        print(f"{artist_track_count} tracks from {artist_name} collected in {artist_elapsed} seconds")

    # don't include the header if appending data
    if out_mode == 'a':
        header_mode = False
    else:
        header_mode = True

    # write df to csv file
    final_df.to_csv(out_path, mode=out_mode, header=header_mode, index=False)
    print(f"Final data frame shape: {final_df.shape}")


def main():
    total_start = time.time()

    # credentials
    client_id = "get_ur_own_id"
    client_secret = "get_ur_own_secret_key"
    client_credentials_manager = spotipy.SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)

    # spotipy object
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)

    # load artists to collect, dict in form of {"artist": "uri"}
    artists_dict = load_uri_data("data/artist_uris.csv") 

    # data features to collect (excluding anything that defines who the artist is)
    col_features = ["danceability", "energy", "key", "loudness", "mode", "speechiness", "instrumentalness", "liveness", "valence", "tempo", "duration_ms", "time_signature", "artist"]

    # where to write data collected
    out_path = "data/tracks.csv"

    # collect the data and write to csv
    get_spotify_data(sp, artists_dict, col_features, out_path, out_mode='w')

    total_elapsed = time.time() - total_start
    print(f"\nAll data collected and written in {total_elapsed} seconds")


if __name__ == "__main__":
    main()


