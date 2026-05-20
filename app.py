import streamlit as st
from content_based_filtering import recommend
from collaborative_filtering import collaborative_recommendation

from scipy.sparse import load_npz
from numpy import load

import pandas as pd


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Spotify Recommendation System",
    page_icon="🎵",
    layout="wide"
)

# ---------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stApp {
    background: linear-gradient(to bottom right, #191414, #121212);
    color: white;
}

.big-title {
    font-size: 3rem;
    font-weight: bold;
    color: #1DB954;
    text-align: center;
    margin-bottom: 10px;
}

.sub-title {
    text-align: center;
    font-size: 1.1rem;
    color: #B3B3B3;
    margin-bottom: 40px;
}

.song-card {
    background-color: #181818;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
    border: 1px solid #282828;
}

.song-card:hover {
    border: 1px solid #1DB954;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD CONTENT BASED DATA
# ---------------------------------------------------

transformed_data_path = "data/transformed/transformed_data.npz"
cleaned_data_path = "data/cleaned/cleaned_data.csv"

data = pd.read_csv(cleaned_data_path)

transformed_data = load_npz(transformed_data_path)

# ---------------------------------------------------
# LOAD COLLABORATIVE FILTERING DATA
# ---------------------------------------------------

filtered_data_path = "data/cleaned/collab_filtered_data.csv"
interaction_matrix_path = "data/transformed/interaction_matrix.npz"
track_ids_path = "data/transformed/track_ids.npy"

filtered_data = pd.read_csv(filtered_data_path)

interaction_matrix = load_npz(
    interaction_matrix_path
)

track_ids = load(
    track_ids_path,
    allow_pickle=True
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '''
    <div class="big-title">
    🎧 Spotify Recommendation System
    </div>
    ''',
    unsafe_allow_html=True
)

st.markdown(
    '''
    <div class="sub-title">
    Discover songs similar to your favorite tracks
    using AI-powered recommendation systems 🚀
    </div>
    ''',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🎵 Recommendation Settings")

# recommendation type
filtering_type = st.sidebar.selectbox(
    "Recommendation Technique",
    [
        "Content-Based Filtering",
        "Collaborative Filtering"
    ]
)

# dataset selection
if filtering_type == "Content-Based Filtering":
    current_data = data
else:
    current_data = filtered_data

# unique songs
song_list = sorted(
    current_data["name"].dropna().unique()
)

# song dropdown
selected_song = st.sidebar.selectbox(
    "Select Song",
    song_list
)

# artist dropdown
artist_list = sorted(
    current_data.loc[
        current_data["name"] == selected_song,
        "artist"
    ].dropna().unique()
)

selected_artist = st.sidebar.selectbox(
    "Select Artist",
    artist_list
)

# recommendation count
k = st.sidebar.slider(
    "Number of Recommendations",
    min_value=5,
    max_value=20,
    value=10,
    step=1
)

# recommendation button
recommend_button = st.sidebar.button(
    "🎧 Generate Recommendations"
)

# ---------------------------------------------------
# MAIN SECTION
# ---------------------------------------------------

if recommend_button:

    st.success(
        f"""
        Showing recommendations for
        '{selected_song.title()}'
        by
        '{selected_artist.title()}'
        """
    )

    # ---------------------------------------------------
    # CONTENT BASED FILTERING
    # ---------------------------------------------------

    if filtering_type == "Content-Based Filtering":

        recommendations = recommend(
            selected_song,
            selected_artist,
            data,
            transformed_data,
            k
        )

    # ---------------------------------------------------
    # COLLABORATIVE FILTERING
    # ---------------------------------------------------

    else:

        recommendations = collaborative_recommendation(
            song_name=selected_song,
            artist_name=selected_artist,
            track_ids=track_ids,
            songs_data=filtered_data,
            interaction_matrix=interaction_matrix,
            k=k
        )

    # ---------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # ---------------------------------------------------

    for ind, recommendation in recommendations.iterrows():

        song_name = recommendation['name'].title()
        artist_name = recommendation['artist'].title()

        with st.container():

            st.markdown(
                f"""
                <div class="song-card">
                    <h3>🎵 {song_name}</h3>
                    <p>👨‍🎤 Artist: {artist_name}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            # audio preview
            st.audio(
                recommendation['spotify_preview_url']
            )

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("---")

st.caption(
    """
    Built with ❤️ using Streamlit,
    Content-Based Filtering,
    Collaborative Filtering,
    TF-IDF,
    Cosine Similarity,
    and Recommendation Systems
    """
)