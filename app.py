import streamlit as st

from content_based_filtering import recommend
from collaborative_filtering import collaborative_recommendation
from hybrid_recommendation import HybridRecommenderSystem

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
    font-size: 3.2rem;
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
    transition: 0.3s;
}

.song-card:hover {
    border: 1px solid #1DB954;
    transform: scale(1.01);
}

.section-title {
    color: #1DB954;
    font-size: 1.6rem;
    margin-top: 10px;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# LOAD CONTENT BASED DATA
# ---------------------------------------------------

transformed_data_path = "data/transformed/transformed_data.npz"
cleaned_data_path = "data/cleaned/cleaned_data.csv"
track_ids_path = "data/transformed/track_ids.npy"

data = pd.read_csv(cleaned_data_path)

transformed_data = load_npz(
    transformed_data_path
)

# ---------------------------------------------------
# LOAD COLLABORATIVE FILTERING DATA
# ---------------------------------------------------

filtered_data_path = "data/cleaned/collab_filtered_data.csv"

interaction_matrix_path = (
    "data/transformed/interaction_matrix.npz"
)

filtered_data = pd.read_csv(
    filtered_data_path
)

interaction_matrix = load_npz(
    interaction_matrix_path
)


track_ids = load(
    track_ids_path,
    allow_pickle=True
)


# ---------------------------------------------------
# LOAD HYBRID DATA
# ---------------------------------------------------

transformed_hybrid_data_path = (
    "data/transformed/transformed_hybrid_data.npz"
)

transformed_hybrid_data = load_npz(
    transformed_hybrid_data_path
)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    """
    <div class="big-title">
    🎧 Spotify Recommendation System
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="sub-title">
    Discover songs using Content-Based,
    Collaborative and Hybrid Recommendation Systems 🚀
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title(
    "🎵 Recommendation Settings"
)

# recommendation type
filtering_type = st.sidebar.selectbox(
    "Recommendation Technique",
    [
        "Content-Based Filtering",
        "Collaborative Filtering",
        "Hybrid Recommendation System"
    ]
)

# dataset selection
if filtering_type == "Content-Based Filtering":
    current_data = data

else:
    current_data = filtered_data

# unique songs
song_list = sorted(
    current_data["name"]
    .dropna()
    .unique()
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
    ]
    .dropna()
    .unique()
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

# hybrid weights
if filtering_type == "Hybrid Recommendation System":

    st.sidebar.markdown("---")

    st.sidebar.subheader(
        "⚖️ Hybrid Weights"
    )

    weight_content = st.sidebar.slider(
        "Content-Based Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.1
    )

    weight_collaborative = round(
        1 - weight_content,
        1
    )

    st.sidebar.info(
        f"""
        Collaborative Weight:
        {weight_collaborative}
        """
    )

# recommendation button
recommend_button = st.sidebar.button(
    "🎧 Generate Recommendations"
)

# ---------------------------------------------------
# MAIN SECTION
# ---------------------------------------------------

if recommend_button:

    st.markdown(
        f"""
        <div class="section-title">
        Recommendations for
        '{selected_song.title()}'
        by
        '{selected_artist.title()}'
        </div>
        """,
        unsafe_allow_html=True
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

    elif filtering_type == "Collaborative Filtering":

        recommendations = collaborative_recommendation(
            song_name=selected_song,
            artist_name=selected_artist,
            track_ids=track_ids,
            songs_data=filtered_data,
            interaction_matrix=interaction_matrix,
            k=k
        )

    # ---------------------------------------------------
    # HYBRID FILTERING
    # ---------------------------------------------------

    else:

        hybrid_recommender = (
            HybridRecommenderSystem(
                song_name=selected_song,
                artist_name=selected_artist,
                number_of_recommendations=k,
                weight_content_based=weight_content,
                weight_collaborative=weight_collaborative,
                songs_data=filtered_data,
                transformed_matrix=transformed_hybrid_data,
                interaction_matrix=interaction_matrix
            )
        )

        recommendations = (
            hybrid_recommender
            .give_recommendations()
        )

    # ---------------------------------------------------
    # DISPLAY RECOMMENDATIONS
    # ---------------------------------------------------

    for ind, recommendation in recommendations.iterrows():

        song_name = recommendation[
            "name"
        ].title()

        artist_name = recommendation[
            "artist"
        ].title()

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
                recommendation[
                    "spotify_preview_url"
                ]
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
    Hybrid Recommendation Systems,
    TF-IDF,
    Sparse Matrices,
    Cosine Similarity,
    and Machine Learning
    """
)