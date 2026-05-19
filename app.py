import streamlit as st
from content_based_filtering import recommend
from scipy.sparse import load_npz
import pandas as pd


# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Spotify Collaborative Filtering",
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
# LOAD DATA
# ---------------------------------------------------

transformed_data_path = "data/transformed_data.npz"
cleaned_data_path = "data/cleaned_data.csv"

data = pd.read_csv(cleaned_data_path)

transformed_data = load_npz(transformed_data_path)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------

st.markdown(
    '<div class="big-title">🎧 Spotify Collaborative Filtering Recommender</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">Discover songs similar to your favorite tracks using AI-powered recommendation systems 🚀</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("🎵 Recommendation Settings")

# unique songs
song_list = sorted(data["name"].dropna().unique())

# song dropdown
selected_song = st.sidebar.selectbox(
    "Select Song",
    song_list
)

# artist dropdown based on song
artist_list = sorted(
    data.loc[
        data["name"] == selected_song,
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
        f"Showing recommendations for '{selected_song.title()}' by '{selected_artist.title()}'"
    )

    # recommendations
    recommendations = recommend(
        selected_song,
        selected_artist,
        data,
        transformed_data,
        k
    )

    # display songs
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
            st.audio(recommendation['spotify_preview_url'])

# ---------------------------------------------------
# FOOTER
# ---------------------------------------------------

st.write("---")

st.caption(
    "Built with ❤️ using Streamlit, Content-Based Filtering, TF-IDF, and Cosine Similarity"
)