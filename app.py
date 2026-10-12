import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Netflix Movies & TV Shows EDA Dashboard",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Netflix-style dark theme & modern aesthetic
st.markdown("""
<style>
    /* Global styles */
    .stApp {
        background-color: #141414;
        color: #F5F5F1;
    }
    
    /* Headers and titles */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-weight: 700;
    }
    
    .main-title {
        color: #E50914 !important;
        font-size: 2.4rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #B3B3B3;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric KPI Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1f1f1f 0%, #292929 100%);
        border: 1px solid #333333;
        border-radius: 10px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #E50914;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.95rem !important;
        color: #B3B3B3 !important;
        font-weight: 600;
    }
    div[data-testid="stMetricValue"] div {
        color: #FFFFFF !important;
        font-size: 1.9rem !important;
        font-weight: 700;
    }
    
    /* Sidebar customization */
    section[data-testid="stSidebar"] {
        background-color: #0b0b0b;
        border-right: 1px solid #222222;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 44px;
        white-space: pre-wrap;
        background-color: #1f1f1f;
        border-radius: 6px;
        color: #B3B3B3;
        font-weight: 600;
        padding: 8px 16px;
        border: 1px solid #333333;
    }
    .stTabs [aria-selected="true"] {
        background-color: #E50914 !important;
        color: #FFFFFF !important;
        border-color: #E50914 !important;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading and preprocessing
@st.cache_data
def load_and_preprocess_data():
    possible_paths = [
        os.path.join("data", "netflix_titles.csv"),
        os.path.join("..", "data", "netflix_titles.csv"),
        "netflix_titles.csv",
        os.path.join("..", "netflix_titles.csv")
    ]
    csv_file = None
    for p in possible_paths:
        if os.path.exists(p):
            csv_file = p
            break
            
    if csv_file is None:
        st.error("Dataset not found! Please place netflix_titles.csv in /data.")
        st.stop()
        
    df = pd.read_csv(csv_file)
    
    # Clean whitespace
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col] == 'nan', col] = np.nan
        
    # Column shift repair (Louis C.K. standups)
    shifted_mask = df['duration'].isna() & df['rating'].str.contains('min', na=False)
    df.loc[shifted_mask, 'duration'] = df.loc[shifted_mask, 'rating']
    df.loc[shifted_mask, 'rating'] = np.nan
    
    # Deduplication
    df['clean_title'] = df['title'].str.lower().str.strip()
    df = df.drop_duplicates(subset=['clean_title', 'type', 'release_year'], keep='first').copy()
    df.drop(columns=['clean_title'], inplace=True)
    
    # Parse date_added
    df['date_added'] = pd.to_datetime(df['date_added'], format='mixed', errors='coerce')
    mode_date = df['date_added'].mode()[0]
    df['date_added'] = df['date_added'].fillna(mode_date)
    df['year_added'] = df['date_added'].dt.year.astype(int)
    df['month_added'] = df['date_added'].dt.month_name()
    df['content_age_at_addition'] = df['year_added'] - df['release_year']
    
    # Impute missing values
    df['director'] = df['director'].fillna('Unknown')
    df['cast'] = df['cast'].fillna('Unknown')
    df['country'] = df['country'].fillna('Unknown')
    mode_rating = df['rating'].mode()[0]
    df['rating'] = df['rating'].fillna(mode_rating)
    
    # Numeric duration parsing
    df['duration_min'] = df['duration'].apply(
        lambda x: int(x.replace(' min', '')) if isinstance(x, str) and 'min' in x else np.nan
    )
    df['seasons'] = df['duration'].apply(
        lambda x: int(x.replace(' Seasons', '').replace(' Season', '')) if isinstance(x, str) and 'Season' in x else np.nan
    )
    
    # Tokenize lists
    df['country_list'] = df['country'].apply(lambda x: [c.strip() for c in str(x).split(',') if c.strip() and c.strip() != 'nan'])
    df['genre_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',') if g.strip() and g.strip() != 'nan'])
    df['cast_list'] = df['cast'].apply(lambda x: [a.strip() for a in str(x).split(',') if a.strip() and a.strip() not in ['nan', 'Unknown']])
    
    return df

df = load_and_preprocess_data()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.markdown("<h2 style='color:#E50914;'>🎬 Netflix Analytics</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Filter catalog data dynamically:")

# Type Filter
type_options = ["All", "Movie", "TV Show"]
selected_type = st.sidebar.radio("Content Type:", type_options, horizontal=True)

# Release Year Filter
min_year = int(df['release_year'].min())
max_year = int(df['release_year'].max())
selected_year_range = st.sidebar.slider(
    "Release Year Range:",
    min_value=min_year,
    max_value=max_year,
    value=(1990, max_year)
)

# Rating Filter
all_ratings = sorted(df['rating'].unique().tolist())
selected_ratings = st.sidebar.multiselect(
    "Maturity Ratings:",
    options=all_ratings,
    default=all_ratings
)

# Country Filter
country_exploded = df.explode('country_list')
top_countries = [c for c in country_exploded['country_list'].value_counts().head(25).index if c != 'Unknown']
selected_countries = st.sidebar.multiselect(
    "Production Country (Top 25):",
    options=top_countries,
    default=[]
)

# Text Search
search_term = st.sidebar.text_input("🔍 Search Title, Cast, or Director:", "")

# Filter dataframe
filtered_df = df[
    (df['release_year'] >= selected_year_range[0]) &
    (df['release_year'] <= selected_year_range[1])
]

if selected_type != "All":
    filtered_df = filtered_df[filtered_df['type'] == selected_type]

if selected_ratings:
    filtered_df = filtered_df[filtered_df['rating'].isin(selected_ratings)]

if selected_countries:
    filtered_df = filtered_df[filtered_df['country_list'].apply(lambda cl: any(c in selected_countries for c in cl))]

if search_term:
    term_lower = search_term.lower()
    filtered_df = filtered_df[
        filtered_df['title'].str.lower().str.contains(term_lower, na=False) |
        filtered_df['cast'].str.lower().str.contains(term_lower, na=False) |
        filtered_df['director'].str.lower().str.contains(term_lower, na=False) |
        filtered_df['description'].str.lower().str.contains(term_lower, na=False)
    ]

# ----------------- HEADER & KPI METRICS -----------------
st.markdown("<h1 class='main-title'>NETFLIX CATALOG INSIGHTS</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Comprehensive Exploratory Data Analysis & Interactive Analytics Dashboard</p>", unsafe_allow_html=True)

total_titles = len(filtered_df)
movies_count = len(filtered_df[filtered_df['type'] == 'Movie'])
tv_count = len(filtered_df[filtered_df['type'] == 'TV Show'])

pct_movies = (movies_count / total_titles * 100) if total_titles > 0 else 0
pct_tv = (tv_count / total_titles * 100) if total_titles > 0 else 0

exp_c = filtered_df.explode('country_list')
valid_countries = exp_c[exp_c['country_list'] != 'Unknown']['country_list']
top_country_str = valid_countries.mode()[0] if not valid_countries.empty else "N/A"

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Titles", f"{total_titles:,}")
col2.metric("Movies Share", f"{pct_movies:.1f}%", f"{movies_count:,} titles")
col3.metric("TV Shows Share", f"{pct_tv:.1f}%", f"{tv_count:,} titles")
col4.metric("Top Country", top_country_str)
median_dur = filtered_df.loc[filtered_df['type'] == 'Movie', 'duration_min'].median()
col5.metric("Avg Movie Duration", f"{median_dur:.0f} min" if not np.isnan(median_dur) else "N/A")

st.markdown("<hr style='border: 1px solid #292929; margin: 20px 0;'>", unsafe_allow_html=True)

# ----------------- TABBED ANALYTICS -----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Growth & Additions", 
    "🏷️ Genres & Categories", 
    "🔞 Ratings & Maturity", 
    "⏱️ Runtimes & Seasons", 
    "📋 Data Explorer"
])

NETFLIX_RED = "#E50914"
NETFLIX_BLACK = "#221F1F"

with tab1:
    st.subheader("Platform Growth & Content Onboarding Dynamics")
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        yearly_data = filtered_df[(filtered_df['year_added'] >= 2012) & (filtered_df['year_added'] <= 2021)].groupby(['year_added', 'type']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(10, 4.5), facecolor='#141414')
        ax.set_facecolor('#141414')
        if 'Movie' in yearly_data.columns:
            ax.plot(yearly_data.index, yearly_data['Movie'], marker='o', linewidth=2.5, color=NETFLIX_RED, label='Movies')
        if 'TV Show' in yearly_data.columns:
            ax.plot(yearly_data.index, yearly_data['TV Show'], marker='s', linewidth=2.5, color='#B3B3B3', label='TV Shows')
            
        ax.set_title("Content Additions Over Time (2012 – 2021)", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Year Added", color='#B3B3B3')
        ax.set_ylabel("Titles Added", color='#B3B3B3')
        ax.tick_params(colors='#B3B3B3')
        ax.grid(color='#333333', linestyle='--', alpha=0.5)
        ax.legend(facecolor='#221F1F', edgecolor='none', labelcolor='white')
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_b:
        fig_donut, ax_donut = plt.subplots(figsize=(5, 4.5), facecolor='#141414')
        ax_donut.set_facecolor('#141414')
        t_counts = filtered_df['type'].value_counts()
        if not t_counts.empty:
            wedges, texts, autotexts = ax_donut.pie(
                t_counts.values, labels=t_counts.index,
                autopct='%1.1f%%', colors=[NETFLIX_RED, '#333333'],
                startangle=90, textprops={'color': 'white', 'weight': 'bold'},
                wedgeprops={'edgecolor': '#141414', 'linewidth': 2}
            )
            for at in autotexts:
                at.set_color('white')
            centre_circle = plt.Circle((0, 0), 0.65, fc='#141414')
            ax_donut.add_artist(centre_circle)
        ax_donut.set_title("Content Split", color='white', fontsize=12, fontweight='bold', pad=12)
        plt.tight_layout()
        st.pyplot(fig_donut)

with tab2:
    st.subheader("Top Genres & Category Breakdown")
    exp_g = filtered_df.explode('genre_list')
    top_genres = exp_g['genre_list'].value_counts().head(12)
    
    fig, ax = plt.subplots(figsize=(11, 5), facecolor='#141414')
    ax.set_facecolor('#141414')
    bars = ax.barh(top_genres.index[::-1], top_genres.values[::-1], color=NETFLIX_RED, edgecolor='#333333')
    ax.set_title("Top 12 Listed Genres in Filtered Catalog", color='white', fontsize=12, fontweight='bold', pad=12)
    ax.set_xlabel("Number of Titles", color='#B3B3B3')
    ax.tick_params(colors='#B3B3B3')
    ax.grid(axis='x', color='#333333', linestyle='--', alpha=0.5)
    
    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:,}", xy=(w, bar.get_y() + bar.get_height() / 2),
                    xytext=(6, 0), textcoords="offset points", ha='left', va='center',
                    color='white', fontsize=9, fontweight='semibold')
                    
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    st.subheader("Audience Maturity & Rating Classifications")
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        rc = filtered_df['rating'].value_counts().head(8)
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#141414')
        ax.set_facecolor('#141414')
        bars = ax.bar(rc.index, rc.values, color=NETFLIX_RED, edgecolor='#221F1F')
        ax.set_title("Rating Volume (Top 8)", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.set_ylabel("Count", color='#B3B3B3')
        ax.tick_params(colors='#B3B3B3')
        ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.5)
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points", ha='center',
                        color='white', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_r2:
        top_ratings_sub = filtered_df[filtered_df['rating'].isin(['TV-MA', 'TV-14', 'R', 'TV-PG', 'PG-13'])]
        tr_grouped = top_ratings_sub.groupby(['rating', 'type']).size().unstack(fill_value=0)
        
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#141414')
        ax.set_facecolor('#141414')
        tr_grouped.plot(kind='bar', color=[NETFLIX_RED, '#555555'], edgecolor='none', ax=ax, width=0.7)
        ax.set_title("Ratings by Format (Movies vs TV)", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.tick_params(colors='#B3B3B3')
        ax.legend(facecolor='#221F1F', edgecolor='none', labelcolor='white')
        ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

with tab4:
    st.subheader("Film Runtimes & TV Series Longevity")
    col_d1, col_d2 = st.columns(2)
    
    with col_d1:
        movie_durations = filtered_df.loc[filtered_df['type'] == 'Movie', 'duration_min'].dropna()
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#141414')
        ax.set_facecolor('#141414')
        if not movie_durations.empty:
            sns.histplot(movie_durations, bins=25, kde=True, color=NETFLIX_RED, ax=ax, edgecolor='#141414')
            ax.axvline(movie_durations.median(), color='gold', linestyle='--', label=f'Median: {movie_durations.median():.0f}m')
        ax.set_title("Movie Duration Distribution (Minutes)", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Runtime (Minutes)", color='#B3B3B3')
        ax.tick_params(colors='#B3B3B3')
        ax.legend(facecolor='#221F1F', edgecolor='none', labelcolor='white')
        ax.grid(color='#333333', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        
    with col_d2:
        tv_s = filtered_df.loc[filtered_df['type'] == 'TV Show', 'seasons'].value_counts().sort_index().head(8)
        fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#141414')
        ax.set_facecolor('#141414')
        if not tv_s.empty:
            bars = ax.bar(tv_s.index.astype(str), tv_s.values, color='#333333', edgecolor=NETFLIX_RED)
            for bar in bars:
                h = bar.get_height()
                ax.annotate(f"{h:,}", xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points", ha='center',
                            color='white', fontsize=9)
        ax.set_title("TV Show Season Breakdown", color='white', fontsize=12, fontweight='bold', pad=12)
        ax.set_xlabel("Number of Seasons", color='#B3B3B3')
        ax.tick_params(colors='#B3B3B3')
        ax.grid(axis='y', color='#333333', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

with tab5:
    st.subheader("Interactive Catalog Explorer")
    display_cols = ['title', 'type', 'director', 'country', 'release_year', 'rating', 'duration', 'listed_in']
    st.dataframe(
        filtered_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        height=450
    )
    
    # Download filtered CSV
    csv_bytes = filtered_df[display_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="netflix_filtered_subset.csv",
        mime="text/csv"
    )
