# Netflix Movies and TV Shows – Exploratory Data Analysis (EDA)

**Course:** Advance Programming Lab-I (Python)  
**Experiment Number:** 09  
**Dataset Source:** [Kaggle: Netflix Movies and TV Shows (`shivamb/netflix-shows`)](https://www.kaggle.com/datasets/shivamb/netflix-shows)  
**Language / Core Libraries:** Python 3.12, Pandas, NumPy, Matplotlib, Seaborn, WordCloud, Streamlit  

[![GitHub Pages](https://img.shields.io/badge/Live_EDA_Report-GitHub_Pages-E50914?style=for-the-badge&logo=github)](https://youngchevalier.github.io/netflix-movies-and-tv-shows-analysis/)
[![Streamlit Cloud](https://img.shields.io/badge/Deploy-Streamlit_Cloud-FF4B4B?style=for-the-badge&logo=streamlit)](https://share.streamlit.io/deploy?repository=YoungChevalier/netflix-movies-and-tv-shows-analysis&branch=main&mainModule=app.py)

---

## 🌐 Live Deployments & Working Links

- 🌐 **Live EDA HTML Report (GitHub Pages)**: [https://youngchevalier.github.io/netflix-movies-and-tv-shows-analysis/](https://youngchevalier.github.io/netflix-movies-and-tv-shows-analysis/)
- 🚀 **1-Click Streamlit Cloud Deployment**: [https://share.streamlit.io/deploy?repository=YoungChevalier/netflix-movies-and-tv-shows-analysis&branch=main&mainModule=app.py](https://share.streamlit.io/deploy?repository=YoungChevalier/netflix-movies-and-tv-shows-analysis&branch=main&mainModule=app.py)
- 💻 **Local Dashboard**: `http://localhost:8501` (when running `streamlit run app.py`)

---

## 1. Project Overview

This repository contains an end-to-end, reproducible Exploratory Data Analysis (EDA) of the global Netflix titles catalog (`netflix_titles.csv`, 8,807 raw records). The analysis investigates catalog composition, historical growth trajectories, genre specialization, demographic age ratings, geographic production hubs, content runtimes, and genre co-occurrence patterns.

The project delivers:
1. **Executed Jupyter Notebook (`notebooks/netflix_eda.ipynb`)**: Top-to-bottom executable notebook spanning 14 structured sections with code, visualizations, and markdown interpretations for every chart.
2. **Interactive Streamlit Dashboard (`app.py`)**: A Netflix-themed interactive web application with KPI metric cards, dynamic multi-attribute filters, and analytical tabs.
3. **High-Resolution Figures (`outputs/figures/*.png`)**: 20 standalone charts saved at 300 DPI following a custom Netflix visual palette (`#E50914`, `#141414`, `#F5F5F1`).
4. **HTML Report Export (`notebooks/netflix_eda.html`)**: Self-contained export of the executed notebook for viewing without a Jupyter server.

---

## 2. Project Directory Structure

```
Netflix Movies and TV shows Analysis/
├── data/
│   └── netflix_titles.csv              # Source dataset (8,807 rows x 12 cols)
├── notebooks/
│   ├── netflix_eda.ipynb               # Primary deliverable (executed with full outputs)
│   └── netflix_eda.html                # Standalone HTML report export
├── outputs/
│   └── figures/                        # 20 high-resolution publication figures (300 DPI)
│       ├── 01_missing_values_bar.png
│       ├── 02_outlier_detection_boxplots.png
│       ├── 03_type_distribution_count_donut.png
│       ├── 04_content_ratings_distribution.png
│       ├── 05_movie_duration_and_tv_seasons.png
│       ├── 06_release_year_distribution.png
│       ├── 07_top_10_countries.png
│       ├── 08_top_10_genres.png
│       ├── 09_top_10_directors_actors.png
│       ├── 10_content_type_by_rating.png
│       ├── 11_content_growth_by_year_added.png
│       ├── 12_content_age_at_addition.png
│       ├── 13_movie_duration_by_rating.png
│       ├── 14_genres_by_top_countries.png
│       ├── 15_heatmap_genres_x_countries.png
│       ├── 16_heatmap_month_x_year_added.png
│       ├── 17_pairplot_numeric_by_type.png
│       ├── 18_correlation_matrix_numeric.png
│       ├── 19_genre_cooccurrence_matrix.png
│       └── 20_wordcloud_titles.png
├── app.py                              # Interactive Streamlit dashboard
├── requirements.txt                    # Pinned library dependencies
└── README.md                           # Documentation and analytical report
```

---

## 3. Dataset Description

The dataset captures 8,807 movies and TV shows released on Netflix up to mid-2021:

| Column | Data Type | Description |
| :--- | :--- | :--- |
| `show_id` | `str` | Unique platform identifier for each title (e.g., `s1`, `s2`) |
| `type` | `str` | Format classification: `Movie` or `TV Show` |
| `title` | `str` | Title of the film or series |
| `director` | `str` | Name of the director(s); ~29.9% missing |
| `cast` | `str` | Comma-separated list of featured actors/actresses |
| `country` | `str` | Country or countries involved in production |
| `date_added` | `str / datetime` | Date when title was made available on Netflix |
| `release_year` | `int` | Original year of theatrical release or television broadcast |
| `rating` | `str` | MPAA / TV Parental Guidelines age maturity rating |
| `duration` | `str` | Original runtime string (e.g., `90 min` or `2 Seasons`) |
| `listed_in` | `str` | Comma-separated category genres |
| `description` | `str` | Brief synopsis / narrative summary |

---

## 4. Key Preprocessing & Data Cleaning Highlights

1. **Shifted Rating/Duration Data Recovery:**
   - 3 records (`Louis C.K. 2017`, `Louis C.K.: Hilarious`, `Louis C.K.: Live at the Comedy Store`) had duration strings (`74 min`, `84 min`, `66 min`) accidentally placed into the `rating` column while `duration` was `NaN`. These values were programmatically identified and relocated to `duration`.
2. **Whitespace Sanitization & True Null Encoding:**
   - All text columns were stripped of leading and trailing whitespace, and string representations of `"nan"` were standardized to `np.nan`.
3. **Composite Key Deduplication:**
   - While no exact full-row duplicate existed, 4 duplicate titles were uncovered when controlling for case-normalization and whitespace (`Consequences`, `Esperando la carroza`, `Love in a Puff`, `Sin senos sí hay paraíso`). Deduplication reduced the dataset to **8,803 unique titles**.
4. **Duration Splitting:**
   - Split `duration` into `duration_min` (numeric minutes for Movies) and `seasons` (integer season counts for TV Shows).
5. **Multi-Value Field Tokenization:**
   - Comma-delimited fields (`country`, `listed_in`, `cast`, `director`) were parsed into Python list columns to enable accurate `.explode()` frequency and co-occurrence modeling.
6. **Missing Values Imputation:**
   - `director`, `cast`, and `country` were imputed with `"Unknown"` to avoid discarding valuable titles.
   - Missing `rating` (7 records) and `date_added` (10 records) were imputed using catalog mode values.
7. **Outlier Detection via IQR:**
   - Computed IQR bounds for movie runtimes ($[46.5\text{ min}, 154.5\text{ min}]$) and release years ($[2004, 2028]$). Outliers were flagged rather than deleted because vintage films and short/extended features are legitimate catalog assets.

---

## 5. Summary of Empirical Findings & Key Patterns

1. **Catalog Balance:**
   - Movies account for **69.6% (6,128 titles)** of the catalog, while TV Shows comprise **30.4% (2,675 titles)**.
2. **Audience Maturity Distribution:**
   - **TV-MA** is the largest single rating category with **3,211 titles (36.5%)**, followed by **TV-14 with 2,157 titles (24.5%)**. Over **61%** of all programming is tailored for mature teen and adult audiences.
3. **The "Single-Season Drop" Reality:**
   - **67.0% (1,791 out of 2,675 TV shows)** consist of exactly 1 Season. Only 9.2% of shows progress beyond Season 3, reflecting high title experimentation and rapid cancellation cycles for lower-performing series.
4. **Movie Runtime Standardization:**
   - Feature films cluster tightly around a median runtime of **98.0 minutes** (IQR: 87–114 minutes).
5. **Transition to Day-and-Date Streaming:**
   - The median gap between original release year and platform ingestion dropped from **5.0 years in 2015 down to 0.0 years by 2020–2021**. Netflix evolved from an archival syndicator into a primary first-window distributor.
6. **Top Content Producing Nations:**
   - The **United States** leads with 3,689 title credits, followed by **India** (1,046) and the **United Kingdom** (804). Canada, France, Japan, and South Korea constitute the primary international tier.
7. **Genre Co-occurrence Hubs:**
   - The highest frequency cross-genre combinations are **Dramas + International Movies** (1,235 titles) and **Comedies + International Movies** (821 titles).
8. **Catalog Drop Seasonality:**
   - Platform additions peak in **Q4 (October, November, December)** and **July**, synchronizing with global school breaks and holiday indoor leisure periods.

---

## 6. Strategic Business Insights

- **Franchise Retention vs. Churn:** The heavy concentration of 1-season shows indicates effective upfront subscriber acquisition, but long-term subscriber retention requires investing in durable multi-season flagship intellectual property (IP).
- **International Market Engines:** India represents the second-largest content supplier on the service, driven heavily by Bollywood catalog licensing. Concurrently, South Korean drama and Japanese anime demonstrate disproportionately high cross-border retention.
- **Family Demographic Defense:** With >60% of programming targeting mature adults, Netflix must defend family and children co-viewing (`TV-Y`, `PG`) to remain resilient against competitors like Disney+.

---

## 7. How to Run the Project

### Prerequisites
Ensure Python 3.10+ is installed on your system.

### Step 1: Clone or Navigate to the Project Directory
```bash
cd "Netflix Movies and TV shows Analysis"
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Jupyter Notebook (Primary Deliverable)
```bash
jupyter notebook notebooks/netflix_eda.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells.

### Step 4: Launch Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 8. Experiment Certification
- **Course:** Advance Programming Lab-I (Python)
- **Experiment:** Experiment 09 – Exploratory Data Analysis on Netflix Dataset
- **Status:** Complete, executed, and verified.
