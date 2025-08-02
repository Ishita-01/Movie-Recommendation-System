
# 🎬 Movie Recommendation System
A content-based movie recommendation system that suggests similar movies using Natural Language Processing (NLP) techniques and a K-Nearest Neighbors (KNN) algorithm.
It leverages movie metadata (genres, cast, keywords, production companies, and plot overviews) and cosine similarity to generate personalized recommendations.
The project also includes an interactive web interface built with Streamlit and integrates the TMDB API for fetching posters and metadata.

## 🚀 Features

- **Content-based recommendations using:**

  - Combined movie features (overview, genres, keywords, cast, production companies)
  
  - Individual feature-based recommendations (e.g., similar cast, similar genres)

- **Custom preprocessing pipelines:**

    - Tokenization, stemming, and stopword removal using NLTK

  -  Feature vectorization with CountVectorizer

- Cosine similarity & KNN for similarity computation

- Interactive UI built with Streamlit

- TMDB API integration to fetch dynamic posters and metadata

## 🛠️ Tech Stack
**Programming Language:** Python

**Libraries & Tools:** NLP & ML: Scikit-learn, NLTK

**Data Processing:** Pandas, NumPy

**Web Framework:** Streamlit

**API Integration:** TMDB API


## ⚙️ Installation & Usage

### Clone the repository
```bash
git clone https://github.com/Ishita-01/Movie-Recommendation-System.git
cd Movie-Recommendation-System
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit application
```bash
streamlit run main.py
```
