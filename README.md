# NLP News Intelligence

This project scrapes news articles from the web and performs NLP analysis to extract insights. It is designed as a simple pipeline for collecting, enriching, and classifying news data.

## Structure
- `scraper_news.py`: Scrapes news articles from online sources (e.g., BBC).
- `nlp_enriched_news.py`: Performs NLP analysis on the scraped news data.
- `nlp_utils.py`: Text preprocessing utilities (tokenization, stopword removal, stemming).
- `vectorizer.py`: Converts preprocessed text into numerical features using CountVectorizer.
- `training_model.py`: Trains and evaluates a topic classification model.
- `data/`: Stores raw and processed news data (e.g., raw_articles.csv).
- `results/`: Stores NLP-enriched results, model outputs, and plots (e.g., learning_curves.png).
- `requirements.txt`: Lists required Python libraries.

## Key Features
- Web scraping with requests and BeautifulSoup
- Text preprocessing with NLTK
- Feature extraction with scikit-learn's CountVectorizer
- Topic classification using Logistic Regression
- Learning curve visualization with matplotlib
- Model persistence with joblib

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```
2. Download NLTK resources (done automatically by nlp_utils.py if missing).

## Usage
- Run each script as needed for scraping, preprocessing, vectorization, and model training.
- Example:
   ```bash
   python scraper_news.py
   python nlp_utils.py
   python vectorizer.py
   python training_model.py
   ```
