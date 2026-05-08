"""
nlp_enriched_news.py
Purpose: Script for performing NLP analysis on scraped news data.
"""

import joblib
from nlp_utils import preprocess_text
import pandas as pd
from deep_translator import GoogleTranslator
import nltk
nltk.download("vader_lexicon")
nltk.download('punkt')
from nltk.sentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from nltk.tokenize import sent_tokenize
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_orgs(text):
    doc = nlp(text)

    orgs = []
    for ent in doc.ents:
        if ent.label_ == "ORG":
            orgs.append(ent.text)

    return list(set(orgs))  # remove duplicates

model = SentenceTransformer('all-MiniLM-L6-v2')
sia = SentimentIntensityAnalyzer()

SCANDAL_KEYWORDS = [
    "oil spill",
    "pollution",
    "toxic waste",
    "environmental damage",
    "deforestation",
    "chemical leak",
    "water contamination",
    "illegal dumping"
]

keyword_embeddings = model.encode(SCANDAL_KEYWORDS, convert_to_tensor=True)

def compute_scandal_score(text):

    sentences = sent_tokenize(text)

    if not sentences:
        return 0.0

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Compute similarity between each sentence and each keyword
    similarities = util.cos_sim(sentence_embeddings, keyword_embeddings)

    # Take maximum similarity
    max_score = similarities.max().item()

    return max_score

def get_sentiment(text):
    scores = sia.polarity_scores(text)

    compound = scores["compound"]

    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    else:
        return "neutral"

def predict_topic(text, clf, vectorizer):
    tokens = preprocess_text(text)
    clean_text = " ".join(tokens)

    print(f"Cleaned text (first 200 chars): {clean_text[:200]}")
    X = vectorizer.transform([clean_text])
    return clf.predict(X)[0]


def translate_to_english(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except Exception as e:
        print(f"Translation error: {e}")
        return text  # fallback
    
def main():
    clf, vectorizer = joblib.load("topic_classifier.pkl")
    df = pd.read_csv("data/raw_articles.csv")

    topics = []
    sentiments = []
    scandal_scores = []
    orgs_list = []

    # Headline + article (first 200 chars)
    df["text"] = df["headline"].fillna("") + " " + df["body"].fillna("")

    # Translate to English for consistent analysis
    df["text_en"] = df["text"].apply(translate_to_english)

    for i, row in df.iterrows():
        print(f"\nProcessing: {row['url']}")
       
        text_en = row["text_en"]

        # ---- Topic ----
        topic = predict_topic(text_en, clf, vectorizer)
        print(f"Topic: {topic}")

        # ---- Sentiment ----
        sentiment = get_sentiment(text_en)
        print(f"Sentiment: {sentiment}")

        # ---- Scandal ----
        score = compute_scandal_score(text_en)
        print(f"Scandal score: {score:.3f}")

        # ---- ORG detection ----
        orgs = extract_orgs(text_en)
        print(f"Detected ORGs: {orgs}")
        orgs_list.append(orgs)

        topics.append(topic)
        sentiments.append(sentiment)
        scandal_scores.append(score)

    # Save results
    df["topic"] = topics
    df["sentiment"] = sentiments
    df["scandal_score"] = scandal_scores

    # Flag top 10 scandals
    df["top_10"] = False
    top_indices = df["scandal_score"].nlargest(10).index
    df.loc[top_indices, "top_10"] = True

    df.to_csv("results/enhanced_news.csv", index=False)

    print("\n✅ Enriched dataset saved!")


# ✅ THIS MUST BE OUTSIDE THE MAIN FUNCTION TO AVOID RE-LOADING MODEL ON IMPORT
if __name__ == "__main__":
      main()