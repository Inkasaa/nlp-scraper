"""
training_model.py
Train a topic classification model using CountVectorizer and a classifier.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score
import joblib
from nlp_utils import preprocess_text
import os

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # Use correct column names from dataset
    df["text"] = df["Text"].fillna("")
    df["label"] = df["Category"]

    # Preprocess
    df["tokens"] = df["text"].apply(preprocess_text)
    df["clean_text"] = df["tokens"].apply(lambda tokens: " ".join(tokens))

    return df

def train_and_evaluate(df):
    """Train classifier and evaluate accuracy."""
    X = df['clean_text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)
    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test accuracy: {acc:.3f}")
    assert acc > 0.95, "Accuracy must be > 0.95"
    return clf, vectorizer, X_train, y_train

def plot_learning_curve(clf, X, y, save_path):
    """Plot and save learning curve."""
    vectorizer = CountVectorizer()
    X_vec = vectorizer.fit_transform(X)
    train_sizes, train_scores, test_scores = learning_curve(
        clf, X_vec, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=[0.1, 0.33, 0.55, 0.78, 1.0]
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)
    plt.figure()
    plt.plot(train_sizes, train_mean, 'o-', label='Train')
    plt.plot(train_sizes, test_mean, 'o-', label='Test')
    plt.xlabel('Training examples')
    plt.ylabel('Accuracy')
    plt.title('Learning Curve')
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

def save_model(clf, vectorizer, path):
    """Save classifier and vectorizer as a tuple."""
    joblib.dump((clf, vectorizer), path)

if __name__ == "__main__":
    train_path = "data/bbc_news_train.csv"
    test_path = "data/bbc_news_tests.csv"

    train_df = load_and_preprocess(train_path)
    test_df = load_and_preprocess(test_path)

    X_train = train_df["clean_text"]
    y_train = train_df["label"]
    print(f"Labels in training set: {y_train.unique()}")

    X_test = test_df["clean_text"]
    y_test = test_df["label"]

    vectorizer = CountVectorizer()
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)

    print(f"Test accuracy: {acc:.3f}")
    assert acc > 0.95, "Accuracy must be > 0.95"

    plot_learning_curve(clf, X_train, y_train, "results/learning_curves.png")
    save_model(clf, vectorizer, "topic_classifier.pkl")

    print("Model saved!")