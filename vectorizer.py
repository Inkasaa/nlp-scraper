"""
vectorizer.py
Convert preprocessed text into numerical features using CountVectorizer.
"""

from sklearn.feature_extraction.text import CountVectorizer


def vectorize_corpus(tokenized_texts: list[list[str]]):
    """
    Convert a list of tokenized texts into a feature matrix using CountVectorizer.
    Returns the matrix X and the fitted vectorizer.
    """
    # Join tokens back into text
    texts = [" ".join(tokens) for tokens in tokenized_texts]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    return X, vectorizer

if __name__ == "__main__":
    # Example tokenized articles
    sample = [
        ["tesla", "expand", "europe"],
        ["company", "profit", "increase"]
    ]
    X, vectorizer = vectorize_corpus(sample)
    print("Vocabulary:", vectorizer.get_feature_names_out())
    print("Matrix shape:", X.shape)
    print("Matrix (dense, first rows):\n", X.toarray()[:5])
