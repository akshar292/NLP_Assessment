import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ---------------------------------------------------------
# NLTK SETUP
# ---------------------------------------------------------

nltk.download(
    "stopwords",
    quiet=True
)


stop_words = set(
    stopwords.words("english")
)

stemmer = PorterStemmer()


# ---------------------------------------------------------
# PREPROCESSING FUNCTION
# ---------------------------------------------------------

def preprocess_text(text):

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = re.sub(
        r"[^a-z\s]",
        " ",
        text
    )

    # Remove extra whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    # Tokenization
    words = text.split()

    # Remove stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------------
# DATASET
# 30 REVIEWS
# 15 POSITIVE
# 15 NEGATIVE
# ---------------------------------------------------------

data = [

    # POSITIVE
    ("The food was delicious and fresh", "Positive"),
    ("Excellent pizza and fast delivery", "Positive"),
    ("I loved the burger it was tasty", "Positive"),
    ("The order arrived quickly and hot", "Positive"),
    ("Amazing food and great service", "Positive"),
    ("Very happy with my order", "Positive"),
    ("The meal was fresh and delicious", "Positive"),
    ("Fast delivery and excellent food", "Positive"),
    ("The restaurant prepared a tasty meal", "Positive"),
    ("Great experience and friendly delivery", "Positive"),
    ("The food quality was excellent", "Positive"),
    ("I enjoyed the delicious chicken", "Positive"),
    ("Perfect order and quick service", "Positive"),
    ("The pizza was hot and wonderful", "Positive"),
    ("Very good food and service", "Positive"),

    # NEGATIVE
    ("The food was cold and terrible", "Negative"),
    ("My order arrived very late", "Negative"),
    ("The burger was burnt and bad", "Negative"),
    ("Terrible food and slow delivery", "Negative"),
    ("I hated the cold pizza", "Negative"),
    ("The meal was stale and tasteless", "Negative"),
    ("Very poor food quality", "Negative"),
    ("The delivery was extremely late", "Negative"),
    ("The food tasted awful", "Negative"),
    ("Bad service and cold food", "Negative"),
    ("My order was delayed and disappointing", "Negative"),
    ("The chicken was undercooked", "Negative"),
    ("The food was not fresh", "Negative"),
    ("Terrible experience and bad service", "Negative"),
    ("I am unhappy with this order", "Negative")
]


# ---------------------------------------------------------
# PREPROCESS DATA
# ---------------------------------------------------------

texts = [
    preprocess_text(text)
    for text, label in data
]


labels = [
    label
    for text, label in data
]


# ---------------------------------------------------------
# TRAIN TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


# =========================================================
# PIPELINE 1
# TF-IDF + NAIVE BAYES
# =========================================================

naive_bayes_pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer()
    ),

    (
        "classifier",
        MultinomialNB()
    )
])


# =========================================================
# PIPELINE 2
# TF-IDF + LOGISTIC REGRESSION
# =========================================================

logistic_pipeline = Pipeline([

    (
        "tfidf",
        TfidfVectorizer()
    ),

    (
        "classifier",
        LogisticRegression(
            max_iter=1000
        )
    )
])


# ---------------------------------------------------------
# TRAIN MODELS
# ---------------------------------------------------------

naive_bayes_pipeline.fit(
    X_train,
    y_train
)


logistic_pipeline.fit(
    X_train,
    y_train
)


# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

nb_predictions = naive_bayes_pipeline.predict(
    X_test
)


lr_predictions = logistic_pipeline.predict(
    X_test
)


# ---------------------------------------------------------
# METRIC FUNCTION
# ---------------------------------------------------------

def calculate_metrics(
    model_name,
    predictions
):

    return {

        "Model": model_name,

        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),

        "Precision": precision_score(
            y_test,
            predictions,
            pos_label="Positive",
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            predictions,
            pos_label="Positive",
            zero_division=0
        ),

        "F1-Score": f1_score(
            y_test,
            predictions,
            pos_label="Positive",
            zero_division=0
        )
    }


# ---------------------------------------------------------
# CALCULATE RESULTS
# ---------------------------------------------------------

results = [

    calculate_metrics(
        "Naive Bayes",
        nb_predictions
    ),

    calculate_metrics(
        "Logistic Regression",
        lr_predictions
    )
]


# ---------------------------------------------------------
# COMPARISON TABLE
# ---------------------------------------------------------

comparison_df = pd.DataFrame(
    results
)


# ---------------------------------------------------------
# DISPLAY
# ---------------------------------------------------------

print("=" * 70)
print("SECTION B - TASK 4")
print("SENTIMENT ANALYSIS MODEL COMPARISON")
print("=" * 70)


print("\nModel Comparison:\n")


print(
    comparison_df.round(3).to_string(
        index=False
    )
)


# ---------------------------------------------------------
# REQUIRED ONE-LINE COMMENT
# ---------------------------------------------------------

# Deployment comment:
# For this assessment dataset, Logistic Regression is selected
# because it works effectively with TF-IDF text features and
# provides a linear decision boundary for binary classification.