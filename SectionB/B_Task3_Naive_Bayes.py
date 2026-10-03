import re
import nltk
import pandas as pd

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    classification_report,
    confusion_matrix
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
# Same concept as Task 1
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

    # Stopword removal
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # Porter stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    return " ".join(words)


# ---------------------------------------------------------
# DATASET
# 24 COMPLAINTS
# 8 Delivery
# 8 Food Quality
# 8 App
# ---------------------------------------------------------

data = [

    # DELIVERY
    ("My order is very late", "Delivery"),
    ("The delivery partner has not arrived", "Delivery"),
    ("Where is my food order", "Delivery"),
    ("My order has been delayed", "Delivery"),
    ("Food delivery is taking too long", "Delivery"),
    ("The rider delivered my order late", "Delivery"),
    ("My food has not arrived yet", "Delivery"),
    ("The delivery person is still not here", "Delivery"),

    # FOOD QUALITY
    ("The food arrived cold", "Food Quality"),
    ("My pizza was burnt", "Food Quality"),
    ("The burger tasted terrible", "Food Quality"),
    ("The food was cold and stale", "Food Quality"),
    ("My meal was not fresh", "Food Quality"),
    ("The chicken was undercooked", "Food Quality"),
    ("The food quality was very poor", "Food Quality"),
    ("My order tasted bad and cold", "Food Quality"),

    # APP
    ("The app is not opening", "App"),
    ("I cannot login to the application", "App"),
    ("The payment failed in the app", "App"),
    ("The app keeps crashing", "App"),
    ("I cannot place an order using the app", "App"),
    ("The application is showing an error", "App"),
    ("My payment is not working", "App"),
    ("The mobile app has a technical problem", "App")
]


# ---------------------------------------------------------
# SEPARATE TEXT AND LABELS
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
# 80% TRAIN / 20% TEST
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    texts,
    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


# ---------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------

vectorizer = TfidfVectorizer()


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


# ---------------------------------------------------------
# MULTINOMIAL NAIVE BAYES
# ---------------------------------------------------------

model = MultinomialNB()


model.fit(
    X_train_tfidf,
    y_train
)


# ---------------------------------------------------------
# PREDICTION
# ---------------------------------------------------------

y_pred = model.predict(
    X_test_tfidf
)


# ---------------------------------------------------------
# CLASSIFICATION REPORT
# ---------------------------------------------------------

print("=" * 70)
print("SECTION B - TASK 3")
print("NAIVE BAYES COMPLAINT CLASSIFIER")
print("=" * 70)


print("\nClassification Report:\n")


print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

classes = [
    "Delivery",
    "Food Quality",
    "App"
]


cm = confusion_matrix(
    y_test,
    y_pred,
    labels=classes
)


cm_df = pd.DataFrame(
    cm,
    index=classes,
    columns=classes
)


print("\nConfusion Matrix:\n")


print(cm_df)