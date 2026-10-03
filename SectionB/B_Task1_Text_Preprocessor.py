import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download English stopwords
nltk.download("stopwords", quiet=True)


# Create stopwords and stemmer
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()


# ---------------------------------------------------------
# TEXT PREPROCESSING FUNCTION
# ---------------------------------------------------------

def preprocess_text(text):

    # 1. Convert text to lowercase
    text = text.lower()

    # 2. Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenization
    words = text.split()

    # 5. Remove English stopwords
    words = [
        word
        for word in words
        if word not in stop_words
    ]

    # 6. Apply Porter stemming
    words = [
        stemmer.stem(word)
        for word in words
    ]

    # 7. Return final processed string
    return " ".join(words)


# ---------------------------------------------------------
# TESTING
# ---------------------------------------------------------

reviews = [
    "The food was AMAZING! Very tasty and fresh.",
    "My order arrived late, and the pizza was completely cold.",
    "I love the quick delivery and friendly delivery partner."
]


print("=" * 60)
print("SECTION B - TASK 1")
print("FOOD REVIEW TEXT PREPROCESSOR")
print("=" * 60)


for i, review in enumerate(reviews, start=1):

    processed_review = preprocess_text(review)

    print(f"\nReview {i}")
    print("Original  :", review)
    print("Processed :", processed_review)