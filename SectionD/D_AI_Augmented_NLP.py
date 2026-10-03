# ============================================================
# SECTION D - AI-AUGMENTED LEARNING
# MODULE 11 - NATURAL LANGUAGE PROCESSING
# ============================================================

import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ============================================================
# NLTK SETUP
# ============================================================

nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


# ============================================================
# TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 4. Tokenisation
    tokens = text.split()

    # 5. Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # 6. Porter stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    return " ".join(tokens)


# ============================================================
# FOOD DELIVERY REVIEW DATASET
# ============================================================

positive_reviews = [

    "The food was delicious and fresh",
    "I loved the pizza and fast delivery",
    "The burger was very tasty",
    "Excellent food and great service",
    "My order arrived quickly and hot",
    "Amazing food quality",
    "The meal was fresh and delicious",
    "Very happy with my order",
    "Fast delivery and friendly rider",
    "The food tasted wonderful",

    "Great restaurant and excellent food",
    "The pizza was hot and tasty",
    "I enjoyed my meal very much",
    "Perfect food and service",
    "The delivery was quick and smooth",
    "Very good experience",
    "The food was hot and delicious",
    "The delivery arrived on time",
    "Tasty and fresh meal",
    "Great service and delicious food",

    "Quick delivery and hot food",
    "The order came very fast",
    "Really enjoyed the tasty burger",
    "Wonderful meal and excellent service",
    "Fresh food and friendly service",
    "The pizza tasted amazing",
    "Happy with the fast delivery",
    "Good food arrived quickly",
    "Excellent restaurant experience",
    "Very tasty and fresh food",

    "Amazing meal and service",
    "Fast and reliable delivery",
    "The food quality was excellent",
    "The meal was absolutely delicious",
    "The rider was quick and polite",
    "The order was perfect",
    "I really enjoyed the food",
    "The restaurant provided excellent service",
    "The food arrived hot",
    "Very satisfied with my order"
]


negative_reviews = [

    "The food was cold and terrible",
    "My order arrived very late",
    "The burger tasted bad",
    "Terrible food and slow delivery",
    "I hated the cold pizza",
    "The meal was stale",
    "Very poor food quality",
    "The delivery was extremely late",
    "The food tasted awful",
    "Bad service and cold food",

    "My order was delayed",
    "The chicken was undercooked",
    "The food was not fresh",
    "Terrible experience",
    "I am unhappy with this order",
    "Very disappointing service",
    "Food was cold when it arrived",
    "The delivery took too long",
    "The meal tasted horrible",
    "Poor quality food",

    "Late delivery and cold meal",
    "The order was very late",
    "Burnt pizza and bad taste",
    "Terrible restaurant service",
    "Stale food and slow delivery",
    "The burger was cold and awful",
    "Unhappy with the delayed order",
    "Bad food and terrible service",
    "Extremely poor experience",
    "The food was tasteless",

    "Very bad delivery",
    "The meal was not fresh",
    "The food was burnt",
    "The order arrived cold",
    "The service was very poor",
    "I was disappointed with the food",
    "The delivery was slow",
    "The pizza was terrible",
    "The food quality was awful",
    "I am unhappy with the service"
]


# ============================================================
# CREATE DATASET
# ============================================================

reviews = positive_reviews + negative_reviews

labels = (
    ["Positive"] * len(positive_reviews)
    +
    ["Negative"] * len(negative_reviews)
)


# ============================================================
# DATASET VALIDATION
# ============================================================

if len(reviews) != len(labels):

    raise ValueError(
        "Number of reviews and labels must be equal."
    )


print("=" * 70)
print("SECTION D - AI-AUGMENTED NLP LEARNING")
print("=" * 70)

print(
    f"\nTotal reviews: {len(reviews)}"
)

print(
    f"Positive reviews: {len(positive_reviews)}"
)

print(
    f"Negative reviews: {len(negative_reviews)}"
)


# ============================================================
# PREPROCESS DATA
# ============================================================

processed_reviews = [
    preprocess_text(review)
    for review in reviews
]


# ============================================================
# SHOW SAMPLE PREPROCESSING
# ============================================================

print("\n")
print("=" * 70)
print("SAMPLE PREPROCESSING")
print("=" * 70)

for i in range(3):

    print("\nOriginal :")
    print(reviews[i])

    print("Processed:")
    print(processed_reviews[i])

    print("-" * 70)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    processed_reviews,

    labels,

    test_size=0.20,

    random_state=42,

    stratify=labels
)


print("\n")
print("=" * 70)
print("DATASET SPLIT")
print("=" * 70)

print(
    f"\nTraining samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# TF-IDF VECTORIZATION
# ============================================================

vectorizer = TfidfVectorizer(

    # Use single words and two-word combinations
    ngram_range=(1, 2),

    # Give more importance to useful repeated terms
    sublinear_tf=True
)


# IMPORTANT:
# Fit only on training data
X_train_tfidf = vectorizer.fit_transform(
    X_train
)


# Transform test data using the same vectorizer
X_test_tfidf = vectorizer.transform(
    X_test
)


# ============================================================
# LOGISTIC REGRESSION
# ============================================================

model = LogisticRegression(

    max_iter=1000,

    random_state=42
)


# Train model
model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# TEST PREDICTIONS
# ============================================================

y_pred = model.predict(
    X_test_tfidf
)


# ============================================================
# ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n")
print("=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"\nTest Accuracy: {accuracy:.2%}"
)


# ============================================================
# SHOW TEST PREDICTIONS
# ============================================================

print("\n")
print("TEST PREDICTIONS")
print("-" * 70)

for review, actual, predicted in zip(
    X_test,
    y_test,
    y_pred
):

    print(
        f"Actual: {actual:<10} "
        f"Predicted: {predicted:<10}"
    )


# ============================================================
# NEW USER REVIEWS
# ============================================================

print("\n")
print("=" * 70)
print("NEW REVIEW SENTIMENT PREDICTION")
print("=" * 70)

print(
    "\nEnter exactly 3 new food delivery reviews."
)


for i in range(1, 4):

    print(f"\nReview {i}")

    user_review = input(
        "Enter review: "
    ).strip()


    # --------------------------------------------------------
    # EDGE CASE 1 - EMPTY INPUT
    # --------------------------------------------------------

    if not user_review:

        print(
            "Error: Review cannot be empty."
        )

        continue


    # --------------------------------------------------------
    # PREPROCESS USER REVIEW
    # --------------------------------------------------------

    processed_review = preprocess_text(
        user_review
    )


    # --------------------------------------------------------
    # EDGE CASE 2 - NO USABLE WORDS
    # --------------------------------------------------------

    if not processed_review:

        print(
            "Error: No usable words found after preprocessing."
        )

        continue


    # --------------------------------------------------------
    # TF-IDF TRANSFORMATION
    # --------------------------------------------------------

    review_tfidf = vectorizer.transform(
        [processed_review]
    )


    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    prediction = model.predict(
        review_tfidf
    )[0]


    # --------------------------------------------------------
    # CONFIDENCE
    # --------------------------------------------------------

    probabilities = model.predict_proba(
        review_tfidf
    )[0]

    confidence = max(
        probabilities
    )


    # --------------------------------------------------------
    # DISPLAY RESULT
    # --------------------------------------------------------

    print("\nProcessed Review:")
    print(processed_review)

    print(
        "\nPredicted Sentiment:",
        prediction
    )

    print(
        f"Confidence: {confidence:.2%}"
    )


# ============================================================
# END
# ============================================================

print("\n")
print("=" * 70)
print("SECTION D COMPLETED")
print("=" * 70)