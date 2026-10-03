import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer


# ---------------------------------------------------------
# FOOD DELIVERY REVIEW DATASET
# ---------------------------------------------------------

reviews = [
    "The pizza was hot and delicious",
    "The burger was tasty and fresh",
    "The pizza arrived cold and late",
    "Very tasty burger with fresh vegetables",
    "The delivery was late and the food was cold",
    "Excellent food and fast delivery",
    "The food was average and okay",
    "Terrible service and cold food"
]


# ---------------------------------------------------------
# CREATE TF-IDF VECTORIZER
# ---------------------------------------------------------

vectorizer = TfidfVectorizer(
    stop_words="english"
)


# Convert reviews into TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(reviews)


# Get feature names
feature_names = vectorizer.get_feature_names_out()


# ---------------------------------------------------------
# CREATE LABELLED DATAFRAME
# ---------------------------------------------------------

tfidf_df = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=feature_names,
    index=[
        f"Review {i}"
        for i in range(1, len(reviews) + 1)
    ]
)


# ---------------------------------------------------------
# DISPLAY FULL TF-IDF MATRIX
# ---------------------------------------------------------

print("=" * 70)
print("SECTION B - TASK 2")
print("TF-IDF MATRIX")
print("=" * 70)

print("\nFull TF-IDF Matrix:\n")

print(
    tfidf_df.round(3)
)


# ---------------------------------------------------------
# TOP 3 WORDS FOR EACH REVIEW
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("TOP 3 WORDS FOR EACH REVIEW")
print("=" * 70)


for i in range(len(reviews)):

    scores = (
        tfidf_matrix[i]
        .toarray()
        .flatten()
    )

    # Get indexes of highest scores
    top_indexes = scores.argsort()[::-1][:3]

    print(f"\nReview {i + 1}:")
    print(reviews[i])

    for index in top_indexes:

        word = feature_names[index]
        score = scores[index]

        print(
            f"  {word:<15} "
            f"Score: {score:.4f}"
        )