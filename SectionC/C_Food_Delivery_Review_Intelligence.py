import re
import nltk

from collections import Counter

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression


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
    """
    Preprocessing steps:
    1. Lowercase
    2. Remove punctuation
    3. Remove extra whitespace
    4. Tokenization
    5. Stopword removal
    6. Stemming
    """

    # Lowercase
    text = text.lower()

    # Remove punctuation and special characters
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

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


# ============================================================
# TRAINING DATA
# ============================================================

training_data = [

    # --------------------------------------------------------
    # POSITIVE - FOOD QUALITY
    # --------------------------------------------------------

    ("The food was hot and delicious",
     "Positive",
     "Food Quality"),

    ("The pizza was fresh and tasty",
     "Positive",
     "Food Quality"),

    ("Excellent food quality",
     "Positive",
     "Food Quality"),

    ("The meal was delicious",
     "Positive",
     "Food Quality"),

    ("The burger was fresh and tasty",
     "Positive",
     "Food Quality"),

    ("The chicken tasted amazing",
     "Positive",
     "Food Quality"),

    # --------------------------------------------------------
    # POSITIVE - DELIVERY
    # --------------------------------------------------------

    ("The order arrived quickly",
     "Positive",
     "Delivery"),

    ("Fast delivery and friendly rider",
     "Positive",
     "Delivery"),

    ("My order was delivered on time",
     "Positive",
     "Delivery"),

    ("Quick and smooth delivery",
     "Positive",
     "Delivery"),

    ("The rider delivered the food quickly",
     "Positive",
     "Delivery"),

    ("Very fast delivery service",
     "Positive",
     "Delivery"),

    # --------------------------------------------------------
    # POSITIVE - APP
    # --------------------------------------------------------

    ("The app works perfectly",
     "Positive",
     "App"),

    ("Easy ordering through the application",
     "Positive",
     "App"),

    ("The app is fast and simple",
     "Positive",
     "App"),

    ("Payment worked perfectly in the app",
     "Positive",
     "App"),

    ("The application is easy to use",
     "Positive",
     "App"),

    ("Ordering from the app was simple",
     "Positive",
     "App"),

    # --------------------------------------------------------
    # POSITIVE - GENERAL
    # --------------------------------------------------------

    ("I enjoyed my meal",
     "Positive",
     "General"),

    ("Great overall experience",
     "Positive",
     "General"),

    ("Very happy with the service",
     "Positive",
     "General"),

    ("Good experience",
     "Positive",
     "General"),

    ("Amazing overall service",
     "Positive",
     "General"),

    ("I am very satisfied",
     "Positive",
     "General"),

    # --------------------------------------------------------
    # NEGATIVE - FOOD QUALITY
    # --------------------------------------------------------

    ("The food was cold and stale",
     "Negative",
     "Food Quality"),

    ("The pizza was burnt",
     "Negative",
     "Food Quality"),

    ("The meal tasted terrible",
     "Negative",
     "Food Quality"),

    ("The food quality was poor",
     "Negative",
     "Food Quality"),

    ("The burger was cold",
     "Negative",
     "Food Quality"),

    ("The chicken was undercooked",
     "Negative",
     "Food Quality"),

    # --------------------------------------------------------
    # NEGATIVE - DELIVERY
    # --------------------------------------------------------

    ("My order was very late",
     "Negative",
     "Delivery"),

    ("The rider did not arrive",
     "Negative",
     "Delivery"),

    ("Delivery took too long",
     "Negative",
     "Delivery"),

    ("My food has not arrived",
     "Negative",
     "Delivery"),

    ("The order was delayed",
     "Negative",
     "Delivery"),

    ("The delivery was extremely late",
     "Negative",
     "Delivery"),

    # --------------------------------------------------------
    # NEGATIVE - APP
    # --------------------------------------------------------

    ("The app keeps crashing",
     "Negative",
     "App"),

    ("I cannot login to the app",
     "Negative",
     "App"),

    ("Payment failed in the application",
     "Negative",
     "App"),

    ("The app shows an error",
     "Negative",
     "App"),

    ("The application is not working",
     "Negative",
     "App"),

    ("I cannot place an order using the app",
     "Negative",
     "App"),

    # --------------------------------------------------------
    # NEGATIVE - GENERAL
    # --------------------------------------------------------

    ("I am unhappy with this service",
     "Negative",
     "General"),

    ("This was a terrible experience",
     "Negative",
     "General"),

    ("Very disappointing order",
     "Negative",
     "General"),

    ("I did not enjoy the experience",
     "Negative",
     "General"),

    ("The service was terrible",
     "Negative",
     "General"),

    ("Very bad overall experience",
     "Negative",
     "General")
]


# ============================================================
# FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM
# ============================================================

class FoodDeliveryReviewSystem:

    def __init__(self):

        # Store customer reviews
        self.reviews = []

        # ----------------------------------------------------
        # Prepare training text
        # ----------------------------------------------------

        self.training_texts = [
            preprocess_text(text)
            for text, sentiment, category
            in training_data
        ]

        # Sentiment labels
        self.sentiment_labels = [
            sentiment
            for text, sentiment, category
            in training_data
        ]

        # Issue category labels
        self.category_labels = [
            category
            for text, sentiment, category
            in training_data
        ]

        # ----------------------------------------------------
        # Sentiment Model
        # TF-IDF + Logistic Regression
        # ----------------------------------------------------

        self.sentiment_model = Pipeline([
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

        # ----------------------------------------------------
        # Category Model
        # TF-IDF + Naive Bayes
        # ----------------------------------------------------

        self.category_model = Pipeline([
            (
                "tfidf",
                TfidfVectorizer()
            ),
            (
                "classifier",
                MultinomialNB()
            )
        ])

        self.trained = False


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    def train_models(self):

        minimum_samples = 10

        # Edge case required by assessment
        if len(self.training_texts) < minimum_samples:

            print(
                "Not enough training samples available."
            )

            self.trained = False

            return

        # Train sentiment model
        self.sentiment_model.fit(
            self.training_texts,
            self.sentiment_labels
        )

        # Train category model
        self.category_model.fit(
            self.training_texts,
            self.category_labels
        )

        self.trained = True


    # ========================================================
    # OPTION 1 - ADD NEW REVIEW
    # ========================================================

    def add_review(self):

        print("\n")
        print("=" * 60)
        print("ADD NEW REVIEW")
        print("=" * 60)

        review = input(
            "Enter customer review: "
        ).strip()

        # Check empty input
        if not review:

            print(
                "\nReview cannot be empty."
            )

            return

        # Preprocess review
        processed = preprocess_text(review)

        # Check after preprocessing
        if not processed:

            print(
                "\nNo usable words found after preprocessing."
            )

            return

        # Store review
        self.reviews.append({
            "original": review,
            "processed": processed,
            "sentiment": None,
            "category": None
        })

        print(
            "\nReview added successfully!"
        )

        print(
            "Processed text:",
            processed
        )


    # ========================================================
    # OPTION 2 - CLASSIFY REVIEW
    # ========================================================

    def classify_review(self):

        print("\n")
        print("=" * 60)
        print("CLASSIFY REVIEW")
        print("=" * 60)

        # Check model
        if not self.trained:

            print(
                "\nClassification cannot be performed."
            )

            print(
                "Minimum training samples are not available."
            )

            return

        # Get review
        review = input(
            "Enter customer review: "
        ).strip()

        # Empty input
        if not review:

            print(
                "\nReview cannot be empty."
            )

            return

        # Preprocess
        processed = preprocess_text(review)

        # Empty after preprocessing
        if not processed:

            print(
                "\nNo usable words found."
            )

            return

        # ----------------------------------------------------
        # Predict sentiment
        # ----------------------------------------------------

        sentiment = self.sentiment_model.predict(
            [processed]
        )[0]

        # ----------------------------------------------------
        # Predict category
        # ----------------------------------------------------

        category = self.category_model.predict(
            [processed]
        )[0]

        # ----------------------------------------------------
        # Display result
        # ----------------------------------------------------

        print("\n")
        print("-" * 60)
        print("CLASSIFICATION RESULT")
        print("-" * 60)

        print(
            "Original Review :",
            review
        )

        print(
            "Processed Text  :",
            processed
        )

        print(
            "Sentiment       :",
            sentiment
        )

        print(
            "Issue Category  :",
            category
        )

        print("-" * 60)

        # Store classified review
        self.reviews.append({
            "original": review,
            "processed": processed,
            "sentiment": sentiment,
            "category": category
        })


    # ========================================================
    # CLASSIFY REVIEWS ADDED USING OPTION 1
    # ========================================================

    def classify_stored_reviews(self):

        if not self.trained:

            return

        for review in self.reviews:

            # Only classify reviews which
            # have not already been classified
            if review["sentiment"] is None:

                processed = review["processed"]

                review["sentiment"] = (
                    self.sentiment_model.predict(
                        [processed]
                    )[0]
                )

                review["category"] = (
                    self.category_model.predict(
                        [processed]
                    )[0]
                )


    # ========================================================
    # OPTION 3 - SUMMARY REPORT
    # ========================================================

    def summary_report(self):

        print("\n")
        print("=" * 60)
        print("SUMMARY REPORT")
        print("=" * 60)

        # No reviews
        if not self.reviews:

            print(
                "\nNo reviews have been added yet."
            )

            return

        # Classify unclassified reviews
        self.classify_stored_reviews()

        # ----------------------------------------------------
        # TOTAL REVIEWS
        # ----------------------------------------------------

        total_reviews = len(
            self.reviews
        )

        print(
            "\nTotal Reviews Added:",
            total_reviews
        )

        # ----------------------------------------------------
        # SENTIMENT COUNT
        # ----------------------------------------------------

        sentiment_counts = Counter(
            review["sentiment"]
            for review in self.reviews
        )

        print("\n")
        print("SENTIMENT COUNT")
        print("-" * 30)

        print(
            "Positive:",
            sentiment_counts.get(
                "Positive",
                0
            )
        )

        print(
            "Negative:",
            sentiment_counts.get(
                "Negative",
                0
            )
        )

        # ----------------------------------------------------
        # ISSUE CATEGORY COUNT
        # ----------------------------------------------------

        category_counts = Counter(
            review["category"]
            for review in self.reviews
        )

        print("\n")
        print("ISSUE CATEGORY COUNT")
        print("-" * 30)

        categories = [
            "Delivery",
            "Food Quality",
            "App",
            "General"
        ]

        for category in categories:

            print(
                f"{category}: "
                f"{category_counts.get(category, 0)}"
            )

        # ----------------------------------------------------
        # TOP 5 CONTENT WORDS
        # ----------------------------------------------------

        all_words = []

        for review in self.reviews:

            all_words.extend(
                review["processed"].split()
            )

        word_counts = Counter(
            all_words
        )

        top_words = word_counts.most_common(5)

        print("\n")
        print("5 MOST FREQUENT CONTENT WORDS")
        print("-" * 30)

        if top_words:

            for word, count in top_words:

                print(
                    f"{word}: {count}"
                )

        else:

            print(
                "No content words available."
            )


    # ========================================================
    # MENU
    # ========================================================

    def run(self):

        # Train models before starting
        self.train_models()

        while True:

            print("\n")
            print("=" * 60)
            print(
                "FOOD DELIVERY REVIEW INTELLIGENCE SYSTEM"
            )
            print("=" * 60)

            print(
                "1. Add a new review"
            )

            print(
                "2. Classify a review"
            )

            print(
                "3. View summary report"
            )

            print(
                "4. Exit"
            )

            choice = input(
                "\nEnter your choice (1-4): "
            ).strip()

            # Option 1
            if choice == "1":

                self.add_review()

            # Option 2
            elif choice == "2":

                self.classify_review()

            # Option 3
            elif choice == "3":

                self.summary_report()

            # Option 4
            elif choice == "4":

                print(
                    "\nThank you. Program ended."
                )

                break

            # Invalid choice
            else:

                print(
                    "\nInvalid choice!"
                )

                print(
                    "Please enter 1, 2, 3, or 4."
                )


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    system = FoodDeliveryReviewSystem()

    system.run()