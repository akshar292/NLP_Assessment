"""
MODULE 11 - NATURAL LANGUAGE PROCESSING (NLP)
Assessment: M11-A1
SECTION A - CONCEPT APPLICATION
"""


# ============================================================
# S1 - NLP TECHNIQUES FOR CUSTOMER SUPPORT CHATBOT
# ============================================================

print("""
============================================================
S1 - NLP TECHNIQUES FOR CUSTOMER SUPPORT CHATBOT
============================================================

Answer:

Two important NLP techniques are:

1. Text Classification / Intent Classification
2. Sentiment Analysis

1. Text Classification / Intent Classification:
The chatbot needs to identify the user's intention from the
message.

For example:
- "Where is my order?" -> Order Tracking
- "I want to cancel my order" -> Order Cancellation
- "My food arrived cold and late" -> Complaint

This technique is necessary because the chatbot must route the
customer message to the correct service or response.

If intent classification is removed, the chatbot may send the
customer to the wrong department or provide an incorrect
response. Therefore, chatbot accuracy would decrease.

2. Sentiment Analysis:
Sentiment analysis identifies the emotional tone of the message,
such as positive, negative, or neutral.

For example:
"My food arrived cold and late" has a negative sentiment.

This is useful for identifying unhappy customers and handling
complaints appropriately.

If sentiment analysis is removed, the chatbot may understand
the topic but fail to recognize how serious or negative the
customer's experience is. This can reduce the quality and
accuracy of the response.
""")


# ============================================================
# S2 - TEXT PREPROCESSING PIPELINE
# ============================================================

print("""
============================================================
S2 - TEXT PREPROCESSING PIPELINE
============================================================

Answer:

For 50,000 raw customer reviews, I would apply the following
text preprocessing pipeline:

1. Convert text to lowercase
2. Remove unnecessary special characters and punctuation
3. Handle numbers where appropriate
4. Tokenize the text
5. Remove stopwords
6. Apply stemming or lemmatization
7. Reconstruct the cleaned text

Example:

Original:
"The FOOD is amazing!!! I ordered 2 pizzas."

After preprocessing:
"food amazing order pizza"

Order of processing:

First, convert all text to lowercase so that words such as
"Food" and "food" are treated consistently.

Second, remove punctuation and unnecessary special characters
to reduce noise.

Third, handle numbers depending on whether they are useful for
the NLP task.

Fourth, tokenize the text into individual words.

Fifth, remove common stopwords such as "the", "is", and "a".
These words occur frequently but usually provide limited useful
information for many text-classification tasks.

Finally, apply stemming or lemmatization to reduce words to
their basic forms.

Why stopword removal is useful:

If stopwords are not removed, the dataset contains many common
words that may add noise and increase the number of features.
This can increase the size of the feature space and make the
model focus on less informative words. Therefore, downstream
model performance can be affected, especially for traditional
bag-of-words or TF-IDF based models.
""")


# ============================================================
# S3 - BAG OF WORDS VS TF-IDF
# ============================================================

print("""
============================================================
S3 - BAG OF WORDS VS TF-IDF
============================================================

Answer:

Bag of Words (BoW):

Bag of Words represents a document using the frequency of words
in the document.

For example:

"spicy chicken chicken"

could be represented using word counts.

Advantages:
- Simple to understand
- Easy to implement
- Works well for basic text representation

Limitation:
It does not consider how important a word is across the whole
collection of documents.

TF-IDF:

TF-IDF stands for Term Frequency-Inverse Document Frequency.

It gives higher importance to words that are frequent in a
particular document but less common across the complete
collection of documents.

For the food search example:

"spicy chicken without onion"

TF-IDF can give more importance to terms that are useful for
distinguishing one menu description from another.

For this ranking task, I would use TF-IDF because the goal is
to rank menu descriptions according to their relevance to the
customer's query.

One limitation still remains:

TF-IDF does not understand the semantic meaning or context of
words. For example, it may not understand that two different
words can have similar meanings. It also does not naturally
understand word order or deeper language relationships.
""")


# ============================================================
# S4 - NAIVE BAYES ASSUMPTION
# ============================================================

print("""
============================================================
S4 - NAIVE BAYES CLASSIFIER
============================================================

Answer:

The core assumption of Naive Bayes is that the input features
are conditionally independent given the class.

In simple terms, the model assumes that the presence of one
word does not depend strongly on the presence of another word
when the class is known.

Example:

Suppose a support ticket says:

"My food arrived cold and late."

The words "cold" and "late" can both provide information about
the complaint.

However, in real language these words may be related to the
same customer experience. They are not always truly independent.

Another example:

"The delivery was late because the restaurant prepared the
food late."

Words such as "delivery", "late", "restaurant", and "food" have
relationships with each other.

This violates the independence assumption.

Effect on classification:

If the independence assumption is strongly violated, Naive
Bayes may incorrectly estimate the probability of a class.
This can lead to incorrect classification and reduce model
accuracy.
""")


# ============================================================
# S5 - IMBALANCED DATASET
# ============================================================

print("""
============================================================
S5 - IMBALANCED DATASET
============================================================

Answer:

The dataset contains:

Positive  = 80%
Neutral   = 15%
Negative  = 5%

Accuracy alone can be misleading because a model could predict
the majority Positive class most of the time and still obtain
high accuracy.

Two metrics I would prioritize are:

1. Precision
2. Recall

Precision:

Precision measures how many of the reviews predicted as a
particular class are actually members of that class.

Recall:

Recall measures how many of the actual samples belonging to a
class were correctly identified by the model.

F1-score could also be considered because it combines precision
and recall.

Technique to handle class imbalance:

One technique is oversampling the minority classes.

For example, the Negative class has only 5% of the reviews.
Oversampling can increase the representation of minority-class
examples in the training data.

Another possible technique is class weighting, where more
importance is given to minority classes during model training.

The technique should be applied only to the training data to
avoid data leakage.
""")


# ============================================================
# S6 - TRANSFORMER ATTENTION MECHANISM
# ============================================================

print("""
============================================================
S6 - TRANSFORMERS AND ATTENTION
============================================================

Answer:

The attention mechanism allows a Transformer to consider
different words in the input sequence and determine which words
are important in relation to each other.

For long text, attention can directly connect information from
different parts of the sequence.

Traditional RNN-based models process sequences step by step.
As the sequence becomes longer, it can become difficult for an
RNN to preserve information from earlier parts of the sequence.

Transformers use attention to examine relationships between
tokens more effectively and can process sequence information
more efficiently in parallel.

Practical production challenge:

A Transformer model can require significant computational
resources, especially when handling several hundred requests
per minute.

This can increase:
- Response latency
- CPU/GPU usage
- Memory usage
- Infrastructure cost

Common mitigation:

One common solution is model optimization and efficient serving.
For example, the model can be smaller or quantized, and
production systems can use batching, caching, or GPU-based
inference to handle high traffic more efficiently.
""")


# ============================================================
# END
# ============================================================

print("""
============================================================
SECTION A COMPLETED
============================================================
""")