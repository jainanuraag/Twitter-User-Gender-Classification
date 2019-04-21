import pandas as pd
import random
import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer

# Data guidelines according to Canvas
train_data_size = .6
test_data_size = .2
val_data_size = .2

# Stop words from Natural Language Toolkit will be ignored in tweet text
stop_words = {
'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there',
'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they',
'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into',
'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who',
'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below',
'are', 'we', 'these', 'your', 'his', 'through', 'don', 'nor', 'me',
'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our',
'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she',
'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been',
'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that',
'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now',
'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only',
'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
'if', 'theirs', 'my', 'against', 'a', 'doing', 'it', 'how', 'further',
'was', 'here', 'than', 'by'
}

# Utility function taken and adapted from:
# https://github.com/rasto2211/Twitter-User-Gender-Classification/blob/master/notebooks/exploration.ipynb
# Normalizes text for analysis by removing URLs, special characters, and double spaces
def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+', ' ', text)
    # Remove URLs
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)
    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '', text)
    # Remove double spaces.
    text = re.sub('\s+', ' ', text)
    # Make text lowercase
    return text.lower()

# Extract data
df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='latin1')

# Process data to partition into training, validation, and test data
# Choose data where gender is male or female and classification confidence is about 1
chosen_rows = df[df["gender"].isin(["male", "female"]) & (df["gender:confidence"] > 0.99)].index.tolist()
n = len(chosen_rows)

# Shuffle data to ensure randomness
random.shuffle(chosen_rows)

# Normalize and filter out stop words from tweets
df['edited_text'] = [normalize_text(tweet) for tweet in df['text']]
filtered_text = []
for tweet in df['edited_text']:
    # Duplicate words removed from individual tweets
    words_in_tweet = set(tweet.split())
    filtered_text.append(" ".join(list(filter(lambda word: word not in stop_words, words_in_tweet))))
df['edited_text'] = filtered_text

# Partition chosen_rows into training, validation, and test datasets
train_data_nrows = round(train_data_size * n)
train_data = chosen_rows[:train_data_nrows]
train_data = df.ix[train_data, :]
validation_upper_limit = (train_data_nrows + round(val_data_size * n))
validation_data = chosen_rows[train_data_nrows: validation_upper_limit]
validation_data = df.ix[validation_data, :]
test_data = chosen_rows[validation_upper_limit:]
test_data = df.ix[test_data, :]

# Calculate Prior Probabilities
n_males = sum(train_data['gender'].isin(['male']))
n = len(train_data.index)
n_females = n - n_males
p_males = n_males / n
p_females = n_females / n

# Build Vocabulary
# Vocabulary is Set of unique words in the training data
vocabulary = set()
for tweet in train_data['edited_text']:
    vocabulary = vocabulary | set(tweet.split())
vocabulary_len = len(vocabulary)

# Build Male Text
# Male Text is list of all words said by males
male_training_data = train_data[train_data['gender'].isin(['male'])]
male_text = []
for tweet in male_training_data['edited_text']:
    male_text += tweet.split()

# Build Female Text
# Female Text is list of all words said by females
female_training_data = train_data[train_data['gender'].isin(['female'])]
female_text = []
for tweet in female_training_data['edited_text']:
    female_text += tweet.split()

# Count occurences of words within each text
male_counts = {word: male_text.count(word) for word in vocabulary}
female_counts = {word: female_text.count(word) for word in vocabulary}

# Build denominators for Multinomial Naive Bayes
male_distinct_n = len(set(male_text))
female_distinct_n = len(set(female_text))
male_denominator = male_distinct_n + vocabulary_len
female_denominator = female_distinct_n + vocabulary_len

# Calculate accuracy on Validation data
def validate():
    prediction = []  # stores predicted values
    correct = []  # boolean list determine whether prediction was correct
    # Check all validation data
    for index, row in validation_data.iterrows():
        tweet = row['text']
        true_gender = row['gender']
        edited_text = normalize_text(tweet)
        words_in_tweet = set(edited_text.split())
        edited_text = list(filter(lambda word: word not in stop_words, words_in_tweet))

        # Initialize probabilities as p(males) and p(females)
        male_prob = p_males
        female_prob = p_females

        # Compute male and female probabilities
        for word in edited_text:
            # Error if word not in male_counts so we give it a value of 1
            try:
                male_count = male_counts[word] + 1
            except KeyError:
                male_count = 1
            male_prob *= male_count / male_denominator
            # Error if word not in female_counts so we give it a value of 1
            try:
                female_count = female_counts[word] + 1
            except KeyError:
                female_count = 1
            female_prob *= female_count / female_denominator

        # Add prediction for tweet based on max probability
        tweet_prediction = 'male' if male_prob > female_prob else 'female'
        prediction.append(tweet_prediction)

        # Determine if prediction is correct
        correct.append(tweet_prediction == true_gender)

    return sum(correct) / len(correct)

# Classify a user's tweet
def classify(tweet):
    # Clean user tweet
    edited_text = normalize_text(tweet)
    words_in_tweet = set(edited_text.split())
    edited_text = list(filter(lambda word: word not in stop_words, words_in_tweet))

    # Initialize probabilities as p(males) and p(females)
    male_prob = p_males
    female_prob = p_females

    # Compute male and female probabilities
    for word in edited_text:
        # Error if word not in male_counts so we give it a value of 1
        try:
            male_count = male_counts[word] + 1
        except KeyError:
            male_count = 1
        male_prob *= male_count / male_denominator
        # Error if word not in female_counts so we give it a value of 1
        try:
            female_count = female_counts[word] + 1
        except KeyError:
            female_count = 1
        female_prob *= female_count / female_denominator

    # Return prediction for tweet based on max probability
    tweet_prediction = 'Male' if male_prob > female_prob else 'Female'
    return tweet_prediction

# Sklearn Multinomial Naive Bayes Implementation
# Returns accuracy score when using sklearn's Multinomial Naive Bayes' classifier on validation data
# Code adapted from https://github.com/rasto2211/Twitter-User-Gender-Classification/blob/master/notebooks/exploration.ipynb
def sklearn_MNB():
    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(train_data["edited_text"])
    x_train = vectorizer.transform(train_data["edited_text"])  # training data x values
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data["gender"])  # training data y values
    x_val = vectorizer.transform(validation_data["edited_text"])
    y_val = encoder.transform(validation_data["gender"])

    mnb = MultinomialNB()  # create classifier object
    mnb = mnb.fit(x_train, y_train)  # fit data to classifier
    return accuracy_score(y_val, mnb.predict(x_val))


# Code adapted from https://github.com/rasto2211/Twitter-User-Gender-Classification/blob/master/notebooks/exploration.ipynb
# Used https://towardsdatascience.com/multi-class-text-classification-with-sklearn-and-nltk-in-python-a-software-engineering-use-case-779d4a28ba5
def sklearn_MNB_predict(tweet):
    # Clean user tweet, make into pandas series for CountVectorizer transform
    edited_text = normalize_text(tweet)
    words_in_tweet = set(edited_text.split())
    edited_text = pd.Series([" ".join(list(filter(lambda word: word not in stop_words, words_in_tweet)))])

    vectorizer = CountVectorizer()
    vectorizer = vectorizer.fit(train_data["edited_text"])
    x_train = vectorizer.transform(train_data["edited_text"])  # training data x values
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(train_data["gender"])  # training data y values

    edited_text = vectorizer.transform(edited_text)

    mnb = MultinomialNB()  # create classifier object
    mnb.fit(x_train, y_train)  # fit data to classifier

    prediction = 'Male' if [1] == mnb.predict(edited_text) else 'Female'
    return prediction
