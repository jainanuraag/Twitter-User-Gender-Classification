
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import re

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

df = pd.read_csv("gender-classifier-DFE-791531.csv", encoding='latin1')


# Utility function taken from: https://github.com/rasto2211/Twitter-User-Gender-Classification/blob/master/notebooks/exploration.ipynb
# Normalizes text for analysis by removing URLs, special characters, and double spaces

def normalize_text(text):
    # Remove non-ASCII chars.
    text = re.sub('[^\x00-\x7F]+',' ', text)
    
    # Remove URLs
    text = re.sub('https?:\/\/.*[\r\n]*', ' ', text)
    
    # Remove special chars.
    text = re.sub('[?!+%{}:;.,"\'()\[\]_]', '',text)
    
    # Remove double spaces.
    text = re.sub('\s+',' ',text)
    return text

df['edited_text'] = [normalize_text(text) for text in df['text']]
#print(df['edited_text'])


# Choose data only where gender is either male or female and gender classification confidence is about 1
chosen_rows = df[df["gender"].isin(["male", "female"]) & (df["gender:confidence"] > 0.99)].index.tolist()

# Shuffle data to ensure randomness
random.shuffle(chosen_rows)

# Data Guidelines according to Canvas
n = len(chosen_rows)
train_data_size = .6
test_data_size = .2
validation_data_size = .2

# Partition chosen_rows
train_data_nrows = round(train_data_size * n)
train_data = chosen_rows[:train_data_nrows]
validation_data_upper_limit = (train_data_nrows + round(validation_data_size * n))
validation_data = chosen_rows[train_data_nrows : validation_data_upper_limit]
test_data = chosen_rows[validation_data_upper_limit:]

# Our own MNB implementation
train_data_1 = df.ix[train_data, :]["edited_text"]
validation_data_1 = df.ix[validation_data, :]["edited_text"]
vectorizer_1 = CountVectorizer().fit_transform(train_data_1)

# Making classifier
vectorizer = CountVectorizer()
#train_counts = vectorizer.fit_transform(df.ix[train_data, :]["edited_text"])
vectorizer = vectorizer.fit(df.ix[train_data, :]["edited_text"])
x_train = vectorizer.transform(df.ix[train_data, "edited_text"])
encoder = LabelEncoder()
y_train = encoder.fit_transform(df.loc[train_data, "gender"])
#print(x_train)
#print(y_train)



nb = MultinomialNB()
nb = nb.fit(x_train, y_train)
x_val = vectorizer.transform(df.ix[validation_data, "edited_text"])
y_val = encoder.transform(df.ix[validation_data, "gender"])
print(classification_report(y_val, nb.predict(x_val), target_names=encoder.classes_))
print(f"accuracy score: {accuracy_score(y_val, nb.predict(x_val))}")

