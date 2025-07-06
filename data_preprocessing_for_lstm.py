import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Load the sentiment-labeled dataset
data = pd.read_csv('data/sentiment_labeled_tweets.csv')

# Check for and handle missing or NaN values in the cleaned_text column
data['cleaned_text'] = data['cleaned_text'].fillna('')  # Replace NaN with an empty string

# Map sentiment labels to numerical values
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
data['sentiment_label'] = data['sentiment'].map(sentiment_mapping)

# Tokenize the text data for LSTM
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['cleaned_text'])

# Convert text to sequences
X = tokenizer.texts_to_sequences(data['cleaned_text'])

# Pad sequences to ensure uniform input length
X = pad_sequences(X, maxlen=100)  # Adjust 'maxlen' as needed based on average tweet length

# Define the target variable
y = data['sentiment_label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Save processed data and tokenizer for later use
import joblib
joblib.dump((X_train, X_test, y_train, y_test), 'data/processed_data.pkl')
joblib.dump(tokenizer, 'data/tokenizer.pkl')

print("Data preprocessing completed and saved.")
