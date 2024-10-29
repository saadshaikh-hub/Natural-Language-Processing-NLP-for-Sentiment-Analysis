# src/data_processing.py

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import os

# Ensure nltk stopwords and tokenizer are downloaded
import nltk
nltk.download('stopwords')
nltk.download('punkt')

# Load dataset
def load_data(filepath):
    """
    Loads the Sentiment140 dataset from the specified file path.
    """
    return pd.read_csv(filepath, encoding='ISO-8859-1', header=None, 
                       names=['target', 'id', 'date', 'flag', 'user', 'text'])

# Text cleaning function
def preprocess_text(text):
    """
    Cleans the input text by removing URLs, special characters, and stopwords.
    """
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove @mentions and hashtags
    text = re.sub(r'\@\w+|\#', '', text)
    # Remove special characters and numbers
    text = re.sub(r'[^A-Za-z ]+', '', text)
    # Tokenize text and remove stopwords
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Stem tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Apply preprocessing to the dataset
def preprocess_data(df):
    """
    Processes the dataset by applying text cleaning to each tweet.
    """
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    # Convert sentiment label: 0 for negative, 1 for positive
    df['target'] = df['target'].apply(lambda x: 0 if x == 0 else 1)
    return df[['target', 'cleaned_text']]

# Save processed data
def save_processed_data(df, filepath):
    """
    Saves the processed dataset to the specified file path.
    """
    df.to_csv(filepath, index=False)

# Main function
if __name__ == "__main__":
    # Load raw data
    raw_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\data\raw\sentiment140.csv"
    processed_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\data\processed/processed_data.csv"
    
    if not os.path.exists(raw_filepath):
        raise FileNotFoundError("Raw dataset not found in data/raw. Please add sentiment140.csv.")

    # Load, process, and save data
    data = load_data(raw_filepath)
    processed_data = preprocess_data(data)
    save_processed_data(processed_data, processed_filepath)
    print("Data preprocessing complete. Processed data saved to data/processed/processed_data.csv")
