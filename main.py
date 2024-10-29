import sys
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import necessary functions
from model_management import save_model, load_model
from hyperparameter_tuning import tune_hyperparameters

# File paths
data_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\data\raw\sentiment140.csv"
model_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\sentiment_model.joblib"
vectorizer_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\vectorizer.joblib"

# Step 1: Load and inspect the dataset columns
df = pd.read_csv(data_filepath, encoding='latin-1')
print("Column names in dataset:", df.columns)  # Print column names to inspect

# Rename or select the appropriate columns based on the actual names
# Assuming sentiment column is the first column and text is the last
df = df.iloc[:, [0, 5]]  # Adjust if needed (0 = target, 5 = text based on sentiment140 layout)
df.columns = ['target', 'text']  # Rename columns for easier reference

# Drop any rows with missing values in these columns
df = df.dropna()

# Define feature and target variables
X = df['text']
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Initialize the vectorizer and transform the training data
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)

# Step 3: Train or tune the model
model = LogisticRegression(max_iter=200)
model = tune_hyperparameters(X_train_vectorized, y_train)

# Step 4: Save the model and vectorizer after training
save_model(model, vectorizer, model_filepath, vectorizer_filepath)

print("Model and vectorizer saved successfully.")