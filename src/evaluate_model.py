import sys
import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split  # Import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import load_model from model_management
from model_management import load_model

# File paths for the saved model and vectorizer
model_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\sentiment_model.joblib"
vectorizer_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\vectorizer.joblib"

# Load the saved model and vectorizer
model, vectorizer = load_model(model_filepath, vectorizer_filepath)

# Load and preprocess the test data
data_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\data\raw\sentiment140.csv"
df = pd.read_csv(data_filepath, encoding='latin-1').iloc[:, [0, 5]]  # Adjust if needed
df.columns = ['target', 'text']
df = df.dropna()

# Split data for testing
_, X_test, _, y_test = train_test_split(df['text'], df['target'], test_size=0.2, random_state=42)

# Transform test data
X_test_vectorized = vectorizer.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_vectorized)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
