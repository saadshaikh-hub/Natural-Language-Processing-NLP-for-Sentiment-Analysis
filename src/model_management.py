import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer

def save_model(model, vectorizer, model_filepath, vectorizer_filepath):
    joblib.dump(model, model_filepath)
    joblib.dump(vectorizer, vectorizer_filepath)
    print("Model and vectorizer saved.")

def load_model(model_filepath, vectorizer_filepath):
    model = joblib.load(model_filepath)
    vectorizer = joblib.load(vectorizer_filepath)
    return model, vectorizer

if __name__ == "__main__":
    # Define paths
    model_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\sentiment_model.joblib"
    vectorizer_filepath = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis\models\vectorizer.joblib"
    
    # Example to save the model
    # Replace `your_model` and `your_vectorizer` with your actual model and vectorizer variables
    # save_model(your_model, your_vectorizer, model_filepath, vectorizer_filepath)

    # Example to load the model
    # model, vectorizer = load_model(model_filepath, vectorizer_filepath)
