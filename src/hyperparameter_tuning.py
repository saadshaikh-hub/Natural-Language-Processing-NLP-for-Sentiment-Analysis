from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
import joblib

def tune_hyperparameters(X_train_vectorized, y_train):
    param_grid = {
        'C': [0.1, 1, 10],
        'max_iter': [100, 200, 300]
    }
    grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
    grid_search.fit(X_train_vectorized, y_train)
    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

if __name__ == "__main__":
    # Placeholder for data loading or tuning execution
    # Example:
    # X_train_vectorized, y_train = ...  # Load your training data here
    
    # Call the tuning function with actual data
    # best_model = tune_hyperparameters(X_train_vectorized, y_train)
    
    print("Hyperparameter tuning script executed.")
