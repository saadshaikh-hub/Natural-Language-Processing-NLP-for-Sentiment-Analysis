import os

# Define the root directory for the project
project_root = r"G:\My website\Machine Learning\Natural Language Processing (NLP) for Sentiment Analysis"

# Define the directory structure
directories = [
    "data/raw",
    "data/processed",
    "notebooks",
    "src",
    "models"
]

# Define file paths and contents
files = {
    "notebooks/sentiment_analysis_eda.ipynb": "# Jupyter notebook for exploratory data analysis",
    "src/data_processing.py": "# Script for loading and preprocessing the dataset\n\nif __name__ == '__main__':\n    pass",
    "src/model_training.py": "# Script for model training\n\nif __name__ == '__main__':\n    pass",
    "src/evaluate.py": "# Script for model evaluation\n\nif __name__ == '__main__':\n    pass",
    "src/utils.py": "# Utility functions (data loaders, preprocessing functions, etc.)",
    "main.py": "# Main script to run the full pipeline\n\nif __name__ == '__main__':\n    pass",
    "requirements.txt": "pandas\nnumpy\nscikit-learn\nnltk\njoblib\nmatplotlib\nseaborn\n",
}

# Create directories
for dir in directories:
    os.makedirs(os.path.join(project_root, dir), exist_ok=True)

# Create files with initial content
for file, content in files.items():
    file_path = os.path.join(project_root, file)
    with open(file_path, "w") as f:
        f.write(content)

print("Project structure created successfully!")
