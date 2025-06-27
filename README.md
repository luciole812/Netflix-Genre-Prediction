# Netflix Genre Classification

## Project Overview

This project predicts the **genre classification** of Netflix titles using supervised machine learning. It applies the CRISP-DM framework and uses the Netflix public dataset from Kaggle.

## Objective

The goal is to help streaming platforms automate content classification for regulatory compliance and recommendation engines.

## Dataset

- **Name:** Netflix Movies and TV Shows
- **Source:** [Kaggle - Netflix Titles Dataset](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **Size:** ~8,800 rows, 12 columns

The dataset contains metadata for Netflix shows including type, duration, genres, release year, country, etc.

## Models Used
Three supervised classification models were tested:
 - Logistic Regression
 - Random Forest
 - XGBoost (Gradient Boosting)
 Evaluation is based on accuracy and macro F1-score, to account for class imbalance.

## Requirements
Make sure you have Python 3.8+ and the following libraries installed: pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib

## How to Run
1. Set up the environment
- Create and activate a virtual environment:
  virtualenv venv
  source venv/bin/activate
- Install dependencies:
  pip install -r requirements.txt
2. Launch Jupyter Notebook (for interactive development)
jupyter notebook \
    --notebook-dir="." \
    --ip=0.0.0.0 \
    --port=3225
3. Debug a Python script (e.g., test.py)
- Run the script with debug mode enabled:
  python -m debugpy --listen 4444 test.py
4. Run the FastAPI App (development mode)
fastapi dev main.py

Enable Push to GitHub
Make sure your Git configuration is set up properly:
nano ~/.gitconfig
- or open with VS Code:
code ~/.gitconfig
Add or verify your GitHub username, email, and any SSH settings needed.

5. Preprocessing (1_preprocessing.ipynb)
This step:
- Cleans text (lowercase, removes digits/punctuation/stopwords),
- Extracts the main genre from the genre list,
- Removes rare genres (fewer than 5 occurrences),
- Applies TF-IDF vectorization to the description,
- One-hot encodes categorical variables (e.g., type),
- Splits the data (80% train / 20% test),
- Saves matrices and encoders (TfidfVectorizer, LabelEncoder).
- Place the files netflix_titles.csv and stopwords_en.txt in the data/ folder.
6. Classification Model (2_classification_model.ipynb)
- Loads X_train and y_train,
- Trains the three models,
- Saves them in the models/ folder.
- Loads X_test, y_test, the label encoder and all trained models,
- Computes accuracy and macro F1-score,
- Displays the classification report and confusion matrix for each model.

## Project Structure

```bash
netflix-age-rating-prediction/
│
├── data/
│ └── netflix_titles.csv
├── notebooks/
│ ├── 1_data_preprocessing.ipynb
│ └── 2_classification_model.ipynb
├── report.pdf
├── requirements.txt 
└── README.md
