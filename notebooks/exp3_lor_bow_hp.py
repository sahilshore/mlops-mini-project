# exp3_lor_bow_hp.py
# hyperparameter tuning

# Import necessary libraries
import mlflow
import mlflow.sklearn
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import os
import dagshub

mlflow.set_tracking_uri('https://dagshub.com/sahilshore/mlops-mini-project.mlflow')
dagshub.init(repo_owner='sahilshore', repo_name='mlops-mini-project', mlflow=True)

# Load the data
# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv').drop(columns=['tweet_id'])


# Define text preprocessing functions
def lemmatization(text):
    """Lemmatize the text."""
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text."""
    stop_words = set(stopwords.words("english"))
    text = [word for word in str(text).split() if word not in stop_words]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    text = text.split()
    text = [word.lower() for word in text]
    return " ".join(text)

def removing_punctuations(text):
    """Remove punctuations from the text."""
    text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
    text = text.replace('؛', "")
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(df):
    """Normalize the text data."""
    try:
        df['content'] = df['content'].apply(lower_case)
        df['content'] = df['content'].apply(remove_stop_words)
        df['content'] = df['content'].apply(removing_numbers)
        df['content'] = df['content'].apply(removing_punctuations)
        df['content'] = df['content'].apply(removing_urls)
        df['content'] = df['content'].apply(lemmatization)
        return df
    except Exception as e:
        print(f'Error during text normalization: {e}')
        raise


x = df['sentiment'].isin(['happiness','sadness'])
df = df[x]

# Normalize the text data
df = normalize_text(df)

df['sentiment'] = df['sentiment'].replace({'sadness':0, 'happiness':1}).astype(int)
from sklearn.pipeline import Pipeline
from mlflow.models import infer_signature

# 1. FIXED: Split the RAW text data first to prevent data leakage
X_train, X_test, y_train, y_test = train_test_split(df['content'], df['sentiment'], test_size=0.2, random_state=42)

# Set the experiment name
mlflow.set_experiment("LoR Hyperparameter Tuning")

# 2. FIXED: Create a pipeline so the vectorizer is bundled with the model
base_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

# 3. FIXED: Prefix hyperparameter grid keys with 'classifier__' so GridSearchCV knows which step to tune
param_grid = {
    'classifier__C': [0.1, 1, 10],
    'classifier__penalty': ['l1', 'l2'],
    'classifier__solver': ['liblinear']
}

# Start the parent run for hyperparameter tuning
with mlflow.start_run(run_name="LoR Grid Search"):

    # Perform grid search on the PIPELINE
    grid_search = GridSearchCV(base_pipeline, param_grid, cv=5, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Log each parameter combination as a child run
    for params, mean_score, std_score in zip(grid_search.cv_results_['params'], grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['std_test_score']):
        with mlflow.start_run(run_name=f"LR_child", nested=True):
            
            # 4. FIXED: Apply the parameters to a fresh pipeline for evaluation
            model = Pipeline([
                ('vectorizer', CountVectorizer()),
                ('classifier', LogisticRegression())
            ])
            model.set_params(**params) # Applies the parameters dynamically
            model.fit(X_train, y_train)
            
            # Model evaluation
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Log parameters and metrics (clean up param names for MLflow UI)
            clean_params = {k.replace('classifier__', ''): v for k, v in params.items()}
            mlflow.log_params(clean_params)
            mlflow.log_metric("mean_cv_score", mean_score)
            mlflow.log_metric("std_cv_score", std_score)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Print the results for verification
            print(f"Mean CV Score: {mean_score:.4f}, Std CV Score: {std_score:.4f}")
            print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")

    # Log the best run details in the parent run
    best_params = {k.replace('classifier__', ''): v for k, v in grid_search.best_params_.items()}
    best_score = grid_search.best_score_
    
    mlflow.log_params(best_params)
    mlflow.log_metric("best_f1_score", best_score)
    
    print(f"\nBest Params: {best_params}")
    print(f"Best F1 Score: {best_score}")

    # Save and log the notebook
    mlflow.log_artifact(__file__)

    # 5. FIXED: Infer signature and log the full Pipeline model
    signature = infer_signature(X_train.head().to_frame(), grid_search.best_estimator_.predict(X_train.head()))
    
    mlflow.sklearn.log_model(
        sk_model=grid_search.best_estimator_, 
        name="model", 
        signature=signature
    )