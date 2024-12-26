# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import VotingClassifier


# Load and preprocess data
def load_and_preprocess_data(file_path):
    # Load data
    data = pd.read_csv(file_path)

    # Fill in missing data (if any)
    if data.isnull().sum().sum() > 0:
        data = data.fillna(data.median())

    # Use a random sample size of dataset
    data = data.sample(frac=0.5)

    # Separate features and target variables
    features = data.drop(columns=["Class"])  # "Class" being the target column
    target = data["Class"]

    # Normalise numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    return features_scaled, target


# Dataset enhancement
def enhance_dataset(data):
    # Transaction amount log transformation
    if "Amount" in data.columns:
        data["Log_Amount"] = data["Amount"].apply(lambda x: 0 if x <= 0 else np.log(x))

    # Transaction time in hours (if 'Time' column exists)
    if "Time" in data.columns:
        data['Hour'] = data['Time'] % (24 * 3600) // 3600  # Convert seconds to hours

    return data


# Model training and evaluation
def train_and_evaluate_best_model(x, y):
    # Split the data into a training and testing subset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

    # Check class distribution
    class_distribution = pd.Series(y_train).value_counts(normalize=True)
    print("Class distribution in training set:")
    print(class_distribution)

    # Define models to try
    models = {
        'SVM': SVC(random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42)
    }

    # If the dataset is imbalanced, use class weights
    if min(class_distribution) < 0.2:  # Adjust this threshold as needed
        print("Dataset is imbalanced. Using class weights.")
        for model_name in models:
            if 'class_weight' in models[model_name].get_params():
                models[model_name].set_params(class_weight='balanced')

    # Define hyperparameters for each model
    param_grids = {
        'Random Forest': {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', 'balanced_subsample', None]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'class_weight': ['balanced', None]
        },
        'Logistic Regression': {
            'C': [0.1, 1, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear'],
            'class_weight': ['balanced', None]
        }
    }

    best_model = None
    best_score = 0
    results = {}

    # Stratified K-Fold for handling imbalanced dataset
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Perform GridSearch with cross-validation
        grid_search = GridSearchCV(model, param_grids[name], cv=cv, scoring='roc_auc', n_jobs=-1)
        grid_search.fit(x_train, y_train)

        # Get best model and its score
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        cv_score = grid_search.best_score_

        # Perform cross-validation on best model
        cv_scores = cross_val_score(best_estimator, x_train, y_train, cv=cv, scoring='roc_auc')

        results[name] = {
            'model': best_estimator,
            'best_params': best_params,
            'cv_score': cv_score,
            'cv_scores': cv_scores
        }

        print(f"{name} model evaluated")

        if cv_score > best_score:
            best_score = cv_score
            best_model = best_estimator

    print(f"model chosen: {best_model}")

    # Train the best model on full dataset
    best_model.fit(x_train, y_train)

    # Make predictions
    y_pred = best_model.predict(x_test)
    y_proba = best_model.predict_proba(x_test)[:, 1]

    # Evaluate the model
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return {'model': best_model, 'metrics': metrics}


def train_and_evaluate_ensemble_models(x, y):
    # Split the data into a training and testing subset
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)

    # Check class distribution
    class_distribution = pd.Series(y_train).value_counts(normalize=True)
    print("Class distribution in training set:")
    print(class_distribution)

    # Define models
    svm_model = SVC(probability=True, random_state=42, C=1, kernel='rbf', class_weight='balanced')
    logistic_model = LogisticRegression(random_state=42, C=1, solver='lbfgs', class_weight='balanced')
    random_forest_model = RandomForestClassifier(random_state=42, n_estimators=200, max_depth=None,
                                                 class_weight='balanced')

    # Create an ensemble model using VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('svm', svm_model),
            ('logistic', logistic_model),
            ('random_forest', random_forest_model)
        ],
        voting='soft'  # Use soft voting to consider probabilities
    )

    # Train the ensemble model
    ensemble_model.fit(x_train, y_train)

    # Make predictions
    y_pred = ensemble_model.predict(x_test)
    y_proba = ensemble_model.predict_proba(x_test)[:, 1]

    # Evaluate the ensemble model
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
    }

    return {'model': ensemble_model, 'metrics': metrics}


# Main function
def main():
    # File path to your dataset
    file_path = "write file path here"

    # Load and preprocess data
    print("Loading and preprocessing data...")
    x, y = load_and_preprocess_data(file_path)

    # Perform feature engineering
    print("Engineering features...")
    x = pd.DataFrame(x)  # Convert X back to DataFrame for feature engineering
    x = enhance_dataset(x)

    # Train and evaluate the ensemble model
    print("Training and evaluating model...")
    results = train_and_evaluate_ensemble_models(x, y)
    # Alternatively, using just one (the best) model
    # train_and_evaluate_best_model(x, y)

    # Extract results
    model = results['model']  # for future applications
    metrics = results['metrics']

    # Print evaluation metrics
    print("\nModel Evaluation Results:")
    print("Confusion Matrix:")
    print(metrics['confusion_matrix'])
    print("\nClassification Report:")
    print(metrics['classification_report'])
    print(f"\nROC AUC Score: {metrics['roc_auc_score']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
