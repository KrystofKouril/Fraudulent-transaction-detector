# Import necessary libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, make_scorer, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif

# Load and preprocess data
def load_and_preprocess_data(file_path, save_scaler_path=None):
    data = pd.read_csv(file_path)

    numeric_columns = data.select_dtypes(include=[np.number]).columns
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())

    features = enhance_dataset(data.drop(columns=["Class"]))
    target = data["Class"]

    assert features.isnull().sum().sum() == 0

    # Store feature names before scaling
    feature_names = features.columns

    scaler = StandardScaler()
    scaler.fit(features)

    # Save feature names in the scaler object
    scaler.feature_names_in_ = feature_names

    features_scaled = scaler.transform(features)

    # Apply enhanced feature selection
    features_selected, feature_mask = enhanced_feature_selection(features_scaled, target)

    if save_scaler_path:
        joblib.dump(scaler, save_scaler_path)
        print(f"Scaler saved to {save_scaler_path}")

    # Save feature mask for later use
    joblib.dump(feature_mask, 'feature_mask.pkl')
    print("Feature mask saved to feature_mask.pkl")

    return features_selected, target

# Dataset enhancement
def enhance_dataset(data):
    if "Amount" in data.columns:
        data["Log_Amount"] = data["Amount"].apply(lambda x: 0 if x <= 0 else np.log(x))
        data["Amount_to_mean_ratio"] = data["Amount"] / data["Amount"].mean()

    if "Time" in data.columns:
        data['Hour'] = data['Time'] % (24 * 3600) // 3600  # Convert seconds to hours
        data["Time_since_last_txn"] = data["Time"].diff().fillna(0)
        data["Time_since_last_txn"] = data["Time_since_last_txn"].clip(lower=0)
        data["Time_rolling_mean"] = data["Time"].rolling(window=5, min_periods=1).mean()

    return data

# Save the trained model
def save_model(model, file_path):
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")

# Load a trained model
def load_model(file_path):
    model = joblib.load(file_path)
    print(f"Model loaded from {file_path}")
    return model

# Load a saved scaler
def load_scaler(file_path):
    scaler = joblib.load(file_path)
    print(f"Scaler loaded from {file_path}")
    return scaler

def enhanced_feature_selection(x, y):
    # Initial feature selection using mutual information
    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(x, y)
    scores = selector.scores_

    # Get feature importance threshold
    threshold = np.percentile(scores, 70)  # Keep top 30% features

    # Select features above threshold
    selected_features = scores >= threshold

    return x[:, selected_features], selected_features

# Custom scoring function to balance precision and recall with emphasis on fraud detection
def custom_scorer(y_true, y_pred):
    precision = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall = recall_score(y_true, y_pred, pos_label=1, zero_division=0)

    # Only consider scores with acceptable false positive rates
    fp = confusion_matrix(y_true, y_pred)[0][1]
    total_negative = np.sum(y_true == 0)
    fpr = fp / total_negative

    if fpr > 0.0005:
        return 0

    beta = 0.5
    return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall) if (
                                                                                                          precision + recall) > 0 else 0

def train_and_evaluate_model(x, y, model_save_path=None):
    # First split the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    # Synthetic oversampling
    smote = SMOTE(
        sampling_strategy=0.05,
        random_state=42,
        k_neighbors=5,
        n_jobs=-1
    )

    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    class_weights = {
        0: 1.0,
        1: 2.0
    }

    model = HistGradientBoostingClassifier(
        max_iter=1500,
        learning_rate=0.001,
        max_depth=5,
        min_samples_leaf=50,
        l2_regularization=3.5,
        class_weight=class_weights
    )

    param_grid = {
        'learning_rate': [0.001, 0.002, 0.003],
        'max_depth': [5, 6, 7],
        'min_samples_leaf': [35, 40, 45],
        'l2_regularization': [2.0, 2.5, 3.0]
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=cv,
        scoring=make_scorer(custom_scorer),
        n_jobs=-1,
        verbose=2
    )

    print("Training model...")
    grid_search.fit(x_train_resampled, y_train_resampled)

    best_model = grid_search.best_estimator_

    if model_save_path:
        save_model(best_model, model_save_path)

    # Threshold optimization
    y_proba = best_model.predict_proba(x_test)[:, 1]

    # Grid search for optimal threshold
    thresholds = np.linspace(0.2, 0.95, 300)
    best_threshold = 0.5
    best_score = 0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        fp = cm[0][1]
        total_negative = cm[0][0] + cm[0][1]
        fpr = fp / total_negative

        recall = recall_score(y_test, y_pred, pos_label=1)
        precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)

        # Only consider thresholds that maintain acceptable false positive rate
        if fpr <= 0.0003:
            score = (recall * 0.7 + precision * 0.3)
            if score > best_score:
                best_score = score
                best_threshold = threshold

    # Final predictions using optimized threshold
    y_pred = (y_proba >= best_threshold).astype(int)

    metrics = {
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, zero_division=0),
        'roc_auc_score': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'optimal_threshold': best_threshold
    }

    print("\nBest parameters found:", grid_search.best_params_)
    print(f"\nOptimal probability threshold: {best_threshold:.3f}")

    return {'model': best_model, 'metrics': metrics}

def evaluate_model_performance(true_labels, predictions, probabilities):
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Legitimate", "Fraudulent"],
                yticklabels=["Legitimate", "Fraudulent"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    precision, recall, thresholds = precision_recall_curve(true_labels, probabilities)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="blue", label=f"PR Curve (AUC = {pr_auc:.2f})")
    plt.plot([0, 1], [1, 0], color="gray", linestyle="--")  # Random classifier baseline
    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.show()

    # Fraud Detection Performance
    fraud_indices = true_labels == 1  # Assuming '1' indicates fraud
    fraud_detected = sum((predictions == 1) & fraud_indices)

    print(f"Total Fraudulent Transactions: {sum(fraud_indices)}")
    print(f"Correctly Identified Fraudulent Transactions: {fraud_detected}")

def predict_new_data(model_path, scaler_path, new_data_path):
    # Load and enhance new dataset
    data = pd.read_csv(new_data_path)
    data = enhance_dataset(data)

    # Load the scaler first
    scaler = load_scaler(scaler_path)

    # Get the feature names used during training
    original_feature_names = scaler.feature_names_in_

    # Ensure all required columns are present
    for col in original_feature_names:
        if col not in data.columns:
            print(f"Adding missing column: {col}")
            data[col] = 0

    # Select only the features that were used during training
    features = data[original_feature_names]

    # Scale the features
    features_scaled = scaler.transform(features)

    # Load feature mask and apply the same feature selection
    try:
        feature_mask = joblib.load('feature_mask.pkl')
        features_selected = features_scaled[:, feature_mask]
        print(f"Selected features shape: {features_selected.shape}")
    except FileNotFoundError:
        print("Feature mask not found, using all features")
        features_selected = features_scaled

    # Load model with threshold
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(features_selected)
    probabilities = model.predict_proba(features_selected)[:, 1]

    # If ground truth is available in the dataset, evaluate performance
    if "Class" in data.columns:
        true_labels = data["Class"]
        evaluate_model_performance(true_labels, predictions, probabilities)

    return data


# Main function
def main():
    mode = input("Enter mode (train/infer): ").strip().lower()

    if mode == "train":
        file_path = "train_data.csv"
        print("Loading and preprocessing training data...")
        x_train, y_train = load_and_preprocess_data(
            file_path=file_path,
            save_scaler_path="scaler.pkl"
        )

        print("Training and evaluating model...")
        results = train_and_evaluate_model(
            x_train,
            y_train,
            model_save_path="fraud_detection_model.pkl")

        metrics = results['metrics']
        print("\nModel Evaluation Results:")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print("\nClassification Report:")
        print(metrics['classification_report'])
        print(f"\nROC AUC Score: {metrics['roc_auc_score']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")

    elif mode == "infer":
        new_data_path = "test_data.csv" #input("Enter path to new dataset: ").strip()

        print("Making predictions on new dataset...")
        results_df = predict_new_data(
            model_path="fraud_detection_model.pkl",
            scaler_path="scaler.pkl",
            new_data_path=new_data_path,
        )

if __name__ == "__main__":
    main()
