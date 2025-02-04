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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, auc, precision_recall_curve
from sklearn.metrics import precision_score, recall_score, f1_score
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
    # Transaction amount log transformation
    if "Amount" in data.columns:
        data["Log_Amount"] = data["Amount"].apply(lambda x: 0 if x <= 0 else np.log(x))
        data["Amount_to_mean_ratio"] = data["Amount"] / data["Amount"].mean()

    # Transaction time in hours (if 'Time' column exists)
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
    threshold = np.percentile(scores, 60)

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

    if fpr > 0.001:
        return 0

    beta = 1
    return (1 + beta ** 2) * (precision * recall) / ((beta ** 2 * precision) + recall) if (
                                                                                                          precision + recall) > 0 else 0
# Higher recall, lower precision
def train_first_stage_model(x, y, model_save_path=None):
    model = HistGradientBoostingClassifier(
        max_iter=2000,
        learning_rate=0.005,
        max_depth=8,
        min_samples_leaf=30,
        l2_regularization=2.0,
        class_weight={0: 1.0, 1: 5}
    )
    model.fit(x, y)
    if model_save_path:
        save_model(model, model_save_path)
    return model

# Focus on precision to lower false positives
def train_second_stage_model(x, y, model_save_path=None):
    model = HistGradientBoostingClassifier(
        max_iter=2500,
        learning_rate=0.002,
        max_depth=6,
        min_samples_leaf=50,
        l2_regularization=3.5,
        class_weight={0: 1.5, 1: 1.0}
    )
    model.fit(x, y)
    if model_save_path:
        save_model(model, model_save_path)
    return model


def train_and_evaluate_model(x, y, model_save_path=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

    smote = SMOTE(sampling_strategy=0.2, random_state=42, k_neighbors=5, n_jobs=-1)
    x_train_resampled, y_train_resampled = smote.fit_resample(x_train, y_train)

    print("Training first stage model...")
    first_stage_model = train_first_stage_model(x_train_resampled, y_train_resampled,
                                                model_save_path="first_stage_model.pkl")

    first_stage_proba = first_stage_model.predict_proba(x_test)[:, 1]
    first_stage_predictions = (first_stage_proba >= 0.25).astype(int)  # Lower threshold for higher recall

    suspicious_indices = first_stage_predictions == 1
    x_suspicious = x_test[suspicious_indices]
    y_suspicious = y_test[suspicious_indices]

    if len(x_suspicious) > 0:
        print("Training second stage model...")
        second_stage_model = train_second_stage_model(x_suspicious, y_suspicious,
                                                      model_save_path="second_stage_model.pkl")
        second_stage_proba = second_stage_model.predict_proba(x_suspicious)[:, 1]
        second_stage_predictions = (second_stage_proba >= 0.6).astype(int)

        final_predictions = np.zeros_like(y_test)
        final_predictions[suspicious_indices] = second_stage_predictions
    else:
        final_predictions = first_stage_predictions

    # Calculate metrics
    metrics = {
        'confusion_matrix': confusion_matrix(y_test, final_predictions),
        'classification_report': classification_report(y_test, final_predictions, zero_division=0),
        'roc_auc_score': roc_auc_score(y_test, first_stage_proba),
        'precision': precision_score(y_test, final_predictions, average='weighted', zero_division=0),
        'recall': recall_score(y_test, final_predictions, average='weighted', zero_division=0),
        'f1': f1_score(y_test, final_predictions, average='weighted', zero_division=0)
    }

    return {'first_stage_model': first_stage_model, 'second_stage_model': second_stage_model, 'metrics': metrics}


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

    # Precision-Recall Curve
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


def predict_new_data(first_stage_model_path, second_stage_model_path, scaler_path, new_data_path):
    # Load models and scaler
    first_stage_model = load_model(first_stage_model_path)
    second_stage_model = load_model(second_stage_model_path)
    scaler = load_scaler(scaler_path)

    # Load and enhance new dataset
    data = pd.read_csv(new_data_path)
    data = enhance_dataset(data)

    # Ensure all required columns are present
    original_feature_names = scaler.feature_names_in_
    for col in original_feature_names:
        if col not in data.columns:
            print(f"Adding missing column: {col}")
            data[col] = 0

    # Select only the features that were used during training
    features = data[original_feature_names]

    # Scale the features
    features_scaled = scaler.transform(features)

    # Load and apply feature mask
    try:
        feature_mask = joblib.load('feature_mask.pkl')
        features_selected = features_scaled[:, feature_mask]
        print(f"Selected features shape: {features_selected.shape}")
    except FileNotFoundError:
        print("Feature mask not found, using all features")
        features_selected = features_scaled

    # First stage predictions
    first_stage_proba = first_stage_model.predict_proba(features_selected)[:, 1]
    first_stage_predictions = (first_stage_proba >= 0.3).astype(int)

    # Second stage for suspicious transactions
    suspicious_indices = first_stage_predictions == 1
    if np.any(suspicious_indices):
        second_stage_proba = second_stage_model.predict_proba(features_selected[suspicious_indices])[:, 1]
        second_stage_predictions = (second_stage_proba >= 0.7).astype(int)

        final_predictions = np.zeros_like(first_stage_predictions)
        final_predictions[suspicious_indices] = second_stage_predictions
    else:
        final_predictions = first_stage_predictions

    # If ground truth is available, evaluate performance
    if "Class" in data.columns:
        true_labels = data["Class"]
        evaluate_model_performance(true_labels, final_predictions, first_stage_proba)

    return final_predictions


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
            first_stage_model_path="first_stage_model.pkl",
            second_stage_model_path="second_stage_model.pkl",
            scaler_path="scaler.pkl",
            new_data_path=new_data_path,
        )

if __name__ == "__main__":
    main()
