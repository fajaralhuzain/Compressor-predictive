import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

def train_model(input_path='data/compressor_data.csv', model_dir='models', plot_dir='plots'):
    """
    Trains a Random Forest model for predictive maintenance.
    
    Args:
        input_path (str): Path to the dataset CSV.
        model_dir (str): Directory to save trained model and scaler.
        plot_dir (str): Directory to save evaluation plots.
    """
    if not os.path.exists(input_path):
        print(f"Error: dataset not found at {input_path}")
        return

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    df = pd.read_csv(input_path)
    
    # Feature Selection
    features = ['Operating_Hours', 'Temperature_C', 'Vibration_mm_s', 'Pressure_PSI', 'Last_Service_Days']
    target = 'Failure_in_Next_30_Days'
    
    X = df[features]
    y = df[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Training (Random Forest)
    # Using class_weight to handle imbalance
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42
    )
    
    print("Training model (Random Forest)...")
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    # Evaluation
    print("\nModel Evaluation:")
    print("----------------")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(plot_dir, 'confusion_matrix.png'))
    plt.close()
    print(f"Saved confusion matrix to {plot_dir}/confusion_matrix.png")
    
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(8, 6))
    plt.title('Feature Importance')
    plt.bar(range(X.shape[1]), importances[indices], align='center')
    plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'feature_importance.png'))
    plt.close()
    print(f"Saved feature importance plot to {plot_dir}/feature_importance.png")
    
    # Save Model and Scaler
    joblib.dump(model, os.path.join(model_dir, 'random_forest_model.pkl'))
    joblib.dump(scaler, os.path.join(model_dir, 'scaler.pkl'))
    print(f"Saved model and scaler to {model_dir}/")

if __name__ == "__main__":
    train_model()
