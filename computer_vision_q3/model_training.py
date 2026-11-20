# File: model_training.py
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from feature_extraction import load_dataset

def train_and_evaluate():
    print(" Loading dataset and extracting features...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset('organized_dataset')
    
    # Scale features (Crucial for SVM, LR, and NN)
    print("\n Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Define the 4 classifiers required by the project prompt
    classifiers = {
        "Support Vector Machine": SVC(kernel='rbf', random_state=42),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
    }
    
    best_acc = 0.0
    best_model_name = ""
    best_model = None
    
    print("\n Training and Evaluating Classifiers...\n" + "="*60)
    for name, clf in classifiers.items():
        print(f"\n Training: {name}")
        clf.fit(X_train_scaled, y_train)
        
        # Evaluate on Test Set
        y_pred = clf.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"   Test Accuracy: {acc:.4f}")
        print("    Classification Report:")
        # Print a clean classification report for your final report write-up
        print(classification_report(y_test, y_pred, target_names=['defect_free', 'hole', 'mark']))
        
        # Track the best model
        if acc > best_acc:
            best_acc = acc
            best_model_name = name
            best_model = clf
            
    print("="*60)
    print(f" Best Model: {best_model_name} with Test Accuracy: {best_acc:.4f}")
    
    # Save the best model and scaler for the localization step
    joblib.dump(best_model, 'best_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("\n Saved 'best_model.pkl' and 'scaler.pkl' to disk for the localization step.")

if __name__ == "__main__":
    train_and_evaluate()