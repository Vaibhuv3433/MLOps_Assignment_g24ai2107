from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

def main():
    print("MLOps Assignment - Model Training")
    print("Roll Number: g24ai2107")
    
    print("1. Loading Olivetti faces dataset...")
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    
    print("2. Splitting data (70% train, 30% test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("3. Training DecisionTreeClassifier...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    print("4. Saving model...")
    joblib.dump(model, 'savedmodel.pth')
    
    print("Training completed successfully!")

if __name__ == "__main__":
    main()
