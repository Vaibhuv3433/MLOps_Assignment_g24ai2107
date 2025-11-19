from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

def main():
    print("MLOps Assignment - Model Testing")
    print("Roll Number: g24ai2107")
    
    if not os.path.exists('savedmodel.pth'):
        print("Model file not found! Run train.py first.")
        return
    
    print("1. Loading saved model...")
    model = joblib.load('savedmodel.pth')
    
    print("2. Loading dataset...")
    data = fetch_olivetti_faces()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    print("3. Making predictions...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"4. Test Accuracy: {accuracy:.4f}")
    
    print("Testing completed successfully!")

if __name__ == "__main__":
    main()
