from flask import Flask, request, jsonify
import numpy as np
import joblib
import os
from sklearn.datasets import fetch_olivetti_faces

app = Flask(__name__)

model = None

def load_model():
    global model
    if os.path.exists('savedmodel.pth'):
        model = joblib.load('savedmodel.pth')
        print("Model loaded successfully!")
    else:
        print("Model file not found!")

load_model()

@app.route('/')
def home():
    return '''
    <html>
    <head>
        <title>MLOps Assignment - Face Classification</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; }
            .container { max-width: 800px; margin: 0 auto; }
            .demo-box { border: 2px solid #4CAF50; padding: 20px; text-align: center; margin: 20px 0; background: #f9fff9; }
            .result { background: #f0f0f0; padding: 15px; margin: 20px 0; border-radius: 5px; }
            .info { background: #e8f4fd; padding: 15px; border-radius: 5px; }
            .btn { background: #4CAF50; color: white; padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Olivetti Faces Classification</h1>
            <p><strong>Roll Number:</strong> g24ai2107</p>
            
            <div class="demo-box">
                <h3>Demo: Test the Model</h3>
                <p>Click the button below to test the model with a random face from the dataset</p>
                <form action="/demo_predict" method="post">
                    <button type="submit" class="btn">Test with Random Face</button>
                </form>
            </div>
            
            <div class="info">
                <h3>About the Model</h3>
                <ul>
                    <li><strong>Dataset:</strong> Olivetti Faces (400 images, 40 people)</li>
                    <li><strong>Model:</strong> Decision Tree Classifier</li>
                    <li><strong>Classes:</strong> 40 different individuals</li>
                    <li><strong>Input:</strong> 64x64 grayscale images (4096 features)</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    '''

@app.route('/demo_predict', methods=['POST'])
def demo_predict():
    try:
        data = fetch_olivetti_faces()
        X, y = data.data, data.target
        
        random_idx = np.random.randint(0, len(X))
        sample = X[random_idx].reshape(1, -1)
        true_label = y[random_idx]
        
        prediction = model.predict(sample)[0]
        confidence = np.max(model.predict_proba(sample))
        
        return f'''
        <div class="container">
            <h2>Demo Prediction Result</h2>
            <div class="result">
                <p><strong>Random Sample Selected:</strong> #{random_idx}</p>
                <p><strong>Predicted Person ID:</strong> {prediction}</p>
                <p><strong>Actual Person ID:</strong> {true_label}</p>
                <p><strong>Confidence:</strong> {confidence:.2%}</p>
                <p><strong>Prediction:</strong> {'✅ CORRECT' if prediction == true_label else '❌ INCORRECT'}</p>
            </div>
            <a href="/">← Back to Home</a>
            <br><br>
            <form action="/demo_predict" method="post">
                <button type="submit" class="btn">Test Another Random Face</button>
            </form>
        </div>
        '''
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
