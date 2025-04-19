from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model
model_path = '../../models/Random_Forest.pkl'  # Adjust the path as necessary
with open(model_path, 'rb') as file:
    model = joblib.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json()

        # Create a DataFrame from the input data
        df = pd.DataFrame(data)

        # Ensure all necessary columns are included (check against model input features)
        expected_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 
                            'V9', 'V10', 'V11', 'V12', 'V13', 'V14', 'V15', 
                            'V16', 'V17', 'V18', 'V19', 'V20', 'V21', 
                            'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 
                            'Amount']
        
        # Validate input
        if not all(col in df.columns for col in expected_columns):
            return jsonify({'error': 'Input data is missing some columns'}), 400

        # Make predictions
        predictions = model.predict(df)
        
        # If you want probabilities as well
        # probabilities = model.predict_proba(df)[:, 1]  # Probability of class 1

        # Return predictions as a JSON response
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)  # Set debug=False for production
