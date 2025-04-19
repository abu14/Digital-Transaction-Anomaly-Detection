# 💳 Transaction Anomaly Detection

![GitHub license](https://img.shields.io/github/license/abu14/Digital-Transaction-Anomaly-Detection)
![GitHub issues](https://img.shields.io/github/issues/abu14/Digital-Transaction-Anomaly-Detection)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/abenezer-tesfaye-191579214/)

ML pipeline for detecting fraudulent transactions using PCA-processed credit card data. Features full EDA, feature engineering, model interpretation (LIME), and Flask/Dash deployment.

## 🚀 Features
- **Automated EDA** with logging (histograms, boxplots, correlation matrices)
- **Feature Engineering**: Transaction velocity/frequency analysis
- **Model Interpretation**: LIME explainability for predictions
- **Production-Ready**: Flask API + Dash dashboard for real-time predictions
- **Data Security**: PCI-compliant data preprocessing (V1-V28 PCA features)

## 📦 Requirements
```bash
  pandas==2.0.3
  scikit-learn==1.3.0
  matplotlib==3.7.2
  seaborn==0.12.2
  lime==0.2.0.1
  flask==2.3.2
  dash==2.14.0
  ```

### ⚙️ **Usage**
Follow these steps to set up and run the project:

1. Clone the repository:
    ```bash
      git clone https://github.com/abu14/Digital-Transaction-Anomaly-Detection.git
      cd Digital-Transaction-Anomaly-Detection
    ```

2. Install dependencies:
   ```bash
    pip install -r requirements.txt
   ```
3. Run Flask API:
   ```bash
    python app/flask_app.py
   ```
4. Launch Dash dashboard::
   ```bash
    python app/dash_app.py
   ```


  
### 🔧 **Code Snippets**
#### **Scraping Website Content**

> Data Analyss and error logging
  ```python
  def perform_full_eda(df):
      logger.info("Starting EDA")
      univariate_analysis_numeric(df)  # Histograms/boxplots
      plot_datetime_aggregations(df)   # Time pattern analysis
  ```


#### **Flask Prediction API**
  ```python
  @app.route('/predict', methods=['POST'])
  def predict():
      """Real-time prediction endpoint"""
      try:
          data = request.get_json()
          df = pd.DataFrame([data])
          
          # Feature engineering pipeline
          df = create_fraud_features(df)
          features = preprocessor.transform(df)
          
          prediction = model.predict(features)
          explanation = generate_explanation(model, df.iloc[0])
          
          return jsonify({
              'fraud_probability': float(prediction[0]),
              'explanation': explanation
          })
      except Exception as e:
          return jsonify({'error': str(e)}), 500
  ```


#### **LIME Model Explanations**
  ```python
  def generate_explanation(model, sample):
      """Generate LIME explanation for predictions"""
      explainer = lime_tabular.LimeTabularExplainer(
          training_data=X_train.values,
          feature_names=feature_names,
          class_names=['Legit', 'Fraud'],
          mode='classification'
      )
      
      exp = explainer.explain_instance(
          sample.values,
          model.predict_proba,
          num_features=10
      )
      
      return exp.as_list()
  ```


### 📄 **License**
This project is licensed under the MIT License.  See [LICENSE](./LICENSE) file for more details.

<!-- CONTACT -->
### 💬 **Contact**

📧 tesfayeabenezer64@gmail.com

🔗 Project Repository [Link](https://github.com/abu14/)
