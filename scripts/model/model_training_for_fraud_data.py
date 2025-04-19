import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Define the path to the Logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

# Create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define file paths
log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

# Create handlers
info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)

error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)

# Create a formatter and set it for the handlers
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)

# Create a logger and set its level
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)


def feature_and_target_separation(df):
    logger.info("feature and target")
    try:
        logger.info("fraud data target separation")

        x_fraud_data = df.drop(columns=['class'])
        y_fraud_data = df['class']

        return x_fraud_data,y_fraud_data
    except Exception as e:
        logger.info(f"error occured : {e}")
def train_test_splite(x_fraud_data,y_fraud_data):
    logger.info("spliting the data to evaluate the model's performance on unseen data")
    try:
        logger.info("spliting the fraud data into train and test")
        X_train_fraud , X_test_fraud , y_train_fraud,y_test_fraud = train_test_split(x_fraud_data,y_fraud_data, test_size=0.2,random_state=42)
        return X_train_fraud,X_test_fraud,y_train_fraud,y_test_fraud
    except Exception as e:
        logger.error(f"error occured : {e}")

def model_Traingin_and_Evaluation(X_train_data, X_test_data, y_train_data, y_test_data):
    logger.info("Training the model and evaluating their performance")
    
    try:
        # Preprocessing
        logger.info("Defining the preprocessing steps")
        
        # Identifying categorical and numerical features
        categorical_features = ['source_Direct', 'source_SEO', 'browser_FireFox', 'browser_IE', 
                                'browser_Opera', 'browser_Safari', 'signup_dayofweek', 
                                'purchase_dayofweek']
        numerical_features = ['purchase_value', 'age', 'transaction_count', 'hour_of_day', 
                             'day_of_week', 'time_since_last_purchase']

        # Ensure that the categorical features are of string type
        for col in categorical_features:
            X_train_data[col] = X_train_data[col].astype(str)
            X_test_data[col] = X_test_data[col].astype(str)

        # Creating preprocessing pipelines for numerical and categorical data
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
            ('scaler', StandardScaler())                   # Scale numerical features
        ])

        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
            ('onehot', OneHotEncoder(handle_unknown='ignore'))                      # One-hot encode categorical variables
        ])

        # Combine preprocessing steps
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        logger.info("Defining the models")
        models = {
            'Logistic Regression': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(max_iter=1000))]),
            'Decision Tree': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', DecisionTreeClassifier())]),
            'Random Forest': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier())]),
            'Gradient Boosting': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', GradientBoostingClassifier())]),
            'Multi-Layer Perception': Pipeline(steps=[('preprocessor', preprocessor), ('classifier', MLPClassifier(max_iter=1000))])
        }

        # Initializing dictionary to store evaluation metrics for each model
        results = {}
        # Initializing the list to store the trained models
        trained_models = []

        # Function to evaluate a model
        def evaluate_model(model, X_train, X_test, y_train, y_test):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            }

        logger.info("Training and evaluating each model")
        for name, model in models.items():
            logger.info(f"Training {name}")
            results[name] = evaluate_model(model, X_train_data, X_test_data, y_train_data, y_test_data)
            trained_models.append(model)

        # Creating a DataFrame for the results
        results_df = pd.DataFrame(results).T
        print(results_df)
        return trained_models

    except Exception as e:
        logger.error(f"Error occurred: {e}")
        return None

