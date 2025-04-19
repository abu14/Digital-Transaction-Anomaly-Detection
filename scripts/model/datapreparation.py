import os , logging 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score ,recall_score , f1_score, roc_auc_score


logging.basicConfig(
    level=logging.DEBUG,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'  # Format of the log messages
)
# Create a logger object
logger = logging.getLogger(__name__)

# define the path to the Logs directory one level up
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..','logs')

# create the logs directory if it doesn't exist
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
# define file paths
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
logger.setLevel(logging.INFO)  # Capture all info and above
logger.addHandler(info_handler)
logger.addHandler(error_handler)


def feature_and_target_separation(df,df1):
    logger.info("feature and target")
    try:
        logger.info("creadit card feature separation")
        X_creditcard = df.drop(columns=['Class'])
        y_creaditcard = df['Class']
        logger.info("fraud data target separation")

        x_fraud_data = df1.drop(columns=['class'])
        y_fraud_data = df1['class']

        return X_creditcard,y_creaditcard,x_fraud_data,y_fraud_data
    except Exception as e:
        logger.info(f"error occured : {e}")
def train_test_splite(X_creditcard,y_creaditcard,x_fraud_data,y_fraud_data):
    logger.info("spliting the data to evaluate the model's performance on unseen data")
    try:
        logger.info("spliting the credit card data into train test ")
        X_train_credit,x_test_credit,y_train_credit,y_test_credit = train_test_split(X_creditcard,y_creaditcard , test_size = 0.2 , random_state = 42)
        logger.info("spliting the fraud data into train and test")
        X_train_fraud , X_test_fraud , y_train_fraud,y_test_fraud = train_test_split(x_fraud_data,y_fraud_data, test_size=0.2,random_state=42)
        return X_train_credit,x_test_credit,y_train_credit,y_test_credit,X_train_fraud,X_test_fraud,y_train_fraud,y_test_fraud
    except Exception as e:
        logger.error(f"error occured : {e}")
def model_Traingin_and_Evaluation(X_train_data , X_test_data , y_train_data,y_test_data):
    logger.info("training the model and evaluating there performance")
    try:
        logger.info("defining the models")
        models = {
            'Logistic Regression':LogisticRegression(max_iter=1000),
            'Decision Tree' : DecisionTreeClassifier(),
            'Random Forest':RandomForestClassifier(),
            'Gradient Bossiting':GradientBoostingClassifier(),
            'Multi-Layer perception':MLPClassifier(max_iter=1000)
        }
        # Initializing dictionary to store evaluation metrics for each model
        results ={}
        # intializinfg the list to store the trained models
        modell = []

        # Function to evaluate a model
        def evaluate_model(model,X_train,X_test,y_train,y_test):
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            return {
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1 Score': f1_score(y_test, y_pred),
                'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            }
        logger.info("Training and evaluating each model")
        for name , model in models.items():
            print(f"Training {name}")
            results[name]=evaluate_model(model,X_train_data , X_test_data , y_train_data,y_test_data)
            modell.append(model)
        results_df = pd.DataFrame(results).T
        print(results_df)
        return modell
    except Exception as e:
        logger.error(f"error occured : {e}")
        return None





