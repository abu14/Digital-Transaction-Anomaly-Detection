import lime
from lime import lime_tabular
import os , logging

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

def explain(model,X_test,X_train):
    logger.info("explaionng the model")
    # Initialize LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train.values,                      # Use training data to build LIME explainer
        feature_names=X_train.columns,       # Column names for easy interpretation
        class_names=['Non-Fraud', 'Fraud'],  # Class names for output
        mode='classification'
    )    
    # Choose an instance from X_test to explain, e.g., the first one
    i = 0  # index of the instance
    instance = X_test.values[i]  # extract feature values of the instance

    # Get explanation
    exp = explainer.explain_instance(
        data_row=instance,                    # Instance to explain
        predict_fn=model.predict_proba,  # Model's prediction function
        num_features=10                       # Number of features to display in the explanation
    )

    # Display the explanation
    exp.show_in_notebook(show_table=True)
    