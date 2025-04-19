import shap
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



def explain_the_model(model, x_test):
    logger.info("loading the model for the model explainability")
    try:
        # Transform x_test to the same scale and structure as the training data
        # For RandomForestClassifier, no additional transformation is usually needed
        # but ensure x_test is a DataFrame with the same features in the same order
        shap_values = shap.TreeExplainer(model).shap_values(x_test)

        logger.info("creating SHAP summary plot")
        shap.summary_plot(shap_values, x_test)  # Use the correct shap_values
        return shap_values
    except Exception as e:
        logger.error(f"error occurred while explaining model: {e}")



def force_plot(explainer, shap_values, x_test, index=0):
    logger.info("visualizing the contribution of features for a single prediction")
    try:
        # Ensure shap_values and x_test have the correct format
        # For binary classification, shap_values[1] will provide the SHAP values for class 1
        shap.initjs()  # Initialize the JS visualizations
        shap.force_plot(
            explainer.expected_value[1],  # Expected value for class 1
            shap_values[1][index, :],      # SHAP values for class 1 for the specified index
            x_test.iloc[index, :],         # Use iloc to access the corresponding row in x_test
            matplotlib=True                 # Set to True to use Matplotlib for rendering
        )
    except Exception as e:
        logger.error(f"error occurred while creating force plot: {e}")

