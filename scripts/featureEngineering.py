import os , logging 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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



def feature_engineering(df):
    try:
        # Convert 'signup_time' and 'purchase_time' to datetime format
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        # Extract hour of the day and day of the week from 'purchase_time'
        df['hour_of_day'] = df['purchase_time'].dt.hour
        df['day_of_week'] = df['purchase_time'].dt.dayofweek

        # Sort values by user_id and purchase_time to calculate transaction frequency and velocity
        df = df.sort_values(by=['user_id', 'purchase_time'])

        # Transaction frequency: Count the number of transactions for each user
        df['transaction_count'] = df.groupby('user_id')['user_id'].transform('count')

        # Transaction velocity: Time difference between consecutive purchases for each user (in seconds)
        df['time_since_last_purchase'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds()

        # Fill NaN values (first transaction of each user) with 0
        df['time_since_last_purchase'] = df['time_since_last_purchase'].fillna(0)

        return df

    except Exception as e:
        print(f"An error occurred during feature engineering: {e}")
        return df
