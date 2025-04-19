#import libaries and normalize the data
import os, logging
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import datetime as dt

# Logging setup
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_info = os.path.join(log_dir, 'info.log')
log_file_error = os.path.join(log_dir, 'error.log')

info_handler = logging.FileHandler(log_file_info)
info_handler.setLevel(logging.INFO)
error_handler = logging.FileHandler(log_file_error)
error_handler.setLevel(logging.ERROR)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
info_handler.setFormatter(formatter)
error_handler.setFormatter(formatter)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(info_handler)
logger.addHandler(error_handler)

def cleaning_df(df):
    logger.info("Cleaning the data frame before normalizing and scaling")
    try:
        if 'Unnamed: 0' in df.columns:
            df = df.drop(columns=['Unnamed: 0'])
        return df
    except Exception as e:
        logger.error(f"Error occurred during cleaning: {e}")
        return None

def normalizing_and_encoding(df):
    logger.info("Normalizing and encoding")
    try:
        # Convert datetime columns and extract time-related features
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Extract time-related features
        df['signup_hour'] = df['signup_time'].dt.hour
        df['signup_dayofweek'] = df['signup_time'].dt.dayofweek
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek

        # Columns for transformation
        catagorical_columns = ['source', 'browser', 'sex']
        numerical_columns = ['purchase_value', 'age', 'transaction_count']

        # Preprocessor with ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_columns),
                ('cat', OneHotEncoder(max_categories=10, drop='first', sparse_output=False), catagorical_columns)
            ],
            remainder='passthrough'  # This keeps other columns intact
        )

        # Processed data as array
        processed_data = preprocessor.fit_transform(df)

        # Retrieve feature names from transformer and concatenate with passthrough columns
        num_feature_names = numerical_columns
        cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(catagorical_columns)
        
        passthrough_columns = [col for col in df.columns if col not in numerical_columns + catagorical_columns]
        all_feature_names = list(num_feature_names) + list(cat_feature_names) + passthrough_columns

        # Convert to DataFrame
        processed_df = pd.DataFrame(processed_data, columns=all_feature_names)
        
        # Ensure user_id and non-transformed columns are included
        processed_df = pd.concat([df[['user_id']], processed_df], axis=1)
        
        return processed_df
    except Exception as e:
        logger.error(f"Error occurred during normalization and encoding: {e}")
        return None

