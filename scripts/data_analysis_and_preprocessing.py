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


def load_data(path):
    logger.info("loading the data")
    try:
        return pd.read_csv(path)
    except Exception as e:
        logger.error(f"error occured while loading the data: {e}")
def data_set_overview(df):
    logger.info("data set overview")
    try:
        logger.info("description of the data set")
        print(df.describe())
        logger.info("data type of the data frame columns :")
        print(df.dtype)
    except Exception as e:
        logger.error(f"error occured while performing the overview of the data set: {e}")
def missing_values(df):
    logger.info("displying the missing values is there is any")
    try:
        print(df.isnull().sum())
    except Exception as e:
        logger.error(f"error occured while dispying the missing values: {e}")
def data_cleaning(df):
    logger.info("cleaning the data")
    try:
        logger.info("removing the duplicatged values")
        df.drop_duplicates()
        logger.info("correcting the data dypes")
        numeric_columns = df.select_dtypes(include=['int64','float64']).columns
        for n in numeric_columns:
            df[n] = pd.to_numeric(df[n],errors='coerce')
        return df
    except Exception as e:
        logger.error(f"error occured while cleaning the data {e}")
        return None
# Univariate Analysis for Numerical Columns
def univariate_analysis_numeric(df):
    logger.info("Performing univariate analysis on numerical columns")
    try:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        for column in numeric_columns:
            plt.figure(figsize=(12, 6))

            # Histogram
            logger.info(f"Plotting histogram for {column}")
            plt.subplot(1, 2, 1)
            sns.histplot(df[column], kde=True, bins=30)
            plt.title(f"Histogram of {column}")

            # Boxplot
            logger.info(f"Plotting boxplot for {column}")
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[column])
            plt.title(f"Boxplot of {column}")

            plt.tight_layout()
            plt.show()
    except Exception as e:
        logger.error(f"Error during univariate analysis for numerical columns: {e}")

def univariate_analysis_categorical(df):
    logger.info("Performing univariate analysis on categorical columns")
    
    try:
        # Convert potential datetime columns
        if 'signup_time' in df.columns:
            df['signup_time'] = pd.to_datetime(df['signup_time'])
        if 'purchase_time' in df.columns:
            df['purchase_time'] = pd.to_datetime(df['purchase_time'])

        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_columns:
            if pd.api.types.is_datetime64_any_dtype(df[column]):
                logger.info(f"Skipping datetime column: {column}")
                continue

            plt.figure(figsize=(8, 6))
            logger.info(f"Plotting countplot for {column}")
            sns.countplot(x=df[column])
            plt.title(f'Count Plot of {column}')
            plt.xticks(rotation=45)
            plt.show()

    except Exception as e:
        logger.error(f"Error during univariate analysis for categorical columns: {e}")

# Example feature extraction for datetime columns
def plot_datetime_aggregations(df):
    logger.info("Performing univariate analysis for datetime features")
    try:
        if 'signup_time' in df.columns:
            # Extract features from datetime columns
            df['signup_hour'] = df['signup_time'].dt.hour
            df['signup_dayofweek'] = df['signup_time'].dt.dayofweek

            # Plot distribution of hours and days of week
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.countplot(x=df['signup_hour'])
            plt.title("Signups by Hour of the Day")
            
            plt.subplot(1, 2, 2)
            sns.countplot(x=df['signup_dayofweek'])
            plt.title("Signups by Day of the Week")
            plt.show()

        if 'purchase_time' in df.columns:
            df['purchase_hour'] = df['purchase_time'].dt.hour
            df['purchase_dayofweek'] = df['purchase_time'].dt.dayofweek

            # Plot distribution for purchase hour and day of week
            plt.figure(figsize=(12, 5))
            plt.subplot(1, 2, 1)
            sns.countplot(x=df['purchase_hour'])
            plt.title("Purchases by Hour of the Day")

            plt.subplot(1, 2, 2)
            sns.countplot(x=df['purchase_dayofweek'])
            plt.title("Purchases by Day of the Week")
            plt.show()

    except Exception as e:
        logger.error(f"Error during univariate analysis for datetime features: {e}")

# Bivariate Analysis for Numerical vs. Numerical (Scatter plot and Correlation matrix)
def bivariate_analysis_numeric(df):
    logger.info("Performing bivariate analysis for numerical columns")
    try:
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        for i, col1 in enumerate(numeric_columns):
            for col2 in numeric_columns[i+1:]:
                plt.figure(figsize=(8, 6))
                
                # Scatter plot
                logger.info(f"Plotting scatterplot for {col1} vs {col2}")
                sns.scatterplot(x=df[col1], y=df[col2])
                plt.title(f'Scatter plot between {col1} and {col2}')
                plt.show()

        # Correlation matrix
        correlation_matrix = df[numeric_columns].corr()
        logger.info("Plotting correlation matrix")
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Matrix")
        plt.show()

    except Exception as e:
        logger.error(f"Error during bivariate analysis for numerical columns: {e}")

# Bivariate Analysis for Categorical vs. Numerical (Boxplot)
def bivariate_analysis_categorical_numeric(df):
    logger.info("Performing bivariate analysis for categorical and numerical columns")
    try:
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns

        for cat_col in categorical_columns:
            for num_col in numeric_columns:
                plt.figure(figsize=(8, 6))

                # Boxplot
                logger.info(f"Plotting boxplot for {num_col} by {cat_col}")
                sns.boxplot(x=df[cat_col], y=df[num_col])
                plt.title(f'Boxplot of {num_col} across {cat_col}')
                plt.xticks(rotation=45)
                plt.show()

    except Exception as e:
        logger.error(f"Error during bivariate analysis for categorical and numerical columns: {e}")

# Function to automate univariate and bivariate analysis
def perform_full_eda(df):
    logger.info("Starting full Exploratory Data Analysis (EDA)")

    # Univariate Analysis
    univariate_analysis_numeric(df)
    univariate_analysis_categorical(df)

    # Bivariate Analysis
    # bivariate_analysis_numeric(df)
    # bivariate_analysis_categorical_numeric(df)

    logger.info("EDA completed")
import pandas as pd
import ipaddress

def ip_to_int(ip_string):
    """
    Convert an IP address from string format (e.g., '192.168.0.1') to integer format.
    
    Parameters:
    ip_string (str): The IP address in string format.
    
    Returns:
    int: The IP address as an integer.
    """
    return int(ipaddress.ip_address(ip_string))

def merge_datasets_for_geolocation(df_fraud, df_ip_country):
    """
    Merge Fraud_Data.csv and IpAddress_to_Country.csv for geolocation analysis.
    
    Parameters:
    df_fraud (pd.DataFrame): DataFrame containing fraud transaction data with 'ip_address' column.
    df_ip_country (pd.DataFrame): DataFrame containing IP address ranges and country data.
    
    Returns:
    pd.DataFrame: Merged DataFrame with geolocation (country) information based on IP address.
    """
    try:
        # Convert IP address in Fraud_Data.csv to integer format
        df_fraud['ip_address_int'] = df_fraud['ip_address'].apply(ip_to_int)

        # Merge: For each IP address in fraud data, find where it falls within the lower and upper bounds in IP address to country data
        merged_df = pd.merge_asof(
            df_fraud.sort_values('ip_address_int'),  # Sort by IP address for merge_asof to work
            df_ip_country.sort_values('lower_bound_ip_address'),  # Sort by lower bound IP for merge_asof
            left_on='ip_address_int',  # Column in Fraud_Data.csv to match
            right_on='lower_bound_ip_address',  # Column in IpAddress_to_Country.csv to match
            direction='backward',  # Find closest matching lower_bound_ip_address less than or equal to the ip_address_int
            suffixes=('_fraud', '_country')
        )

        # Filter the rows where the 'ip_address_int' is within the IP range (lower_bound_ip_address and upper_bound_ip_address)
        merged_df = merged_df[
            (merged_df['ip_address_int'] >= merged_df['lower_bound_ip_address']) &
            (merged_df['ip_address_int'] <= merged_df['upper_bound_ip_address'])
        ]

        return merged_df

    except Exception as e:
        print(f"An error occurred during merging: {e}")
        return df_fraud


