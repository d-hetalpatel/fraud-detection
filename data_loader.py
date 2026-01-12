"""
Data Loader Module
Handles loading data from CSV files
"""

import pandas as pd
from logger_config import logger


def load_data(path):
    """
    Load dataset from CSV file
    
    Args:
        path (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    try:
        logger.info(f"Loading dataset from {path}")
        df = pd.read_csv(path)
        logger.info(f"Dataset loaded successfully: {len(df)} records")
        
        # Validate required columns
        required_columns = ['InvoiceID', 'InvoiceDateTime', 'Quantity', 
                          'UnitPrice', 'VendorID', 'ItemCode']
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        
        if missing_cols:
            logger.warning(f"Missing columns: {missing_cols}")
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info(f"All required columns present")
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise


def validate_data(df):
    """
    Validate loaded data
    
    Args:
        df (pd.DataFrame): DataFrame to validate
        
    Returns:
        bool: True if valid, raises exception otherwise
    """
    logger.info("Validating data...")
    
    # Check for null values in critical columns
    critical_cols = ['Quantity', 'UnitPrice', 'VendorID', 'ItemCode']
    
    for col in critical_cols:
        null_count = df[col].isnull().sum()
        if null_count > 0:
            logger.warning(f"Column '{col}' has {null_count} null values")
    
    # Check data types
    if not pd.api.types.is_numeric_dtype(df['Quantity']):
        logger.error("Quantity column must be numeric")
        raise ValueError("Quantity must be numeric")
    
    if not pd.api.types.is_numeric_dtype(df['UnitPrice']):
        logger.error("UnitPrice column must be numeric")
        raise ValueError("UnitPrice must be numeric")
    
    logger.info("Data validation passed")
    return True