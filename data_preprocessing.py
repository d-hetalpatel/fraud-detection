"""
Data Preprocessing Module
Feature engineering for fraud detection
EXACT SAME LOGIC as train_pipeline_new.py - DO NOT CHANGE
"""

import pandas as pd
import numpy as np
from logger_config import logger


def feature_engineering(df):
    """
    Feature engineering for invoice anomaly detection
    EXACT SAME as train_pipeline_new.py
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        tuple: (X features dataframe, full dataframe with all columns)
    """
    logger.info("Starting feature engineering")

    # ------------------------
    # Datetime parsing
    # ------------------------
    df["InvoiceDateTime"] = pd.to_datetime(df["InvoiceDateTime"])

    # Invoice amount
    df['InvoiceAmount'] = df['Quantity'] * df['UnitPrice']

    # Time features
    df['InvoiceHour'] = df['InvoiceDateTime'].dt.hour
    df['InvoiceWeekday'] = df['InvoiceDateTime'].dt.weekday

    # Vendor behavior features
    df['VendorInvoiceCount'] = df.groupby('VendorID')['InvoiceID'].transform('count')
    df['VendorAvgAmount'] = df.groupby('VendorID')['InvoiceAmount'].transform('mean')
    df['VendorStdAmount'] = df.groupby('VendorID')['InvoiceAmount'].transform('std').fillna(0)

    # Item behavior
    df['ItemFreq'] = df.groupby('ItemCode')['ItemCode'].transform('count')
    df['VendorItemFreq'] = df.groupby(['VendorID','ItemCode'])['ItemCode'].transform('count')

    # Deviation feature (VERY IMPORTANT)
    df['DeviationFromVendorMean'] = df['InvoiceAmount'] - df['VendorAvgAmount']

    # ------------------------
    # Final feature list - EXACT SAME ORDER
    # ------------------------
    features = [
        'Quantity', 'UnitPrice', 'InvoiceAmount',
        'VendorInvoiceCount', 'VendorAvgAmount', 'VendorStdAmount',
        'ItemFreq', 'VendorItemFreq',
        'InvoiceHour', 'InvoiceWeekday',
        'DeviationFromVendorMean'
    ]

    X = df[features]

    logger.info(f"Feature engineering completed: {len(features)} features")
    logger.info(f"Features: {', '.join(features)}")

    return X, df