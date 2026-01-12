"""
Model Saver Module
Saves trained models and statistics
"""

import joblib
import os
from logger_config import logger


def save_model_and_stats(model, scaler, stats, X, 
                         model_path="models/fraud_model.pkl",
                         scaler_path="models/fraud_scaler.pkl",
                         stats_path="models/training_stats.pkl"):
    """
    Save trained model, scaler, and statistics
    
    Args:
        model: Trained Isolation Forest
        scaler: Fitted StandardScaler
        stats: Training statistics dict
        X: Feature dataframe
        model_path: Path to save model
        scaler_path: Path to save scaler
        stats_path: Path to save stats
    """
    logger.info("Saving model and statistics...")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Save model
    joblib.dump(model, model_path)
    logger.info(f"Model saved successfully at {model_path}")
    
    # Save scaler
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved successfully at {scaler_path}")
    
    # Save training stats
    training_info = {
        'stats': stats,
        'feature_names': list(X.columns),
        'feature_stats': {
            'means': X.mean().to_dict(),
            'stds': X.std().to_dict(),
            'mins': X.min().to_dict(),
            'maxs': X.max().to_dict()
        }
    }
    
    joblib.dump(training_info, stats_path)
    logger.info(f"Training statistics saved to {stats_path}")
    
    logger.info("All files saved successfully!")


def load_model_and_scaler(model_path="models/fraud_model.pkl",
                          scaler_path="models/fraud_scaler.pkl"):
    """
    Load trained model and scaler
    
    Args:
        model_path: Path to model file
        scaler_path: Path to scaler file
        
    Returns:
        tuple: (model, scaler)
    """
    logger.info("Loading model and scaler...")
    
    try:
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        scaler = joblib.load(scaler_path)
        logger.info(f"Scaler loaded from {scaler_path}")
        
        return model, scaler
        
    except FileNotFoundError as e:
        logger.error(f"Model file not found: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise