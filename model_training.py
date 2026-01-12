"""
Model Training Module
Trains Isolation Forest model
EXACT SAME LOGIC as train_pipeline_new.py - DO NOT CHANGE
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

from logger_config import logger


def build_and_train_model(X, contamination=0.05, n_estimators=1000):
    """
    Build and train Isolation Forest model
    EXACT SAME as train_pipeline_new.py
    
    Args:
        X: Feature dataframe
        contamination: Expected proportion of anomalies
        n_estimators: Number of trees
        
    Returns:
        tuple: (trained model, fitted scaler)
    """
    logger.info("Building and training model...")
    
    # Build scaler
    logger.info("Fitting StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    logger.info("Scaling completed")
    
    # Build and train Isolation Forest
    logger.info(f"Training Isolation Forest model (contamination={contamination}, n_estimators={n_estimators})...")
    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
    
    iso.fit(X_scaled)
    logger.info("Model training completed")
    
    return iso, scaler


def evaluate_model(model, scaler, X, y_true=None, contamination=0.05):
    """
    Evaluate model performance
    EXACT SAME as train_pipeline_new.py
    
    Args:
        model: Trained Isolation Forest
        scaler: Fitted scaler
        X: Features
        y_true: Ground truth labels (optional)
        contamination: Contamination rate
        
    Returns:
        dict: Evaluation metrics
    """
    logger.info("Evaluating model performance")
    
    # Scale features
    X_scaled = scaler.transform(X)
    
    # Get predictions and scores
    predictions = model.predict(X_scaled)
    anomaly_scores = model.decision_function(X_scaled)
    
    # Apply threshold-based detection (top 5% most anomalous)
    threshold = np.percentile(anomaly_scores, contamination * 100)
    anomaly_labels = (anomaly_scores < threshold).astype(int)
    
    # Basic statistics
    n_anomalies = anomaly_labels.sum()
    n_normal = len(anomaly_labels) - n_anomalies
    contamination_rate = (n_anomalies / len(anomaly_labels)) * 100
    
    logger.info(f"Total records: {len(X)}")
    logger.info(f"Detected anomalies: {n_anomalies} ({contamination_rate:.2f}%)")
    logger.info(f"Normal records: {n_normal}")
    logger.info(f"Anomaly score threshold: {threshold:.6f}")
    
    # If ground truth labels are available
    if y_true is not None:
        logger.info("=== Classification Report ===")
        report = classification_report(y_true, anomaly_labels, target_names=['Normal', 'Fraud'])
        print(report)
        logger.info(report)
        
        logger.info("=== Confusion Matrix ===")
        cm = confusion_matrix(y_true, anomaly_labels)
        print(cm)
        logger.info(f"\n{cm}")
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'contamination_rate': float(contamination_rate),
            'threshold': float(threshold)
        }
        
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        
        return metrics, anomaly_labels
    
    return {
        'n_anomalies': int(n_anomalies),
        'n_normal': int(n_normal),
        'contamination_rate': float(contamination_rate),
        'threshold': float(threshold)
    }, anomaly_labels


def plot_feature_importance(X, anomaly_labels, output_path='feature_distributions.png'):
    """
    Visualize feature distributions for normal vs anomaly
    EXACT SAME as train_pipeline_new.py
    
    Args:
        X: Features
        anomaly_labels: Anomaly labels
        output_path: Path to save plot
    """
    logger.info("Generating feature importance plots")
    
    df_plot = X.copy()
    df_plot['Anomaly'] = anomaly_labels
    
    # Select top features to plot
    feature_cols = X.columns[:6]  # Plot first 6 features
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for idx, col in enumerate(feature_cols):
        df_plot.boxplot(column=col, by='Anomaly', ax=axes[idx])
        axes[idx].set_title(f'{col}')
        axes[idx].set_xlabel('Anomaly (0=Normal, 1=Fraud)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Feature distribution plot saved as '{output_path}'")
    plt.close()