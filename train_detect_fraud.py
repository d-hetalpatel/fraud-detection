"""
Main Training Script for Fraud Detection
Uses modular components: data_loader, data_preprocessing, model_training, model_saver
EXACT SAME LOGIC as train_pipeline_new.py - DO NOT CHANGE
"""

import argparse
from logger_config import logger
from data_loader import load_data, validate_data
from data_preprocessing import feature_engineering
from model_training import build_and_train_model, evaluate_model, plot_feature_importance
from model_saver import save_model_and_stats


def train_fraud_detection_model(data_path, 
                                contamination=0.05, 
                                n_estimators=1000, 
                                has_labels=False):
    """
    Complete training pipeline
    EXACT SAME LOGIC as train_pipeline_new.py
    
    Args:
        data_path: Path to training CSV file
        contamination: Expected proportion of anomalies (default: 0.05 = 5%)
        n_estimators: Number of trees in Isolation Forest (default: 1000)
        has_labels: Whether dataset has ground truth 'IsFraud' column
        
    Returns:
        tuple: (trained model, scaler, statistics)
    """
    logger.info("=" * 70)
    logger.info("STARTING FRAUD DETECTION MODEL TRAINING")
    logger.info("=" * 70)
    
    # Step 1: Load data
    logger.info("\n[STEP 1/5] Loading data...")
    df = load_data(data_path)
    validate_data(df)
    
    # Step 2: Feature engineering
    logger.info("\n[STEP 2/5] Feature engineering...")
    X, df_full = feature_engineering(df)
    
    # Check for ground truth labels
    y_true = None
    if has_labels and 'IsFraud' in df_full.columns:
        y_true = df_full['IsFraud']
        fraud_count = y_true.sum()
        normal_count = (~y_true.astype(bool)).sum()
        logger.info(f"Ground truth labels found: {fraud_count} frauds, {normal_count} normal")
        logger.info(f"Fraud rate: {fraud_count/len(y_true)*100:.2f}%")
    else:
        logger.info("No ground truth labels found (unsupervised mode)")
    
    # Step 3: Train model
    logger.info("\n[STEP 3/5] Training model...")
    model, scaler = build_and_train_model(X, contamination, n_estimators)
    
    # Step 4: Evaluate model
    logger.info("\n[STEP 4/5] Evaluating model...")
    stats, anomaly_labels = evaluate_model(model, scaler, X, y_true, contamination)
    
    # Step 5: Generate visualizations
    logger.info("\n[STEP 5/5] Generating visualizations...")
    plot_feature_importance(X, anomaly_labels)
    
    # Save everything
    logger.info("\nSaving model and statistics...")
    save_model_and_stats(model, scaler, stats, X)
    
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETED SUCCESSFULLY")
    logger.info("=" * 70)
    
    if has_labels and y_true is not None:
        logger.info("\n📊 Final Results:")
        logger.info(f"   Precision: {stats['precision']:.1%}")
        logger.info(f"   Recall: {stats['recall']:.1%}")
        logger.info(f"   F1-Score: {stats['f1_score']:.3f}")
        logger.info(f"   Detected: {stats['contamination_rate']:.2f}%")
    
    logger.info("\n✓ Model saved to: models/fraud_model.pkl")
    logger.info("✓ Scaler saved to: models/fraud_scaler.pkl")
    logger.info("✓ Stats saved to: models/training_stats.pkl")
    logger.info("✓ Plot saved to: feature_distributions.png")
    
    logger.info("\n🚀 Next step: Run streamlit app")
    logger.info("   streamlit run app_with_ocr.py")
    
    return model, scaler, stats


def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(
        description='Train Invoice Fraud Detection Model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default settings (5% contamination)
  python train_detect_fraud.py --data synthetic_invoice_fraud_data.csv
  
  # Train with labeled data and see evaluation metrics
  python train_detect_fraud.py --data synthetic_invoice_fraud_data.csv --has-labels
  
  # Train with custom contamination (7%)
  python train_detect_fraud.py --data your_data.csv --contamination 0.07
  
  # Train with more trees for better accuracy
  python train_detect_fraud.py --data your_data.csv --estimators 2000 --has-labels
        """
    )
    
    parser.add_argument(
        '--data', 
        type=str, 
        default='synthetic_invoice_fraud_data.csv',
        help='Path to training data CSV file'
    )
    
    parser.add_argument(
        '--contamination', 
        type=float, 
        default=0.05,
        help='Expected proportion of anomalies (default: 0.05 = 5%%)'
    )
    
    parser.add_argument(
        '--estimators', 
        type=int, 
        default=1000,
        help='Number of trees in Isolation Forest (default: 1000)'
    )
    
    parser.add_argument(
        '--has-labels', 
        action='store_true',
        help='Dataset has ground truth IsFraud column for evaluation'
    )
    
    args = parser.parse_args()
    
    # Log configuration
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING CONFIGURATION")
    logger.info("=" * 70)
    logger.info(f"Data file: {args.data}")
    logger.info(f"Contamination: {args.contamination} ({args.contamination*100}%)")
    logger.info(f"N_estimators: {args.estimators}")
    logger.info(f"Has labels: {args.has_labels}")
    logger.info("=" * 70 + "\n")
    
    # Train model
    try:
        model, scaler, stats = train_fraud_detection_model(
            data_path=args.data,
            contamination=args.contamination,
            n_estimators=args.estimators,
            has_labels=args.has_labels
        )
        
        logger.info("\n✅ Training completed successfully!")
        
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()