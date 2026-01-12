# fraud-detection
A Python-based Streamlit application that detects potentially fraudulent invoices using OCR (Optical Character Recognition) and machine learning anomaly detection
Invoice Fraud Detection System with OCR & ML Anomaly Detection
The system extracts invoice details from both CSV files and scanned images, performs feature engineering, calculates risk scores, and classifies invoices as normal or suspicious with detailed fraud reasoning.

Key Features:
OCR extraction from invoice images with advanced preprocessing for improved accuracy
Automatic detection of Invoice ID, Vendor ID, Item Code, Quantity, Unit Price, and Date
Anomaly detection using a pre-trained ML model with threshold-based scoring
Fraud classification with detailed risk scoring and reasons
Interactive Streamlit dashboard for CSV uploads and image analysis
Downloadable fraud reports in CSV format
Configurable detection thresholds and preprocessing options

Tech Stack:
Python 3.x
Streamlit
OpenCV & PIL for image processing
Pytesseract for OCR
Scikit-learn for anomaly detection
Pandas & NumPy for data handling

Use Cases:
Accounts payable fraud prevention
Vendor invoice validation
Financial anomaly detection in enterprises
