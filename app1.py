"""
Streamlit App for Invoice Fraud Detection with OCR Support
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import re
import cv2

from logger_config import logger
from data_preprocessing import feature_engineering
from model_saver import load_model_and_scaler

import platform
import shutil

# Check if Tesseract is in the system path first
if shutil.which("tesseract") is None:
    # Only if it's NOT found (likely your local Windows machine), set the path
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Paths
MODEL_PATH = "models/fraud_model.pkl"
SCALER_PATH = "models/fraud_scaler.pkl"
#pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="Invoice Fraud Detection", layout="wide", page_icon="🚨")

# Custom CSS
st.markdown("""
    <style>
    .fraud-alert {
        background-color: #ffebee;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #f44336;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
    }
    .ocr-box {
        background-color: #f5f5f5;
        padding: 15px;
        border-radius: 5px;
        border: 1px solid #ddd;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚨 Invoice Fraud Detection System")
st.markdown("**Upload CSV or Extract from Invoice Images (OCR)**")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    logger.info("Loading trained model and scaler")
    model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
    return model, scaler

try:
    model, scaler = load_model()
    st.sidebar.success("✓ Model loaded successfully")
except Exception as e:
    st.sidebar.error(f"❌ Model not found. Please train first!")
    st.stop()


# ---------------- IMAGE PREPROCESSING PIPELINE ----------------
def preprocess_image_for_ocr(image):
    """
    Advanced image preprocessing pipeline to improve OCR accuracy.
    Handles various image quality issues.
    """
    logger.info("Starting image preprocessing pipeline...")
    
    # Convert PIL Image to numpy array
    img_array = np.array(image)
    
    # Convert to grayscale if color image
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        logger.info("Converted to grayscale")
    else:
        gray = img_array
    
    # Apply multiple preprocessing techniques
    preprocessed_images = []
    
    # 1. Original grayscale
    preprocessed_images.append(("Original Grayscale", gray))
    
    # 2. Otsu's thresholding (binary)
    _, thresh_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("Otsu Threshold", thresh_otsu))
    
    # 3. Adaptive thresholding
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv2.THRESH_BINARY, 11, 2)
    preprocessed_images.append(("Adaptive Threshold", adaptive))
    
    # 4. Denoising + Thresholding
    denoised = cv2.fastNlMeansDenoising(gray, h=10)
    _, thresh_denoised = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("Denoised + Threshold", thresh_denoised))
    
    # 5. Morphological operations (remove noise)
    kernel = np.ones((1, 1), np.uint8)
    morph = cv2.morphologyEx(thresh_otsu, cv2.MORPH_CLOSE, kernel)
    preprocessed_images.append(("Morphological", morph))
    
    # 6. Sharpening
    sharpened = cv2.filter2D(gray, -1, np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    _, thresh_sharp = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    preprocessed_images.append(("Sharpened", thresh_sharp))
    
    logger.info(f"Created {len(preprocessed_images)} preprocessed versions")
    
    return preprocessed_images


# ---------------- OCR HELPER FUNCTIONS ----------------
def clean_ocr_id(text):
    """Clean OCR text for IDs - replace common OCR mistakes"""
    if not text:
        return text
    
    cleaned = text.strip()
    
    # Replace common OCR character mistakes with digits
    # @ symbol is often misread 0
    cleaned = cleaned.replace('@', '0')
    
    # O (letter) to 0 (zero) in specific contexts
    cleaned = re.sub(r'([A-Z]{3})O+(\d)', r'\g<1>0\2', cleaned)  # VENO43 -> VEN043
    cleaned = re.sub(r'([A-Z]{3})O{2,}', r'\g<1>00', cleaned)    # INVOO -> INV00
    #cleaned = re.sub(r'(\d)O(\d)', r'\g<1>0\2', cleaned)         # 1O1 -> 101
    #cleaned = re.sub(r'O(\d)', r'0\1', cleaned)                  # O11 -> 011
    #cleaned = re.sub(r'(\d)O', r'\g<1>0', cleaned)               # 1O -> 10
    
    # Handle lowercase o as well
    #cleaned = re.sub(r'([A-Z]{3})o+(\d)', r'\g<1>0\2', cleaned)
    #cleaned = re.sub(r'(\d)o(\d)', r'\g<1>0\2', cleaned)
    #cleaned = re.sub(r'o(\d)', r'0\1', cleaned)
    
    return cleaned


def extract_invoice_data_from_image(image, use_preprocessing=True):
    """
    Enhanced OCR extraction with optional image preprocessing.
    Tries multiple preprocessing techniques to find the best OCR result.
    """
    logger.info("Performing OCR on image...")
    
    best_result = None
    best_score = 0
    all_texts = []
    
    if use_preprocessing:
        # Try multiple preprocessed versions
        preprocessed_images = preprocess_image_for_ocr(image)
        
        for name, processed_img in preprocessed_images:
            try:
                # Convert numpy array back to PIL Image
                pil_img = Image.fromarray(processed_img)
                
                # Perform OCR
                custom_config = r'--oem 3 --psm 6'
                text = pytesseract.image_to_string(pil_img, config=custom_config)
                
                logger.info(f"OCR with {name}: {len(text)} chars")
                all_texts.append((name, text))
                
                # Score this result based on how many fields we can extract
                score = score_ocr_result(text)
                logger.info(f"{name} score: {score}")
                
                if score > best_score:
                    best_score = score
                    best_result = (name, text, pil_img)
                    
            except Exception as e:
                logger.warning(f"Error with {name}: {e}")
                continue
        
        if best_result:
            logger.info(f"Best OCR result from: {best_result[0]} (score: {best_score})")
            text = best_result[1]
        else:
            logger.warning("All preprocessing failed, using original")
            text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')
    else:
        # Use original image without preprocessing
        custom_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(image, config=custom_config)
    
    logger.info(f"OCR extracted text length: {len(text)}")
    logger.info(f"OCR raw text:\n{text}")
    
    invoice_data = {
        'InvoiceID': '',
        'InvoiceDateTime': '',#datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'VendorID': '',
        'ItemCode': '',
        'Quantity': 0,
        'UnitPrice': 0.0,
        'InvalidOCR': False
    }
    
    # Strip empty lines and leading/trailing spaces
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    logger.info(f"Processing {len(lines)} lines of text")
    logger.info(lines)
    
    # Parse line by line for better accuracy
    for i, line in enumerate(lines):
        logger.info(f"Line {i}: '{line}'")
        #line = re.sub(r'\s+', ' ', line).replace('：', ':')#added for date extraction
        # Invoice ID - Look for pattern "Invoice ID : XXXXX"
        if 'invoice' in line.lower() and 'id' in line.lower() and not invoice_data['InvoiceID']:
            # Try to extract after the colon
            match = re.search(r':\s*([A-Z0-9]+)', line, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                extracted = clean_ocr_id(extracted)
                invoice_data['InvoiceID'] = extracted
                logger.info(f"Found InvoiceID: '{extracted}'")
        
        # Vendor ID - Look for pattern "Vendor ID : XXXXX"
        elif 'vendor' in line.lower() and 'id' in line.lower() and not invoice_data['VendorID']:
            # Try to extract everything after the colon (including @ symbols that might be 0s)
            match = re.search(r':\s*([A-Z0-9@]+)', line, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                extracted = clean_ocr_id(extracted)
                invoice_data['VendorID'] = extracted
                logger.info(f"Found VendorID: '{extracted}'")
        
        # ItemCode - Look for pattern "ItemCode : XXXXX"
        elif 'item' in line.lower() and 'code' in line.lower() and not invoice_data['ItemCode']:
            # Try to extract after the colon (handle various separators like : or —)
            match = re.search(r'[:\-—]\s*([A-Z0-9]+)', line, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()
                extracted = clean_ocr_id(extracted)
                invoice_data['ItemCode'] = extracted
                logger.info(f"Found ItemCode: '{extracted}'")
        
        
        
        elif 'date' in line.lower() and ':' in line and not invoice_data['InvoiceDateTime']:#invoice_data['InvoiceDateTime'] == datetime.now().strftime('%Y-%m-%d %H:%M:%S'):
            #match = re.search(r':\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2}\s*[AP]M)', line, re.IGNORECASE)
            #match = re.search(r':\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}:\d{2})', line)
            #match = re.search(r':\s*(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2})',line)#somehow works
            match = re.search(r':\s*(\d{1,2}/\d{1,2}/\d{4})\s+(\d{1,2}:\d{2})',line)

            logger.info(line)
            logger.info(match)
            if match:
                #date_str = match.group(1).strip()
                date_str = f"{match.group(1)} {match.group(2)}"
                try:
                    parsed_date = datetime.strptime(date_str, '%m/%d/%Y %H:%M')
                    invoice_data['InvoiceDateTime'] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                    #parsed_date = datetime.strptime(date_str, '%m/%d/%Y %I:%M:%S %p')
                    #invoice_data['InvoiceDateTime'] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                    #parsed_date = datetime.strptime(date_str, '%m/%d/%Y %H:%M:%S')
                    #invoice_data['InvoiceDateTime'] = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
                    logger.info(f"Found Date: '{date_str}'")
                except Exception as e:
                    logger.warning(f"Date parsing failed: {e}")
        
        # Quantity - Look for pattern "Quantity : X"
        elif 'quantity' in line.lower() and invoice_data['Quantity'] == 0:
            # Handle both : and = as separators
            match = re.search(r'[:\=]\s*(\d+)', line, re.IGNORECASE)
            if match:
                quantity_str = match.group(1).strip()
                invoice_data['Quantity'] = int(quantity_str)
                logger.info(f"Found Quantity: {quantity_str}")
        
        # Unit Price - Look for pattern "Unit Price : Rs XXX.XX"
     

        elif 'unit' in line.lower() and 'price' in line.lower() and invoice_data['UnitPrice'] == 0.0:
            match = re.search(r'[:=]\s*Rs\s*([\d]+(?:\.\d+)?)', line, re.IGNORECASE)
            if match:
                price_str = match.group(1).strip()
                invoice_data['UnitPrice'] = float(price_str)
                logger.info(f"Found UnitPrice: {price_str}")

    
    # Defaults if OCR fails
    if not invoice_data['InvoiceID']:
        invoice_data['InvoiceID'] = f"OCR_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        logger.warning(f"Using default InvoiceID: {invoice_data['InvoiceID']}")
    
    if not invoice_data['VendorID']:
        invoice_data['VendorID'] = "UNKNOWN_VENDOR"
        logger.warning("Using default VendorID: UNKNOWN_VENDOR")
    
    if not invoice_data['ItemCode']:
        invoice_data['ItemCode'] = "UNKNOWN_ITEM"
        logger.warning("Using default ItemCode: UNKNOWN_ITEM")
    
    if invoice_data['Quantity'] == 0:
        invoice_data['Quantity'] = 1
        logger.warning("Using default Quantity: 1")

    # Check if OCR is invalid
    is_invalid = all([
        invoice_data['InvoiceID'].startswith('OCR_'),
        invoice_data['VendorID'] == "UNKNOWN_VENDOR",
        invoice_data['ItemCode'] == "UNKNOWN_ITEM",
        invoice_data['Quantity'] == 1,
        invoice_data['UnitPrice'] == 0.0
    ])
    invoice_data['InvalidOCR'] = is_invalid

    logger.info(f"Final OCR extraction: {invoice_data}")
    return invoice_data, text


def score_ocr_result(text):
    """
    Score OCR result based on how many expected fields are detected.
    Higher score = better OCR result.
    """
    score = 0
    text_lower = text.lower()
    
    # Check for presence of key fields
    if 'invoice' in text_lower and 'id' in text_lower:
        score += 20
    if 'vendor' in text_lower and 'id' in text_lower:
        score += 20
    if 'item' in text_lower and 'code' in text_lower:
        score += 20
    if 'quantity' in text_lower:
        score += 15
    if 'unit' in text_lower and 'price' in text_lower:
        score += 15
    if 'date' in text_lower:
        score += 10
    
    # Bonus for finding actual numbers
    if re.search(r'[A-Z]{3}\d{4,}', text):  # ID patterns like INV0001
        score += 10
    if re.search(r'\d+\.\d{2}', text):  # Price patterns
        score += 5
    
    return score


# ---------------- Fraud categorization ----------------
def categorize_fraud_advanced(df):
    """Advanced fraud categorization with risk scoring"""
    
    df['FraudType'] = 'Normal'
    df['RiskLevel'] = 'Low'
    df['RiskScore'] = 0
    df['FraudReasons'] = ''
    
    fraud_mask = df['Anomaly'] == 1
    
    if fraud_mask.sum() == 0:
        return df
    
    fraud_df = df[fraud_mask].copy()
    
    reasons_list = []
    fraud_types = []
    risk_scores = []

    if len(fraud_df) == 1:
        reasons = []
        score = 0
        fraud_type = 'Suspicious'

        
        idx = fraud_df.index[0]  # get the actual row index

        computed_amount = fraud_df.loc[idx, 'Quantity'] * fraud_df.loc[idx, 'UnitPrice']
        invoice_amount = fraud_df.loc[idx, 'InvoiceAmount']

        diff_pct = abs(computed_amount - invoice_amount) / max(invoice_amount, 1)

        if diff_pct > 0.05:  # >5% mismatch
            reasons.append("Amount mismatch with Quantity × UnitPrice")
            score += 25
            fraud_type = "Amount Mismatch"

        if fraud_df.loc[idx, 'InvoiceAmount'] > 100000:
            reasons.append("Unusually high amount")
            score += 25
        
        if fraud_df.loc[idx, 'Quantity'] > 500:
            reasons.append("Excessive quantity")
            score += 20
            fraud_type = 'Quantity Fraud'
        
        hour = fraud_df.loc[idx, 'InvoiceHour']
        if hour < 6 or hour > 22:
            reasons.append(f"Off-hours ({hour}:00)")
            score += 10
            fraud_type = 'Timing Anomaly'
        
        if fraud_df.loc[idx, 'InvoiceWeekday'] >= 5:
            reasons.append("Weekend transaction")
            score += 5
        
        anomaly_score_contrib = int(abs(fraud_df.loc[idx, 'anomaly_score']) * 10)
        score += anomaly_score_contrib
        
        reasons_list.append('; '.join(reasons) if reasons else 'Normal')
        fraud_types.append(fraud_type)
        risk_scores.append(min(score, 100))
    else:  
    
        for idx in fraud_df.index:
            reasons = []
            score = 0
            fraud_type = 'Suspicious'
            
            deviation_pct = abs(fraud_df.loc[idx, 'DeviationFromVendorMean'] / (fraud_df.loc[idx, 'VendorAvgAmount'] + 1))
            if deviation_pct > 2:
                reasons.append(f"Amount {deviation_pct:.1f}x vendor avg")
                score += 30
                fraud_type = 'Amount Anomaly'
            elif deviation_pct > 1:
                reasons.append(f"Amount {deviation_pct:.1f}x vendor avg")
                score += 20
            
            if fraud_df.loc[idx, 'InvoiceAmount'] > df['InvoiceAmount'].quantile(0.95):
                reasons.append("Unusually high amount")
                score += 25
            
            if fraud_df.loc[idx, 'Quantity'] > df['Quantity'].quantile(0.98):
                reasons.append("Excessive quantity")
                score += 20
                fraud_type = 'Quantity Fraud'
            
            if fraud_df.loc[idx, 'VendorInvoiceCount'] < 3:
                reasons.append("New/rare vendor")
                score += 15
            # fraud_type = 'Vendor Risk' 
            
            if fraud_df.loc[idx, 'VendorItemFreq'] == 1:
                reasons.append("First-time vendor-item combo")
                score += 15
            
            hour = fraud_df.loc[idx, 'InvoiceHour']
            if hour < 6 or hour > 22:
                reasons.append(f"Off-hours ({hour}:00)")
                score += 10
                fraud_type = 'Timing Anomaly'
            
            if fraud_df.loc[idx, 'InvoiceWeekday'] >= 5:
                reasons.append("Weekend transaction")
                score += 5
            
            anomaly_score_contrib = int(abs(fraud_df.loc[idx, 'anomaly_score']) * 10)
            score += anomaly_score_contrib
            
            reasons_list.append('; '.join(reasons) if reasons else 'Statistical anomaly')
            fraud_types.append(fraud_type)
            risk_scores.append(min(score, 100))
        
    df.loc[fraud_mask, 'FraudReasons'] = reasons_list
    df.loc[fraud_mask, 'FraudType'] = fraud_types
    df.loc[fraud_mask, 'RiskScore'] = risk_scores
    
    df.loc[(df['RiskScore'] >= 70) & fraud_mask, 'RiskLevel'] = 'Critical'
    df.loc[(df['RiskScore'] >= 50) & (df['RiskScore'] < 70) & fraud_mask, 'RiskLevel'] = 'High'
    df.loc[(df['RiskScore'] >= 30) & (df['RiskScore'] < 50) & fraud_mask, 'RiskLevel'] = 'Medium'
    df.loc[(df['RiskScore'] < 30) & fraud_mask, 'RiskLevel'] = 'Low'
    
    return df


# ---------------- Sidebar ----------------
st.sidebar.header("📋 Data Input Method")
input_method = st.sidebar.radio(
    "Choose input method:",
    ["Upload CSV File", "OCR from Invoice Image"],
    key="input_method_selector"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Settings")
threshold_pct = st.sidebar.slider("Detection Threshold (%)", 1.0, 20.0, 5.0, 0.5)

# OCR settings
if input_method == "OCR from Invoice Image":
    st.sidebar.markdown("### 🔍 OCR Settings")
    use_preprocessing = st.sidebar.checkbox("Enable Image Preprocessing", value=True, 
                                            help="Applies multiple image enhancement techniques to improve OCR accuracy")
    
    if use_preprocessing:
        st.sidebar.info("📌 The system will try 6 different preprocessing methods and select the best result automatically.")


# ---------------- Main App ----------------
if input_method == "Upload CSV File":
    st.subheader("📁 Upload Invoice CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            logger.info(f"CSV uploaded: {len(df)} records")
            
            X, df = feature_engineering(df)
            X_scaled = scaler.transform(X)
            df['anomaly_score'] = model.decision_function(X_scaled)
            threshold = np.percentile(df['anomaly_score'], threshold_pct)
            df['Anomaly'] = (df['anomaly_score'] <= threshold).astype(int)
            
            df = categorize_fraud_advanced(df)
            
            # Overview Metrics
            st.markdown("### 📈 Overview Metrics")
            col1, col2, col3, col4, col5 = st.columns(5)
            total_invoices = len(df)
            total_anomalies = df['Anomaly'].sum()
            fraud_rate = (total_anomalies / total_invoices) * 100 if total_invoices > 0 else 0
            total_fraud_amount = df[df['Anomaly'] == 1]['InvoiceAmount'].sum()
            critical_count = len(df[df['RiskLevel'] == 'Critical'])
            avg_risk = df[df['Anomaly'] == 1]['RiskScore'].mean() if total_anomalies > 0 else 0
            
            col1.metric("Total Invoices", f"{total_invoices:,}")
            col2.metric("Detected Frauds", f"{total_anomalies:,}", delta=f"{fraud_rate:.1f}%")
            col3.metric("Fraud Amount", f"₹{total_fraud_amount:,.2f}")
            col4.metric("Critical Cases", f"{critical_count:,}")
            col5.metric("Avg Risk Score", f"{avg_risk:.1f}")
            
            st.markdown("---")
            
            # Detailed table
            st.subheader("🚩 Flagged Invoices")
            if total_anomalies > 0:
                fraud_display = df[df['Anomaly'] == 1].sort_values('RiskScore', ascending=False)
                display_cols = ['InvoiceID', 'InvoiceDateTime', 'VendorID', 'ItemCode',
                               'Quantity', 'UnitPrice', 'InvoiceAmount',
                               'FraudType', 'RiskLevel', 'RiskScore', 'FraudReasons']
                st.dataframe(fraud_display[display_cols], use_container_width=True, height=400)
                
                st.download_button(
                    "⬇️ Download Fraud Report",
                    data=fraud_display.to_csv(index=False),
                    file_name=f"fraud_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.success("✅ No fraudulent invoices detected!")
                
        except Exception as e:
            logger.error(f"Error: {e}")
            st.error(f"❌ Error: {str(e)}")

else:  # OCR from Invoice Image
    st.subheader("📷 Extract Invoice Data from Image (OCR)")
    
    st.info("""
    **Instructions:**
    1. Upload a clear image of the invoice
    2. System will extract invoice details using OCR
    3. Review extracted data and make corrections if needed
    4. Click 'Analyze for Fraud' to detect anomalies
    """)
    
    uploaded_image = st.file_uploader("Upload Invoice Image", type=["png", "jpg", "jpeg"])
    
    if uploaded_image:
        try:
            image = Image.open(uploaded_image)
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.image(image, caption="Uploaded Invoice",width="stretch")# use_column_width=True)
            
            with col2:
                with st.spinner("🔍 Performing OCR with image preprocessing..."):
                    invoice_data, ocr_text = extract_invoice_data_from_image(image, use_preprocessing)
                
                if invoice_data['InvalidOCR']:
                    st.warning("⚠️ OCR extraction incomplete. Please review and correct the data below.")
                else:
                    st.success("✓ OCR extraction successful!")
                
                st.markdown("### Extracted Data")
                st.markdown('<div class="ocr-box">', unsafe_allow_html=True)
                
                invoice_id = st.text_input("Invoice ID", value=invoice_data['InvoiceID'])
                invoice_date = st.text_input("Invoice Date", value=invoice_data['InvoiceDateTime'])
                vendor_id = st.text_input("Vendor ID", value=invoice_data['VendorID'])
                item_code = st.text_input("Item Code", value=invoice_data['ItemCode'])
                quantity = st.number_input("Quantity", value=int(invoice_data['Quantity']), min_value=0)
                unit_price = st.number_input("Unit Price (₹)", value=float(invoice_data['UnitPrice']), min_value=0.0)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                with st.expander("📄 View Raw OCR Text"):
                    st.text_area("Extracted Text", ocr_text, height=200)
            
            if st.button("🔍 Analyze for Fraud", type="primary"):
                ocr_df = pd.DataFrame([{
                    'InvoiceID': invoice_id,
                    'InvoiceDateTime': invoice_date,
                    'VendorID': vendor_id,
                    'ItemCode': item_code,
                    'Quantity': quantity,
                    'UnitPrice': unit_price
                }])
                
                logger.info(f"Analyzing OCR invoice: {invoice_id}")
                
                X, df_processed = feature_engineering(ocr_df)
                X_scaled = scaler.transform(X)
                df_processed['anomaly_score'] = model.decision_function(X_scaled)
                threshold = np.percentile(df_processed['anomaly_score'], threshold_pct)
                df_processed['Anomaly'] = (df_processed['anomaly_score'] <= threshold).astype(int)
                
                df_processed = categorize_fraud_advanced(df_processed)
                
                st.markdown("---")
                st.markdown("### 🎯 Fraud Detection Result")
                
                is_fraud = df_processed['Anomaly'].iloc[0] == 1
                risk_score = df_processed['RiskScore'].iloc[0]

                # Simple validation
                if risk_score < 15:
                    is_fraud = False
                else:
                    is_fraud = True
                
                if is_fraud:
                    st.markdown('<div class="fraud-alert">', unsafe_allow_html=True)
                    st.error("🚨 **FRAUD DETECTED!**")
                    
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Risk Level", df_processed['RiskLevel'].iloc[0])
                    col2.metric("Risk Score", f"{df_processed['RiskScore'].iloc[0]:.0f}/100")
                    col3.metric("Fraud Type", df_processed['FraudType'].iloc[0])
                    
                    st.markdown("**Reasons:**")
                    reasons = df_processed['FraudReasons'].iloc[0].split(';')
                    for reason in reasons:
                        if reason.strip():
                            st.markdown(f"- {reason.strip()}")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="safe-alert">', unsafe_allow_html=True)
                    st.success("✅ **INVOICE APPEARS NORMAL**")
                    st.info("No fraudulent patterns detected in this invoice.")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                st.markdown("### 📊 Invoice Details")
                st.json({
                    'Invoice ID': invoice_id,
                    'Date': invoice_date,
                    'Vendor': vendor_id,
                    'Item': item_code,
                    'Quantity': quantity,
                    'Unit Price': f"₹{unit_price:.2f}",
                    'Total Amount': f"₹{quantity * unit_price:.2f}",
                    'Anomaly Score': f"{df_processed['anomaly_score'].iloc[0]:.6f}",
                    'Is Fraud': 'Yes' if is_fraud else 'No'
                })
                
        except Exception as e:
            logger.error(f"OCR Error: {e}", exc_info=True)
            st.error(f"❌ OCR Error: {str(e)}")
            st.info("Please upload a clear image with readable text")


# ---------------- Footer ----------------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Invoice Fraud Detection System</p>
</div>

""", unsafe_allow_html=True)

