"""
Generate Realistic Synthetic Invoice Fraud Dataset

This creates a dataset with:
- 10,000 invoices total
- ~15% actual fraud with realistic patterns
- Multiple fraud types that Z-scores can detect
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
N_INVOICES = 10000
FRAUD_RATE = 0.15  # 15% fraud
N_VENDORS = 150
N_ITEMS = 200

print("=" * 70)
print("GENERATING SYNTHETIC INVOICE FRAUD DATASET")
print("=" * 70)

# ============================================
# 1. CREATE VENDORS WITH TYPICAL BEHAVIOR
# ============================================
print("\n1. Creating vendor profiles...")

vendors = []
for i in range(N_VENDORS):
    vendor_profile = {
        'VendorID': f'V{i+1:04d}',
        'typical_amount_mean': np.random.lognormal(mean=6, sigma=1.5),  # $400-$1000 typical
        'typical_amount_std': np.random.uniform(50, 300),
        'typical_quantity_mean': np.random.uniform(5, 50),
        'typical_quantity_std': np.random.uniform(2, 15),
        'invoice_frequency': np.random.choice(['high', 'medium', 'low'], p=[0.3, 0.5, 0.2])
    }
    vendors.append(vendor_profile)

vendor_df = pd.DataFrame(vendors)
print(f"✓ Created {len(vendor_df)} vendor profiles")

# ============================================
# 2. CREATE ITEMS WITH TYPICAL PRICES
# ============================================
print("\n2. Creating item catalog...")

items = []
for i in range(N_ITEMS):
    item_profile = {
        'ItemCode': f'ITEM{i+1:04d}',
        'typical_unit_price': np.random.lognormal(mean=3.5, sigma=1),  # $10-$100 typical
        'price_std': np.random.uniform(5, 30),
        'category': np.random.choice(['Electronics', 'Office', 'Equipment', 'Supplies'])
    }
    items.append(item_profile)

item_df = pd.DataFrame(items)
print(f"✓ Created {len(item_df)} items")

# ============================================
# 3. GENERATE NORMAL INVOICES
# ============================================
print("\n3. Generating normal invoices...")

invoices = []
start_date = datetime(2024, 1, 1)

n_normal = int(N_INVOICES * (1 - FRAUD_RATE))

for i in range(n_normal):
    # Select vendor
    vendor = vendor_df.sample(1).iloc[0]
    
    # Select item
    item = item_df.sample(1).iloc[0]
    
    # Generate normal behavior
    # Quantity follows vendor's typical pattern
    quantity = max(1, int(np.random.normal(
        vendor['typical_quantity_mean'],
        vendor['typical_quantity_std']
    )))
    
    # Price follows item's typical pattern
    unit_price = max(1, np.random.normal(
        item['typical_unit_price'],
        item['price_std']
    ))
    
    # Amount = Quantity × Price (normal calculation)
    invoice_amount = quantity * unit_price
    
    # Timestamp - mostly business hours
    days_offset = int(np.random.randint(0, 180))
    hour_probs = np.array([
        0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 (late night - rare)
        0.03, 0.05, 0.07, 0.09, 0.10, 0.11,  # 6-11 (morning - increasing)
        0.10, 0.10, 0.09, 0.08, 0.07, 0.05,  # 12-17 (afternoon - decreasing)
        0.03, 0.02, 0.02, 0.01, 0.01, 0.01   # 18-23 (evening - rare)
    ])
    hour_probs = hour_probs / hour_probs.sum()  # Normalize to sum to 1
    hour = int(np.random.choice(range(24), p=hour_probs))
    
    invoice_date = start_date + timedelta(days=days_offset, hours=hour)
    
    invoices.append({
        'InvoiceID': f'INV{i+1:06d}',
        'InvoiceDateTime': invoice_date,
        'VendorID': vendor['VendorID'],
        'ItemCode': item['ItemCode'],
        'Quantity': quantity,
        'UnitPrice': round(unit_price, 2),
        'InvoiceAmount': round(invoice_amount, 2),
        'IsFraud': 0,
        'FraudType': 'Normal'
    })

print(f"✓ Generated {len(invoices)} normal invoices")

# ============================================
# 4. GENERATE FRAUD INVOICES (Multiple Types)
# ============================================
print("\n4. Generating fraud invoices with realistic patterns...")

n_fraud = int(N_INVOICES * FRAUD_RATE)

fraud_types = {
    'amount_inflation': 0.30,      # 30% - Inflated amounts
    'quantity_fraud': 0.25,         # 25% - Excessive quantities
    'price_manipulation': 0.20,     # 20% - Inflated prices
    'new_vendor_item': 0.15,        # 15% - Suspicious new combos
    'timing_fraud': 0.10            # 10% - Off-hours suspicious activity
}

for i in range(n_fraud):
    fraud_type = np.random.choice(list(fraud_types.keys()), p=list(fraud_types.values()))
    
    vendor = vendor_df.sample(1).iloc[0]
    item = item_df.sample(1).iloc[0]
    
    # Start with normal values
    quantity = max(1, int(np.random.normal(vendor['typical_quantity_mean'], vendor['typical_quantity_std'])))
    unit_price = max(1, np.random.normal(item['typical_unit_price'], item['price_std']))
    
    # Apply fraud pattern
    if fraud_type == 'amount_inflation':
        # Amount is 3-10x normal for this vendor
        multiplier = np.random.uniform(3, 10)
        invoice_amount = vendor['typical_amount_mean'] * multiplier
        # Adjust quantity/price to match
        quantity = max(1, int(quantity * np.sqrt(multiplier)))
        unit_price = invoice_amount / quantity
        
    elif fraud_type == 'quantity_fraud':
        # Quantity is 5-20x normal
        multiplier = np.random.uniform(5, 20)
        quantity = int(vendor['typical_quantity_mean'] * multiplier)
        invoice_amount = quantity * unit_price
        
    elif fraud_type == 'price_manipulation':
        # Price is 3-8x normal for this item
        multiplier = np.random.uniform(3, 8)
        unit_price = item['typical_unit_price'] * multiplier
        invoice_amount = quantity * unit_price
        
    elif fraud_type == 'new_vendor_item':
        # Use a rare vendor-item combo with high amount
        # Pick a vendor with low frequency
        vendor = vendor_df[vendor_df['invoice_frequency'] == 'low'].sample(1).iloc[0]
        invoice_amount = vendor['typical_amount_mean'] * np.random.uniform(2, 5)
        quantity = max(1, int(quantity * 1.5))
        unit_price = invoice_amount / quantity
        
    elif fraud_type == 'timing_fraud':
        # High-value transaction at unusual time
        invoice_amount = vendor['typical_amount_mean'] * np.random.uniform(2, 6)
        quantity = max(1, int(quantity * 1.5))
        unit_price = invoice_amount / quantity
    
    # Timestamp
    days_offset = int(np.random.randint(0, 180))
    
    if fraud_type == 'timing_fraud':
        # Off-hours: 0-5 AM or 10 PM - midnight
        hour = int(np.random.choice(list(range(0, 6)) + list(range(22, 24))))
    else:
        # Normal distribution but slightly more off-hours
        hour_probs = np.array([
            0.02, 0.02, 0.02, 0.02, 0.02, 0.03,
            0.04, 0.06, 0.08, 0.09, 0.10, 0.10,
            0.09, 0.09, 0.08, 0.07, 0.06, 0.04,
            0.03, 0.02, 0.02, 0.02, 0.02, 0.02
        ])
        hour_probs = hour_probs / hour_probs.sum()  # Normalize
        hour = int(np.random.choice(range(24), p=hour_probs))
    
    invoice_date = start_date + timedelta(days=days_offset, hours=hour)
    
    invoices.append({
        'InvoiceID': f'INV{n_normal + i + 1:06d}',
        'InvoiceDateTime': invoice_date,
        'VendorID': vendor['VendorID'],
        'ItemCode': item['ItemCode'],
        'Quantity': max(1, int(quantity)),
        'UnitPrice': round(max(0.01, unit_price), 2),
        'InvoiceAmount': round(max(0.01, invoice_amount), 2),
        'IsFraud': 1,
        'FraudType': fraud_type
    })

print(f"✓ Generated {n_fraud} fraud invoices")

# ============================================
# 5. CREATE DATAFRAME AND ADD VARIANCE
# ============================================
print("\n5. Finalizing dataset...")

df = pd.DataFrame(invoices)

# Shuffle the data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Recalculate InvoiceAmount to ensure consistency (add small rounding errors sometimes)
for idx in df.index:
    calculated = df.loc[idx, 'Quantity'] * df.loc[idx, 'UnitPrice']
    
    # 5% chance of small calculation "error" (another fraud signal)
    if random.random() < 0.05 and df.loc[idx, 'IsFraud'] == 1:
        df.loc[idx, 'InvoiceAmount'] = round(calculated * np.random.uniform(1.05, 1.15), 2)
    else:
        df.loc[idx, 'InvoiceAmount'] = round(calculated, 2)

# Sort by date
df = df.sort_values('InvoiceDateTime').reset_index(drop=True)

# ============================================
# 6. STATISTICS AND VALIDATION
# ============================================
print("\n" + "=" * 70)
print("DATASET STATISTICS")
print("=" * 70)

print(f"\nTotal Invoices: {len(df):,}")
print(f"Normal Invoices: {(df['IsFraud'] == 0).sum():,} ({(df['IsFraud'] == 0).sum()/len(df)*100:.1f}%)")
print(f"Fraud Invoices: {(df['IsFraud'] == 1).sum():,} ({(df['IsFraud'] == 1).sum()/len(df)*100:.1f}%)")

print(f"\nDate Range: {df['InvoiceDateTime'].min()} to {df['InvoiceDateTime'].max()}")
print(f"Unique Vendors: {df['VendorID'].nunique()}")
print(f"Unique Items: {df['ItemCode'].nunique()}")

print("\n--- Amount Statistics ---")
print(f"Normal Invoices - Mean: ${df[df['IsFraud']==0]['InvoiceAmount'].mean():.2f}, "
      f"Median: ${df[df['IsFraud']==0]['InvoiceAmount'].median():.2f}")
print(f"Fraud Invoices - Mean: ${df[df['IsFraud']==1]['InvoiceAmount'].mean():.2f}, "
      f"Median: ${df[df['IsFraud']==1]['InvoiceAmount'].median():.2f}")

print("\n--- Fraud Type Distribution ---")
fraud_dist = df[df['IsFraud'] == 1]['FraudType'].value_counts()
for fraud_type, count in fraud_dist.items():
    print(f"{fraud_type}: {count} ({count/len(df[df['IsFraud']==1])*100:.1f}%)")

print("\n--- Sample Records ---")
print("\nNormal Invoices (sample):")
print(df[df['IsFraud'] == 0][['InvoiceID', 'VendorID', 'ItemCode', 'Quantity', 'UnitPrice', 'InvoiceAmount']].head(3))

print("\nFraud Invoices (sample):")
print(df[df['IsFraud'] == 1][['InvoiceID', 'VendorID', 'ItemCode', 'Quantity', 'UnitPrice', 'InvoiceAmount', 'FraudType']].head(3))

# ============================================
# 7. SAVE TO CSV
# ============================================
output_file = 'synthetic_invoice_fraud_data.csv'
df.to_csv(output_file, index=False)

print("\n" + "=" * 70)
print(f"✓ DATASET SAVED: {output_file}")
print("=" * 70)

print(f"\nFile size: {len(df):,} rows × {len(df.columns)} columns")
print(f"Columns: {', '.join(df.columns)}")

print("\n" + "=" * 70)
print("READY FOR TRAINING!")
print("=" * 70)

print("\nNext steps:")
print("1. Train model: python train_pipeline_new.py --data synthetic_invoice_fraud_data.csv --has-labels")
print("2. The model will show actual precision/recall with ground truth")
print("3. Use the trained model in Streamlit app")

print("\n✓ Dataset generation complete!")

# ============================================
# 8. BONUS: Create test sets with different fraud rates
# ============================================
print("\n" + "=" * 70)
print("BONUS: Creating test sets with different fraud rates")
print("=" * 70)

# Test set with 5% fraud
test_5pct = df.sample(1000, random_state=123)
n_fraud_needed = 50
n_normal_needed = 950

test_5pct_df = pd.concat([
    df[df['IsFraud'] == 1].sample(n_fraud_needed, random_state=123),
    df[df['IsFraud'] == 0].sample(n_normal_needed, random_state=123)
]).sample(frac=1, random_state=123)

test_5pct_df.to_csv('test_data_5pct_fraud.csv', index=False)
print(f"✓ Created test_data_5pct_fraud.csv ({n_fraud_needed} frauds / 1000 = 5%)")

# Test set with 25% fraud
test_25pct = df.sample(1000, random_state=456)
n_fraud_needed = 250
n_normal_needed = 750

test_25pct_df = pd.concat([
    df[df['IsFraud'] == 1].sample(n_fraud_needed, random_state=456),
    df[df['IsFraud'] == 0].sample(n_normal_needed, random_state=456)
]).sample(frac=1, random_state=456)

test_25pct_df.to_csv('test_data_25pct_fraud.csv', index=False)
print(f"✓ Created test_data_25pct_fraud.csv ({n_fraud_needed} frauds / 1000 = 25%)")

print("\nYou can now test your model on datasets with different fraud rates!")
print("This demonstrates that the model works regardless of fraud percentage.")
