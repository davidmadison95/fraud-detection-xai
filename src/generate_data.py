"""
Synthetic Financial Transaction Data Generator
Generates realistic transaction data with fraud patterns for model training.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def synth_transactions(n_samples=100000, fraud_rate=0.015, random_state=42):
    """
    Generate synthetic financial transaction data with fraud patterns.
    
    Parameters:
    -----------
    n_samples : int
        Total number of transactions to generate
    fraud_rate : float
        Proportion of fraudulent transactions (default: 1.5%)
    random_state : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        Synthetic transaction dataset
    """
    np.random.seed(random_state)
    
    # Calculate fraud and normal counts
    n_fraud = int(n_samples * fraud_rate)
    n_normal = n_samples - n_fraud
    
    print(f"Generating {n_samples:,} transactions...")
    print(f"  - Normal: {n_normal:,} ({(1-fraud_rate)*100:.1f}%)")
    print(f"  - Fraud: {n_fraud:,} ({fraud_rate*100:.1f}%)")
    
    # Generate base timestamp
    start_date = datetime(2023, 1, 1)
    
    # Generate normal transactions
    normal_data = generate_normal_transactions(n_normal, start_date)
    
    # Generate fraudulent transactions
    fraud_data = generate_fraud_transactions(n_fraud, start_date)
    
    # Combine and shuffle
    df = pd.concat([normal_data, fraud_data], ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    
    # Add transaction IDs
    df.insert(0, 'transaction_id', [f'TXN_{i:08d}' for i in range(len(df))])
    
    print(f"\nDataset created successfully!")
    print(f"Shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    return df


def generate_normal_transactions(n, start_date):
    """Generate normal transaction patterns."""
    
    # Timestamp - business hours weighted
    hours = np.random.choice(range(24), n, p=get_hour_distribution())
    days = np.random.randint(0, 365, n)
    timestamps = [start_date + timedelta(days=int(d), hours=int(h), 
                  minutes=np.random.randint(0, 60)) for d, h in zip(days, hours)]
    
    # Amount - log-normal distribution (most transactions are small)
    amounts = np.random.lognormal(mean=3.5, sigma=1.2, size=n)
    amounts = np.clip(amounts, 5, 5000)
    
    # Merchant categories
    categories = np.random.choice(
        ['grocery', 'restaurant', 'gas_station', 'retail', 'pharmacy', 
         'entertainment', 'electronics', 'travel', 'utilities', 'other'],
        n,
        p=[0.25, 0.20, 0.15, 0.12, 0.08, 0.07, 0.05, 0.04, 0.02, 0.02]
    )
    
    # Merchant country - mostly domestic
    countries = np.random.choice(
        ['US', 'CA', 'GB', 'MX', 'other'],
        n,
        p=[0.85, 0.06, 0.04, 0.03, 0.02]
    )
    
    # Channel
    channels = np.random.choice(
        ['online', 'in_store', 'mobile', 'atm'],
        n,
        p=[0.45, 0.40, 0.13, 0.02]
    )
    
    # Card present (higher for in-store)
    card_present = (channels == 'in_store') | ((channels == 'atm') & (np.random.rand(n) > 0.1))
    card_present = card_present | ((channels == 'online') & (np.random.rand(n) < 0.05))
    
    # Customer features
    n_customers = n // 50  # Average 50 transactions per customer
    customer_ids = np.random.randint(1000000, 9999999, n_customers)
    customer_id = np.random.choice(customer_ids, n)
    
    # Customer age (days since account creation)
    customer_age_days = np.random.gamma(shape=2, scale=200, size=n).astype(int)
    customer_age_days = np.clip(customer_age_days, 30, 3650)
    
    # Transaction history features
    customer_txn_30d = np.random.poisson(lam=12, size=n) + 1
    avg_amount_30d = amounts * np.random.uniform(0.8, 1.2, n)
    std_amount_30d = avg_amount_30d * np.random.uniform(0.2, 0.6, n)
    
    # Country mismatch (rare for normal transactions)
    country_mismatch = np.random.rand(n) < 0.02
    
    # Time features
    hour_of_day = np.array([ts.hour for ts in timestamps])
    is_weekend = np.array([ts.weekday() >= 5 for ts in timestamps])
    
    # Label
    is_fraud = np.zeros(n, dtype=int)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'amount_usd': np.round(amounts, 2),
        'merchant_category': categories,
        'merchant_country': countries,
        'channel': channels,
        'card_present': card_present.astype(int),
        'customer_id': customer_id,
        'customer_age_days': customer_age_days,
        'customer_txn_30d': customer_txn_30d,
        'avg_amount_30d': np.round(avg_amount_30d, 2),
        'std_amount_30d': np.round(std_amount_30d, 2),
        'country_mismatch': country_mismatch.astype(int),
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend.astype(int),
        'is_fraud': is_fraud
    })
    
    return df


def generate_fraud_transactions(n, start_date):
    """Generate fraudulent transaction patterns with anomalies."""
    
    # Timestamp - fraud occurs more at night
    hours = np.random.choice(range(24), n, p=get_fraud_hour_distribution())
    days = np.random.randint(0, 365, n)
    timestamps = [start_date + timedelta(days=int(d), hours=int(h), 
                  minutes=np.random.randint(0, 60)) for d, h in zip(days, hours)]
    
    # Amount - higher amounts, different distribution
    amounts = np.random.lognormal(mean=4.5, sigma=1.0, size=n)
    amounts = np.clip(amounts, 50, 10000)
    
    # Merchant categories - fraud prefers high-value categories
    categories = np.random.choice(
        ['electronics', 'jewelry', 'travel', 'retail', 'entertainment', 
         'online_marketplace', 'luxury', 'other'],
        n,
        p=[0.30, 0.20, 0.15, 0.12, 0.10, 0.08, 0.03, 0.02]
    )
    
    # Merchant country - more international for fraud
    countries = np.random.choice(
        ['US', 'CN', 'RU', 'NG', 'BR', 'other'],
        n,
        p=[0.40, 0.15, 0.15, 0.10, 0.10, 0.10]
    )
    
    # Channel - mostly online
    channels = np.random.choice(
        ['online', 'mobile', 'in_store', 'atm'],
        n,
        p=[0.70, 0.20, 0.08, 0.02]
    )
    
    # Card present - usually not present for fraud
    card_present = np.random.rand(n) < 0.05
    
    # Customer features - stolen/compromised accounts
    n_customers = n // 3  # Fewer customers, more txns per compromised account
    customer_ids = np.random.randint(1000000, 9999999, n_customers)
    customer_id = np.random.choice(customer_ids, n)
    
    # Customer age - targeting older, established accounts
    customer_age_days = np.random.gamma(shape=3, scale=300, size=n).astype(int)
    customer_age_days = np.clip(customer_age_days, 90, 3650)
    
    # Transaction history - sudden spike (fraud burst pattern)
    customer_txn_30d = np.random.poisson(lam=8, size=n) + 1
    
    # Amount deviation - fraud amounts differ from historical patterns
    avg_amount_30d = amounts * np.random.uniform(0.3, 0.7, n)  # Historical avg is lower
    std_amount_30d = avg_amount_30d * np.random.uniform(0.8, 1.5, n)  # Higher variance
    
    # Country mismatch - common for fraud
    country_mismatch = np.random.rand(n) < 0.35
    
    # Time features
    hour_of_day = np.array([ts.hour for ts in timestamps])
    is_weekend = np.array([ts.weekday() >= 5 for ts in timestamps])
    
    # Label
    is_fraud = np.ones(n, dtype=int)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'amount_usd': np.round(amounts, 2),
        'merchant_category': categories,
        'merchant_country': countries,
        'channel': channels,
        'card_present': card_present.astype(int),
        'customer_id': customer_id,
        'customer_age_days': customer_age_days,
        'customer_txn_30d': customer_txn_30d,
        'avg_amount_30d': np.round(avg_amount_30d, 2),
        'std_amount_30d': np.round(std_amount_30d, 2),
        'country_mismatch': country_mismatch.astype(int),
        'hour_of_day': hour_of_day,
        'is_weekend': is_weekend.astype(int),
        'is_fraud': is_fraud
    })
    
    return df


def get_hour_distribution():
    """Hourly distribution for normal transactions (business hours peak)."""
    probs = np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.02,  # 0-5 AM
                      0.03, 0.05, 0.06, 0.07, 0.08, 0.08,  # 6-11 AM
                      0.07, 0.06, 0.05, 0.05, 0.05, 0.06,  # 12-5 PM
                      0.07, 0.06, 0.04, 0.03, 0.02, 0.01]) # 6-11 PM
    return probs / probs.sum()


def get_fraud_hour_distribution():
    """Hourly distribution for fraud transactions (late night peak)."""
    probs = np.array([0.06, 0.07, 0.08, 0.07, 0.05, 0.04,  # 0-5 AM (night fraud)
                      0.03, 0.02, 0.03, 0.04, 0.05, 0.05,  # 6-11 AM
                      0.05, 0.05, 0.04, 0.04, 0.04, 0.04,  # 12-5 PM
                      0.04, 0.04, 0.05, 0.06, 0.07, 0.06]) # 6-11 PM
    return probs / probs.sum()


if __name__ == "__main__":
    # Generate dataset
    df = synth_transactions(n_samples=100000, fraud_rate=0.015, random_state=42)
    
    # Save to CSV
    output_path = '../data/raw/transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✅ Dataset saved to: {output_path}")
    
    # Display sample
    print("\n📊 Sample transactions:")
    print(df.head(10))
    
    print("\n📈 Dataset statistics:")
    print(df.describe())
    
    print("\n🏷️ Fraud distribution:")
    print(df['is_fraud'].value_counts())
