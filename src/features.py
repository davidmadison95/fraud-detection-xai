"""
Feature Engineering Module
Handles feature transformations and preprocessing for fraud detection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib


class FraudFeatureEngine:
    """
    Feature engineering pipeline for fraud detection.
    Handles categorical encoding and numerical scaling.
    """
    
    def __init__(self):
        """Initialize feature engineering pipeline."""
        self.preprocessor = None
        self.feature_names = None
        
        # Define feature groups
        self.categorical_features = [
            'merchant_category',
            'merchant_country', 
            'channel'
        ]
        
        self.numerical_features = [
            'amount_usd',
            'customer_age_days',
            'customer_txn_30d',
            'avg_amount_30d',
            'std_amount_30d',
            'card_present',
            'country_mismatch',
            'hour_of_day',
            'is_weekend'
        ]
    
    def create_preprocessor(self):
        """
        Create sklearn preprocessing pipeline.
        
        Returns:
        --------
        ColumnTransformer : Preprocessing pipeline
        """
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), 
                 self.categorical_features)
            ],
            remainder='drop'
        )
        
        self.preprocessor = preprocessor
        return preprocessor
    
    def fit_transform(self, df):
        """
        Fit preprocessor and transform data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with raw features
        
        Returns:
        --------
        np.ndarray : Transformed feature matrix
        """
        if self.preprocessor is None:
            self.create_preprocessor()
        
        X_transformed = self.preprocessor.fit_transform(df)
        
        # Store feature names
        self.feature_names = self._get_feature_names()
        
        return X_transformed
    
    def transform(self, df):
        """
        Transform data using fitted preprocessor.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with raw features
        
        Returns:
        --------
        np.ndarray : Transformed feature matrix
        """
        if self.preprocessor is None:
            raise ValueError("Preprocessor not fitted. Call fit_transform first.")
        
        return self.preprocessor.transform(df)
    
    def _get_feature_names(self):
        """Extract feature names after transformation."""
        feature_names = []
        
        # Numerical features
        feature_names.extend(self.numerical_features)
        
        # Categorical features (one-hot encoded)
        cat_encoder = self.preprocessor.named_transformers_['cat']
        for i, cat_feature in enumerate(self.categorical_features):
            categories = cat_encoder.categories_[i][1:]  # Skip first (dropped)
            feature_names.extend([f"{cat_feature}_{cat}" for cat in categories])
        
        return feature_names
    
    def save(self, filepath):
        """Save preprocessor to disk."""
        joblib.dump(self.preprocessor, filepath)
        print(f"✓ Preprocessor saved to: {filepath}")
    
    def load(self, filepath):
        """Load preprocessor from disk."""
        self.preprocessor = joblib.load(filepath)
        self.feature_names = self._get_feature_names()
        print(f"✓ Preprocessor loaded from: {filepath}")


def engineer_features(df):
    """
    Create additional engineered features.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    
    Returns:
    --------
    pd.DataFrame : Dataframe with additional features
    """
    df = df.copy()
    
    # Amount deviation from personal average
    df['amount_vs_avg'] = (df['amount_usd'] - df['avg_amount_30d']) / (df['std_amount_30d'] + 1)
    
    # High amount flag
    df['high_amount_flag'] = (df['amount_usd'] > df['avg_amount_30d'] + 2 * df['std_amount_30d']).astype(int)
    
    # Unusual time flag (late night / early morning)
    df['unusual_time'] = ((df['hour_of_day'] < 6) | (df['hour_of_day'] > 22)).astype(int)
    
    # New customer flag
    df['new_customer'] = (df['customer_age_days'] < 90).astype(int)
    
    # Low transaction history flag
    df['low_txn_history'] = (df['customer_txn_30d'] < 3).astype(int)
    
    return df


def prepare_features(df, target_col='is_fraud'):
    """
    Prepare features and target for modeling.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    target_col : str
        Name of target column
    
    Returns:
    --------
    tuple : (X, y, feature_names)
    """
    # Separate features and target
    X = df.drop(columns=[target_col, 'transaction_id', 'timestamp', 'customer_id'], errors='ignore')
    y = df[target_col] if target_col in df.columns else None
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("="*60)
    
    # Load sample data
    df = pd.read_csv('data/raw/transactions.csv')
    print(f"Loaded {len(df):,} transactions")
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Initialize feature engine
    feature_engine = FraudFeatureEngine()
    
    # Fit and transform
    X_transformed = feature_engine.fit_transform(X)
    
    print(f"\n✓ Features transformed")
    print(f"✓ Shape: {X_transformed.shape}")
    print(f"✓ Number of features: {len(feature_engine.feature_names)}")
    print(f"\nFeature names:")
    for i, name in enumerate(feature_engine.feature_names[:10]):
        print(f"  {i+1}. {name}")
    print("  ...")
