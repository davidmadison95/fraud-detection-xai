#!/usr/bin/env python
"""
Quick Model Training Script
Trains the fraud detection model without SHAP explainer.
"""

import sys
sys.path.append('/home/claude/fraud-xai/src')

import xgboost as xgb
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd
from features import create_preprocessor, get_feature_columns

def quick_train():
    print("=" * 70)
    print("QUICK MODEL TRAINING")
    print("=" * 70)
    
    # Load data
    print("\n[1/4] Loading data...")
    df = pd.read_csv('/home/claude/fraud-xai/data/raw/transactions.csv')
    
    # Get features
    numeric_features, categorical_features = get_feature_columns()
    feature_columns = numeric_features + categorical_features
    
    X = df[feature_columns]
    y = df['is_fraud']
    
    print(f"✓ Loaded {len(df):,} transactions")
    print(f"✓ Fraud rate: {y.mean():.2%}")
    
    # Split data
    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Train: {len(X_train):,} samples")
    print(f"✓ Test: {len(X_test):,} samples")
    
    # Preprocess
    print("\n[3/4] Preprocessing...")
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    print(f"✓ Features after preprocessing: {X_train_processed.shape[1]}")
    
    # Train model
    print("\n[4/4] Training XGBoost model...")
    
    fraud_count = y_train.sum()
    normal_count = len(y_train) - fraud_count
    scale_pos_weight = normal_count / fraud_count
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='aucpr',
        use_label_encoder=False
    )
    
    model.fit(X_train_processed, y_train, verbose=False)
    
    # Evaluate
    from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
    y_pred_proba = model.predict_proba(X_test_processed)[:, 1]
    
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC:  {pr_auc:.4f}")
    
    # Save
    print("\n[5/5] Saving model and preprocessor...")
    joblib.dump(model, '/home/claude/fraud-xai/models/fraud_model.pkl')
    joblib.dump(preprocessor, '/home/claude/fraud-xai/models/preprocessor.pkl')
    
    print("✓ Model saved to /home/claude/fraud-xai/models/fraud_model.pkl")
    print("✓ Preprocessor saved to /home/claude/fraud-xai/models/preprocessor.pkl")
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    
    return model, preprocessor

if __name__ == "__main__":
    quick_train()
