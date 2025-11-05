"""
Complete Training Pipeline
End-to-end pipeline for fraud detection model training.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.generate_data import synth_transactions
from src.features import FraudFeatureEngineer, engineer_derived_features
from src.train import FraudDetectionModel
from src.explain import FraudExplainer, generate_all_explanations
from sklearn.model_selection import train_test_split


def run_complete_pipeline():
    """
    Execute complete fraud detection pipeline:
    1. Generate synthetic data
    2. Engineer features
    3. Train model
    4. Evaluate performance
    5. Generate explainability visualizations
    """
    
    print("="*70)
    print("FINANCIAL FRAUD DETECTION - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Data Generation
    print("\n📊 STEP 1: Generating Synthetic Transaction Data")
    print("-" * 70)
    
    # Check if data exists
    data_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'transactions.csv')
    
    if os.path.exists(data_path):
        print(f"Loading existing dataset from {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Generating new synthetic dataset...")
        df = synth_transactions(n_samples=100000, fraud_rate=0.015, random_state=42)
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"✅ Dataset saved to {data_path}")
    
    print(f"\nDataset shape: {df.shape}")
    print(f"Fraud rate: {df['is_fraud'].mean()*100:.2f}%")
    
    # Step 2: Feature Engineering
    print("\n🔧 STEP 2: Feature Engineering")
    print("-" * 70)
    
    # Add derived features
    df_enhanced = engineer_derived_features(df)
    
    # Split data
    feature_cols = [col for col in df_enhanced.columns 
                   if col not in ['transaction_id', 'timestamp', 'customer_id', 'is_fraud']]
    
    X = df_enhanced[feature_cols]
    y = df_enhanced['is_fraud']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {len(X_train):,} samples")
    print(f"Test set: {len(X_test):,} samples")
    
    # Preprocess features
    feature_engineer = FraudFeatureEngineer()
    X_train_processed, _ = feature_engineer.fit_transform(
        pd.concat([X_train, y_train], axis=1), 
        target_col='is_fraud'
    )
    X_test_processed = feature_engineer.transform(X_test)
    
    print(f"✅ Features engineered: {X_train_processed.shape[1]} features")
    
    # Save preprocessor
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    feature_engineer.save(os.path.join(models_dir, 'preprocessor.pkl'))
    
    # Step 3: Model Training
    print("\n🤖 STEP 3: Training XGBoost Model")
    print("-" * 70)
    
    model = FraudDetectionModel(random_state=42)
    model.train(X_train_processed, y_train, X_test_processed, y_test, verbose=False)
    
    # Step 4: Model Evaluation
    print("\n📈 STEP 4: Model Evaluation")
    print("-" * 70)
    
    metrics = model.evaluate(
        X_test_processed, 
        y_test,
        feature_names=feature_engineer.get_feature_names()
    )
    
    # Save model
    model.save(os.path.join(models_dir, 'fraud_model.pkl'))
    
    # Step 5: Explainability
    print("\n🔍 STEP 5: Generating SHAP Explanations")
    print("-" * 70)
    
    # Create explainer
    explainer = FraudExplainer(
        model.model,
        X_train_processed[:100],  # Background sample
        feature_engineer.get_feature_names()
    )
    
    # Generate all visualizations
    reports_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    generate_all_explanations(explainer, X_test_processed, output_dir=reports_dir)
    
    # Step 6: Summary
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Generated Artifacts:")
    print(f"  - Dataset: {data_path}")
    print(f"  - Model: {os.path.join(models_dir, 'fraud_model.pkl')}")
    print(f"  - Preprocessor: {os.path.join(models_dir, 'preprocessor.pkl')}")
    print(f"  - SHAP Visualizations: {reports_dir}/")
    
    print("\n🎯 Model Performance Summary:")
    print(f"  - ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"  - Recall @ Top 1%: {metrics['recall_at_1pct']:.4f}")
    
    print("\n🚀 Next Steps:")
    print("  1. Launch Streamlit Dashboard:")
    print("     cd src && streamlit run app_dashboard.py")
    print("\n  2. Start Flask API:")
    print("     cd src && python serve_api.py")
    print("\n  3. Explore notebooks in notebooks/ directory")
    
    print("\n" + "="*70)
    
    return {
        'model': model,
        'preprocessor': feature_engineer,
        'explainer': explainer,
        'metrics': metrics
    }


if __name__ == "__main__":
    results = run_complete_pipeline()
