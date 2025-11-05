"""
Model Training Module
Trains XGBoost fraud detection model with imbalanced data handling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    precision_recall_curve, f1_score, classification_report
)
from xgboost import XGBClassifier
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class FraudDetectionModel:
    """
    XGBoost-based fraud detection model with imbalanced data handling.
    """
    
    def __init__(self, params=None, random_state=42):
        """
        Initialize fraud detection model.
        
        Parameters:
        -----------
        params : dict
            XGBoost parameters (optional)
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        
        # Default parameters optimized for fraud detection
        default_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 3,
            'gamma': 0.1,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'scale_pos_weight': 60,  # Adjust for ~1.5% fraud rate
            'objective': 'binary:logistic',
            'eval_metric': 'aucpr',
            'random_state': random_state,
            'n_jobs': -1
        }
        
        self.params = params if params is not None else default_params
        self.model = XGBClassifier(**self.params)
        self.metrics = {}
        self.feature_importance = None
        
    def train(self, X_train, y_train, X_val=None, y_val=None, verbose=True):
        """
        Train the fraud detection model.
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training labels
        X_val : array-like
            Validation features (optional)
        y_val : array-like
            Validation labels (optional)
        verbose : bool
            Print training progress
            
        Returns:
        --------
        self
        """
        print("🚀 Training fraud detection model...")
        print(f"Training samples: {len(X_train):,} | Fraud rate: {y_train.mean()*100:.2f}%")
        
        # Prepare eval set if validation data provided
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
            print(f"Validation samples: {len(X_val):,} | Fraud rate: {y_val.mean()*100:.2f}%")
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=verbose
        )
        
        print("✅ Model training completed!")
        
        # Store feature importance
        self.feature_importance = self.model.feature_importances_
        
        return self
    
    def predict_proba(self, X):
        """
        Predict fraud probabilities.
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        np.ndarray
            Fraud probabilities
        """
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=0.5):
        """
        Predict fraud labels using threshold.
        
        Parameters:
        -----------
        X : array-like
            Features
        threshold : float
            Classification threshold
            
        Returns:
        --------
        np.ndarray
            Binary predictions
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def evaluate(self, X_test, y_test, feature_names=None):
        """
        Comprehensive model evaluation.
        
        Parameters:
        -----------
        X_test : array-like
            Test features
        y_test : array-like
            Test labels
        feature_names : list
            Feature names for importance ranking
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        print("\n📊 Evaluating model performance...")
        
        # Get predictions
        y_proba = self.predict_proba(X_test)
        y_pred = self.predict(X_test, threshold=0.5)
        
        # Calculate metrics
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        # Precision and Recall at top 1%
        top_1_percent_idx = int(len(y_test) * 0.01)
        sorted_idx = np.argsort(y_proba)[::-1][:top_1_percent_idx]
        
        if len(sorted_idx) > 0:
            precision_at_1 = y_test.iloc[sorted_idx].mean() if isinstance(y_test, pd.Series) else y_test[sorted_idx].mean()
            recall_at_1 = y_test.iloc[sorted_idx].sum() / y_test.sum() if isinstance(y_test, pd.Series) else y_test[sorted_idx].sum() / y_test.sum()
        else:
            precision_at_1 = 0
            recall_at_1 = 0
            
        metrics['precision_at_1pct'] = precision_at_1
        metrics['recall_at_1pct'] = recall_at_1
        
        self.metrics = metrics
        
        # Print results
        print("\n" + "="*50)
        print("🎯 MODEL PERFORMANCE METRICS")
        print("="*50)
        print(f"ROC-AUC:              {metrics['roc_auc']:.4f}")
        print(f"PR-AUC:               {metrics['pr_auc']:.4f}")
        print(f"F1 Score:             {metrics['f1_score']:.4f}")
        print(f"Precision @ Top 1%:   {metrics['precision_at_1pct']:.4f}")
        print(f"Recall @ Top 1%:      {metrics['recall_at_1pct']:.4f}")
        print("="*50)
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))
        
        # Feature importance
        if feature_names is not None and self.feature_importance is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': self.feature_importance
            }).sort_values('importance', ascending=False)
            
            print("\n🔝 Top 10 Most Important Features:")
            print(importance_df.head(10).to_string(index=False))
            
        return metrics
    
    def cross_validate(self, X, y, cv=5, feature_names=None):
        """
        Perform stratified cross-validation.
        
        Parameters:
        -----------
        X : array-like
            Features
        y : array-like
            Labels
        cv : int
            Number of folds
        feature_names : list
            Feature names
            
        Returns:
        --------
        dict
            Cross-validation scores
        """
        print(f"\n🔄 Performing {cv}-fold stratified cross-validation...")
        
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        scoring = {
            'roc_auc': 'roc_auc',
            'pr_auc': 'average_precision',
            'f1': 'f1'
        }
        
        cv_results = cross_validate(
            self.model, X, y,
            cv=skf,
            scoring=scoring,
            n_jobs=-1,
            verbose=0
        )
        
        # Print results
        print("\n" + "="*50)
        print("📊 CROSS-VALIDATION RESULTS")
        print("="*50)
        print(f"ROC-AUC:  {cv_results['test_roc_auc'].mean():.4f} (± {cv_results['test_roc_auc'].std():.4f})")
        print(f"PR-AUC:   {cv_results['test_pr_auc'].mean():.4f} (± {cv_results['test_pr_auc'].std():.4f})")
        print(f"F1 Score: {cv_results['test_f1'].mean():.4f} (± {cv_results['test_f1'].std():.4f})")
        print("="*50)
        
        return cv_results
    
    def save(self, filepath):
        """Save model to disk."""
        joblib.dump(self.model, filepath)
        print(f"✅ Model saved to {filepath}")
        
        # Save metrics if available
        if self.metrics:
            metrics_path = filepath.replace('.pkl', '_metrics.json')
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics, f, indent=4)
            print(f"✅ Metrics saved to {metrics_path}")
    
    def load(self, filepath):
        """Load model from disk."""
        self.model = joblib.load(filepath)
        self.feature_importance = self.model.feature_importances_
        print(f"✅ Model loaded from {filepath}")
        
        # Load metrics if available
        metrics_path = filepath.replace('.pkl', '_metrics.json')
        try:
            with open(metrics_path, 'r') as f:
                self.metrics = json.load(f)
            print(f"✅ Metrics loaded from {metrics_path}")
        except FileNotFoundError:
            print("⚠️  Metrics file not found.")


def train_fraud_model(X_train, y_train, X_test, y_test, feature_names=None, save_dir='../models/'):
    """
    Complete training pipeline for fraud detection model.
    
    Parameters:
    -----------
    X_train, X_test : array-like
        Training and test features
    y_train, y_test : array-like
        Training and test labels
    feature_names : list
        Names of features
    save_dir : str
        Directory to save model
        
    Returns:
    --------
    FraudDetectionModel
        Trained model
    """
    # Initialize model
    model = FraudDetectionModel()
    
    # Train
    model.train(X_train, y_train, X_test, y_test, verbose=False)
    
    # Evaluate
    metrics = model.evaluate(X_test, y_test, feature_names)
    
    # Save
    import os
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, 'fraud_model.pkl'))
    
    return model


if __name__ == "__main__":
    print("Model training module loaded successfully.")
    print("\nKey components:")
    print("  - FraudDetectionModel: Main model class")
    print("  - train_fraud_model: Complete training pipeline")
