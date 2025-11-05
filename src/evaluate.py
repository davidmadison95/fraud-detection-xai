"""
Model Evaluation Module
Comprehensive evaluation and performance analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
import os

sns.set_style("whitegrid")


def plot_confusion_matrix(y_true, y_pred, output_path=None):
    """
    Plot confusion matrix heatmap.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_pred : array-like
        Predicted labels
    output_path : str, optional
        Path to save plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {output_path}")
    
    plt.show()


def plot_roc_curve(y_true, y_proba, output_path=None):
    """
    Plot ROC curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    output_path : str, optional
        Path to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 7))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ ROC curve saved to: {output_path}")
    
    plt.show()


def plot_precision_recall_curve(y_true, y_proba, output_path=None):
    """
    Plot Precision-Recall curve.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    output_path : str, optional
        Path to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, color='darkgreen', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    
    # Baseline (random classifier)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], color='navy', lw=2, 
             linestyle='--', label=f'Baseline (Fraud Rate = {baseline:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left", fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Precision-Recall curve saved to: {output_path}")
    
    plt.show()


def plot_fraud_score_distribution(y_true, y_proba, output_path=None):
    """
    Plot distribution of fraud scores by class.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    output_path : str, optional
        Path to save plot
    """
    plt.figure(figsize=(12, 6))
    
    # Legitimate transactions
    plt.hist(y_proba[y_true == 0], bins=50, alpha=0.6, color='blue',
             label='Legitimate', density=True)
    
    # Fraudulent transactions
    plt.hist(y_proba[y_true == 1], bins=50, alpha=0.6, color='red',
             label='Fraud', density=True)
    
    plt.xlabel('Fraud Probability Score', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Distribution of Fraud Scores by True Class', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='upper center', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Score distribution saved to: {output_path}")
    
    plt.show()


def plot_feature_importance(feature_importance_df, top_n=15, output_path=None):
    """
    Plot feature importance bar chart.
    
    Parameters:
    -----------
    feature_importance_df : pd.DataFrame
        DataFrame with 'feature' and 'importance' columns
    top_n : int
        Number of top features to display
    output_path : str, optional
        Path to save plot
    """
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Feature Importance', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Feature importance plot saved to: {output_path}")
    
    plt.show()


def plot_threshold_analysis(y_true, y_proba, output_path=None):
    """
    Analyze model performance across different thresholds.
    
    Parameters:
    -----------
    y_true : array-like
        True labels
    y_proba : array-like
        Predicted probabilities
    output_path : str, optional
        Path to save plot
    """
    thresholds = np.linspace(0, 1, 100)
    precision_list = []
    recall_list = []
    f1_list = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
    
    plt.figure(figsize=(12, 7))
    plt.plot(thresholds, precision_list, label='Precision', linewidth=2)
    plt.plot(thresholds, recall_list, label='Recall', linewidth=2)
    plt.plot(thresholds, f1_list, label='F1 Score', linewidth=2)
    
    # Mark optimal F1 threshold
    optimal_idx = np.argmax(f1_list)
    optimal_threshold = thresholds[optimal_idx]
    plt.axvline(x=optimal_threshold, color='red', linestyle='--', 
                label=f'Optimal Threshold ({optimal_threshold:.3f})')
    
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.title('Model Performance vs. Classification Threshold', 
              fontsize=16, fontweight='bold')
    plt.legend(loc='best', fontsize=11)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Threshold analysis saved to: {output_path}")
    
    plt.show()
    
    return optimal_threshold


def generate_evaluation_report(model, X_test, y_test, output_dir='reports'):
    """
    Generate comprehensive evaluation report with visualizations.
    
    Parameters:
    -----------
    model : FraudDetectionModel
        Trained model
    X_test : array-like
        Test features
    y_test : array-like
        Test labels
    output_dir : str
        Directory to save reports
    """
    print("="*60)
    print("GENERATING EVALUATION REPORT")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get predictions
    y_proba = model.predict_proba(X_test)
    y_pred = model.predict(X_test, threshold=0.5)
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(y_test, y_pred, 
                         os.path.join(output_dir, 'confusion_matrix.png'))
    
    # Plot ROC curve
    print("Generating ROC curve...")
    plot_roc_curve(y_test, y_proba,
                  os.path.join(output_dir, 'roc_curve.png'))
    
    # Plot Precision-Recall curve
    print("Generating Precision-Recall curve...")
    plot_precision_recall_curve(y_test, y_proba,
                               os.path.join(output_dir, 'pr_curve.png'))
    
    # Plot score distribution
    print("Generating score distribution...")
    plot_fraud_score_distribution(y_test, y_proba,
                                 os.path.join(output_dir, 'score_distribution.png'))
    
    # Plot feature importance
    if model.feature_importance is not None:
        print("Generating feature importance plot...")
        plot_feature_importance(model.feature_importance,
                              output_path=os.path.join(output_dir, 'feature_importance.png'))
    
    # Plot threshold analysis
    print("Generating threshold analysis...")
    optimal_threshold = plot_threshold_analysis(y_test, y_proba,
                                               os.path.join(output_dir, 'threshold_analysis.png'))
    
    print(f"\n✓ All evaluation reports saved to: {output_dir}")
    print(f"✓ Optimal threshold for F1: {optimal_threshold:.3f}")
    
    return optimal_threshold


if __name__ == "__main__":
    from train import FraudDetectionModel
    import joblib
    
    print("Loading model and test data...")
    
    # Load model
    model = FraudDetectionModel()
    model.load('models/fraud_model.pkl', 'models/preprocessor.pkl')
    
    # Load test data
    df = pd.read_csv('data/raw/transactions.csv')
    from features import prepare_features
    X, y = prepare_features(df)
    
    # Transform features
    X_transformed = model.feature_engine.transform(X)
    
    # Generate report
    generate_evaluation_report(model, X_transformed, y, output_dir='reports')
