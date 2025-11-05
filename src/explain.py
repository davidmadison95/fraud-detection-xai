"""
SHAP Explainability Module
Provides global and local explanations using SHAP values.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os


class FraudExplainer:
    """
    SHAP-based explainability for fraud detection model.
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize explainer.
        
        Parameters:
        -----------
        model : trained model
            Model with predict_proba method
        feature_names : list, optional
            Feature names for visualization
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
    
    def create_explainer(self, X_background, max_samples=100):
        """
        Create SHAP explainer using background data.
        
        Parameters:
        -----------
        X_background : array-like
            Background dataset for SHAP
        max_samples : int
            Maximum background samples (for speed)
        """
        print("Creating SHAP explainer...")
        
        # Subsample background data if needed
        if len(X_background) > max_samples:
            indices = np.random.choice(len(X_background), max_samples, replace=False)
            X_background = X_background[indices]
        
        # Create TreeExplainer (optimized for tree models)
        self.explainer = shap.TreeExplainer(self.model)
        
        print(f"✓ Explainer created with {len(X_background)} background samples")
    
    def compute_shap_values(self, X):
        """
        Compute SHAP values for input data.
        
        Parameters:
        -----------
        X : array-like
            Input features
        
        Returns:
        --------
        np.ndarray : SHAP values
        """
        if self.explainer is None:
            raise ValueError("Explainer not created. Call create_explainer first.")
        
        print(f"Computing SHAP values for {len(X)} samples...")
        self.shap_values = self.explainer.shap_values(X)
        print("✓ SHAP values computed")
        
        return self.shap_values
    
    def plot_summary(self, X, shap_values=None, max_display=10, output_path=None):
        """
        Create SHAP summary plot (global feature importance).
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        shap_values : array-like, optional
            Pre-computed SHAP values
        max_display : int
            Maximum features to display
        output_path : str, optional
            Path to save plot
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Convert to DataFrame if feature names available
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, 
            X_df,
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Feature Importance Summary', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP summary plot saved to: {output_path}")
        
        plt.show()
    
    def plot_beeswarm(self, X, shap_values=None, max_display=10, output_path=None):
        """
        Create SHAP beeswarm plot.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        shap_values : array-like, optional
            Pre-computed SHAP values
        max_display : int
            Maximum features to display
        output_path : str, optional
            Path to save plot
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Convert to DataFrame if feature names available
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        plt.figure(figsize=(10, 8))
        shap.plots.beeswarm(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                data=X_df.values if isinstance(X_df, pd.DataFrame) else X_df,
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.title('SHAP Beeswarm Plot', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP beeswarm plot saved to: {output_path}")
        
        plt.show()
    
    def plot_waterfall(self, X, index=0, shap_values=None, output_path=None):
        """
        Create SHAP waterfall plot for a single prediction.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        index : int
            Index of sample to explain
        shap_values : array-like, optional
            Pre-computed SHAP values
        output_path : str, optional
            Path to save plot
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Convert to DataFrame if feature names available
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        plt.figure(figsize=(10, 8))
        shap.plots.waterfall(
            shap.Explanation(
                values=shap_values[index],
                base_values=self.explainer.expected_value,
                data=X_df.iloc[index].values if isinstance(X_df, pd.DataFrame) else X_df[index],
                feature_names=self.feature_names
            ),
            show=False
        )
        plt.title(f'SHAP Waterfall Plot - Transaction {index}', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP waterfall plot saved to: {output_path}")
        
        plt.show()
    
    def plot_force(self, X, index=0, shap_values=None, output_path=None):
        """
        Create SHAP force plot for a single prediction.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        index : int
            Index of sample to explain
        shap_values : array-like, optional
            Pre-computed SHAP values
        output_path : str, optional
            Path to save plot (as HTML)
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Convert to DataFrame if feature names available
        if self.feature_names is not None and not isinstance(X, pd.DataFrame):
            X_df = pd.DataFrame(X, columns=self.feature_names)
        else:
            X_df = X
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            shap_values[index],
            X_df.iloc[index] if isinstance(X_df, pd.DataFrame) else X_df[index],
            feature_names=self.feature_names,
            matplotlib=False
        )
        
        if output_path:
            shap.save_html(output_path, force_plot)
            print(f"✓ SHAP force plot saved to: {output_path}")
        
        return force_plot
    
    def plot_bar(self, X, shap_values=None, max_display=10, output_path=None):
        """
        Create SHAP bar plot (mean absolute SHAP values).
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        shap_values : array-like, optional
            Pre-computed SHAP values
        max_display : int
            Maximum features to display
        output_path : str, optional
            Path to save plot
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        plt.figure(figsize=(10, 8))
        shap.plots.bar(
            shap.Explanation(
                values=shap_values,
                base_values=self.explainer.expected_value,
                feature_names=self.feature_names
            ),
            max_display=max_display,
            show=False
        )
        plt.title('Mean Absolute SHAP Values', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"✓ SHAP bar plot saved to: {output_path}")
        
        plt.show()
    
    def get_top_features(self, X, index=0, shap_values=None, top_n=5):
        """
        Get top contributing features for a single prediction.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        index : int
            Index of sample
        shap_values : array-like, optional
            Pre-computed SHAP values
        top_n : int
            Number of top features
        
        Returns:
        --------
        pd.DataFrame : Top features with SHAP values
        """
        if shap_values is None:
            shap_values = self.compute_shap_values(X)
        
        # Get SHAP values for this sample
        sample_shap = shap_values[index]
        
        # Create DataFrame
        if self.feature_names is not None:
            feature_df = pd.DataFrame({
                'feature': self.feature_names,
                'shap_value': sample_shap,
                'abs_shap': np.abs(sample_shap)
            })
        else:
            feature_df = pd.DataFrame({
                'feature': [f'feature_{i}' for i in range(len(sample_shap))],
                'shap_value': sample_shap,
                'abs_shap': np.abs(sample_shap)
            })
        
        # Sort by absolute SHAP value
        feature_df = feature_df.sort_values('abs_shap', ascending=False)
        
        return feature_df.head(top_n)
    
    def generate_explanation_report(self, X, output_dir='reports/shap'):
        """
        Generate comprehensive SHAP explanation report.
        
        Parameters:
        -----------
        X : array-like or pd.DataFrame
            Input features
        output_dir : str
            Directory to save plots
        """
        print("="*60)
        print("GENERATING SHAP EXPLANATION REPORT")
        print("="*60)
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Compute SHAP values once
        shap_values = self.compute_shap_values(X)
        
        # Global explanations
        print("\nGenerating global explanations...")
        self.plot_summary(X, shap_values, output_path=os.path.join(output_dir, 'shap_summary.png'))
        self.plot_bar(X, shap_values, output_path=os.path.join(output_dir, 'shap_bar.png'))
        
        # Local explanations for top fraud cases
        print("\nGenerating local explanations...")
        
        # Get predictions
        y_proba = self.model.predict_proba(X)
        top_fraud_indices = np.argsort(y_proba)[-5:][::-1]
        
        for i, idx in enumerate(top_fraud_indices):
            print(f"  Explaining transaction {idx} (fraud prob: {y_proba[idx]:.4f})")
            self.plot_waterfall(X, index=idx, shap_values=shap_values,
                              output_path=os.path.join(output_dir, f'waterfall_txn_{idx}.png'))
            self.plot_force(X, index=idx, shap_values=shap_values,
                          output_path=os.path.join(output_dir, f'force_txn_{idx}.html'))
        
        print(f"\n✓ All SHAP explanations saved to: {output_dir}")


if __name__ == "__main__":
    from train import FraudDetectionModel
    from features import prepare_features
    import joblib
    
    print("Loading model and data...")
    
    # Load model
    model_wrapper = FraudDetectionModel()
    model_wrapper.load('models/fraud_model.pkl', 'models/preprocessor.pkl')
    
    # Load data
    df = pd.read_csv('data/raw/transactions.csv')
    X, y = prepare_features(df)
    
    # Transform features
    X_transformed = model_wrapper.feature_engine.transform(X)
    
    # Create explainer
    explainer = FraudExplainer(
        model=model_wrapper.model,
        feature_names=model_wrapper.feature_engine.feature_names
    )
    
    # Create SHAP explainer with background data
    explainer.create_explainer(X_transformed[:1000])
    
    # Generate report
    explainer.generate_explanation_report(
        X_transformed[:500],  # Explain subset for speed
        output_dir='reports/shap'
    )
