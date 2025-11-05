"""
Streamlit Dashboard for Fraud Detection
Interactive analyst dashboard with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
import matplotlib.pyplot as plt
from train import FraudDetectionModel
from explain import FraudExplainer
from features import prepare_features
import io

# Page configuration
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .fraud-alert {
        background-color: #ffebee;
        padding: 1rem;
        border-left: 4px solid #f44336;
        border-radius: 0.3rem;
    }
    .safe-alert {
        background-color: #e8f5e9;
        padding: 1rem;
        border-left: 4px solid #4caf50;
        border-radius: 0.3rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load trained model and preprocessor."""
    model = FraudDetectionModel()
    try:
        model.load('models/fraud_model.pkl', 'models/preprocessor.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None


@st.cache_resource
def create_explainer(_model, X_background):
    """Create SHAP explainer."""
    explainer = FraudExplainer(
        model=_model.model,
        feature_names=_model.feature_engine.feature_names
    )
    explainer.create_explainer(X_background, max_samples=100)
    return explainer


def plot_fraud_distribution(df_results):
    """Create fraud probability distribution plot."""
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=df_results['fraud_probability'],
        nbinsx=50,
        name='Transactions',
        marker_color='steelblue',
        opacity=0.7
    ))
    
    fig.update_layout(
        title='Fraud Probability Distribution',
        xaxis_title='Fraud Probability',
        yaxis_title='Count',
        height=400,
        showlegend=False
    )
    
    return fig


def plot_top_frauds(df_results, top_n=10):
    """Create bar chart of top fraud cases."""
    top_frauds = df_results.nlargest(top_n, 'fraud_probability')
    
    fig = go.Figure(go.Bar(
        x=top_frauds['fraud_probability'],
        y=top_frauds['transaction_id'],
        orientation='h',
        marker=dict(
            color=top_frauds['fraud_probability'],
            colorscale='Reds',
            showscale=True,
            colorbar=dict(title="Fraud Score")
        )
    ))
    
    fig.update_layout(
        title=f'Top {top_n} Suspicious Transactions',
        xaxis_title='Fraud Probability',
        yaxis_title='Transaction ID',
        height=400,
        yaxis=dict(autorange="reversed")
    )
    
    return fig


def display_shap_summary(explainer, X_sample, shap_values):
    """Display SHAP summary plot."""
    st.subheader("🎯 Global Feature Importance (SHAP)")
    
    # Create DataFrame
    X_df = pd.DataFrame(X_sample, columns=explainer.feature_names)
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(shap_values, X_df, max_display=10, show=False)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()


def display_shap_waterfall(explainer, X_sample, shap_values, index):
    """Display SHAP waterfall plot for single transaction."""
    st.subheader(f"💧 SHAP Explanation - Transaction {index}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    shap_exp = shap.Explanation(
        values=shap_values[index],
        base_values=explainer.explainer.expected_value,
        data=X_sample[index],
        feature_names=explainer.feature_names
    )
    
    shap.plots.waterfall(shap_exp, show=False)
    plt.tight_layout()
    
    st.pyplot(fig)
    plt.close()


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Fraud Detection Dashboard</h1>', 
                unsafe_allow_html=True)
    st.markdown("**AI-Powered Financial Transaction Analysis with Explainable AI**")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Load model
        st.info("Loading model...")
        model = load_model()
        
        if model is None:
            st.error("Failed to load model. Please train the model first.")
            st.stop()
        else:
            st.success("✓ Model loaded successfully")
        
        st.markdown("---")
        
        # Threshold slider
        threshold = st.slider(
            "Fraud Detection Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Adjust sensitivity: Lower = more alerts, Higher = fewer alerts"
        )
        
        st.markdown("---")
        
        # File upload
        st.header("📁 Data Upload")
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV",
            type=['csv'],
            help="Upload a CSV file with transaction data"
        )
        
        # Sample data option
        use_sample = st.checkbox("Use sample data", value=False)
    
    # Main content
    if uploaded_file is not None or use_sample:
        
        # Load data
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✓ Loaded {len(df):,} transactions from uploaded file")
        else:
            try:
                df = pd.read_csv('data/raw/transactions.csv').head(1000)
                st.info(f"Using sample dataset: {len(df):,} transactions")
            except:
                st.error("Sample data not found. Please upload a CSV file.")
                st.stop()
        
        # Prepare features
        with st.spinner("Processing features..."):
            X, y = prepare_features(df)
            X_transformed = model.feature_engine.transform(X)
            
            # Get predictions
            fraud_proba = model.predict_proba(X_transformed)
            fraud_pred = (fraud_proba >= threshold).astype(int)
        
        # Add results to dataframe
        df_results = df.copy()
        df_results['fraud_probability'] = fraud_proba
        df_results['fraud_prediction'] = fraud_pred
        
        # Key metrics
        st.header("📊 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions",
                f"{len(df):,}",
                help="Total number of transactions analyzed"
            )
        
        with col2:
            n_flagged = fraud_pred.sum()
            flagged_pct = (n_flagged / len(df)) * 100
            st.metric(
                "Flagged as Fraud",
                f"{n_flagged:,}",
                f"{flagged_pct:.2f}%",
                delta_color="inverse"
            )
        
        with col3:
            avg_risk = fraud_proba.mean()
            st.metric(
                "Average Risk Score",
                f"{avg_risk:.3f}",
                help="Mean fraud probability across all transactions"
            )
        
        with col4:
            high_risk = (fraud_proba > 0.8).sum()
            st.metric(
                "High Risk (>80%)",
                f"{high_risk:,}",
                help="Transactions with >80% fraud probability"
            )
        
        st.markdown("---")
        
        # Visualizations
        st.header("📈 Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution plot
            fig_dist = plot_fraud_distribution(df_results)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Top frauds
            fig_top = plot_top_frauds(df_results, top_n=10)
            st.plotly_chart(fig_top, use_container_width=True)
        
        st.markdown("---")
        
        # Top flagged transactions
        st.header("🚨 Top Flagged Transactions")
        
        top_frauds = df_results.nlargest(20, 'fraud_probability')
        
        # Format display
        display_cols = ['transaction_id', 'amount_usd', 'merchant_category', 
                       'merchant_country', 'channel', 'fraud_probability', 'fraud_prediction']
        display_df = top_frauds[display_cols].copy()
        display_df['fraud_probability'] = display_df['fraud_probability'].apply(lambda x: f"{x:.4f}")
        
        # Color code predictions
        def highlight_fraud(row):
            if row['fraud_prediction'] == 1:
                return ['background-color: #ffcdd2'] * len(row)
            else:
                return [''] * len(row)
        
        st.dataframe(
            display_df.style.apply(highlight_fraud, axis=1),
            use_container_width=True,
            height=400
        )
        
        st.markdown("---")
        
        # SHAP Explanations
        st.header("🧠 Explainable AI - SHAP Analysis")
        
        with st.spinner("Computing SHAP explanations..."):
            # Create explainer
            explainer = create_explainer(model, X_transformed[:200])
            
            # Compute SHAP values for sample
            n_explain = min(100, len(X_transformed))
            shap_values = explainer.compute_shap_values(X_transformed[:n_explain])
        
        # Global explanation
        with st.expander("🌍 Global Feature Importance", expanded=True):
            display_shap_summary(explainer, X_transformed[:n_explain], shap_values)
        
        # Local explanations
        with st.expander("🔬 Individual Transaction Explanation", expanded=True):
            st.write("Select a transaction to see detailed SHAP explanation:")
            
            # Transaction selector
            transaction_options = list(range(min(50, len(df_results))))
            selected_idx = st.selectbox(
                "Transaction Index",
                transaction_options,
                format_func=lambda x: f"Transaction {x} (Fraud Prob: {fraud_proba[x]:.4f})"
            )
            
            if selected_idx is not None:
                # Display transaction details
                st.subheader(f"Transaction Details - Index {selected_idx}")
                
                txn_details = df_results.iloc[selected_idx]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Transaction ID:** {txn_details['transaction_id']}")
                    st.write(f"**Amount:** ${txn_details['amount_usd']:.2f}")
                    st.write(f"**Category:** {txn_details['merchant_category']}")
                
                with col2:
                    st.write(f"**Country:** {txn_details['merchant_country']}")
                    st.write(f"**Channel:** {txn_details['channel']}")
                    st.write(f"**Card Present:** {bool(txn_details['card_present'])}")
                
                with col3:
                    fraud_score = fraud_proba[selected_idx]
                    if fraud_score > 0.7:
                        st.markdown(f'<div class="fraud-alert"><strong>⚠️ HIGH RISK</strong><br>Fraud Probability: {fraud_score:.4f}</div>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="safe-alert"><strong>✓ LOW RISK</strong><br>Fraud Probability: {fraud_score:.4f}</div>', 
                                  unsafe_allow_html=True)
                
                st.markdown("---")
                
                # SHAP waterfall
                if selected_idx < n_explain:
                    display_shap_waterfall(explainer, X_transformed[:n_explain], 
                                         shap_values, selected_idx)
                    
                    # Top features
                    st.subheader("Top Contributing Features")
                    top_features = explainer.get_top_features(
                        X_transformed[:n_explain], 
                        selected_idx, 
                        shap_values, 
                        top_n=5
                    )
                    
                    for _, row in top_features.iterrows():
                        impact = "increases" if row['shap_value'] > 0 else "decreases"
                        st.write(f"- **{row['feature']}**: {impact} fraud risk (SHAP: {row['shap_value']:.4f})")
        
        st.markdown("---")
        
        # Download results
        st.header("💾 Export Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Download full results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="📥 Download All Results (CSV)",
                data=csv,
                file_name="fraud_detection_results.csv",
                mime="text/csv"
            )
        
        with col2:
            # Download flagged only
            flagged_df = df_results[df_results['fraud_prediction'] == 1]
            csv_flagged = flagged_df.to_csv(index=False)
            st.download_button(
                label="🚨 Download Flagged Transactions (CSV)",
                data=csv_flagged,
                file_name="flagged_transactions.csv",
                mime="text/csv"
            )
    
    else:
        # Welcome screen
        st.info("👈 Please upload a transaction CSV file or use sample data to begin analysis")
        
        st.markdown("### 📋 Expected CSV Format")
        st.markdown("""
        Your CSV should include the following columns:
        - `transaction_id`: Unique transaction identifier
        - `timestamp`: Transaction timestamp
        - `amount_usd`: Transaction amount in USD
        - `merchant_category`: Merchant category code
        - `merchant_country`: Merchant country
        - `channel`: Transaction channel (online, pos, atm, phone)
        - `card_present`: Whether card was physically present (0/1)
        - `customer_id`: Customer identifier
        - `customer_age_days`: Customer account age in days
        - `customer_txn_30d`: Number of transactions in last 30 days
        - `avg_amount_30d`: Average transaction amount (30 days)
        - `std_amount_30d`: Standard deviation of amounts (30 days)
        - `country_mismatch`: Country mismatch flag (0/1)
        - `hour_of_day`: Hour of transaction (0-23)
        - `is_weekend`: Weekend transaction flag (0/1)
        """)


if __name__ == "__main__":
    main()
