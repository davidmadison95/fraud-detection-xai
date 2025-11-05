# 🔍 Financial Fraud Detection System — Explainable AI

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io/)

An enterprise-grade machine learning system for detecting fraudulent financial transactions with explainable AI capabilities. This project demonstrates end-to-end ML pipeline development, from data generation to deployment, featuring XGBoost classification, SHAP explanations, and interactive dashboards.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Project Structure](#-project-structure)
- [Usage Guide](#-usage-guide)
- [Model Performance](#-model-performance)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [Model Card](#-model-card)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This fraud detection system uses advanced machine learning techniques to identify potentially fraudulent financial transactions in real-time. The system addresses the critical challenge of imbalanced datasets (fraud rate ~1.5%) through sophisticated sampling techniques and provides transparent decision-making through SHAP (SHapley Additive exPlanations) visualizations.

### Business Value

- **Reduce Fraud Losses**: Identify suspicious transactions before they complete
- **Improve Customer Trust**: Lower false positives through explainable decisions
- **Regulatory Compliance**: Provide audit trails and transparent decision reasoning
- **Real-Time Scoring**: Sub-second prediction latency via REST API

---

## ✨ Key Features

### Machine Learning
- **XGBoost Classifier** with optimized hyperparameters for imbalanced data
- **Stratified K-Fold Cross-Validation** (5-fold) for robust evaluation
- **Class Imbalance Handling** via `scale_pos_weight` parameter
- **Feature Engineering** with transaction patterns and customer behavior

### Explainability (XAI)
- **SHAP Summary Plots** - Global feature importance across all predictions
- **SHAP Waterfall Charts** - Individual transaction explanations
- **SHAP Force Plots** - Interactive HTML visualizations
- **Feature Contribution Analysis** - Understand which features drive risk scores

### Interactive Dashboard
- **Streamlit Web Application** with professional UI
- **Real-Time Analysis** - Upload CSV and get instant results
- **Risk Distribution Visualization** using Plotly
- **Adjustable Threshold Slider** for sensitivity tuning
- **Transaction Deep-Dive** with SHAP explanations per transaction

### Production API
- **Flask REST API** with `/score` and `/batch_score` endpoints
- **JSON Request/Response** format
- **Health Monitoring** endpoints
- **Batch Processing** support for high-throughput scenarios

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Layer                               │
│  • Synthetic Transaction Generator                           │
│  • Feature Engineering Pipeline                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Model Training Layer                        │
│  • XGBoost Classifier (Imbalance-Aware)                     │
│  • Stratified Cross-Validation                              │
│  • Hyperparameter Optimization                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│               Explainability Layer (SHAP)                    │
│  • TreeExplainer for XGBoost                                │
│  • Global & Local Explanations                              │
│  • Visualization Generation                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
         ┌─────────────┴─────────────┐
         │                           │
┌────────▼─────────┐      ┌─────────▼──────────┐
│  Streamlit UI    │      │   Flask REST API   │
│  • Dashboard     │      │   • /score         │
│  • Upload CSV    │      │   • /batch_score   │
│  • Visualizations│      │   • Real-time      │
└──────────────────┘      └────────────────────┘
```

---

## 🚀 Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager
- 4GB+ RAM recommended

### Setup Instructions

1. **Clone or download the project files**

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import xgboost, shap, streamlit, flask; print('All packages installed successfully!')"
   ```

---

## 🎬 Quick Start

### Step 1: Generate Synthetic Data
```bash
python generate_data.py
```
This creates `data/raw/transactions.csv` with 100,000 transactions (~1.5% fraud rate).

### Step 2: Train the Model
```bash
python train.py
```
Output:
- `models/fraud_model.pkl` - Trained XGBoost model
- `models/preprocessor.pkl` - Feature transformation pipeline
- `models/metrics.json` - Performance metrics

Expected metrics:
- **ROC-AUC**: ~0.95
- **PR-AUC**: ~0.85
- **Recall @ Top 1%**: ~75%

### Step 3: Generate SHAP Explanations
```bash
python explain.py
```
Creates visualizations in `reports/shap/`:
- `shap_summary.png` - Global feature importance
- `waterfall_txn_*.png` - Individual transaction explanations
- `force_txn_*.html` - Interactive force plots

### Step 4: Launch Dashboard
```bash
streamlit run app_dashboard.py
```
Open browser to `http://localhost:8501`

### Step 5: Start API Server
```bash
python serve_api.py
```
API available at `http://localhost:5000`

---

## 📁 Project Structure

```
fraud-xai/
├── data/
│   ├── raw/
│   │   └── transactions.csv          # Generated transaction data
│   └── processed/                     # Processed datasets (optional)
│
├── models/
│   ├── fraud_model.pkl               # Trained XGBoost model
│   ├── preprocessor.pkl              # Feature transformer
│   └── metrics.json                  # Model evaluation metrics
│
├── reports/
│   ├── shap/
│   │   ├── shap_summary.png          # Global SHAP importance
│   │   ├── shap_bar.png              # Mean absolute SHAP values
│   │   ├── waterfall_txn_*.png       # Transaction explanations
│   │   └── force_txn_*.html          # Interactive force plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── pr_curve.png
│   └── model_card.md                 # Model documentation
│
├── src/ (or root level)
│   ├── generate_data.py              # Synthetic data generation
│   ├── features.py                   # Feature engineering
│   ├── train.py                      # Model training pipeline
│   ├── evaluate.py                   # Model evaluation
│   ├── explain.py                    # SHAP explainability
│   ├── app_dashboard.py              # Streamlit dashboard
│   └── serve_api.py                  # Flask REST API
│
├── notebooks/ (optional)
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_train_model.ipynb          # Interactive training
│   └── 03_explainability.ipynb       # SHAP analysis
│
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## 📖 Usage Guide

### Using the Streamlit Dashboard

1. **Launch the dashboard**:
   ```bash
   streamlit run app_dashboard.py
   ```

2. **Upload your data**:
   - Click "Browse files" in the sidebar
   - Upload a CSV with required columns (see format below)
   - Or check "Use sample data"

3. **Adjust fraud threshold**:
   - Use slider in sidebar (0.0 to 1.0)
   - Lower = more sensitive (more alerts)
   - Higher = more specific (fewer alerts)

4. **Review results**:
   - View key metrics at the top
   - Explore fraud probability distribution
   - Examine top flagged transactions
   - Deep-dive into SHAP explanations

5. **Export results**:
   - Download full results CSV
   - Download flagged transactions only

### Using the Flask API

#### Start the API server:
```bash
python serve_api.py
```

#### Score a single transaction:
```bash
curl -X POST http://localhost:5000/score \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": "TXN_12345",
    "amount_usd": 542.10,
    "merchant_category": "electronics",
    "merchant_country": "US",
    "channel": "online",
    "card_present": 0,
    "customer_age_days": 365,
    "customer_txn_30d": 12,
    "avg_amount_30d": 150.50,
    "std_amount_30d": 75.25,
    "country_mismatch": 1,
    "hour_of_day": 23,
    "is_weekend": 0
  }'
```

**Response:**
```json
{
  "transaction_id": "TXN_12345",
  "fraud_probability": 0.8542,
  "fraud_prediction": 1,
  "risk_level": "HIGH",
  "threshold": 0.5,
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Batch scoring:
```bash
curl -X POST http://localhost:5000/batch_score \
  -H "Content-Type: application/json" \
  -d '{
    "records": [
      { "amount_usd": 100, "merchant_category": "groceries", ... },
      { "amount_usd": 2000, "merchant_category": "electronics", ... }
    ]
  }'
```

### Required Data Format

Your CSV should include these columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `transaction_id` | string | Unique identifier | TXN_00001234 |
| `timestamp` | datetime | Transaction time | 2024-01-15 14:30:00 |
| `amount_usd` | float | Amount in USD | 542.10 |
| `merchant_category` | string | Category code | electronics |
| `merchant_country` | string | Country code | US |
| `channel` | string | online/pos/atm/phone | online |
| `card_present` | int | 0 or 1 | 0 |
| `customer_id` | string | Customer identifier | CUST_001234 |
| `customer_age_days` | int | Account age in days | 365 |
| `customer_txn_30d` | int | Txns in last 30 days | 12 |
| `avg_amount_30d` | float | Avg amount (30 days) | 150.50 |
| `std_amount_30d` | float | Std dev amount (30 days) | 75.25 |
| `country_mismatch` | int | 0 or 1 | 1 |
| `hour_of_day` | int | 0-23 | 14 |
| `is_weekend` | int | 0 or 1 | 0 |

---

## 📊 Model Performance

### Evaluation Metrics (Test Set)

| Metric | Score | Interpretation |
|--------|-------|----------------|
| **ROC-AUC** | 0.95+ | Excellent discrimination |
| **PR-AUC** | 0.85+ | Strong precision-recall tradeoff |
| **Recall @ Top 1%** | 70-80% | Catches most fraud in top tier |
| **Precision @ Top 1%** | 60-70% | Low false positives in alerts |
| **F1 Score** | 0.70+ | Good balance |

### Feature Importance (Top 10)

Based on SHAP analysis:

1. **amount_usd** - Transaction amount (primary driver)
2. **country_mismatch** - Mismatch between customer and merchant country
3. **customer_age_days** - New accounts = higher risk
4. **avg_amount_30d** - Deviation from typical behavior
5. **card_present** - Card-not-present transactions
6. **merchant_category** - Certain categories higher risk (electronics, travel)
7. **hour_of_day** - Late night / early morning elevated risk
8. **customer_txn_30d** - Transaction frequency patterns
9. **channel** - Online vs. in-person
10. **is_weekend** - Weekend transaction patterns

---

## 🔌 API Reference

### Endpoints

#### `GET /`
**Home/Info endpoint**
- Returns API information and available endpoints

#### `GET /health`
**Health check**
- Returns: `{"status": "healthy", "model_loaded": true}`

#### `POST /score`
**Score single transaction**
- Input: JSON object with transaction features
- Output: Fraud probability, prediction, risk level

#### `POST /batch_score`
**Score multiple transactions**
- Input: `{"records": [...]}`
- Output: Array of results + summary statistics

#### `GET /model_info`
**Model information**
- Returns: Model type, feature count, performance metrics

---

## 🖼️ Screenshots

### Streamlit Dashboard
![Dashboard Overview](reports/dashboard_overview.png)
*Main dashboard showing key metrics and fraud distribution*

### SHAP Global Explanation
![SHAP Summary](reports/shap/shap_summary.png)
*Global feature importance using SHAP values*

### Individual Transaction Explanation
![SHAP Waterfall](reports/shap/waterfall_txn_0.png)
*Detailed explanation for a single high-risk transaction*

---

## 📄 Model Card

See [reports/model_card.md](reports/model_card.md) for comprehensive model documentation including:
- Model architecture details
- Training data characteristics
- Performance benchmarks
- Ethical considerations
- Known limitations
- Bias analysis
- Intended use cases

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Enhanced Features**
   - Graph-based fraud ring detection (NetworkX)
   - Time-series anomaly detection
   - Ensemble methods (LightGBM, CatBoost)

2. **Production Features**
   - Docker containerization
   - CI/CD pipeline
   - Model monitoring dashboard
   - A/B testing framework

3. **Documentation**
   - Video tutorials
   - Additional notebooks
   - Use case examples

---

## 📝 License

This project is licensed under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

- **XGBoost** - Tianqi Chen and Carlos Guestrin
- **SHAP** - Scott Lundberg
- **Streamlit** - Streamlit Inc.
- **scikit-learn** - scikit-learn developers

---

## 📧 Contact

For questions, issues, or collaboration opportunities:
- Create an issue in the repository
- Email: your.email@example.com
- LinkedIn: [Your Profile]

---

## 🔖 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{fraud_detection_xai,
  title={Financial Fraud Detection System with Explainable AI},
  author={David Madison},
  year={2025},
  url={https://github.com/davidmadison95/fraud-detection-xai}
}
```

---

**Built with ❤️ for transparent and trustworthy AI**
