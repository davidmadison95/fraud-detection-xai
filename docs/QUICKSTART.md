# Quick Start Guide

Get up and running with the Fraud Detection System in 5 minutes!

## ⚡ Fastest Path (Automated Setup)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run automated setup (generates data + trains model)
python setup.py
# Select Option 1: Complete Setup

# 3. Launch dashboard
streamlit run app_dashboard.py
```

Visit http://localhost:8501 and start analyzing transactions!

---

## 📋 Step-by-Step Manual Setup

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Generate Synthetic Data
```bash
python generate_data.py
```
**Output:** `data/raw/transactions.csv` (100K transactions)

### Step 3: Train Model
```bash
python train.py
```
**Output:** 
- `models/fraud_model.pkl`
- `models/preprocessor.pkl`
- `models/metrics.json`

**Expected Results:**
- ROC-AUC: ~0.95
- PR-AUC: ~0.85
- Training time: ~1-2 minutes

### Step 4: Generate Explainability Reports
```bash
python explain.py
```
**Output:** `reports/shap/` with visualizations

### Step 5: Launch Dashboard
```bash
streamlit run app_dashboard.py
```
Open browser to: http://localhost:8501

---

## 🚀 Using the Dashboard

1. **Upload Data**
   - Click "Browse files" in sidebar
   - Upload your transaction CSV
   - Or check "Use sample data"

2. **Adjust Threshold**
   - Use slider: 0.0 (sensitive) to 1.0 (specific)
   - Default: 0.5

3. **Analyze Results**
   - View fraud distribution
   - Examine top suspicious transactions
   - Deep-dive with SHAP explanations

4. **Export Results**
   - Download all results
   - Download flagged transactions only

---

## 🔌 Using the API

### Start API Server
```bash
python serve_api.py
```
Server runs on: http://localhost:5000

### Score a Transaction
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

### Test API
```bash
python test_api.py
```

---

## 📊 Sample Workflow

```bash
# Complete workflow in one go
python setup.py  # Select Option 1

# Then choose your interface:
streamlit run app_dashboard.py  # Interactive dashboard
# OR
python serve_api.py  # REST API
```

---

## 🐛 Troubleshooting

### Issue: Model not found
**Solution:**
```bash
python train.py
```

### Issue: Data not found
**Solution:**
```bash
python generate_data.py
```

### Issue: Package installation errors
**Solution:**
```bash
pip install --upgrade pip
pip install -r requirements.txt --upgrade
```

### Issue: Dashboard won't start
**Solution:**
```bash
# Check if port 8501 is available
streamlit run app_dashboard.py --server.port 8502
```

### Issue: API won't start
**Solution:**
```bash
# Check if port 5000 is available
# Edit serve_api.py and change port in app.run()
```

---

## 📁 Required CSV Format

Your CSV must have these columns:

```
transaction_id, timestamp, amount_usd, merchant_category, 
merchant_country, channel, card_present, customer_id, 
customer_age_days, customer_txn_30d, avg_amount_30d, 
std_amount_30d, country_mismatch, hour_of_day, is_weekend
```

**Example:**
```csv
transaction_id,timestamp,amount_usd,merchant_category,merchant_country,channel,card_present,customer_id,customer_age_days,customer_txn_30d,avg_amount_30d,std_amount_30d,country_mismatch,hour_of_day,is_weekend
TXN_00000001,2024-11-04 14:32:15,45.99,groceries,US,pos,1,CUST_001234,850,15,52.30,18.50,0,14,0
TXN_00000002,2024-11-04 23:15:42,2500.00,electronics,CN,online,0,CUST_005678,45,2,75.50,25.30,1,23,1
```

---

## ⏱️ Expected Runtime

| Step | Time | Output Size |
|------|------|-------------|
| Install dependencies | 2-5 min | ~500 MB |
| Generate data | 10 sec | ~15 MB |
| Train model | 1-2 min | ~5 MB |
| Generate SHAP | 2-3 min | ~10 MB |
| Dashboard startup | 5 sec | - |
| API startup | 2 sec | - |

---

## 🎯 Next Steps

After setup:

1. **Explore the Dashboard**
   - Upload your own transaction data
   - Experiment with different thresholds
   - Review SHAP explanations

2. **Integrate the API**
   - Test with sample transactions
   - Build client applications
   - Set up batch processing

3. **Customize the Model**
   - Adjust hyperparameters in `train.py`
   - Add new features in `features.py`
   - Retrain with real data

4. **Deploy to Production**
   - Containerize with Docker
   - Set up monitoring
   - Implement CI/CD pipeline

---

## 📚 Documentation

- Full README: `README.md`
- Model Card: `reports/model_card.md`
- API Docs: Available at http://localhost:5000/ when API is running
- Notebooks: `notebooks/` directory

---

## 🆘 Support

If you encounter issues:

1. Check the troubleshooting section above
2. Review error messages carefully
3. Ensure all dependencies are installed
4. Verify Python version (3.9+)

---

**Happy Fraud Hunting! 🔍**
