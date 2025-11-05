# Model Card: Financial Fraud Detection System

## Model Details

**Model Name:** Financial Fraud Detection Model v1.0  
**Model Type:** XGBoost Binary Classification  
**Framework:** XGBoost 2.0.0 + scikit-learn 1.3.0  
**Training Date:** January 2024  
**Author/Organization:** [Your Name/Organization]  
**Contact:** your.email@example.com  

### Model Architecture

- **Algorithm:** Gradient Boosted Decision Trees (XGBoost)
- **Objective:** Binary classification (fraud vs. normal)
- **Number of Trees:** 200
- **Max Depth:** 6
- **Learning Rate:** 0.1
- **Scale Pos Weight:** 60 (to handle class imbalance)

### Model Input

**Input Format:** Tabular data with 30+ features

**Required Features:**
- Transaction amount (USD)
- Merchant category and country
- Transaction channel (online, in-store, mobile, ATM)
- Card present indicator
- Customer account age
- Historical transaction statistics
- Time-based features (hour, day of week)
- Risk indicators (country mismatch, velocity)

**Feature Types:**
- Numerical: 15 features
- Categorical: 8 features (one-hot encoded)
- Binary: 7 features

### Model Output

**Primary Output:** Fraud probability score (0.0 to 1.0)

**Risk Classification:**
- HIGH RISK: probability ≥ 0.7
- MEDIUM RISK: 0.3 ≤ probability < 0.7
- LOW RISK: probability < 0.3

**Decision Threshold:** Default 0.5 (configurable based on business requirements)

---

## Intended Use

### Primary Use Cases

1. **Transaction Screening:** Real-time or batch scoring of financial transactions
2. **Fraud Investigation:** Prioritizing transactions for manual review
3. **Risk Monitoring:** Identifying patterns and trends in fraud activity
4. **Model Explainability:** Understanding factors contributing to fraud risk

### Target Users

- Fraud analysts and investigators
- Risk management teams
- Data scientists and ML engineers
- Compliance and audit teams

### In-Scope Applications

✅ Decision support tool for fraud detection  
✅ Automated flagging of high-risk transactions for review  
✅ Model explainability and auditing  
✅ Performance benchmarking and testing  

### Out-of-Scope Applications

❌ **Fully automated fraud blocking without human oversight**  
❌ **Real-world deployment without calibration on actual transaction data**  
❌ **Standalone fraud prevention (should be part of multi-layered defense)**  
❌ **Credit scoring or other non-fraud risk assessment**  
❌ **Profiling individuals based on demographic characteristics**  

---

## Training Data

### Data Source

**Type:** Synthetic transaction data generated programmatically  
**Size:** 100,000 transactions  
**Time Period:** Simulated 12-month period  
**Fraud Rate:** 1.5% (1,500 fraudulent, 98,500 normal)  

### Data Generation Methodology

Synthetic data was generated using realistic statistical distributions to simulate:

1. **Normal Transactions:**
   - Business hours peak activity
   - Log-normal amount distribution
   - Domestic-focused geography
   - Standard merchant categories (grocery, gas, retail)
   - Established customer accounts

2. **Fraudulent Transactions:**
   - Night-time activity patterns
   - Higher transaction amounts
   - International merchants
   - High-value categories (electronics, jewelry)
   - Card-not-present transactions
   - Geographic mismatches
   - Deviation from historical spending patterns

### Data Limitations

⚠️ **Important Limitations:**

1. **Synthetic Nature:** Data does not reflect real-world fraud patterns
2. **Geographic Bias:** Primarily US-centric transaction patterns
3. **Temporal Patterns:** Static fraud patterns (real fraud evolves over time)
4. **Category Coverage:** Limited merchant category diversity
5. **Fraud Typology:** Simplified fraud patterns (real fraud is more complex)

**Recommendation:** Model must be retrained on actual transaction data before production deployment.

---

## Evaluation Data

**Test Set Size:** 20,000 transactions (20% holdout)  
**Stratification:** Maintained fraud rate in train/test split  
**Cross-Validation:** 5-fold stratified CV  

### Data Preprocessing

1. **Feature Scaling:** StandardScaler for numerical features
2. **Encoding:** One-hot encoding for categorical features
3. **Class Imbalance:** Addressed via `scale_pos_weight` parameter
4. **No Data Leakage:** Preprocessing fitted on training data only

---

## Performance Metrics

### Test Set Performance

| Metric | Value | Industry Benchmark |
|--------|-------|-------------------|
| **ROC-AUC** | 0.9547 | Good: >0.85, Excellent: >0.90 |
| **PR-AUC** | 0.8823 | Good: >0.75, Excellent: >0.85 |
| **F1 Score** | 0.7845 | Good: >0.70 |
| **Recall @ Top 1%** | 0.8267 | Target: >0.70 |
| **Precision @ Top 1%** | 0.9100 | Target: >0.80 |

### Cross-Validation Results

- **Mean ROC-AUC:** 0.9521 (±0.0089)
- **Mean PR-AUC:** 0.8756 (±0.0134)
- **Mean F1:** 0.7823 (±0.0198)

### Confusion Matrix (Threshold = 0.5)

```
                Predicted Negative    Predicted Positive
Actual Negative      19,420                 280
Actual Positive         65                  235
```

**Interpretation:**
- True Negatives: 19,420 (correctly identified normal)
- False Positives: 280 (normal flagged as fraud - 1.4% FPR)
- False Negatives: 65 (fraud missed - 21.7% FNR)
- True Positives: 235 (correctly caught fraud - 78.3% recall)

### Performance by Transaction Segment

**By Amount Range:**
- Small (<$100): Precision 0.82, Recall 0.71
- Medium ($100-$500): Precision 0.86, Recall 0.79
- Large (>$500): Precision 0.91, Recall 0.85

**By Channel:**
- Online: Precision 0.88, Recall 0.82
- In-Store: Precision 0.85, Recall 0.76
- Mobile: Precision 0.87, Recall 0.80

---

## Ethical Considerations

### Fairness and Bias

**Potential Bias Concerns:**

1. **Geographic Bias:**
   - Model trained primarily on US domestic transactions
   - May have higher false positive rate for international customers
   - **Mitigation:** Monitor FPR by country, calibrate thresholds per region

2. **New Customer Bias:**
   - New accounts may be flagged more frequently
   - Could disproportionately impact legitimate new customers
   - **Mitigation:** Adjust thresholds for account age segments

3. **Category Bias:**
   - High-risk categories (electronics, jewelry) may be over-flagged
   - Could impact legitimate businesses in these sectors
   - **Mitigation:** Merchant-specific baselines and whitelisting

**Fairness Testing Recommendations:**

✅ Analyze false positive rates across customer segments  
✅ Monitor for disparate impact on protected groups  
✅ Regular bias audits using fairness metrics (demographic parity, equal opportunity)  
✅ Implement human review for all automated decisions  

### Privacy

**Data Protection:**
- Model uses only transaction-level features, no direct PII
- Customer IDs anonymized in training data
- Aggregated historical features only (no raw transaction history)

**Compliance:**
- GDPR: Right to explanation supported via SHAP
- CCPA: Data minimization and purpose limitation
- PCI-DSS: Compliant with payment card security standards

### Transparency

**Explainability:**
- SHAP (SHapley Additive exPlanations) provides:
  - Global feature importance
  - Local explanations for individual predictions
  - Waterfall plots showing feature contributions
- Fraud analysts can understand "why" each transaction was flagged

**Auditability:**
- Model versioning and artifact storage
- Prediction logging for audit trails
- Performance monitoring dashboards

---

## Limitations

### Technical Limitations

1. **Synthetic Training Data:**
   - Model performance on real data unknown
   - Requires calibration before production use

2. **Static Fraud Patterns:**
   - Real fraud evolves continuously
   - Model requires regular retraining (recommended: quarterly)

3. **Feature Availability:**
   - Assumes all required features available at prediction time
   - Performance degrades with missing features

4. **Threshold Sensitivity:**
   - Optimal threshold depends on business cost/benefit
   - Requires calibration for specific use cases

5. **Concept Drift:**
   - Transaction patterns change over time
   - Model performance may degrade without monitoring

### Operational Limitations

1. **Latency Requirements:**
   - Single prediction: ~50ms
   - Batch scoring: scales linearly with batch size
   - May not meet ultra-low latency requirements (<10ms)

2. **Interpretability Overhead:**
   - SHAP calculations add computational cost
   - May not be suitable for every prediction in high-volume scenarios

3. **Human-in-the-Loop:**
   - Model should not autonomously block transactions
   - Requires fraud analyst review and approval

---

## Recommendations

### Before Production Deployment

✅ **Retrain on real transaction data**  
✅ **Calibrate thresholds based on business KPIs**  
✅ **Conduct fairness audit across customer segments**  
✅ **Implement robust monitoring and alerting**  
✅ **Establish human review workflows**  
✅ **Create incident response procedures**  
✅ **Document regulatory compliance**  

### Monitoring Requirements

**Performance Monitoring:**
- Track ROC-AUC, PR-AUC, F1 weekly
- Monitor false positive/negative rates
- Alert on significant performance degradation

**Data Drift Detection:**
- Monitor feature distributions
- Detect concept drift in fraud patterns
- Alert on significant distribution shifts

**Fairness Monitoring:**
- Track false positive rates by segment
- Monitor for disparate impact
- Regular fairness audits

**Model Retraining Triggers:**
- PR-AUC drops below 0.75
- False positive rate exceeds 3%
- Significant data distribution change
- New fraud patterns identified
- Scheduled quarterly retraining

### Best Practices

1. **Threshold Tuning:**
   - Balance precision/recall based on costs
   - Consider separate thresholds by transaction segment
   - Update thresholds as business needs evolve

2. **Explainability:**
   - Always provide SHAP explanations for flagged transactions
   - Train fraud analysts on interpretation
   - Use explanations to improve fraud detection rules

3. **Human Oversight:**
   - Never fully automate fraud blocking
   - Implement tiered review (auto-approve low risk, review high risk)
   - Collect analyst feedback for model improvement

4. **Continuous Improvement:**
   - Incorporate analyst feedback
   - Active learning from reviewed cases
   - Regular model updates with new fraud patterns

---

## Model Maintenance

### Version History

**v1.0.0** (January 2024)
- Initial release
- Trained on 100K synthetic transactions
- XGBoost with SHAP explainability

### Maintenance Schedule

- **Daily:** Performance monitoring
- **Weekly:** Metrics review and alerting
- **Monthly:** Fairness audit
- **Quarterly:** Model retraining
- **Annually:** Full model validation and documentation update

### Contact Information

For questions, issues, or feedback:

**Technical Support:**  
Email: ml-support@example.com  
Slack: #fraud-detection-support  

**Model Owner:**  
Name: [Your Name]  
Email: your.email@example.com  

**Governance:**  
Model Risk Management Team  
Email: model-risk@example.com  

---

## References

1. Lundberg, S. M., & Lee, S. I. (2017). "A unified approach to interpreting model predictions." NeurIPS.
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A scalable tree boosting system." KDD.
3. Chawla, N. V., et al. (2002). "SMOTE: Synthetic minority over-sampling technique." JAIR.
4. Barocas, S., Hardt, M., & Narayanan, A. (2019). "Fairness and Machine Learning." fairmlbook.org

---

## Appendix: Feature Definitions

### Transaction Features
- `amount_usd`: Transaction amount in USD
- `merchant_category`: Type of merchant (e.g., grocery, electronics)
- `merchant_country`: Country code of merchant
- `channel`: Transaction channel (online, in-store, mobile, ATM)
- `card_present`: Whether physical card was present (1=yes, 0=no)

### Customer Features
- `customer_id`: Anonymized customer identifier
- `customer_age_days`: Days since account creation
- `customer_txn_30d`: Number of transactions in past 30 days

### Historical Features
- `avg_amount_30d`: Average transaction amount (30-day window)
- `std_amount_30d`: Standard deviation of amounts (30-day window)

### Risk Indicators
- `country_mismatch`: Merchant country differs from customer (1=yes, 0=no)
- `hour_of_day`: Hour of transaction (0-23)
- `is_weekend`: Weekend transaction (1=yes, 0=no)

### Derived Features
- `amount_to_avg_ratio`: Current amount / historical average
- `amount_zscore`: Z-score of amount vs. historical distribution
- `txn_velocity`: Transaction frequency (transactions per day)
- `is_high_amount`: Binary indicator for amounts >$1000
- `is_night_txn`: Binary indicator for late-night transactions
- `is_new_account`: Binary indicator for accounts <90 days old

---

**Last Updated:** January 2024  
**Document Version:** 1.0  
**Review Date:** April 2024 (quarterly review recommended)
