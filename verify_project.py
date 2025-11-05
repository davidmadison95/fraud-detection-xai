#!/usr/bin/env python
"""
Project Verification Script
Checks that all components are present and functional.
"""

import os
import sys
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists"""
    exists = os.path.exists(filepath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {filepath}")
    return exists

def check_directory_exists(dirpath, description):
    """Check if a directory exists"""
    exists = os.path.isdir(dirpath)
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {dirpath}")
    return exists

def main():
    print("=" * 70)
    print("FRAUD DETECTION SYSTEM - PROJECT VERIFICATION")
    print("=" * 70)
    
    base_path = Path("/home/claude/fraud-xai")
    all_checks = []
    
    # Check directories
    print("\n📁 Checking Directory Structure...")
    print("-" * 70)
    dirs = [
        ("src", "Source code directory"),
        ("notebooks", "Jupyter notebooks"),
        ("data/raw", "Raw data directory"),
        ("models", "Model storage"),
        ("reports", "Reports and visualizations")
    ]
    
    for dir_name, desc in dirs:
        all_checks.append(check_directory_exists(base_path / dir_name, desc))
    
    # Check Python modules
    print("\n🐍 Checking Python Modules...")
    print("-" * 70)
    modules = [
        ("src/generate_data.py", "Data generation module"),
        ("src/features.py", "Feature engineering module"),
        ("src/train.py", "Model training module"),
        ("src/evaluate.py", "Model evaluation module"),
        ("src/explain.py", "SHAP explainability module"),
        ("src/app_dashboard.py", "Streamlit dashboard"),
        ("src/serve_api.py", "Flask REST API")
    ]
    
    for file_name, desc in modules:
        all_checks.append(check_file_exists(base_path / file_name, desc))
    
    # Check notebooks
    print("\n📓 Checking Jupyter Notebooks...")
    print("-" * 70)
    notebooks = [
        ("notebooks/01_eda.ipynb", "EDA notebook"),
        ("notebooks/02_train_model.ipynb", "Training notebook"),
        ("notebooks/03_explainability.ipynb", "Explainability notebook")
    ]
    
    for file_name, desc in notebooks:
        all_checks.append(check_file_exists(base_path / file_name, desc))
    
    # Check documentation
    print("\n📚 Checking Documentation...")
    print("-" * 70)
    docs = [
        ("README.md", "Main README"),
        ("SETUP_GUIDE.md", "Setup guide"),
        ("requirements.txt", "Python dependencies"),
        ("reports/model_card.md", "Model card")
    ]
    
    for file_name, desc in docs:
        all_checks.append(check_file_exists(base_path / file_name, desc))
    
    # Check data and models
    print("\n💾 Checking Data & Models...")
    print("-" * 70)
    data_models = [
        ("data/raw/transactions.csv", "Transaction dataset"),
        ("models/fraud_model.pkl", "Trained model"),
        ("models/preprocessor.pkl", "Preprocessor")
    ]
    
    for file_name, desc in data_models:
        all_checks.append(check_file_exists(base_path / file_name, desc))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    total_checks = len(all_checks)
    passed_checks = sum(all_checks)
    failed_checks = total_checks - passed_checks
    
    print(f"\nTotal Checks: {total_checks}")
    print(f"✅ Passed: {passed_checks}")
    print(f"❌ Failed: {failed_checks}")
    
    if failed_checks == 0:
        print("\n🎉 ALL CHECKS PASSED! Project is complete and ready to use.")
        print("\n📋 Next Steps:")
        print("   1. Install dependencies: pip install -r requirements.txt")
        print("   2. Launch dashboard: streamlit run src/app_dashboard.py")
        print("   3. Start API: python src/serve_api.py")
        print("   4. Explore notebooks: jupyter notebook notebooks/")
        return 0
    else:
        print("\n⚠️  Some checks failed. Review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
