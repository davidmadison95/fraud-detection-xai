"""
Setup and Run Script
Complete setup and execution of the fraud detection system.
"""

import subprocess
import sys
import os
from pathlib import Path


def print_header(message):
    """Print formatted header."""
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70 + "\n")


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"⚙️  {description}...")
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(e.stderr)
        return False


def create_directories():
    """Create necessary directories."""
    print_header("Creating Directory Structure")
    
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'reports/shap',
        'notebooks'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")


def check_python_version():
    """Check Python version."""
    print_header("Checking Python Version")
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print("❌ Python 3.9 or higher is required")
        return False
    
    print("✓ Python version is compatible")
    return True


def install_dependencies():
    """Install required packages."""
    print_header("Installing Dependencies")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing Python packages"
    )


def generate_data():
    """Generate synthetic transaction data."""
    print_header("Generating Synthetic Data")
    
    if os.path.exists('data/raw/transactions.csv'):
        response = input("Data already exists. Regenerate? (y/n): ")
        if response.lower() != 'y':
            print("⏭️  Skipping data generation")
            return True
    
    return run_command(
        f"{sys.executable} generate_data.py",
        "Generating transaction data"
    )


def train_model():
    """Train the fraud detection model."""
    print_header("Training Fraud Detection Model")
    
    if os.path.exists('models/fraud_model.pkl'):
        response = input("Model already exists. Retrain? (y/n): ")
        if response.lower() != 'y':
            print("⏭️  Skipping model training")
            return True
    
    return run_command(
        f"{sys.executable} train.py",
        "Training XGBoost model"
    )


def generate_explanations():
    """Generate SHAP explanations."""
    print_header("Generating SHAP Explanations")
    
    if not os.path.exists('models/fraud_model.pkl'):
        print("❌ Model not found. Please train the model first.")
        return False
    
    response = input("Generate SHAP explanations? (y/n): ")
    if response.lower() != 'y':
        print("⏭️  Skipping SHAP generation")
        return True
    
    return run_command(
        f"{sys.executable} explain.py",
        "Generating SHAP visualizations"
    )


def run_dashboard():
    """Launch Streamlit dashboard."""
    print_header("Launching Streamlit Dashboard")
    
    print("🚀 Starting dashboard on http://localhost:8501")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run(
            f"streamlit run app_dashboard.py",
            shell=True
        )
    except KeyboardInterrupt:
        print("\n✓ Dashboard stopped")


def run_api():
    """Launch Flask API."""
    print_header("Launching Flask API")
    
    print("🚀 Starting API server on http://localhost:5000")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        subprocess.run(
            f"{sys.executable} serve_api.py",
            shell=True
        )
    except KeyboardInterrupt:
        print("\n✓ API server stopped")


def show_menu():
    """Display main menu."""
    print("\n" + "="*70)
    print("  FRAUD DETECTION SYSTEM - MAIN MENU")
    print("="*70)
    print("\n1. Complete Setup (Install + Generate + Train)")
    print("2. Generate Data Only")
    print("3. Train Model Only")
    print("4. Generate SHAP Explanations")
    print("5. Launch Streamlit Dashboard")
    print("6. Launch Flask API")
    print("7. Test API (requires API to be running)")
    print("8. Exit")
    print()


def complete_setup():
    """Run complete setup process."""
    print_header("COMPLETE SETUP - Fraud Detection System")
    
    steps = [
        (check_python_version, "Python version check"),
        (create_directories, "Directory creation"),
        (install_dependencies, "Dependency installation"),
        (generate_data, "Data generation"),
        (train_model, "Model training"),
    ]
    
    for func, name in steps:
        if not func():
            print(f"\n❌ Setup failed at: {name}")
            return False
    
    print_header("✓ SETUP COMPLETED SUCCESSFULLY")
    print("\nNext steps:")
    print("  • Run: python setup.py → Option 4 (Generate SHAP)")
    print("  • Run: python setup.py → Option 5 (Launch Dashboard)")
    print("  • Run: python setup.py → Option 6 (Launch API)")
    
    return True


def test_api():
    """Run API tests."""
    print_header("Testing API Endpoints")
    
    if not os.path.exists('test_api.py'):
        print("❌ test_api.py not found")
        return False
    
    return run_command(
        f"{sys.executable} test_api.py",
        "Running API tests"
    )


def main():
    """Main execution function."""
    while True:
        show_menu()
        
        try:
            choice = input("Select an option (1-8): ").strip()
            
            if choice == '1':
                complete_setup()
            elif choice == '2':
                create_directories()
                generate_data()
            elif choice == '3':
                train_model()
            elif choice == '4':
                generate_explanations()
            elif choice == '5':
                run_dashboard()
            elif choice == '6':
                run_api()
            elif choice == '7':
                test_api()
            elif choice == '8':
                print("\n👋 Goodbye!")
                sys.exit(0)
            else:
                print("\n❌ Invalid option. Please select 1-8.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║                                                                ║
    ║         FRAUD DETECTION SYSTEM - Setup & Run                  ║
    ║         Explainable AI for Financial Transactions             ║
    ║                                                                ║
    ╚════════════════════════════════════════════════════════════════╝
    """)
    
    main()
