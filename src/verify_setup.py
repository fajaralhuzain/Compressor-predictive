import os
import joblib
import pandas as pd

def verify_project():
    print("Verifying Project Setup...")
    
    files_to_check = [
        'data/compressor_data.csv',
        'src/data_generator.py',
        'src/eda.py',
        'src/model_trainer.py',
        'src/app.py',
        'models/random_forest_model.pkl',
        'models/scaler.pkl',
        'plots/correlation_matrix.png',
        'plots/confusion_matrix.png'
    ]
    
    missing_files = []
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ Found {file_path}")
        else:
            print(f"❌ Missing {file_path}")
            missing_files.append(file_path)
            
    if missing_files:
        print("\nVerification Failed! Missing files.")
        return False
        
    print("\nVerifying Model Loading...")
    try:
        model = joblib.load('models/random_forest_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        print("✅ Model and Scaler loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model/scaler: {e}")
        return False
        
    print("\nProject Verification Successful! 🚀")
    print("You can run the dashboard using: streamlit run src/app.py")
    return True

if __name__ == "__main__":
    verify_project()
