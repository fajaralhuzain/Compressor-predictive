import pandas as pd
import numpy as np
import random
import os

def generate_data(num_rows=5000):
    """
    Generates a realistic dummy dataset for compressor predictive maintenance.
    
    Args:
        num_rows (int): Number of rows to generate.
        
    Returns:
        pd.DataFrame: Generated dataset.
    """
    np.random.seed(42)
    random.seed(42)
    
    # Machine IDs
    machine_ids = [f'M-{i:04d}' for i in range(1, num_rows + 1)]
    
    # Client Names
    clients = ['Alpha Corp', 'Beta Industries', 'Gamma Solutions', 'Delta Mfg', 'Epsilon Tech']
    client_names = [random.choice(clients) for _ in range(num_rows)]
    
    # Operating Hours (0 to 10000)
    operating_hours = np.random.randint(0, 10000, num_rows)
    
    # Temperature (C) - Normal ~60-80, High > 90
    temperature = np.random.normal(70, 10, num_rows)
    temperature = np.clip(temperature, 40, 120)
    
    # Vibration (mm/s) - Normal ~0-3, High > 5
    vibration = np.random.exponential(1.5, num_rows)
    vibration = np.clip(vibration, 0, 10)
    
    # Pressure (PSI) - Normal ~100-120
    pressure = np.random.normal(110, 15, num_rows)
    pressure = np.clip(pressure, 80, 150)
    
    # Last Service Days
    last_service = np.random.randint(1, 365, num_rows)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Machine_ID': machine_ids,
        'Client_Name': client_names,
        'Operating_Hours': operating_hours,
        'Temperature_C': temperature,
        'Vibration_mm_s': vibration,
        'Pressure_PSI': pressure,
        'Last_Service_Days': last_service
    })
    
    # Define Failure Logic (Target Variable)
    # Failure is more likely if Temp > 90 OR Vibration > 5 OR Operating Hours > 8000
    # We'll use a logistic-like probability to assign failure
    
    def calculate_failure_prob(row):
        prob = 0.05 # Base probability
        
        if row['Temperature_C'] > 90:
            prob += 0.4
        if row['Vibration_mm_s'] > 5:
            prob += 0.3
        if row['Operating_Hours'] > 8000:
            prob += 0.2
        if row['Last_Service_Days'] > 300:
            prob += 0.1
            
        return min(prob, 0.95)
    
    df['Failure_Prob'] = df.apply(calculate_failure_prob, axis=1)
    df['Failure_in_Next_30_Days'] = df['Failure_Prob'].apply(lambda x: 1 if np.random.rand() < x else 0)
    
    # Drop the temporary probability column
    df = df.drop(columns=['Failure_Prob'])
    
    return df

if __name__ == "__main__":
    print("Generating data...")
    df = generate_data()
    
    output_dir = 'data'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'compressor_data.csv')
    
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    print(df.head())
    print("-" * 30)
    print(df['Failure_in_Next_30_Days'].value_counts(normalize=True))
