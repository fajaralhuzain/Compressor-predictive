import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_eda(input_path='data/compressor_data.csv', output_dir='plots'):
    """
    Performs Exploratory Data Analysis on the compressor dataset.
    Generates correlation heatmap and distribution plots.
    
    Args:
        input_path (str): Path to the dataset CSV.
        output_dir (str): Directory to save plots.
    """
    if not os.path.exists(input_path):
        print(f"Error: dataset not found at {input_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_csv(input_path)
    
    print("Dataset Info:")
    print(df.info())
    print("\nDescribe:")
    print(df.describe())
    
    # 1. Correlation Matrix of Numerical Features
    numerical_cols = ['Operating_Hours', 'Temperature_C', 'Vibration_mm_s', 'Pressure_PSI', 'Last_Service_Days', 'Failure_in_Next_30_Days']
    corr_matrix = df[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
    plt.close()
    print(f"Saved correlation matrix to {output_dir}/correlation_matrix.png")
    
    # 2. Pairplot to visualize relationships with Target
    sns.pairplot(df[numerical_cols], hue='Failure_in_Next_30_Days', palette={0: 'blue', 1: 'red'})
    plt.savefig(os.path.join(output_dir, 'pairplot.png'))
    plt.close()
    print(f"Saved pairplot to {output_dir}/pairplot.png")

    # 3. Distribution of Key Features by Failure Status
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sns.histplot(data=df, x='Temperature_C', hue='Failure_in_Next_30_Days', kde=True, ax=axes[0, 0], palette={0: 'blue', 1: 'red'})
    axes[0, 0].set_title('Temperature Distribution by Failure')
    
    sns.histplot(data=df, x='Vibration_mm_s', hue='Failure_in_Next_30_Days', kde=True, ax=axes[0, 1], palette={0: 'blue', 1: 'red'})
    axes[0, 1].set_title('Vibration Distribution by Failure')
    
    sns.histplot(data=df, x='Pressure_PSI', hue='Failure_in_Next_30_Days', kde=True, ax=axes[1, 0], palette={0: 'blue', 1: 'red'})
    axes[1, 0].set_title('Pressure Distribution by Failure')

    sns.histplot(data=df, x='Operating_Hours', hue='Failure_in_Next_30_Days', kde=True, ax=axes[1, 1], palette={0: 'blue', 1: 'red'})
    axes[1, 1].set_title('Operating Hours Distribution by Failure')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_distributions.png'))
    plt.close()
    print(f"Saved feature distributions to {output_dir}/feature_distributions.png")

if __name__ == "__main__":
    run_eda()
