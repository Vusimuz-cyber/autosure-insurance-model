import pandas as pd
import numpy as np

def update_dataset(input_path="../data/insurance_dataset.csv", output_path="../data/insurance_dataset_updated.csv"):
    df = pd.read_csv(input_path)
    
    # Generate random registration numbers (e.g., GP123456)
    provinces = ['GP', 'WC', 'KZN', 'EC', 'FS', 'MP', 'LP', 'NW', 'NC']
    df['regNumber'] = [f"{np.random.choice(provinces)}{np.random.randint(100000, 999999)}" for _ in range(len(df))]
    
    # Assign colors (weighted for realism)
    colors = ['White', 'Silver', 'Black', 'Blue', 'Red', 'Grey', 'Other']
    color_weights = [0.3, 0.25, 0.2, 0.15, 0.05, 0.03, 0.02]  # White/Silver most common
    df['color'] = np.random.choice(colors, size=len(df), p=color_weights)
    
    # Assign peakHours (0/1, 30% drive during peak hours)
    df['peakHours'] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    
    df.to_csv(output_path, index=False)
    print(f"Updated dataset saved to {output_path}")

if __name__ == "__main__":
    update_dataset()