# src/build_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime

from utils import load_reference_data

def create_dataset(cars_path='data/cars.csv', areas_path='data/areas.csv', out_path='data/insurance_dataset.csv', samples_per_combo=4, random_state=42):
    cars, areas = load_reference_data(cars_path, areas_path)
    # cross join
    cars['key'] = 1
    areas['key'] = 1
    df = cars.merge(areas, on='key').drop('key', axis=1)

    # replicate rows to increase dataset size
    df = pd.concat([df]*samples_per_combo, ignore_index=True)
    N = len(df)
    rng = np.random.RandomState(random_state)

    # Simulated user fields
    df['Driver_Age'] = rng.randint(18, 70, size=N)
    df['Claims'] = rng.choice([0,0,0,1,2], size=N)  # skew to 0
    df['Vehicle_Value'] = rng.randint(60000, 4000000, size=N)  # wide range so model learns extremes
    df['Vehicle_Year'] = rng.randint(2000, datetime.now().year+1, size=N)
    df['Annual_Mileage'] = rng.randint(2000, 50000, size=N)
    df['Usage'] = rng.choice(['Daily Commute','Business Use','Occasional','Weekend Only'], size=N)
    df['Parking'] = rng.choice(['Street','Garage','Secured Lot'], size=N)
    df['Tracker'] = rng.choice([0,1], size=N, p=[0.7,0.3])
    df['Alarm'] = rng.choice([0,1], size=N, p=[0.6,0.4])
    df['Immobilizer'] = rng.choice([0,1], size=N, p=[0.8,0.2])
    df['Married'] = rng.choice([0,1], size=N, p=[0.6,0.4])
    df['Coverage'] = rng.choice(['Comprehensive','Smart','Third-Party'], size=N, p=[0.5,0.3,0.2])

    # numeric columns from CSV
    df['Theft_Rate'] = df['Theft Rate (%)'].astype(float)
    df['Area_Hijack'] = df['Hijacking_Rate'].astype(float)
    df['Crime_Level_Num'] = df.get('Crime_Level_Num', pd.Series(1.0, index=df.index)).astype(float)

    # realistic premium calculation (monthly) — uses vehicle value heavily
    def calc_premium(row):
        # base annual rate between 1.5% and 3.0% depending on value tier, then /12
        v = row['Vehicle_Value']
        if v >= 2000000:
            annual_rate = 0.08  # 8% per year for ultra-luxury (highly insurable)
        elif v >= 1000000:
            annual_rate = 0.05  # 5% per year for high-end
        elif v >= 600000:
            annual_rate = 0.035
        elif v >= 300000:
            annual_rate = 0.025
        else:
            annual_rate = 0.018

        base_monthly = (v * annual_rate) / 12.0

        # driver factor
        age = row['Driver_Age']
        if age < 25:
            age_factor = 1.6
        elif age <= 35:
            age_factor = 1.2
        elif age <= 60:
            age_factor = 1.0
        else:
            age_factor = 1.3

        # usage
        usage = row['Usage']
        usage_factor = 1.15 if usage == 'Daily Commute' else 1.4 if usage == 'Business Use' else 0.9 if usage == 'Occasional' else 0.85

        # claims
        claims_factor = 1 + (row['Claims'] * 0.30)

        # vehicle age
        car_age = datetime.now().year - int(row['Vehicle_Year'])
        car_age_factor = 1.0 if car_age <=3 else 1.1 if car_age <=10 else 1.25

        # security
        security_factor = 0.85 if row['Tracker'] else 1.0
        if row['Alarm']:
            security_factor *= 0.95
        if row['Immobilizer']:
            security_factor *= 0.92

        # area & theft risk
        area_factor = 1.0 + (row['Area_Hijack'] / 100.0)   # e.g. 4.5% -> 1.045
        theft_factor = 1.0 + (row['Theft_Rate'] / 100.0)   # e.g. 2.5% -> 1.025

        # performance car multiplier (detect words in model)
        model_str = str(row['Model']).lower()
        brand_str = str(row['Brand']).lower()
        perf_factor = 1.0
        if any(x in model_str for x in ['m3','m4','m5','amg','rs','g63','gt','r8','cayman','carrera']):
            perf_factor = 3.0
        elif v > 1000000:
            perf_factor = 2.5
        elif v > 600000:
            perf_factor = 1.8
        elif v > 300000:
            perf_factor = 1.4

        # coverage multiplier
        cov = row['Coverage']
        cov_factor = 1.0 if cov == 'Comprehensive' else 0.75 if cov == 'Smart' else 0.5

        premium = base_monthly * age_factor * usage_factor * claims_factor * car_age_factor * (1.0/theft_factor) * (1.0/security_factor) * area_factor * perf_factor * cov_factor

        # floor
        if premium < 200:
            premium = 200.0
        return premium

    df['Premium'] = df.apply(calc_premium, axis=1).round(2)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} rows")
    return df

if __name__ == "__main__":
    create_dataset()
