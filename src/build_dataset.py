# src/build_dataset.py
import pandas as pd
import numpy as np
from datetime import datetime

from utils import load_reference_data

def create_dataset(
    cars_path='data/cars.csv',
    areas_path='data/areas.csv',
    out_path='data/insurance_dataset.csv',
    samples_per_combo=4,
    random_state=42
):
    cars, areas = load_reference_data(cars_path, areas_path)

    # Cross join cars × areas so every combination is represented
    cars['key'] = 1
    areas['key'] = 1
    df = cars.merge(areas, on='key').drop('key', axis=1)

    # Replicate rows to increase dataset size
    df = pd.concat([df] * samples_per_combo, ignore_index=True)
    N = len(df)
    rng = np.random.RandomState(random_state)

    # ── Simulated user fields ──────────────────────────────────────────────────
    df['Driver_Age']    = rng.randint(18, 70, size=N)
    df['Claims']        = rng.choice([0, 0, 0, 1, 2], size=N)          # skew to 0
    df['Vehicle_Value'] = rng.randint(60_000, 4_000_000, size=N)        # wide range
    df['Vehicle_Year']  = rng.randint(2000, datetime.now().year + 1, size=N)
    df['Annual_Mileage']= rng.randint(2_000, 50_000, size=N)
    df['Tracker']       = rng.choice([0, 1], size=N, p=[0.7, 0.3])
    df['Alarm']         = rng.choice([0, 1], size=N, p=[0.6, 0.4])
    df['Immobilizer']   = rng.choice([0, 1], size=N, p=[0.8, 0.2])
    df['Married']       = rng.choice([0, 1], size=N, p=[0.6, 0.4])
    df['Coverage']      = rng.choice(
        ['Comprehensive', 'Smart', 'Third-Party'], size=N, p=[0.5, 0.3, 0.2]
    )

    # FIX ── Usage & Parking values now match Flutter dropdown options exactly
    df['Usage'] = rng.choice(
        ['Daily Commute', 'Business', 'Occasional', 'Weekend Only'], size=N
    )
    df['Parking'] = rng.choice(
        ['Street', 'Garage', 'Carport'], size=N          # was 'Secured Lot' — now 'Carport'
    )

    # ── Numeric columns from CSV ───────────────────────────────────────────────
    df['Theft_Rate']      = df['Theft Rate (%)'].astype(float)
    df['Area_Hijack']     = df['Hijacking_Rate'].astype(float)
    df['Crime_Level_Num'] = (
        df.get('Crime_Level_Num', pd.Series(1.0, index=df.index)).astype(float)
    )

    # ── Premium calculation (monthly) ─────────────────────────────────────────
    def calc_premium(row):
        v = row['Vehicle_Value']

        # Base annual rate by vehicle value tier → monthly
        if v >= 2_000_000:
            annual_rate = 0.08
        elif v >= 1_000_000:
            annual_rate = 0.05
        elif v >= 600_000:
            annual_rate = 0.035
        elif v >= 300_000:
            annual_rate = 0.025
        else:
            annual_rate = 0.018

        base_monthly = (v * annual_rate) / 12.0

        # Driver age factor
        age = row['Driver_Age']
        if age < 25:
            age_factor = 1.6
        elif age <= 35:
            age_factor = 1.2
        elif age <= 60:
            age_factor = 1.0
        else:
            age_factor = 1.3

        # Usage factor (aligned to Flutter dropdown values)
        usage = row['Usage']
        if usage == 'Business':
            usage_factor = 1.40
        elif usage == 'Daily Commute':
            usage_factor = 1.15
        elif usage == 'Weekend Only':
            usage_factor = 0.85
        else:  # Occasional
            usage_factor = 0.90

        # Claims factor
        claims_factor = 1 + (row['Claims'] * 0.30)

        # Vehicle age factor
        car_age = datetime.now().year - int(row['Vehicle_Year'])
        if car_age <= 3:
            car_age_factor = 1.0
        elif car_age <= 10:
            car_age_factor = 1.1
        else:
            car_age_factor = 1.25

        # ── FIX: security_factor < 1.0 means discount, so MULTIPLY (not divide) ──
        security_factor = 0.85 if row['Tracker'] else 1.0
        if row['Alarm']:
            security_factor *= 0.95
        if row['Immobilizer']:
            security_factor *= 0.92

        # ── FIX: theft_factor > 1.0 means higher risk, so MULTIPLY (not divide) ──
        theft_factor = 1.0 + (row['Theft_Rate'] / 100.0)

        # Area hijack factor (was already correct)
        area_factor = 1.0 + (row['Area_Hijack'] / 100.0)

        # Performance / luxury vehicle multiplier
        model_str = str(row['Model']).lower()
        perf_factor = 1.0
        if any(x in model_str for x in ['m3', 'm4', 'm5', 'amg', 'rs', 'g63', 'gt', 'r8', 'cayman', 'carrera']):
            perf_factor = 3.0
        elif v > 1_000_000:
            perf_factor = 2.5
        elif v > 600_000:
            perf_factor = 1.8
        elif v > 300_000:
            perf_factor = 1.4

        # Coverage multiplier
        cov = row['Coverage']
        if cov == 'Comprehensive':
            cov_factor = 1.0
        elif cov == 'Smart':
            cov_factor = 0.75
        else:  # Third-Party
            cov_factor = 0.50

        # Parking factor (aligned to Flutter dropdown values)
        parking = row['Parking']
        if parking == 'Street':
            parking_factor = 1.20
        elif parking == 'Carport':
            parking_factor = 1.08
        else:  # Garage
            parking_factor = 1.00

        premium = (
            base_monthly
            * age_factor
            * usage_factor
            * claims_factor
            * car_age_factor
            * theft_factor       # FIX: was (1.0/theft_factor)
            * security_factor    # FIX: was (1.0/security_factor)
            * area_factor
            * perf_factor
            * parking_factor
            * cov_factor
        )

        return max(premium, 200.0)

    df['Premium'] = df.apply(calc_premium, axis=1).round(2)
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path} with {len(df)} rows")
    return df


if __name__ == "__main__":
    create_dataset()