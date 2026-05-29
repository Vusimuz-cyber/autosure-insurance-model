import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def train():
    # =========================
    # PATH SETUP
    # =========================
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"

    MODEL_DIR.mkdir(exist_ok=True)

    # 🔥 FORCE UPDATED DATASET ONLY
    dataset_path = DATA_DIR / "insurance_dataset_updated.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Updated dataset not found: {dataset_path}")

    print(f"📂 Using updated dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    print(f"✅ Dataset loaded with {len(df)} rows")

    df.columns = [col.strip() for col in df.columns]

    # =========================
    # FEATURES
    # =========================
    cat_cols = [
        'Brand',
        'Model',
        'Province',
        'Place',
        'Usage',
        'Parking',
        'Coverage',
        'regNumber',
        'color'
    ]

    num_cols = [
        'Theft_Rate',
        'Area_Hijack',
        'Crime_Level_Num',
        'Driver_Age',
        'Claims',
        'Vehicle_Value',
        'Vehicle_Year',
        'Annual_Mileage',
        'Tracker',
        'Alarm',
        'Immobilizer',
        'Married',
        'peakHours'
    ]

    target_col = "Premium"

    # =========================
    # VALIDATION (SAFE BUT STRICT)
    # =========================
    missing_cols = [c for c in cat_cols + num_cols + [target_col] if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in UPDATED dataset: {missing_cols}")

    X = df[cat_cols + num_cols]
    y = df[target_col]

    print("⚙️ Building pipeline...")

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        (
            "rf",
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1
            )
        )
    ])

    print("🚀 Training model...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("📊 Evaluating model...")

    preds = model.predict(X_test)

    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"✅ MSE: {mse}")
    print(f"✅ R² Score: {r2}")

    model_path = MODEL_DIR / "insurance_model.pkl"
    joblib.dump(model, model_path)

    print(f"💾 Model saved to: {model_path}")

    return model


if __name__ == "__main__":
    train()