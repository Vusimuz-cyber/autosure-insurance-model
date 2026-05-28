import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

def train(dataset_path="../data/insurance_dataset_updated.csv", out_model="../models/insurance_model.pkl"):
    df = pd.read_csv(dataset_path)
    # Feature columns
    cat_cols = ['Brand', 'Model', 'Province', 'Place', 'Usage', 'Parking', 'Coverage', 'regNumber', 'color']
    num_cols = ['Theft_Rate', 'Area_Hijack', 'Crime_Level_Num', 'Driver_Age', 'Claims', 'Vehicle_Value', 'Vehicle_Year', 'Annual_Mileage', 'Tracker', 'Alarm', 'Immobilizer', 'Married', 'peakHours']

    X = df[cat_cols + num_cols]
    y = df['Premium']

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols),
        ],
        remainder='passthrough'
    )

    model = Pipeline(steps=[
        ('pre', preprocessor),
        ('rf', RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1))
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("MSE:", mean_squared_error(y_test, preds))
    print("R2:", r2_score(y_test, preds))
    joblib.dump(model, out_model)
    print("Saved model to", out_model)
    return model

if __name__ == "__main__":
    train()