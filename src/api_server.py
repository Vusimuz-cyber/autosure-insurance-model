from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from src.utils import load_reference_data, best_car_match, best_area_match
import numpy as np
from typing import Optional

app = FastAPI()

# FIX: allow_credentials=True is incompatible with allow_origins=["*"].
# Either list your exact origins, or drop allow_credentials.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,        # FIX: was True — invalid with wildcard origin
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ── Paths ──────────────────────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(CURRENT_DIR)

MODEL_PATH      = os.path.join(PARENT_DIR, "models", "insurance_model.pkl")
CARS_DATA_PATH  = os.path.join(PARENT_DIR, "data",   "cars.csv")
AREAS_DATA_PATH = os.path.join(PARENT_DIR, "data",   "areas.csv")

print(f"🔍 Looking for model at:      {MODEL_PATH}")
print(f"🔍 Looking for cars data at:  {CARS_DATA_PATH}")
print(f"🔍 Looking for areas data at: {AREAS_DATA_PATH}")

try:
    MODEL = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Could not load model: {e}")
    MODEL = None

try:
    CARS_DF, AREAS_DF = load_reference_data(CARS_DATA_PATH, AREAS_DATA_PATH)
    print("✅ Reference data loaded successfully")
except Exception as e:
    print(f"❌ Could not load reference data: {e}")
    CARS_DF, AREAS_DF = pd.DataFrame(), pd.DataFrame()


# ── Request model ──────────────────────────────────────────────────────────────
class QuoteRequest(BaseModel):
    vehicle_value:    Optional[float] = None
    brand:            Optional[str]   = None
    model:            Optional[str]   = None
    province:         Optional[str]   = None
    place:            Optional[str]   = None
    driver_age:       Optional[int]   = None
    claims_history:   Optional[int]   = None
    vehicle_year:     Optional[int]   = None
    annual_mileage:   Optional[int]   = None
    vehicle_usage:    Optional[str]   = None
    parking_type:     Optional[str]   = None
    has_tracker:      Optional[int]   = None
    has_alarm:        Optional[int]   = None
    has_immobilizer:  Optional[int]   = None
    married:          Optional[int]   = None
    peak_hours_usage: Optional[int]   = None
    color:            Optional[str]   = None
    reg_number:       Optional[str]   = None
    coverage_type:    Optional[str]   = None

    # Legacy field names kept for backward compatibility
    car_input:    Optional[str] = None
    area_input:   Optional[str] = None
    claims:       Optional[int] = None
    usage:        Optional[str] = None
    parking:      Optional[str] = None
    tracker:      Optional[int] = None
    alarm:        Optional[int] = None
    immobilizer:  Optional[int] = None
    regNumber:    Optional[str] = None
    peakHours:    Optional[int] = None
    coverage:     Optional[str] = None


# ── Helpers ────────────────────────────────────────────────────────────────────

def _first_not_none(*values, default):
    """
    FIX: Returns the first value that is not None.
    Using `or` would wrongly skip 0 (e.g. claims=0, tracker=0) and fall
    through to the next field or the default.
    """
    for v in values:
        if v is not None:
            return v
    return default


def _normalise_parking(raw: str) -> str:
    """
    Map any casing / legacy value to the canonical training-data label.
    Training values: 'Street', 'Garage', 'Carport'
    """
    mapping = {
        'street':       'Street',
        'garage':       'Garage',
        'carport':      'Carport',
        'secured lot':  'Carport',   # legacy fallback
        'secured_lot':  'Carport',
    }
    return mapping.get(raw.lower().strip(), 'Street')


def _normalise_usage(raw: str) -> str:
    """
    Map any casing / legacy value to the canonical training-data label.
    Training values: 'Daily Commute', 'Business', 'Occasional', 'Weekend Only'
    """
    r = raw.lower().strip()
    if 'business' in r:
        return 'Business'
    if 'weekend' in r:
        return 'Weekend Only'
    if 'occasional' in r:
        return 'Occasional'
    return 'Daily Commute'


def calculate_fallback_premium(vehicle_value: float, risk_factors: dict) -> float:
    """Calculate a realistic monthly premium when the ML model is unavailable."""
    if vehicle_value <= 0:
        vehicle_value = 150_000

    base_annual_rate = 0.015
    base_premium = vehicle_value * base_annual_rate
    m = 1.0   # risk multiplier

    # Age
    age = risk_factors.get('driver_age', 35)
    if age < 25:
        m *= 1.30
    elif age < 30:
        m *= 1.15
    elif age > 65:
        m *= 1.20

    # Claims
    claims = risk_factors.get('claims_history', 0)
    if claims == 1:
        m *= 1.15
    elif claims == 2:
        m *= 1.30
    elif claims >= 3:
        m *= 1.50

    # Security features (discounts)
    if risk_factors.get('tracker'):
        m *= 0.85
    if risk_factors.get('alarm'):
        m *= 0.90
    if risk_factors.get('immobilizer'):
        m *= 0.88

    # Parking  — FIX: compare after lowercasing
    parking = risk_factors.get('parking_type', 'Street').lower()
    if parking == 'street':
        m *= 1.25
    elif parking == 'carport':
        m *= 1.10

    # Married discount
    if risk_factors.get('married'):
        m *= 0.95

    # Peak hours
    if risk_factors.get('peak_hours_usage'):
        m *= 1.10

    # Usage  — FIX: compare after lowercasing
    usage = risk_factors.get('vehicle_usage', 'Daily Commute').lower()
    if 'business' in usage:
        m *= 1.20
    elif 'daily' in usage:
        m *= 1.05
    elif 'weekend' in usage:
        m *= 0.92
    elif 'occasional' in usage:
        m *= 0.95

    # Vehicle value surcharge
    if vehicle_value > 2_000_000:
        m *= 1.30
    elif vehicle_value > 1_000_000:
        m *= 1.20
    elif vehicle_value > 500_000:
        m *= 1.10

    # Area risk
    theft_rate  = risk_factors.get('theft_rate',  2.5)
    hijack_rate = risk_factors.get('hijack_rate', 2.5)
    if theft_rate > 4.0 or hijack_rate > 4.0:
        m *= 1.25
    elif theft_rate > 3.0 or hijack_rate > 3.0:
        m *= 1.10

    # Annual mileage
    annual_mileage = risk_factors.get('annual_mileage', 15_000)
    if annual_mileage > 30_000:
        m *= 1.15
    elif annual_mileage > 20_000:
        m *= 1.05

    monthly = (base_premium * m) / 12
    min_p = 350
    max_p = min(vehicle_value * 0.025 / 12, 35_000)
    return max(min_p, min(monthly, max_p))


def apply_premium_multiplier(base_premium: float) -> float:
    return base_premium * 1.10


def calculate_realistic_premiums(
    base_premium: float, vehicle_value: float, risk_factors: dict
) -> dict:
    max_comp  = min(vehicle_value * 0.022 / 12, 28_000)
    max_smart = min(vehicle_value * 0.016 / 12, 20_000)
    max_tp    = min(vehicle_value * 0.008 / 12,  6_000)

    comp  = min(base_premium,        max_comp)
    smart = min(base_premium * 0.75, max_smart)
    tp    = min(base_premium * 0.45, max_tp)

    age    = risk_factors.get('driver_age',    35)
    claims = risk_factors.get('claims_history', 0)

    if age < 25:
        comp  = min(comp  * 1.25, max_comp  * 1.10)
        smart = min(smart * 1.25, max_smart * 1.10)
        tp    = min(tp    * 1.25, max_tp    * 1.10)

    if claims >= 2:
        comp  = min(comp  * 1.30, max_comp  * 1.15)
        smart = min(smart * 1.30, max_smart * 1.15)
        tp    = min(tp    * 1.30, max_tp    * 1.15)

    brand = risk_factors.get('brand', '').lower()
    model = risk_factors.get('model', '').lower()
    if any(t in f"{brand} {model}" for t in ['m3','m4','m5','m8','amg','rs','gt','gtr']):
        comp  = min(comp  * 1.20, 30_000)
        smart = min(smart * 1.20, 22_000)
        tp    = min(tp    * 1.20,  7_000)

    return {
        "comprehensive": round(min(comp,  max_comp),  2),
        "smart":         round(min(smart, max_smart), 2),
        "third_party":   round(min(tp,    max_tp),    2),
    }


# ── Endpoint ───────────────────────────────────────────────────────────────────

@app.post("/get_quote")
def get_quote(req: QuoteRequest):
    try:
        print(f"📨 Received quote request: {req.dict()}")

        # FIX: use _first_not_none so that explicit 0s (no tracker, 0 claims)
        # are respected instead of being skipped by Python's truthiness rules.
        vehicle_value = _first_not_none(req.vehicle_value,   default=150_000.0)
        brand         = _first_not_none(req.brand,           default="Unknown")
        model         = _first_not_none(req.model,           default="Unknown")
        province      = _first_not_none(req.province,        default="Unknown")
        place         = _first_not_none(req.place,           default="Unknown")
        driver_age    = _first_not_none(req.driver_age,      default=35)
        claims        = _first_not_none(req.claims_history, req.claims,       default=0)
        vehicle_year  = _first_not_none(req.vehicle_year,    default=2020)
        annual_mileage= _first_not_none(req.annual_mileage,  default=15_000)
        tracker       = _first_not_none(req.has_tracker,  req.tracker,       default=0)
        alarm         = _first_not_none(req.has_alarm,    req.alarm,         default=0)
        immobilizer   = _first_not_none(req.has_immobilizer, req.immobilizer, default=0)
        married       = _first_not_none(req.married,          default=0)
        peak_hours    = _first_not_none(req.peak_hours_usage, req.peakHours,  default=0)
        color         = _first_not_none(req.color,            default="unknown")
        reg_number    = _first_not_none(req.reg_number,  req.regNumber,      default="UNKNOWN")

        raw_usage   = _first_not_none(req.vehicle_usage,  req.usage,   default="Daily Commute")
        raw_parking = _first_not_none(req.parking_type,   req.parking, default="Street")
        raw_coverage= _first_not_none(req.coverage_type,  req.coverage,default="Comprehensive")

        # Normalise to exact training-data labels
        usage    = _normalise_usage(raw_usage)
        parking  = _normalise_parking(raw_parking)
        coverage = raw_coverage  # Flutter already sends 'Comprehensive' / 'Smart' / 'Third-Party'

        # Fuzzy-match car and area from reference CSVs
        car_input  = req.car_input  or f"{brand} {model}".lower()
        area_input = req.area_input or f"{place} {province}".lower()

        car_info  = best_car_match(car_input,  CARS_DF,  score_cutoff=55) if not CARS_DF.empty  else None
        area_info = best_area_match(area_input, AREAS_DF, score_cutoff=55) if not AREAS_DF.empty else None

        if car_info is None:
            theft_rate    = float(CARS_DF["Theft Rate (%)"].mean()) if not CARS_DF.empty else 2.0
            matched_brand = brand
            matched_model = model
        else:
            theft_rate    = float(car_info["Theft Rate (%)"])
            matched_brand = car_info["Brand"]
            matched_model = car_info["Model"]

        if area_info is None:
            area_hijack      = float(AREAS_DF["Hijacking_Rate"].mean())   if not AREAS_DF.empty else 2.0
            crime_level_num  = float(AREAS_DF["Crime_Level_Num"].mean())  if not AREAS_DF.empty else 1.0
            matched_province = province
            matched_place    = place
        else:
            area_hijack      = float(area_info["Hijacking_Rate"])
            crime_level_num  = float(area_info["Crime_Level_Num"])
            matched_province = area_info["Province"]
            matched_place    = area_info["Place"]

        risk_factors = {
            'driver_age':     driver_age,
            'claims_history': claims,
            'tracker':        tracker,
            'alarm':          alarm,
            'immobilizer':    immobilizer,
            'parking_type':   parking,
            'married':        married,
            'peak_hours_usage': peak_hours,
            'vehicle_usage':  usage,
            'annual_mileage': annual_mileage,
            'theft_rate':     theft_rate,
            'hijack_rate':    area_hijack,
            'brand':          brand,
            'model':          model,
        }

        print(f"🧮 {brand} {model} | value=R{vehicle_value:,.0f} | usage={usage} | parking={parking}")

        # ML model prediction (preferred) or fallback
        if MODEL is not None:
            try:
                payload = {
                    "Brand":          matched_brand,
                    "Model":          matched_model,
                    "Province":       matched_province,
                    "Place":          matched_place,
                    "Usage":          usage,           # normalised to training labels
                    "Parking":        parking,         # normalised to training labels
                    "Coverage":       coverage,
                    "Theft_Rate":     theft_rate,
                    "Area_Hijack":    area_hijack,
                    "Crime_Level_Num":crime_level_num,
                    "Driver_Age":     driver_age,
                    "Claims":         claims,
                    "Vehicle_Value":  vehicle_value,
                    "Vehicle_Year":   vehicle_year,
                    "Annual_Mileage": annual_mileage,
                    "Tracker":        tracker,
                    "Alarm":          alarm,
                    "Immobilizer":    immobilizer,
                    "Married":        married,
                    "regNumber":      reg_number,
                    "color":          color,
                    "peakHours":      peak_hours,
                }
                df_pred      = pd.DataFrame([payload])
                base_premium = float(MODEL.predict(df_pred)[0])
                base_premium = apply_premium_multiplier(base_premium)
                method       = "model"
                print(f"🤖 Model prediction: R{base_premium:,.2f}")
            except Exception as model_err:
                print(f"⚠️  Model failed: {model_err} — using fallback")
                base_premium = calculate_fallback_premium(vehicle_value, risk_factors)
                method       = "fallback (model error)"
        else:
            base_premium = calculate_fallback_premium(vehicle_value, risk_factors)
            method       = "fallback (no model)"
            print(f"📊 Fallback: R{base_premium:,.2f}")

        premiums = calculate_realistic_premiums(base_premium, vehicle_value, risk_factors)
        print(
            f"💰 Comprehensive=R{premiums['comprehensive']:,} | "
            f"Smart=R{premiums['smart']:,} | TP=R{premiums['third_party']:,}"
        )

        return {
            "base_monthly":       premiums["comprehensive"],
            "comprehensive":      premiums["comprehensive"],
            "smart":              premiums["smart"],
            "third_party":        premiums["third_party"],
            "matched_brand":      matched_brand,
            "matched_model":      matched_model,
            "matched_place":      matched_place,
            "calculation_method": method,
            "received_data": {
                "vehicle_value": vehicle_value,
                "brand":         brand,
                "model":         model,
                "province":      province,
                "place":         place,
                "usage":         usage,
                "parking":       parking,
            },
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating quote: {e}")


@app.get("/")
def root():
    return {
        "message":          "AutoSure Insurance API is running",
        "model_loaded":     MODEL is not None,
        "cars_data_loaded": not CARS_DF.empty,
        "areas_data_loaded":not AREAS_DF.empty,
    }


@app.get("/test")
def test_endpoint():
    return {
        "status":          "API is working!",
        "timestamp":       pd.Timestamp.now().isoformat(),
        "model_available": MODEL is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)