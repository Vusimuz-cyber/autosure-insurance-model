from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import os
from src.utils import load_reference_data, best_car_match, best_area_match
import numpy as np
from typing import Optional, Dict, Any

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Get the current directory and go one level up
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)  # This goes from /src to /insurance-model

# Load model and data with correct paths - files are in parent directory
MODEL_PATH = os.path.join(PARENT_DIR, "models", "insurance_model.pkl")
CARS_DATA_PATH = os.path.join(PARENT_DIR, "data", "cars.csv") 
AREAS_DATA_PATH = os.path.join(PARENT_DIR, "data", "areas.csv")

print(f"🔍 Looking for model at: {MODEL_PATH}")
print(f"🔍 Looking for cars data at: {CARS_DATA_PATH}")
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

# FLEXIBLE request model that accepts ANY data from Flutter
class QuoteRequest(BaseModel):
    # Accept ANY fields the Flutter app sends
    vehicle_value: Optional[float] = None
    brand: Optional[str] = None
    model: Optional[str] = None
    province: Optional[str] = None
    place: Optional[str] = None
    driver_age: Optional[int] = None
    claims_history: Optional[int] = None
    vehicle_year: Optional[int] = None
    annual_mileage: Optional[int] = None
    vehicle_usage: Optional[str] = None
    parking_type: Optional[str] = None
    has_tracker: Optional[int] = None
    has_alarm: Optional[int] = None
    has_immobilizer: Optional[int] = None
    married: Optional[int] = None
    peak_hours_usage: Optional[int] = None
    color: Optional[str] = None
    reg_number: Optional[str] = None
    coverage_type: Optional[str] = None
    
    # Also accept the OLD field names for backward compatibility
    car_input: Optional[str] = None
    area_input: Optional[str] = None
    claims: Optional[int] = None
    usage: Optional[str] = None
    parking: Optional[str] = None
    tracker: Optional[int] = None
    alarm: Optional[int] = None
    immobilizer: Optional[int] = None
    regNumber: Optional[str] = None
    peakHours: Optional[int] = None
    coverage: Optional[str] = None

def calculate_fallback_premium(vehicle_value: float, risk_factors: dict) -> float:
    """Calculate premium using fallback method - REALISTIC PREMIUMS"""
    if vehicle_value <= 0:
        vehicle_value = 150000  # Default value
    
    # REALISTIC: Base premium from 1.2% to 2% annually for normal vehicles
    # For luxury/high-performance, we'll cap it
    base_annual_rate = 0.015  # 1.5% annually (much more reasonable)
    base_premium = vehicle_value * base_annual_rate
    
    risk_multiplier = 1.0
    
    # Age factor - REALISTIC impact
    age = risk_factors.get('driver_age', 35)
    if age < 25:
        risk_multiplier *= 1.3  # Young drivers pay more but not extreme
    elif age < 30:
        risk_multiplier *= 1.15  # Young adults pay slightly more
    elif age > 65:
        risk_multiplier *= 1.2  # Senior drivers pay more
    
    # Claims history - REALISTIC impact
    claims = risk_factors.get('claims_history', 0) or risk_factors.get('claims', 0)
    if claims == 1:
        risk_multiplier *= 1.15
    elif claims == 2:
        risk_multiplier *= 1.3
    elif claims >= 3:
        risk_multiplier *= 1.5
    
    # Security features - REALISTIC discounts
    if risk_factors.get('has_tracker') or risk_factors.get('tracker'):
        risk_multiplier *= 0.85  # Good discount
    if risk_factors.get('has_alarm') or risk_factors.get('alarm'):
        risk_multiplier *= 0.90
    if risk_factors.get('has_immobilizer') or risk_factors.get('immobilizer'):
        risk_multiplier *= 0.88
    
    # Parking type - REALISTIC impact
    parking = risk_factors.get('parking_type') or risk_factors.get('parking', 'street')
    if parking == 'street':
        risk_multiplier *= 1.25  # Street parking is riskier
    elif parking == 'carport':
        risk_multiplier *= 1.1
    
    # Married discount - MODERATE
    if risk_factors.get('married'):
        risk_multiplier *= 0.95
    
    # Peak hours - REALISTIC impact
    if risk_factors.get('peak_hours_usage') or risk_factors.get('peakHours'):
        risk_multiplier *= 1.1
    
    # Vehicle usage impact - REALISTIC
    usage = risk_factors.get('vehicle_usage', '').lower()
    if 'business' in usage:
        risk_multiplier *= 1.2  # Business use increases premium
    elif 'daily' in usage:
        risk_multiplier *= 1.05   # Daily commute slightly increases premium
    elif 'weekend' in usage:
        risk_multiplier *= 0.92   # Weekend only gets small discount
    
    # High-value vehicle surcharge - REALISTIC but CAPPED
    if vehicle_value > 2000000:  # R2M+ vehicles
        risk_multiplier *= 1.3   # 30% surcharge for super luxury
    elif vehicle_value > 1000000:  # R1M-R2M vehicles
        risk_multiplier *= 1.2   # 20% surcharge for luxury
    elif vehicle_value > 500000:  # R500k-R1M vehicles
        risk_multiplier *= 1.1   # 10% surcharge for premium
    
    # High-risk area multiplier (based on theft and hijack rates)
    theft_rate = risk_factors.get('theft_rate', 2.5)
    hijack_rate = risk_factors.get('hijack_rate', 2.5)
    
    if theft_rate > 4.0 or hijack_rate > 4.0:
        risk_multiplier *= 1.25  # High crime areas
    elif theft_rate > 3.0 or hijack_rate > 3.0:
        risk_multiplier *= 1.1  # Medium crime areas
    
    # High-mileage surcharge
    annual_mileage = risk_factors.get('annual_mileage', 15000)
    if annual_mileage > 30000:
        risk_multiplier *= 1.15
    elif annual_mileage > 20000:
        risk_multiplier *= 1.05
    
    # Convert annual to monthly
    monthly_premium = (base_premium * risk_multiplier) / 12
    
    # REALISTIC minimum premium with reasonable maximums
    min_premium = 350
    max_premium = min(vehicle_value * 0.025 / 12, 35000)  # Max 2.5% annually, capped at R35k
    
    return max(min_premium, min(monthly_premium, max_premium))

def apply_premium_multiplier(base_premium: float) -> float:
    """Apply reasonable multiplier to premiums"""
    return base_premium * 1.1  # Only 10% increase (very reasonable)

def calculate_realistic_premiums(base_premium: float, vehicle_value: float, risk_factors: dict) -> dict:
    """Calculate realistic premium tiers with proper caps for luxury vehicles"""
    
    # Set reasonable maximums based on vehicle value - MORE REALISTIC
    max_comprehensive = min(vehicle_value * 0.022 / 12, 28000)  # Max 2.2% annually, capped at R28k monthly
    max_smart = min(vehicle_value * 0.016 / 12, 20000)  # Max 1.6% annually, capped at R20k monthly
    max_third_party = min(vehicle_value * 0.008 / 12, 6000)  # Max 0.8% annually, capped at R6k monthly
    
    # Calculate base premiums
    comprehensive = min(base_premium, max_comprehensive)
    smart = min(base_premium * 0.75, max_smart)  # 25% discount for smart plan
    third_party = min(base_premium * 0.45, max_third_party)  # 55% discount for third party
    
    # Apply additional risk-based adjustments
    age = risk_factors.get('driver_age', 35)
    claims = risk_factors.get('claims_history', 0)
    
    # Young driver surcharge (capped)
    if age < 25:
        comprehensive = min(comprehensive * 1.25, max_comprehensive * 1.1)
        smart = min(smart * 1.25, max_smart * 1.1)
        third_party = min(third_party * 1.25, max_third_party * 1.1)
    
    # High claims surcharge (capped)
    if claims >= 2:
        comprehensive = min(comprehensive * 1.3, max_comprehensive * 1.15)
        smart = min(smart * 1.3, max_smart * 1.15)
        third_party = min(third_party * 1.3, max_third_party * 1.15)
    
    # High-performance vehicle adjustment (BMW M3, etc.)
    brand = risk_factors.get('brand', '').lower()
    model = risk_factors.get('model', '').lower()
    
    # High-performance surcharge but still reasonable
    if any(term in f"{brand} {model}".lower() for term in ['m3', 'm4', 'm5', 'm8', 'amg', 'rs', 'gt', 'gtr']):
        comprehensive = min(comprehensive * 1.2, 30000)  # 20% surcharge but max R30k
        smart = min(smart * 1.2, 22000)
        third_party = min(third_party * 1.2, 7000)
    
    # Ensure premiums are realistic and rounded
    comprehensive = round(min(comprehensive, max_comprehensive), 2)
    smart = round(min(smart, max_smart), 2)
    third_party = round(min(third_party, max_third_party), 2)
    
    return {
        "comprehensive": comprehensive,
        "smart": smart,
        "third_party": third_party
    }

@app.post("/get_quote")
def get_quote(req: QuoteRequest):
    try:
        print(f"📨 Received quote request with data: {req.dict()}")
        
        # Extract values with fallbacks - handle BOTH old and new field names
        vehicle_value = req.vehicle_value or 150000
        brand = req.brand or "Unknown"
        model = req.model or "Unknown"
        province = req.province or "Unknown"
        place = req.place or "Unknown"
        
        # Handle both field naming conventions
        driver_age = req.driver_age or 35
        claims = req.claims_history or req.claims or 0
        vehicle_year = req.vehicle_year or 2020
        annual_mileage = req.annual_mileage or 15000
        
        # Usage - try multiple field names
        usage = req.vehicle_usage or req.usage or "Daily Commute"
        
        # Parking - try multiple field names
        parking = req.parking_type or req.parking or "Street"
        
        # Security features - try multiple field names
        tracker = req.has_tracker or req.tracker or 0
        alarm = req.has_alarm or req.alarm or 0
        immobilizer = req.has_immobilizer or req.immobilizer or 0
        
        # Other fields
        married = req.married or 0
        peak_hours = req.peak_hours_usage or req.peakHours or 0
        color = req.color or "unknown"
        reg_number = req.reg_number or req.regNumber or "UNKNOWN"
        coverage = req.coverage_type or req.coverage or "Comprehensive"
        
        # Get car and area info using whatever data we have
        car_input = req.car_input or f"{brand} {model}".lower()
        area_input = req.area_input or f"{place} {province}".lower()
        
        print(f"🔍 Looking up car: {car_input}, area: {area_input}")
        
        car_info = best_car_match(car_input, CARS_DF, score_cutoff=55) if not CARS_DF.empty else None
        area_info = best_area_match(area_input, AREAS_DF, score_cutoff=55) if not AREAS_DF.empty else None

        if car_info is None:
            theft_rate = float(CARS_DF["Theft Rate (%)"].mean()) if not CARS_DF.empty else 2.0  # Lower default
            matched_brand = brand
            matched_model = model
            print(f"🚗 No exact car match found, using average theft rate: {theft_rate}")
        else:
            theft_rate = float(car_info["Theft Rate (%)"])
            matched_brand = car_info["Brand"]
            matched_model = car_info["Model"]
            print(f"✅ Matched car: {matched_brand} {matched_model}, theft rate: {theft_rate}")

        if area_info is None:
            area_hijack = float(AREAS_DF["Hijacking_Rate"].mean()) if not AREAS_DF.empty else 2.0  # Lower default
            crime_level_num = float(AREAS_DF["Crime_Level_Num"].mean()) if not AREAS_DF.empty else 1.0  # Lower default
            matched_province = province
            matched_place = place
            print(f"📍 No exact area match found, using average hijack rate: {area_hijack}")
        else:
            area_hijack = float(area_info["Hijacking_Rate"])
            crime_level_num = float(area_info["Crime_Level_Num"])
            matched_province = area_info["Province"]
            matched_place = area_info["Place"]
            print(f"✅ Matched area: {matched_place}, {matched_province}, hijack rate: {area_hijack}")

        # Risk factors for realistic premium calculation
        risk_factors = {
            'driver_age': driver_age,
            'claims_history': claims,
            'tracker': tracker,
            'alarm': alarm,
            'immobilizer': immobilizer,
            'parking_type': parking,
            'married': married,
            'peak_hours_usage': peak_hours,
            'vehicle_usage': usage,
            'annual_mileage': annual_mileage,
            'theft_rate': theft_rate,
            'hijack_rate': area_hijack,
            'brand': brand,
            'model': model
        }

        print(f"🧮 Calculating premium for {brand} {model} valued at R{vehicle_value:,.2f}")

        # Calculate base premium - use model if available, otherwise fallback
        if MODEL is not None:
            try:
                # Prepare payload for model prediction
                payload = {
                    "Brand": matched_brand,
                    "Model": matched_model,
                    "Province": matched_province,
                    "Place": matched_place,
                    "Usage": usage,
                    "Parking": parking,
                    "Coverage": coverage,
                    "Theft_Rate": theft_rate,
                    "Area_Hijack": area_hijack,
                    "Crime_Level_Num": crime_level_num,
                    "Driver_Age": driver_age,
                    "Claims": claims,
                    "Vehicle_Value": vehicle_value,
                    "Vehicle_Year": vehicle_year,
                    "Annual_Mileage": annual_mileage,
                    "Tracker": tracker,
                    "Alarm": alarm,
                    "Immobilizer": immobilizer,
                    "Married": married,
                    "regNumber": reg_number,
                    "color": color,
                    "peakHours": peak_hours
                }
                
                df = pd.DataFrame([payload])
                pred = MODEL.predict(df)[0]
                base_premium = float(pred)
                # Apply very reasonable multiplier
                base_premium = apply_premium_multiplier(base_premium)
                method = "model (with reasonable multiplier)"
                print(f"🤖 Model prediction (after multiplier): R{base_premium:,.2f}")
            except Exception as model_error:
                print(f"⚠️ Model prediction failed: {model_error}, using fallback")
                base_premium = calculate_fallback_premium(vehicle_value, risk_factors)
                method = "fallback (model failed)"
        else:
            base_premium = calculate_fallback_premium(vehicle_value, risk_factors)
            method = "fallback (no model)"
            print(f"📊 Fallback calculation: R{base_premium:,.2f}")

        # Calculate realistic premium tiers with caps
        premiums = calculate_realistic_premiums(base_premium, vehicle_value, risk_factors)

        print(f"💰 Final premiums - Comprehensive: R{premiums['comprehensive']:,}, Smart: R{premiums['smart']:,}, Third-party: R{premiums['third_party']:,}")

        return {
            "base_monthly": premiums["comprehensive"],
            "comprehensive": premiums["comprehensive"],
            "smart": premiums["smart"],
            "third_party": premiums["third_party"],
            "matched_brand": matched_brand,
            "matched_model": matched_model,
            "matched_place": matched_place,
            "calculation_method": method,
            "received_data": {
                "vehicle_value": vehicle_value,
                "brand": brand,
                "model": model,
                "province": province,
                "place": place
            }
        }
        
    except Exception as e:
        print(f"❌ Error in get_quote: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error calculating quote: {str(e)}")

@app.get("/")
def root():
    return {
        "message": "AutoSure Insurance API is running", 
        "model_loaded": MODEL is not None,
        "cars_data_loaded": not CARS_DF.empty,
        "areas_data_loaded": not AREAS_DF.empty
    }

@app.get("/test")
def test_endpoint():
    """Test endpoint to verify API is working"""
    return {
        "status": "API is working!",
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_available": MODEL is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)