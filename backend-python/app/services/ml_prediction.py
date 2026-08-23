import os
import joblib
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Dynamically find the backend-python/artifacts directory
BACKEND_DIR = Path(__file__).parent.parent.parent
ARTIFACTS_DIR = BACKEND_DIR / "artifacts"

HARVEST_MODEL_PATH = ARTIFACTS_DIR / "harvest_model.joblib"
BIOMASS_MODEL_PATH = ARTIFACTS_DIR / "biomass_model.joblib"

class MLPredictionService:
    def __init__(self):
        self.harvest_model = None
        self.biomass_model = None
        self.load_models()

    def load_models(self):
        try:
            self.harvest_model = joblib.load(HARVEST_MODEL_PATH)
            self.biomass_model = joblib.load(BIOMASS_MODEL_PATH)
            print("✅ ML Models loaded successfully!")
        except FileNotFoundError as e:
            print(f"❌ Error: Could not find .joblib files. Searched in: {ARTIFACTS_DIR}")

    async def predict_farm(self, farm_data: dict, weather_data: dict) -> dict:
        sowing_date = datetime.strptime(farm_data["sowing_date"], "%Y-%m-%d")
        days_since_sowing = max(1, (datetime.now() - sowing_date).days)

        inference_df = pd.DataFrame([{
            'Crop_Standard': farm_data["crop"],
            'location': farm_data["location"],
            'farm_area_acres': float(farm_data["farm_area"]),
            'days_since_sowing': days_since_sowing,
            'Temperature': float(weather_data.get("temperature", 28.0)),
            'Rainfall': float(weather_data.get("rainfall", 400.0)),
            'Humidity': float(weather_data.get("humidity", 60.0)),
            'ndvi': float(weather_data.get("ndvi", 0.75)),
            'evi': float(weather_data.get("evi", 0.50)),
            'N': float(farm_data.get("n_value", 150.0)),
            'P': float(farm_data.get("p_value", 20.0)),
            'K': float(farm_data.get("k_value", 120.0)),
            'pH': float(farm_data.get("ph_value", 7.5))
        }])

        # ... (previous inference code) ...
        
        days_remaining = int(round(self.harvest_model.predict(inference_df)[0]))
        biomass_tons = round(float(self.biomass_model.predict(inference_df)[0]), 1)
        harvest_date = (datetime.now() + timedelta(days=days_remaining)).strftime("%d %b %Y")

        # --- DYNAMIC CONFIDENCE CALCULATION ---
        base_confidence = 96
        
        # Penalty 1: Missing Soil Data (Farmer didn't provide N, P, K, pH)
        if farm_data.get("n_value") is None or farm_data.get("p_value") is None:
            base_confidence -= 8  # We had to guess the soil, so we are less confident
            
        # Penalty 2: Extreme Weather (Harder to predict yields in floods/droughts)
        rainfall = float(weather_data.get("rainfall", 400.0))
        if rainfall > 800.0 or rainfall < 100.0:
            base_confidence -= 5 
            
        # Ensure it stays within realistic bounds
        final_confidence = max(65, min(99, base_confidence))

        return {
            "predicted_harvest_date": harvest_date,
            "harvest_expected_in_days": days_remaining,
            "available_stubble_tons": biomass_tons,
            "confidence": f"{final_confidence}%"
        }

ml_service = MLPredictionService()