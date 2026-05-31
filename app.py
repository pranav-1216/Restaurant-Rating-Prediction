import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Restaurant Rating Prediction API",
    description="Predict restaurant ratings using XGBoost",
    version="1.0"
)

# ==================================================
# LOAD MODELS
# ==================================================
try:
    print("Loading assets...")

    model = joblib.load("advanced_xgb_model.joblib")
    scaler = joblib.load("gps_scaler.joblib")
    kmeans = joblib.load("kmeans_hotspot_model.joblib")

    print("Assets loaded successfully!")

    print("\nMODEL FEATURES:")
    print(model.feature_names_in_)

    print("\nSCALER FEATURES:")
    print(scaler.feature_names_in_)

except Exception as e:
    print(f"ERROR LOADING FILES: {e}")

    if hasattr(model, "feature_names_in_"):
        print("\nModel Features:")
        print(list(model.feature_names_in_))

    if hasattr(scaler, "feature_names_in_"):
        print("\nScaler Features:")
        print(list(scaler.feature_names_in_))

except Exception as e:
    print(f"ERROR LOADING FILES: {e}")


# ==================================================
# INPUT SCHEMA
# ==================================================

class RestaurantData(BaseModel):
    Longitude: float = Field(..., example=77.5946)
    Latitude: float = Field(..., example=12.9716)

    Has_Table_booking: int = Field(..., example=1)
    Has_Online_delivery: int = Field(..., example=1)
    Is_delivering_now: int = Field(..., example=0)
    Switch_to_order_menu: int = Field(..., example=0)

    Price_range: int = Field(..., example=3)
    Cost_in_USD: float = Field(..., example=25.5)

    Cuisines_Encoded: int = Field(..., example=15)
    City_Encoded: int = Field(..., example=2)
    Locality_Encoded: int = Field(..., example=45)


# ==================================================
# ROOT ENDPOINT
# ==================================================

@app.get("/")
def home():
    return {
        "message": "Restaurant Rating Prediction API Running"
    }


# ==================================================
# HEALTH CHECK
# ==================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True
    }


# ==================================================
# PREDICTION ENDPOINT
# ==================================================

@app.post("/predict")
def predict(data: RestaurantData):

    try:

        d = data.model_dump()

        # ------------------------------------------
        # Generate Geo Cluster
        # ------------------------------------------

        coords = [[d["Latitude"], d["Longitude"]]]
        geo_cluster = int(kmeans.predict(coords)[0])

        # ------------------------------------------
        # Create Input Data
        # ------------------------------------------

        input_data = {
            "Longitude": d["Longitude"],
            "Latitude": d["Latitude"],
            "Has Table booking": d["Has_Table_booking"],
            "Has Online delivery": d["Has_Online_delivery"],
            "Is delivering now": d["Is_delivering_now"],
            "Switch to order menu": d["Switch_to_order_menu"],
            "Price range": d["Price_range"],
            "Cost in USD": d["Cost_in_USD"],
            "Cuisines_Encoded": d["Cuisines_Encoded"],
            "City_Encoded": d["City_Encoded"],
            "Locality_Encoded": d["Locality_Encoded"],
            "Geo_Cluster": geo_cluster
        }

        # ------------------------------------------
        # Convert to DataFrame
        # ------------------------------------------

        df = pd.DataFrame([input_data])

        print("\nIncoming DataFrame:")
        print(df)

        # ------------------------------------------
        # Match Model Feature Order
        # ------------------------------------------

        if hasattr(model, "feature_names_in_"):

            required_features = list(model.feature_names_in_)

            missing_features = [
                col for col in required_features
                if col not in df.columns
            ]

            if missing_features:
                return {
                    "status": "error",
                    "message": f"Missing features: {missing_features}"
                }

            df = df[required_features]

        print("\nFinal DataFrame:")
        print(df)

        # ------------------------------------------
        # Scaling
        # ------------------------------------------

        if hasattr(scaler, "feature_names_in_"):

            scaler_features = list(scaler.feature_names_in_)

            scaled_df = df.copy()

            scaled_df[scaler_features] = scaler.transform(
                scaled_df[scaler_features]
            )

        else:
            scaled_df = scaler.transform(df)

        # ------------------------------------------
        # Prediction
        # ------------------------------------------

        prediction = model.predict(scaled_df)

        predicted_rating = round(float(prediction[0]), 2)

        return {
            "status": "Prediction Successful",
            "predicted_rating": predicted_rating,
            "generated_geo_cluster": geo_cluster
        }

    except Exception as e:

        return {
            "status": "error",
            "message": str(e)
        }