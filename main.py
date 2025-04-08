# main.py
from fastapi import FastAPI, Query
import joblib
import numpy as np

# Load model
model = joblib.load("weather_model.pkl")

# Create FastAPI app
app = FastAPI()

@app.get("/predict/")
def predict(
    precipitation: float = Query(..., description="Precipitation"),
    rain: float = Query(..., description="Rain"),
    humidity: float = Query(..., description="Humidity")
):
    # Convert input to numpy array
    input_array = np.array([[precipitation, rain, humidity]])

    # Get prediction
    prediction = model.predict(input_array)[0]

    return {"prediction": int(prediction)}



#Example Usuage:http://127.0.0.1:8000/predict/?precipitation=5.0&rain=3.0&humidity=1.0