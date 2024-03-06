from fastapi import FastAPI 
from joblib import  load
from pydantic import  BaseModel
import pathlib

app = FastAPI()

class PredictionInput(BaseModel):
    # Define input parameters required for prediction
    passenger_count: int
    pickup_longitude: float
    pickup_latitude: float
    dropoff_longitude: float
    dropoff_latitude: float
    store_and_fwd_flag: int
    pickup_day: int
    pickup_month: int
    pickup_year : int
    pickup_Hour : int       
    pickup_Minute: int
    pickup_Second :int

# load the pre-trained Randomforest regression model
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent
model_path = home_dir.as_posix() + '/models/model.joblib'
model = load(model_path)

@app.get("/")
def home():
    # add customized UI where we can upload csv or tsv file to do prediction
    return "working fine"

@app.post("/predict")
def predict(input_data: PredictionInput):
     # Extract features from input_data and make predictions using the loaded model
    features = [input_data.passenger_count,
                input_data.pickup_longitude,
                input_data.pickup_latitude,
                input_data.dropoff_longitude,
                input_data.dropoff_latitude,
                input_data.store_and_fwd_flag,
                input_data.pickup_day,
                input_data.pickup_month,
                input_data.pickup_year,
                input_data.pickup_Hour,
                input_data.pickup_Minute,
                input_data.pickup_Second,
                ]
    prediction = model.predict([features]).item()
    # return the prediciton
    return prediction

if __name__=="__main__":
    import uvicorn 
    uvicorn.run(app, host="0.0.0.0", port= 8088)

