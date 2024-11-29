from app.model.input import TextInput
from app.predictions import predict_emotions
from fastapi import APIRouter

router = APIRouter()

@router.post("/predict")
def predict(input_data: TextInput):
    result = predict_emotions(input_data.text)
    sorted_results = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
    return {"predicted_emotions": sorted_results}