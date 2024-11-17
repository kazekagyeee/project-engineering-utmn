import pytest
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from main import predict_emotions

tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")


def test_predict_emotions_format():
    text = "I am very happy today!"
    result = predict_emotions(text)

    assert isinstance(result, dict), "Результат должен быть словарем"
    assert len(result) == 28, "Должно быть 28 эмоций"

    for emotion, score in result.items():
        assert isinstance(score, float), f"Значение эмоции {emotion} должно быть числом"

def test_predict_emotions_values():
    text = "I am so angry right now."
    result = predict_emotions(text)

    assert result["anger"] > 0.5, "Эмоция 'anger' должна иметь высокое значение для данного текста"
    assert result["joy"] < 0.5, "Эмоция 'joy' должна иметь низкое значение для данного текста"
