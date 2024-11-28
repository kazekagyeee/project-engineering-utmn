import pytest
from main import predict_emotions

EXPECTED_EMOTION_COUNT = 28

def test_predict_emotions_format():
    text = "I am very happy today!"
    result = predict_emotions(text)

    assert isinstance(result, dict), "Результат должен быть словарем"
    assert len(result) == EXPECTED_EMOTION_COUNT, f"Ожидается {EXPECTED_EMOTION_COUNT} эмоций, но получено {len(result)}."

    for emotion, score in result.items():
        assert isinstance(score, float), f"Значение эмоции {emotion} должно быть числом"

def test_predict_emotions_values():
    text = "I am so angry right now."
    result = predict_emotions(text)

    assert result["anger"] > 0.5, "Эмоция 'anger' должна иметь высокое значение для данного текста"
    assert result["joy"] < 0.5, "Эмоция 'joy' должна иметь низкое значение для данного текста"
