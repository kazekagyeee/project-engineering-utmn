# Импортируем библиотеки
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Загрузка модели и токенизатора с Hugging Face
tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")

# Список эмоций, которые распознает модель
emotions = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion", "curiosity",
    "desire", "disappointment", "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "gratitude", "grief", "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Функция для предсказания эмоций
def predict_emotions(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)
    return {emotion: float(probs[0][i]) for i, emotion in enumerate(emotions)}