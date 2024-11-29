# Импортируем библиотеки
import streamlit as st
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

# Заголовок приложения
st.title("Emotion Detection with RoBERTa - GoEmotions")

# Поле для ввода текста
user_input = st.text_area("Введите текст для анализа эмоций:")

# Кнопка для запуска анализа эмоций
if st.button("Анализировать эмоции"):
    if user_input:
        results = predict_emotions(user_input)
        sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
        st.subheader("Предсказанные эмоции (отсортировано по вероятности):")
        for emotion, score in sorted_results.items():
            st.write(f"{emotion}: {score:.4f}")
    else:
        st.write("Пожалуйста, введите текст для анализа.")
