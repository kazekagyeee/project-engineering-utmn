import streamlit as st
from app.predictions import predict_emotions

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
