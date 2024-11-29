from fastapi import FastAPI
from app.routers import api

app = FastAPI()

# Подключаем маршруты
app.include_router(api.router)