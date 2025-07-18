from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import openai
import replicate
import os

# Подключаем ключи
openai.api_key = os.environ["OPENAI_API_KEY"]
os.environ["REPLICATE_API_TOKEN"] = os.environ["REPLICATE_API_TOKEN"]

app = FastAPI()

# Разрешаем CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Модель запроса
class DreamRequest(BaseModel):
    dream: str

# Модель ответа
class DreamResponse(BaseModel):
    interpretation: str
    video_url: str

@app.post("/dream", response_model=DreamResponse)
async def dream_endpoint(request: DreamRequest):
    # Трактовка сна
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты толкователь снов по Евгению Цветкову. Дай трактовку сна по классической русской традиции Цветкова. Не упоминай, что ты ИИ."},
                {"role": "user", "content": request.dream}
            ],
            max_tokens=500
        )
        interpretation = response.choices[0].message.content.strip()
    except Exception:
        interpretation = "Не удалось получить трактовку."

    # Генерация видео
    try:
        output = replicate.run(
            "cjwbw/video-to-video:8e24824b2c246b85bbfe05877e6caa69694491cbfb8b0f063f1fb681818e224d",
            input={"prompt": request.dream}
        )
        video_url = output[0] if isinstance(output, list) and output else ""
    except Exception:
        video_url = ""

    return DreamResponse(interpretation=interpretation, video_url=video_url)
