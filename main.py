from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import openai
import replicate
import os

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["REPLICATE_API_TOKEN"] = os.getenv("REPLICATE_API_TOKEN")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DreamRequest(BaseModel):
    dream: str

class DreamResponse(BaseModel):
    interpretation: str
    video_url: str

@app.post("/dream", response_model=DreamResponse)
async def dream_endpoint(request: DreamRequest):
    print(f"💤 Получен сон: {request.dream}")

    # 1. Трактовка сна через OpenAI
    try:
        print("📘 Запрос к OpenAI...")
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Ты толкователь снов по Евгению Цветкову. Дай трактовку сна по классической русской традиции Цветкова. Не упоминай, что ты ИИ."
                },
                {"role": "user", "content": request.dream}
            ],
            max_tokens=500
        )
        interpretation = response.choices[0].message.content.strip()
        print("✅ Трактовка готова")
    except Exception as e:
        print(f"❌ OpenAI ошибка: {e}")
        interpretation = "Не удалось получить трактовку сна."

    # 2. Генерация видео через Bytedance
    try:
        print("🎥 Генерация видео через bytedance/seedance-1-pro...")
        output = replicate.run(
            "bytedance/seedance-1-pro",
            input={
                "prompt": request.dream,
                "video_length": "5s",
                "resolution": "480p"
            }
        )
        video_url = output[0] if isinstance(output, list) and output else ""
        print(f"✅ Видео URL: {video_url}")
    except Exception as e:
        print(f"❌ Replicate ошибка: {e}")
        video_url = ""

    return DreamResponse(interpretation=interpretation, video_url=video_url)
