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

    try:
        print("📘 Запрос к OpenAI...")
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"system","content":"Ты толкователь снов по Евгению Цветкову. Дай трактовку."},
                {"role":"user","content": request.dream}
            ],
            max_tokens=500
        )
        interpretation = resp.choices[0].message.content.strip()
        print("✅ Трактовка готова")
    except Exception as e:
        print(f"❌ OpenAI ошибка:", e)
        interpretation = "Не удалось получить трактовку."

    try:
        print("🎥 Генерация видео через seedance-1-lite...")
        output = replicate.run(
            "bytedance/seedance-1-lite",
            input={
                "prompt": request.dream,
                "video_length": "5s",
                "resolution": "720p"
            }
        )
        video_url = output[0] if isinstance(output, list) else ""
        print(f"✅ Видео URL: {video_url}")
    except Exception as e:
        print(f"❌ Replicate ошибка:", e)
        video_url = ""

    return DreamResponse(interpretation=interpretation, video_url=video_url)
