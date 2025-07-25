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

    # Трактовка
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты толкователь снов по Евгению Цветкову. Дай краткую трактовку сна без лишней воды."},
                {"role": "user", "content": request.dream}
            ],
            max_tokens=500
        )
        interpretation = response.choices[0].message.content.strip()
        print("✅ Трактовка получена")
    except Exception as e:
        print("❌ OpenAI ошибка:", e)
        interpretation = "Не удалось получить трактовку сна."

    # Видео (pixverse-v4)
    try:
        print("🎥 Генерация видео через pixverse-v4...")
        output = replicate.run(
            "pixverse/pixverse-v4",
            input={
                "prompt": request.dream,
                "num_inference_steps": 25,
                "guidance_scale": 7.5,
                "width": 576,
                "height": 320
            }
        )
        video_url = output[0] if isinstance(output, list) else ""
        print("✅ Видео URL:", video_url)
    except Exception as e:
        print("❌ Replicate ошибка:", e)
        video_url = ""

    return DreamResponse(interpretation=interpretation, video_url=video_url)
