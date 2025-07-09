import os
import replicate
import openai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Получаем ключи из переменных среды
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI
app = FastAPI()

# Разрешаем все запросы
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Входная модель
class DreamRequest(BaseModel):
    dream: str

# Ответная модель
class DreamResponse(BaseModel):
    interpretation: str
    video_url: str

@app.on_event("startup")
def check_env():
    if not REPLICATE_API_TOKEN or not OPENAI_API_KEY:
        print("❌ ENV ERROR: Missing REPLICATE_API_TOKEN or OPENAI_API_KEY")
    else:
        print("✅ ENV OK: Tokens loaded")

# Основной обработчик
@app.post("/dream", response_model=DreamResponse)
async def interpret_dream(request: DreamRequest):
    try:
        # Трактовка сна
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты психолог-аналитик. Объясни сон глубоко, по Юнгу и Фрейду."},
                {"role": "user", "content": request.dream}
            ],
            max_tokens=500
        )
        interpretation = response.choices[0].message["content"].strip()

        # Генерация видео
        output = replicate.run(
            "cjwbw/video-to-video:8e24824b2c246b85bbfe05877e6caa69694491cbfb8b0f063f1fb681818e224d",
            input={"prompt": request.dream}
        )

        return DreamResponse(
            interpretation=interpretation,
            video_url=output
        )

    except Exception as e:
        print(f"❌ Error in /dream handler: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print("🔥 Launching app on port", port)
    uvicorn.run("main:app", host="0.0.0.0", port=port)
