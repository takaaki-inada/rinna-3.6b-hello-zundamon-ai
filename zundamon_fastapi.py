from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from scripts.webui_streaming import generate

app = FastAPI()

# CORS設定
origins = [
    "*",
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateInput(BaseModel):
    text: str
    max_new_tokens: int = 128


# how to start server
# ```
# uvicorn zundamon_fastapi:app --reload --port 8000
# ```

@app.post("/generate")
async def generate_post(input: GenerateInput):
    return StreamingResponse(generate(input.text, max_new_tokens=input.max_new_tokens), media_type="application/json")
