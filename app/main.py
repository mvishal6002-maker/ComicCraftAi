from fastapi import FastAPI
from .routes import router


app = FastAPI(
    title="ComicCraft AI",
    version="1.0.0",
)

app.include_router(router)