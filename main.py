from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from routers.api import router as api_router
from routers.pages import router as pages_router
from services.classifier import load_model_bundle


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "archetype_model.pkl"


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.executor = ThreadPoolExecutor(max_workers=4)
    app.state.templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    app.state.profile_cache = {}
    app.state.inflight_analyses = {}
    app.state.cache_ttl_seconds = 3600
    app.state.model_bundle = load_model_bundle(MODEL_PATH)
    yield
    app.state.executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(title="GitHub Personality Profiler", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.include_router(pages_router)
app.include_router(api_router)
