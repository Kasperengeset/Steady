import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .db import init_db
from .routers import checkins, friends, interventions, patients, sim
from .seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed()
    # Warm up the heavy models in the background so the first check-in isn't slow
    threading.Thread(target=_warmup, daemon=True).start()
    yield


def _warmup() -> None:
    from .pipeline import emotion, prosody

    prosody._smile()
    emotion._model()


app = FastAPI(title="Beyond the Clinic", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)

for r in (checkins.router, interventions.router, patients.router, friends.router, sim.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
