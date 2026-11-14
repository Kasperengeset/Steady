import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import baseline
from . import push as push_mod
from .config import settings
from .db import engine, init_db
from .routers import baselines, checkins, friends, interventions, patients, prompts, push, sim
from .seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    seed()
    # Warm up the heavy models in the background so the first check-in isn't slow
    threading.Thread(target=_warmup, daemon=True).start()
    stop = threading.Event()
    if push_mod.enabled():  # Web Push for the daily prompt (#22, P2): only with VAPID keys in .env
        threading.Thread(target=push_mod.run_loop, args=(stop, engine, baseline.clock), daemon=True).start()
    yield
    stop.set()


def _warmup() -> None:
    from .pipeline import emotion, prosody

    prosody._smile()
    emotion._model()


app = FastAPI(title="Beyond the Clinic", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)

for r in (checkins.router, interventions.router, patients.router, baselines.router, friends.router, sim.router,
          prompts.router, push.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True}
