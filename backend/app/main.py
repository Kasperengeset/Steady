import asyncio
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import appie, baseline, health_sync
from . import push as push_mod
from .config import settings
from .db import engine, init_db
from .routers import baselines, checkins, friends, interventions, patients, prompts, push, sim
from .routers import companion, rewards, social
from .routers import appie as appie_router, health as health_router, views
from .seed import seed


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    appie.migrate_legacy(engine)
    seed()
    # Warm up the heavy models in the background so the first check-in isn't slow
    if settings.warmup_models:
        threading.Thread(target=_warmup, daemon=True).start()
    stop = threading.Event()
    if push_mod.enabled():  # Web Push for the daily prompt (#22, P2): only with VAPID keys in .env
        threading.Thread(target=push_mod.run_loop, args=(stop, engine, baseline.clock), daemon=True).start()
    task = asyncio.create_task(_wearable_worker()) if settings.health_sync_enabled else None
    try:
        yield
    finally:
        stop.set()
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


async def _wearable_worker():
    while True:
        try:
            await asyncio.to_thread(health_sync.sync_connected_patients, engine)
        except Exception:
            logging.getLogger(__name__).warning("Wearable background sync will retry in an hour.")
        await asyncio.sleep(3600)


def _warmup() -> None:
    from .pipeline import emotion, prosody

    try:
        prosody._smile()
        emotion._model()
    except Exception:
        logging.getLogger(__name__).warning("Voice model warm-up unavailable; text check-ins remain available.")


app = FastAPI(title="Beyond the Clinic", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"]
)

for r in (checkins.router, interventions.router, patients.router, baselines.router, friends.router, sim.router,
          prompts.router, push.router):
    app.include_router(r, prefix="/api")
for r in (rewards.router, social.router, companion.router):  # engagement: #24, #26, #25
    app.include_router(r, prefix="/api")
for r in (appie_router.router, health_router.router, views.router):
    app.include_router(r, prefix="/api")


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "service": "steady", "api_version": "appie-integration-v1",
            "capabilities": ["appie", "health-sync", "prompts", "push", "social-consent"]}
