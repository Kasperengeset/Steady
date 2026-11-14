from fastapi import APIRouter

from ..schemas import SimIn
from ..sim.simulator import demo_lead_time, simulate

router = APIRouter(tags=["sim"])


@router.post("/sim/run")
def run(body: SimIn) -> dict:
    """One synthetic patient through the real engine – days[], summary, assumptions."""
    return simulate(days=body.days, seed=body.seed, relapse_day=body.relapse_day)


@router.get("/sim/robustness")
def robustness(seeds: int = 50) -> dict:
    """Same scenario over many random personas – the honest numbers for the pitch."""
    return demo_lead_time(min(seeds, 200))
