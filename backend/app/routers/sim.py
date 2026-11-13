from fastapi import APIRouter

from ..schemas import SimIn
from ..sim.simulator import simulate

router = APIRouter(tags=["sim"])


@router.post("/sim/run")
def run(body: SimIn) -> list[dict]:
    return simulate(body.days, body.seed)
