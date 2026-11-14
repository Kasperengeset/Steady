"""Namespaced avatar and custom-habit facade; canonical social/prompt routes remain separate."""

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlmodel import Session

from .. import appie, baseline, rewards
from ..db import get_session

router = APIRouter(tags=["appie"])


class FeedIn(BaseModel):
    amount: int = Field(default=5, ge=1, le=500)
    request_id: str | None = Field(default=None, min_length=1, max_length=100)


class HabitIn(BaseModel):
    label: str = Field(min_length=2, max_length=40)

    @field_validator("label")
    @classmethod
    def meaningful(cls, value: str) -> str:
        value = " ".join(value.split())
        if len(value) < 2:
            raise ValueError("Habit name must contain at least two characters.")
        return value


class ProfileIn(BaseModel):
    avatar_name: str = Field(min_length=1, max_length=16)

    @field_validator("avatar_name")
    @classmethod
    def meaningful(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Appie's name cannot be empty.")
        return value


@router.get("/patients/{pid}/appie")
def get_appie(pid: int, s: Session = Depends(get_session)) -> dict:
    return appie.view(s, pid)


@router.post("/patients/{pid}/appie/feed")
def feed(pid: int, body: FeedIn, idempotency_key: str | None = Header(default=None, max_length=100), s: Session = Depends(get_session)) -> dict:
    try:
        return appie.feed(s, pid, body.amount, body.request_id or idempotency_key)
    except rewards.InsufficientBalance as exc:
        raise HTTPException(400, str(exc)) from exc


@router.post("/patients/{pid}/appie/habits")
def add_habit(pid: int, body: HabitIn, s: Session = Depends(get_session)) -> dict:
    habit = appie.add_habit(s, pid, body.label)
    return {"key": habit.key, "label": habit.label}


@router.put("/patients/{pid}/appie/habits/{key}")
def rename_habit(pid: int, key: str, body: HabitIn, s: Session = Depends(get_session)) -> dict:
    try:
        habit = appie.change_habit(s, pid, key, body.label)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"key": habit.key, "label": habit.label}


@router.delete("/patients/{pid}/appie/habits/{key}")
def remove_habit(pid: int, key: str, s: Session = Depends(get_session)) -> dict:
    try:
        appie.change_habit(s, pid, key)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc
    return {"ok": True}


@router.post("/patients/{pid}/appie/habits/{key}/tick")
def tick_habit(pid: int, key: str, s: Session = Depends(get_session)) -> dict:
    try:
        return appie.tick_habit(s, pid, key)
    except LookupError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.put("/patients/{pid}/appie/profile")
def profile(pid: int, body: ProfileIn, s: Session = Depends(get_session)) -> dict:
    pet = appie._prepare(s, pid, baseline.clock())
    pet.name = body.avatar_name
    s.add(pet)
    s.commit()
    return {"avatar_name": pet.name}
