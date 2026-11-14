"""The circle (#26, minimal P1): profile or anonymous, invite by code, a feed, preset reactions.

Never shared: scores, drift, diaries, voice, transcripts, features, Fitbit data, clinician notes, the
escalation level. The feed shows only what each person opted in to: avatar, level, streak, "checked in
today", and an emoji. Every response here is built field by field from the allow-list in _member_card,
and tests/test_social.py fails if a health field ever shows up.
"""

import importlib
import secrets
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlmodel import Session, col, select

from .. import baseline, rewards, safety
from ..db import get_session
from ..models import Escalation, Patient
from ..models_social import Circle, IdentityMode, MemberRole, Membership, NudgeCancel, Profile, Reaction

router = APIRouter(tags=["social"])

PRESETS = {
    "thinking_of_you": "Thinking of you",
    "proud_of_you": "Proud of you",
    "walk_together": "Walk together?",
    "heart": "❤️",
    "wave": "👋",
    "sun": "☀️",
    "hug": "🤗",
}
NUDGE_HOLD = timedelta(minutes=30)  # the patient sees the friends card first and can cancel it
MAX_TEXT = 40
MAX_EMOJI = 8


class ProfileIn(BaseModel):
    mode: IdentityMode | None = None
    display_name: str | None = Field(default=None, max_length=MAX_TEXT)
    pseudonym: str | None = Field(default=None, max_length=MAX_TEXT)
    photo_url: str | None = Field(default=None, max_length=500)
    emoji: str | None = Field(default=None, max_length=MAX_EMOJI)
    share_avatar: bool | None = None
    share_level: bool | None = None
    share_streak: bool | None = None
    share_checked_in: bool | None = None
    social_consent: bool | None = None  # True gives consent, False withdraws it


class CircleIn(BaseModel):
    name: str | None = Field(default=None, max_length=MAX_TEXT)


class JoinIn(BaseModel):
    code: str
    name: str | None = Field(default=None, max_length=MAX_TEXT)  # a friend's name (light friend account)
    patient_id: int | None = None  # someone who uses the app joins with their own avatar


class ReactIn(BaseModel):
    token: str
    to_member_id: int
    kind: str


class TokenIn(BaseModel):
    token: str


# ---------------------------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------------------------


def _patient(s: Session, pid: int) -> Patient:
    p = s.get(Patient, pid)
    if p is None:
        raise HTTPException(404, "patient not found")
    return p


def _profile(s: Session, pid: int) -> Profile:
    return s.get(Profile, pid) or Profile(patient_id=pid)


def _profile_out(p: Profile) -> dict:
    return {
        "mode": p.mode.value, "display_name": p.display_name, "pseudonym": p.pseudonym, "photo_url": p.photo_url,
        "emoji": p.emoji, "share_avatar": p.share_avatar, "share_level": p.share_level,
        "share_streak": p.share_streak, "share_checked_in": p.share_checked_in,
        "social_consent": p.social_consent_at is not None,
    }


def _display(s: Session, m: Membership) -> str:
    if m.role == MemberRole.friend:
        return m.friend_name or "Friend"
    prof = _profile(s, m.patient_id)
    if prof.mode == IdentityMode.profile:
        return prof.display_name or s.get(Patient, m.patient_id).name
    return prof.pseudonym or "Anonymous"


def _avatar_ref(s: Session, pid: int) -> dict | None:
    """The avatar (#23) if that model exists yet: a reference for the share card, never health data."""
    Avatar = None
    for mod in ("..models_avatar", "..models"):
        try:
            Avatar = getattr(importlib.import_module(mod, __package__), "Avatar", None)
        except ImportError:
            continue
        if Avatar is not None:
            break
    if Avatar is None:
        from ..appie import level_of, tier_of
        from ..models_appie import AppiePet

        pet = s.get(AppiePet, pid)
        return {"patient_id": pid, "name": pet.name if pet else "Appie",
                "tier": tier_of(level_of(rewards.xp(s, pid)))}
    av = s.exec(select(Avatar).where(Avatar.patient_id == pid)).first()
    if av is None:
        return {"patient_id": pid}
    return {"patient_id": pid, "name": getattr(av, "name", None), "colour": getattr(av, "colour", None),
            "items_json": getattr(av, "items_json", None)}


def _member_card(s: Session, m: Membership, viewer: Membership, now: datetime) -> dict:
    """The ONLY shape a member has in the circle. Allow-list: add a field here only if it is not health data."""
    card = {"member_id": m.id, "role": m.role.value, "name": _display(s, m), "is_you": m.id == viewer.id,
            "photo_url": None, "emoji": None, "avatar": None, "level": None, "streak": None,
            "checked_in_today": None}
    if m.role == MemberRole.patient:
        prof = _profile(s, m.patient_id)
        card["emoji"] = prof.emoji
        if prof.mode == IdentityMode.profile:
            card["photo_url"] = prof.photo_url
        rewards.sync(s, m.patient_id, now)
        pres = rewards.presence(s, m.patient_id, now)
        if prof.share_avatar:
            card["avatar"] = _avatar_ref(s, m.patient_id)
        if prof.share_level:
            from ..appie import level_of

            card["level"] = level_of(rewards.xp(s, m.patient_id))
        if prof.share_streak:
            card["streak"] = pres["streak"]
        if prof.share_checked_in:
            card["checked_in_today"] = pres["checked_in_today"]
    return card


def _latest_nudge(s: Session, pid: int) -> Escalation | None:
    return s.exec(select(Escalation).where(Escalation.patient_id == pid, Escalation.level >= 1)
                  .order_by(col(Escalation.ts).desc())).first()


def _nudge_state(s: Session, patient: Patient, now: datetime) -> dict | None:
    """The friends nudge (escalation level 1+) as a discreet card. Held for NUDGE_HOLD so the patient can cancel."""
    if patient.risk_level < 1:
        return None
    esc = _latest_nudge(s, patient.id)
    if esc is not None and s.exec(select(NudgeCancel).where(NudgeCancel.escalation_id == esc.id)).first():
        return None
    held_until = rewards.aware(esc.ts) + NUDGE_HOLD if esc else rewards.aware(now)
    return {"escalation_id": esc.id if esc else None, "held_until": held_until,
            "visible": rewards.aware(now) >= held_until}


def _member(s: Session, token: str) -> Membership:
    m = s.exec(select(Membership).where(Membership.token == token)).first()
    if m is None or m.left_at is not None:
        raise HTTPException(404, "not a member")
    return m


def _new_code(s: Session) -> str:
    while True:
        code = secrets.token_hex(3).upper()
        if not s.exec(select(Circle).where(Circle.invite_code == code)).first():
            return code


def _patient_membership(s: Session, pid: int) -> tuple[Circle, Membership] | None:
    m = s.exec(select(Membership).where(Membership.patient_id == pid, col(Membership.left_at).is_(None))
               .order_by(col(Membership.joined_at))).first()
    return (s.get(Circle, m.circle_id), m) if m else None


# ---------------------------------------------------------------------------------------------
# Profile and consent
# ---------------------------------------------------------------------------------------------


@router.get("/social/presets")
def presets() -> dict:
    return {"reactions": [{"kind": k, "label": v} for k, v in PRESETS.items()]}


@router.get("/patients/{pid}/profile")
def get_profile(pid: int, s: Session = Depends(get_session)) -> dict:
    _patient(s, pid)
    return _profile_out(_profile(s, pid))


@router.put("/patients/{pid}/profile")
def put_profile(pid: int, body: ProfileIn, s: Session = Depends(get_session)) -> dict:
    """Identity can be changed at any time. Free text with crisis words is not saved; the crisis path runs."""
    patient = _patient(s, pid)
    prof = _profile(s, pid)
    texts = {k: getattr(body, k) for k in ("display_name", "pseudonym", "emoji")}
    crisis = safety.check(s, patient, *texts.values(), where="the circle profile")
    data = body.model_dump(exclude_unset=True)
    if crisis:
        for k in texts:
            data.pop(k, None)
    consent = data.pop("social_consent", None)
    for k, v in data.items():
        setattr(prof, k, v.strip() if isinstance(v, str) else v)
    if consent is True and prof.social_consent_at is None:
        prof.social_consent_at = baseline.clock()
    elif consent is False:
        prof.social_consent_at = None
        prof.share_avatar = prof.share_level = prof.share_streak = prof.share_checked_in = False
    prof.updated_at = baseline.clock()
    s.add(prof)
    s.commit()
    s.refresh(prof)
    return {**_profile_out(prof), "crisis": crisis}


# ---------------------------------------------------------------------------------------------
# Circle
# ---------------------------------------------------------------------------------------------


def _require_consent(s: Session, pid: int) -> None:
    if _profile(s, pid).social_consent_at is None:
        raise HTTPException(403, "the circle needs its own consent first (PUT /profile with social_consent=true)")


@router.post("/patients/{pid}/circle")
def create_circle(pid: int, body: CircleIn, s: Session = Depends(get_session)) -> dict:
    """One circle per person (P1). Returns the existing one if there is one."""
    _patient(s, pid)
    _require_consent(s, pid)
    found = _patient_membership(s, pid)
    if found is None:
        circle = Circle(owner_patient_id=pid, name=(body.name or "My circle").strip(), invite_code=_new_code(s))
        s.add(circle)
        s.flush()
        m = Membership(circle_id=circle.id, role=MemberRole.patient, patient_id=pid, token=secrets.token_urlsafe(16))
        s.add(m)
        s.commit()
    return get_circle(pid, s)


@router.get("/patients/{pid}/circle")
def get_circle(pid: int, s: Session = Depends(get_session)) -> dict:
    """The patient's own circle: invite code, their member token, and their own pending friends card."""
    patient = _patient(s, pid)
    found = _patient_membership(s, pid)
    if found is None:
        raise HTTPException(404, "no circle yet")
    circle, m = found
    nudge = _nudge_state(s, patient, baseline.clock())
    return {
        "circle_id": circle.id, "name": circle.name, "invite_code": circle.invite_code,
        "member_id": m.id, "token": m.token,
        "nudge": None if nudge is None else {
            "pending": not nudge["visible"], "visible_from": nudge["held_until"].isoformat(),
            "can_cancel": nudge["escalation_id"] is not None,
            "message": "Your circle will get a gentle card suggesting they reach out. You can cancel it.",
        },
    }


@router.post("/patients/{pid}/circle/nudge/cancel")
def cancel_nudge(pid: int, s: Session = Depends(get_session)) -> dict:
    """The patient cancels the friends card. The clinician side of the ladder is not affected."""
    patient = _patient(s, pid)
    esc = _latest_nudge(s, pid)
    if patient.risk_level < 1 or esc is None:
        raise HTTPException(404, "no friends card to cancel")
    if not s.exec(select(NudgeCancel).where(NudgeCancel.escalation_id == esc.id)).first():
        s.add(NudgeCancel(patient_id=pid, escalation_id=esc.id))
        s.commit()
    return {"ok": True}


@router.post("/circles/join")
def join(body: JoinIn, s: Session = Depends(get_session)) -> dict:
    circle = s.exec(select(Circle).where(Circle.invite_code == body.code.strip().upper())).first()
    if circle is None:
        raise HTTPException(404, "invite code not found")
    if body.patient_id is not None:
        _patient(s, body.patient_id)
        _require_consent(s, body.patient_id)
        existing = s.exec(select(Membership).where(Membership.circle_id == circle.id,
                                                   Membership.patient_id == body.patient_id,
                                                   col(Membership.left_at).is_(None))).first()
        m = existing or Membership(circle_id=circle.id, role=MemberRole.patient, patient_id=body.patient_id,
                                   token=secrets.token_urlsafe(16))
    else:
        if not (body.name or "").strip():
            raise HTTPException(400, "a name is needed to join as a friend")
        m = Membership(circle_id=circle.id, role=MemberRole.friend, friend_name=body.name.strip(),
                       token=secrets.token_urlsafe(16))
    s.add(m)
    s.commit()
    s.refresh(m)
    return {"circle_id": circle.id, "name": circle.name, "member_id": m.id, "token": m.token}


@router.post("/circles/leave")
def leave(body: TokenIn, s: Session = Depends(get_session)) -> dict:
    m = _member(s, body.token)
    m.left_at = baseline.clock()
    s.add(m)
    s.commit()
    return {"ok": True}


@router.get("/circles/feed")
def feed(token: str, s: Session = Depends(get_session)) -> dict:
    """What a member sees: each member's opt-in card, recent reactions, and discreet reach-out cards."""
    viewer = _member(s, token)
    circle = s.get(Circle, viewer.circle_id)
    now = baseline.clock()
    members = s.exec(select(Membership).where(Membership.circle_id == circle.id,
                                              col(Membership.left_at).is_(None))
                     .order_by(col(Membership.joined_at))).all()
    names = {m.id: _display(s, m) for m in members}
    cards = []
    for m in members:
        if m.role != MemberRole.patient or m.id == viewer.id:
            continue  # the patient's own card is in GET /patients/{id}/circle
        nudge = _nudge_state(s, s.get(Patient, m.patient_id), now)
        if nudge and nudge["visible"]:
            cards.append({"to_member_id": m.id, "message": f"Maybe a good day to reach out to {names[m.id]}?"})
    reactions = s.exec(select(Reaction).where(Reaction.circle_id == circle.id)
                       .order_by(col(Reaction.ts).desc()).limit(30)).all()
    out = {
        "circle": {"id": circle.id, "name": circle.name, "invite_code": circle.invite_code},
        "you": viewer.id,
        "members": [_member_card(s, m, viewer, now) for m in members],
        "reactions": [
            {"id": r.id, "from_member_id": r.from_member_id, "from_name": names.get(r.from_member_id, "Former member"),
             "to_member_id": r.to_member_id, "kind": r.kind, "label": PRESETS.get(r.kind, r.kind),
             "ts": rewards.aware(r.ts).isoformat()}
            for r in reactions
        ],
        "reach_out_cards": cards,
    }
    s.commit()  # rewards.sync may have paid what was due
    return out


@router.post("/circles/react")
def react(body: ReactIn, s: Session = Depends(get_session)) -> dict:
    """Preset reactions only: no free text in the circle (chat is P3)."""
    viewer = _member(s, body.token)
    if body.kind not in PRESETS:
        raise HTTPException(400, "unknown reaction")
    to = s.get(Membership, body.to_member_id)
    if to is None or to.circle_id != viewer.circle_id or to.left_at is not None:
        raise HTTPException(404, "member not found in this circle")
    r = Reaction(circle_id=viewer.circle_id, from_member_id=viewer.id, to_member_id=to.id, kind=body.kind)
    s.add(r)
    s.commit()
    return {"ok": True, "id": r.id}
