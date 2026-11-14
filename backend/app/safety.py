"""Crisis words in free text outside the check-in (circle profile, a reply to the avatar).

Same phrases as the check-in (`crisis_hits` in pipeline/text.py) and the same path: the clinician is
told at once (Escalation level 3, risk_level 3), and the caller returns `crisis: true` so the app opens
the support screen. Never only friends. The text itself is not stored.
"""

from sqlmodel import Session

from .engine.escalation import CRISIS
from .models import ContactKind, Escalation, Patient
from .pipeline.text import crisis_hits


def check(s: Session, patient: Patient, *texts: str | None, where: str) -> bool:
    """True if any text has a crisis phrase; then the crisis path has run (not committed)."""
    if not any(crisis_hits(t) for t in texts if t):
        return False
    s.add(Escalation(patient_id=patient.id, level=CRISIS, recipient=ContactKind.clinician,
                     reason=f"crisis phrase in {where}"))
    patient.risk_level = CRISIS
    s.add(patient)
    return True
