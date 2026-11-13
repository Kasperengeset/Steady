from sqlmodel import Session, select

from .db import engine
from .models import Arm, Contact, ContactKind, Patient

ARMS = [
    ("walk_outside", "Tur ute"),
    ("call_friend", "Ring en venn"),
    ("morning_light", "Morgenlys"),
    ("music", "Musikkøkt"),
    ("breathing", "Pusteøvelse"),
    ("small_task", "Liten aktivitetsoppgave"),
]


def seed() -> None:
    with Session(engine) as s:
        if s.exec(select(Patient)).first():
            return
        mia = Patient(name="Mia")
        s.add(mia)
        s.commit()
        s.refresh(mia)
        s.add(Contact(patient_id=mia.id, name="Jonas", kind=ContactKind.friend))
        s.add(Contact(patient_id=mia.id, name="Dr. Virtanen", kind=ContactKind.clinician))
        for key, label in ARMS:
            s.add(Arm(patient_id=mia.id, key=key, label=label))
        s.commit()
