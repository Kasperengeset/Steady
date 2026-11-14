"""llm.rate_text (#28): strict schema, fallback on errors, and weight-0 features leaving D_text unchanged."""

import json
from types import SimpleNamespace

import pytest

from app.engine import state
from app.pipeline import llm

GOOD = {"about_self_now": True, "language": "en", "mood": 7, "energy_self": 6, "anhedonia": 2, "social_contact": 8,
        "evidence": {"mood": "a calm day", "energy_self": "biked", "anhedonia": "enjoyed", "social_contact": "lunch with Anna"}}


class FakeInteractions:
    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def create(self, **kw):
        self.calls.append(kw)
        if self.error:
            raise self.error
        return SimpleNamespace(output_text=self.reply)


@pytest.fixture()
def fake(monkeypatch):
    monkeypatch.setattr(llm.settings, "gemini_api_key", "test-key")

    def install(**kw):
        fi = FakeInteractions(**kw)
        monkeypatch.setattr(llm, "_client", lambda: SimpleNamespace(interactions=fi))
        return fi

    return install


def test_valid_reply_and_call_settings(fake):
    fi = fake(reply=json.dumps(GOOD))
    out = llm.rate_text("I am not sad today. Had lunch with Anna.")
    assert out["scores"] == {"mood": 7.0, "energy_self": 6.0, "anhedonia": 2.0, "social_contact": 8.0}
    assert out["evidence"]["social_contact"] == "lunch with Anna" and out["language"] == "en"
    kw = fi.calls[0]
    assert kw["store"] is False and kw["generation_config"]["temperature"] == 0.0
    assert kw["response_mime_type"] == "application/json" and kw["response_format"] == llm.RATE_SCHEMA
    assert kw["model"] == llm.settings.rate_model


@pytest.mark.parametrize("bad", [
    "not json",
    json.dumps({**GOOD, "mood": 11}),
    json.dumps({**GOOD, "mood": 5.5}),
    json.dumps({**GOOD, "mood": True}),
    json.dumps({k: v for k, v in GOOD.items() if k != "anhedonia"}),
    json.dumps({**GOOD, "about_self_now": False}),  # about someone else → missing, not a neutral 5
    json.dumps([GOOD]),
])
def test_invalid_or_not_about_self_gives_no_scores(fake, bad):
    fake(reply=bad)
    assert llm.rate_text("text") is None


def test_error_and_no_key_give_no_scores(fake, monkeypatch):
    fake(error=RuntimeError("400 bad request"))
    assert llm.rate_text("text") is None
    monkeypatch.setattr(llm.settings, "gemini_api_key", "")
    assert llm.rate_text("text") is None


def test_weight_zero_content_scores_leave_D_text_unchanged():
    model = state.load_branch("text")
    assert all(model.weights.get(f, 0.0) == 0.0 for f in llm.RATE_FIELDS)
    z = {"text_logit": 1.2, "sentiment": -0.4, "first_person": 0.3, "absolutist": 0.1}
    with_llm = {**z, "mood": 2.5, "energy_self": -1.0, "anhedonia": 3.0, "social_contact": 0.7}
    assert state.branch_deviation(with_llm, model) == pytest.approx(state.branch_deviation(z, model))
