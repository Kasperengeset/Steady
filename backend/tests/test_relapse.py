"""Relapse model (engine/relapse.py): math, robustness and explanation."""

import math

import pytest

from app.engine.relapse import Params, level_from_posterior, step, transition

TWO_STATE = Params(delta={"acoustic": 1.5}, h_relapse=0.005, h_elevated=0.0, r_relapse=0.0, r_elevated=0.0,
                   prior=(0.99, 0.01, 0.0), clip_margin=100)
FULL = Params(delta={"acoustic": 1.58, "text": 1.5}, h_relapse=0.005, h_elevated=0.002, r_relapse=0.02,
              r_elevated=0.1, prior=(0.98, 0.01, 0.01))


def test_worked_example_from_status():
    """1 % → 4 % → 16 % → 28 % → 69 % for four flat days (δ = 1.5, h = 0.005, two states)."""
    p = {"well": 0.99, "relapse": 0.01, "elevated": 0.0}
    got = []
    for D in (1.4, 1.7, 1.2, 1.9):
        s = step(p, {"acoustic": D}, {"acoustic": 1}, 1, prm=TWO_STATE)
        assert s.llr_relapse == pytest.approx(1.5 * D - 1.125)
        p = s.posterior
        got.append(round(p["relapse"], 2))
    assert got == [0.04, 0.16, 0.28, 0.69]


def test_one_extreme_day_is_clipped_and_cannot_reach_the_clinician():
    s = step(None, {"acoustic": 25.0}, {"acoustic": 1}, prm=FULL)  # e.g. a broken recording
    assert s.clipped == ["acoustic"]
    m = 1.58
    assert s.llr_relapse == pytest.approx(m * (m + 1) - m * m / 2)  # capped evidence
    assert s.posterior["relapse"] < 0.6


def test_missing_days_only_apply_the_transition():
    p = {"well": 0.9, "relapse": 0.1, "elevated": 0.0}
    s = step(p, {}, {}, days_elapsed=3, prm=FULL)
    assert s.posterior == pytest.approx(transition(p, 3, FULL))
    assert s.llr_relapse == 0 and s.by_feature == {}


def test_sustained_negative_shift_means_elevated_not_relapse():
    p = None
    for _ in range(5):
        s = step(p, {"acoustic": -1.8}, {"acoustic": 1}, prm=FULL)
        p = s.posterior
    assert p["elevated"] > 0.8 and p["relapse"] < 0.01
    level, elevated = level_from_posterior(p, crisis=False)
    assert level == 0 and elevated  # clinician flag only, no friend alert


def test_feature_contributions_add_up_to_the_days_evidence():
    shares = {"acoustic": {"arousal": 0.6, "energy": 0.5, "jitter": -0.2}, "text": {"text_logit": 1.1}}
    D = {"acoustic": 0.9, "text": 1.1}
    s = step(None, D, {"acoustic": 2, "text": 1}, shares_by_branch=shares, prm=FULL)
    assert sum(s.by_feature.values()) + s.scepticism == pytest.approx(s.llr_relapse)
    assert sum(s.by_branch.values()) == pytest.approx(s.llr_relapse)
    m_ac = 1.58 * math.sqrt(2)  # two spoken check-ins today → mean shift grows by √n
    assert s.by_branch["acoustic"] == pytest.approx(m_ac * 0.9 - m_ac**2 / 2)


def test_crisis_flag_bypasses_the_model():
    assert level_from_posterior({"well": 1.0, "relapse": 0.0, "elevated": 0.0}, crisis=True) == (3, False)
