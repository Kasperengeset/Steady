"""Simulator options for the #27 design checks: off by default, and neutral when their effect is zero."""

from app.sim.simulator import simulate


def _S(r):
    return [(d["S_combined"], d["level_combined"]) for d in r["days"]]


def test_random_times_without_a_time_of_day_effect_change_nothing():
    base = simulate(seed=3, day_noise_factor=1.5)
    assert _S(simulate(seed=3, day_noise_factor=1.5, prompt_hours=(9.0, 21.0), tod_effect=0.0)) == _S(base)


def test_full_adherence_and_zero_engagement_weight_change_nothing():
    base = simulate(seed=4)
    assert _S(simulate(seed=4, adherence=1.0, engagement_weight=0.0)) == _S(base)


def test_missed_days_have_no_score_and_keep_the_level():
    r = simulate(seed=5, adherence=0.6, adherence_drop=0.3)
    missed = [d for d in r["days"][15:] if d["checkins"] == 0]
    assert missed, "expected some missed days"
    days = r["days"]
    for d in missed:
        assert d["S_combined"] is None
        assert d["level_combined"] == days[d["day"] - 1]["level_combined"]
        assert d["D_engagement"] is not None and d["D_engagement"] > 0  # a missed day reads as "worse"


def test_engagement_signal_scores_missed_days():
    r = simulate(seed=5, adherence=0.6, adherence_drop=0.3, engagement_weight=1.0)
    missed = [d for d in r["days"][15:] if d["checkins"] == 0]
    assert missed and all(d["S_anchored"] is not None for d in missed)


def test_extras_and_shared_noise_run():
    r = simulate(seed=6, day_shared=0.5, extra_rate=1.0, extra_weight=0.5)
    assert sum(d["checkins"] for d in r["days"]) > 70
    for mode in ("two_slots", "hour_covariate"):
        r = simulate(seed=6, prompt_hours=(9.0, 21.0), tod_effect=1.0, tod_mode=mode)
        assert r["days"][-1]["S_combined"] is not None
