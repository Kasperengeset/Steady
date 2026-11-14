# Acoustic branch – training summary

Trained 2026-11-14T08:08:15+00:00 on `androids_interview` · 114 speakers (62 depressed) · folds: corpus fold-lists.csv (100% of speakers)

## Speaker-independent AUC

| Model | AUC (mean ± sd over folds) | Pooled out-of-fold |
|---|---|---|
| Our 8 features | 0.762 ± 0.073 | 0.722 |
| All 88 eGeMAPS | 0.857 ± 0.073 | 0.860 |
| Gender + age only | 0.475 ± 0.161 | 0.505 |
| wav2vec2 probe | 0.897 ± 0.065 | 0.891 |

## Coefficients (standardised; + = higher value → more likely depressed)

| Feature | β | Weight in app | vs. literature |
|---|---|---|---|
| `valence` | -0.022 | 0.214 | agrees |
| `arousal` | -0.021 | 0.204 | agrees |
| `energy` | -0.018 | 0.175 | agrees |
| `jitter` | +0.013 | 0.128 | agrees |
| `pause_ratio` | +0.009 | 0.084 | agrees |
| `speech_rate` | -0.007 | 0.067 | agrees |
| `shimmer` | +0.007 | 0.067 | agrees |
| `f0_std` | -0.006 | 0.060 | agrees |

![ROC](roc.png) ![Coefficients](coefficients.png)

**Limitations:** between-person classifier used as a prior for within-person change · small sample · Italian speech (prosody transfers; content is handled by the text branch).
