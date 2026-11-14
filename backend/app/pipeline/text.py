"""Text branch: written diary or the transcript of a spoken check-in -> 4 features + crisis flag.

Same code in training (ml/text/) and in the app, so the exported model is valid for the app's
features by construction (STATUS.md B9).

Features (higher = worse except sentiment, see engine/state.py DIRECTION["text"]):
  text_logit    log-odds from TF-IDF + logistic regression trained on the Reddit Mental Health
                Dataset (r/depression vs everyday and relationship forums, split by author).
                Probability = sigmoid(text_logit). Use the log-odds for z-scores: diary text often
                scores near 1, where probabilities bunch up.
  absolutist    absolutist words per 100 words (Al-Mosaiwi & Johnstone 2018)
  first_person  i, me, my, mine, myself per 100 words
  sentiment     VADER compound, -1 (negative) to +1 (positive)

The model is English only. Pure Python at inference: no scikit-learn, no pickle, so it loads the
same on every machine. Model file: pipeline/models/text_tfidf_lr.json.gz (ml/text/train_v1.py).
"""

import gzip
import html
import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

MODEL_PATH = Path(__file__).with_name("models") / "text_tfidf_lr.json.gz"
THRESHOLD = 0.5

# ---------------------------------------------------------------------------------------------
# Cleaning, masking, tokens
# ---------------------------------------------------------------------------------------------

# Diary length. Training posts are cut into chunks of this size so they look like a diary entry.
CHUNK_WORDS = 100
MIN_WORDS = 50
MAX_CHUNKS_PER_POST = 3

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_REDDIT_REF = re.compile(r"(?<![\w/])/?[ru]/[A-Za-z0-9_-]+")
_REDDIT_WORD = re.compile(r"\breddit\w*\b", re.I)
# Reddit formatting that diaries never contain: tl;dr, "edit:", age/gender tags like (24/f), [f 24].
_REDDIT_ARTIFACTS = re.compile(
    r"\btl\s*;?\s*dr\b"
    r"|\bedit\s*\d*\s*:"
    r"|\b\d{1,2}\s*/?\s*[mf]\b"
    r"|[\(\[]\s*[mf]\s*/?\s*\d{1,2}\s*[\)\]]",
    re.I,
)
_SPACE = re.compile(r"\s+")

# Words that name a forum and so give the training label away ("depress" covers depression,
# depressed, depressive, depressing). The negative forums' own names are masked the same way.
LABEL_TERMS = [r"depress\w*", r"subreddits?"]
FORUM_TERMS = [r"meditat\w*", r"divorc\w*", r"fitness", r"parenting", r"teaching", r"personal ?finance"]
# Treatment and clinical words. People after TMS mention treatment on good days too, so the
# model must not learn "mentions treatment = depressed".
TREATMENT_TERMS = [
    r"therap\w*", r"psychiatr\w*", r"anti-?\s?depressants?", r"meds", r"medications?",
    r"episodes?", r"r?tms", r"ect", r"ketamine", r"esketamine", r"spravato", r"magnetic stimulation",
    r"medicated", r"ssris?", r"snris?", r"diagnos\w*", r"treatments?", r"doctors?",
    r"dr", r"symptoms?", r"side effects?", r"mental health", r"mental illness\w*",
    r"prozac", r"zoloft", r"lexapro", r"sertraline", r"fluoxetine", r"escitalopram",
    r"citalopram", r"wellbutrin", r"bupropion", r"venlafaxine", r"effexor",
    r"cymbalta", r"duloxetine", r"mirtazapine",
]
MASK_TERMS = LABEL_TERMS + FORUM_TERMS + TREATMENT_TERMS
_MASK = re.compile(r"\b(?:" + "|".join(MASK_TERMS) + r")\b", re.I)

# A masked word becomes a break. Without it "my depression is" turns into "my is", and the
# bigram still points at the removed word.
BREAK = "|"
_TOKEN = re.compile(r"(?u)\b\w+\b")


def clean_text(text: str) -> str:
    """Unescape HTML; drop URLs, r/... and u/..., the word reddit and Reddit formatting."""
    text = html.unescape(str(text)).replace("​", " ")
    text = _URL.sub(" ", text)
    text = _REDDIT_REF.sub(" ", text)
    text = _REDDIT_WORD.sub(" ", text)
    text = _REDDIT_ARTIFACTS.sub(" ", text)
    return _SPACE.sub(" ", text).strip()


def mask_text(text: str) -> str:
    """Replace forum and treatment words with a break."""
    return _SPACE.sub(" ", _MASK.sub(f" {BREAK} ", text)).strip()


def prepare(text: str) -> str:
    return mask_text(clean_text(text))


def analyze(text: str) -> list[str]:
    """Lowercase unigrams and bigrams; bigrams never cross a break. Keeps one-letter tokens:
    "i" matters, first-person singular is a known marker."""
    out: list[str] = []
    for segment in text.lower().split(BREAK):
        toks = _TOKEN.findall(segment)
        out += toks
        out += [f"{a} {b}" for a, b in zip(toks, toks[1:])]
    return out


def chunk_words(text: str) -> list[str]:
    """Cut text into chunks of CHUNK_WORDS words; drop a tail under MIN_WORDS (training only)."""
    words = text.split()
    out = []
    for i in range(0, len(words), CHUNK_WORDS):
        piece = words[i : i + CHUNK_WORDS]
        if len(piece) < MIN_WORDS:
            break
        out.append(" ".join(piece))
        if len(out) == MAX_CHUNKS_PER_POST:
            break
    return out


# ---------------------------------------------------------------------------------------------
# Lexicon features
# ---------------------------------------------------------------------------------------------

ABSOLUTIST = {  # Al-Mosaiwi & Johnstone (2018)
    "absolutely", "all", "always", "complete", "completely", "constant", "constantly",
    "definitely", "entire", "ever", "every", "everyone", "everything", "full", "must",
    "never", "nothing", "totally", "whole",
}
FIRST_PERSON = {"i", "me", "my", "mine", "myself"}
_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


@lru_cache(maxsize=1)
def _vader():
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    return SentimentIntensityAnalyzer()


def lexicon_features(prepared: str) -> dict[str, float]:
    """Expects prepare()d text. A word right after a negation is not absolutist ("not always")."""
    words = _WORD.findall(prepared.lower().replace("’", "'"))
    n = max(len(words), 1)
    neg = lambda w: w == "not" or w.endswith("n't")  # noqa: E731
    absolutist = sum(w in ABSOLUTIST and not (i > 0 and neg(words[i - 1])) for i, w in enumerate(words))
    first = sum(w in FIRST_PERSON or w.startswith("i'") for w in words)
    return {
        "absolutist": 100 * absolutist / n,
        "first_person": 100 * first / n,
        "sentiment": _vader().polarity_scores(prepared.replace(BREAK, " "))["compound"],
    }


# ---------------------------------------------------------------------------------------------
# Crisis phrases – deterministic, never waits for a model. Any hit goes to the clinician.
# A negated phrase ("I'm not suicidal") still flags: a clinician reads it, a model does not decide.
# ---------------------------------------------------------------------------------------------

CRISIS_PATTERNS = [
    # English
    r"suicid\w*", r"kill(ing)? myself", r"end(ing)? my life", r"end it all", r"take my (own )?life",
    r"(want|wanna|wish) to die", r"wish i (was|were) dead", r"better off dead",
    r"no reason to (live|go on)", r"don'?t want to (be alive|live|be here anymore|wake up)",
    r"hurt(ing)? myself", r"self[- ]?harm\w*", r"cut(ting)? myself", r"overdose",
    r"can'?t go on",
    # Norwegian
    r"selvmord\w*", r"ta livet (av )?meg", r"ta mitt (eget )?liv", r"vil (bare )?dø",
    r"vil ikke leve", r"skade meg selv",
]
_CRISIS = re.compile(r"\b(?:" + "|".join(CRISIS_PATTERNS) + r")\b", re.I)


def crisis_hits(text: str) -> list[str]:
    return [m.group(0) for m in _CRISIS.finditer(text.replace("’", "'"))]


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


@lru_cache(maxsize=1)
def _model() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} missing – run ml/text/train_v1.py")
    with gzip.open(MODEL_PATH, "rt", encoding="utf-8") as f:
        m = json.load(f)
    m["index"] = {t: i for i, t in enumerate(m["terms"])}
    return m


def model_info() -> dict:
    return {k: v for k, v in _model().items() if k not in ("terms", "idf", "coef", "index")}


def _tfidf(prepared: str) -> dict[int, float]:
    """Same as scikit-learn TfidfVectorizer(sublinear_tf=True, norm='l2') with the exported idf."""
    m = _model()
    counts = Counter(j for tok in analyze(prepared) if (j := m["index"].get(tok)) is not None)
    x = {j: (1.0 + math.log(c)) * m["idf"][j] for j, c in counts.items()}
    norm = math.sqrt(sum(v * v for v in x.values()))
    return {j: v / norm for j, v in x.items()} if norm else {}


def text_logit(prepared: str) -> float:
    m = _model()
    return m["intercept"] + sum(v * m["coef"][j] for j, v in _tfidf(prepared).items())


def explain(prepared: str, top_k: int = 5) -> dict[str, list[tuple[str, float]]]:
    """Words that pushed the score up and down most."""
    m = _model()
    contrib = sorted(((m["terms"][j], v * m["coef"][j]) for j, v in _tfidf(prepared).items()),
                     key=lambda t: t[1])
    return {
        "words_up": [(w, round(c, 3)) for w, c in reversed(contrib) if c > 0][:top_k],
        "words_down": [(w, round(c, 3)) for w, c in contrib if c < 0][:top_k],
    }


def analyze_text(text: str, explain_top: int = 0) -> dict:
    """Everything the app needs from one diary entry or transcript.

    {"features": {text_logit, absolutist, first_person, sentiment},
     "p_depressive": 0..1, "label": str, "n_words": int, "short_input": bool,
     "crisis": bool, "crisis_hits": [...], ("words_up", "words_down" if explain_top)}
    """
    prepared = prepare(text)
    logit = text_logit(prepared)
    p = 1.0 / (1.0 + math.exp(-logit))
    hits = crisis_hits(text)
    n_words = len(text.split())
    out = {
        "features": {"text_logit": logit, **lexicon_features(prepared)},
        "p_depressive": p,
        "label": "depressive language" if p >= THRESHOLD else "everyday language",
        "n_words": n_words,
        "short_input": n_words < MIN_WORDS,
        "crisis": bool(hits),
        "crisis_hits": hits,
    }
    if explain_top:
        out.update(explain(prepared, explain_top))
    return out


def extract(text: str) -> dict[str, float]:
    """Engine contract: {feature name: value} for the text branch."""
    return analyze_text(text)["features"]
