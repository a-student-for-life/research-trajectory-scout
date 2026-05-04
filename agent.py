"""
Research Trajectory Scout
=========================
Daily pipeline: search arXiv + Semantic Scholar → score by research shape
→ reject bad-fit papers → generate Markdown report → commit to GitHub.

Sources:   arXiv (free), Semantic Scholar (free key optional)
Scoring:   keyword match + shape-based bonus + semantic similarity
Runtime:   ~3–5 minutes on GitHub Actions free tier
Cost:      $0
"""

import json
import logging
import os
import re
import time
import yaml
import requests
import feedparser
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

# ── Semantic scoring (optional but strongly recommended) ──────────────────────
# If sentence-transformers is not installed, the agent falls back to
# keyword-only scoring. Quality is noticeably worse in that mode.
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_AVAILABLE = True
except ImportError:
    SEMANTIC_AVAILABLE = False

# ── Paths ─────────────────────────────────────────────────────────────────────
CONFIG_PATH  = "config.yaml"
PROFILE_PATH = "profile.md"
SEEN_PATH    = "data/seen_papers.json"
REPORTS_DIR  = "reports"

# ── Core identity sentence ────────────────────────────────────────────────────
# This single sentence is what every paper is semantically compared against.
# Changing this is the fastest way to retune the agent's taste.
IDENTITY_SENTENCE = (
    "machine learning on underexplored sensing modalities "
    "in real-world resource-constrained environments"
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("scout")


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_seen() -> set:
    if not os.path.exists(SEEN_PATH):
        return set()
    try:
        with open(SEEN_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return set(data) if isinstance(data, list) else set()
    except (json.JSONDecodeError, ValueError):
        log.warning("seen_papers.json was corrupted — resetting.")
        return set()


def save_seen(seen: set) -> None:
    os.makedirs(os.path.dirname(SEEN_PATH), exist_ok=True)
    with open(SEEN_PATH, "w", encoding="utf-8") as f:
        json.dump(sorted(list(seen)), f, indent=2)


def clean_text(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


# ═══════════════════════════════════════════════════════════════════════════════
# API helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _get_with_retry(url: str, params: dict = None, headers: dict = None,
                    timeout: int = 30, retries: int = 3, backoff: float = 5.0):
    """GET with exponential backoff. Returns Response or raises on final failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=headers,
                                timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = backoff * (2 ** attempt)
            log.warning(f"Request failed ({e}), retrying in {wait:.0f}s…")
            time.sleep(wait)


# ═══════════════════════════════════════════════════════════════════════════════
# Search sources
# ═══════════════════════════════════════════════════════════════════════════════

def _arxiv_date_filter(days_back: int) -> str:
    end   = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)
    return f"[{start.strftime('%Y%m%d')}0000 TO {end.strftime('%Y%m%d')}2359]"


def search_arxiv(query: str, max_results: int = 10, days_back: int = 30) -> list:
    """
    Search arXiv with a date window.
    The date filter was present in config but silently ignored in v1 — fixed here.
    """
    date_filter = _arxiv_date_filter(days_back)
    full_query  = f"({query}) AND submittedDate:{date_filter}"
    url = (
        "https://export.arxiv.org/api/query?"
        f"search_query=all:{quote(full_query)}"
        f"&start=0&max_results={max_results}"
        "&sortBy=submittedDate&sortOrder=descending"
    )

    resp = _get_with_retry(url)
    feed = feedparser.parse(resp.text)

    papers = []
    for entry in feed.entries:
        abstract = clean_text(entry.get("summary", ""))
        if not abstract:
            continue
        authors = [a["name"] for a in entry.get("authors", []) if "name" in a]
        papers.append({
            "source":    "arXiv",
            "id":        entry.get("id", ""),
            "title":     clean_text(entry.get("title", "")),
            "abstract":  abstract,
            "published": entry.get("published", "")[:10],
            "link":      entry.get("id", ""),
            "authors":   authors,
        })
    return papers


def search_semantic_scholar(query: str, max_results: int = 5) -> list:
    """
    Search Semantic Scholar's public API.
    Set the SEMANTIC_SCHOLAR_API_KEY environment variable (free at
    semanticscholar.org) to raise the rate limit from 1 req/s to 10 req/s.
    """
    url    = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query":  query,
        "limit":  max_results,
        "fields": "title,abstract,year,authors,url,publicationDate,externalIds",
    }
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        resp = _get_with_retry(url, params=params, headers=headers)
        data = resp.json()
    except Exception as e:
        log.warning(f"Semantic Scholar query failed: {e}")
        return []

    papers = []
    for item in data.get("data", []):
        abstract = clean_text(item.get("abstract") or "")
        if not abstract:
            continue  # can't score without an abstract

        paper_id = item.get("paperId", "")
        authors  = [a.get("name", "") for a in item.get("authors", [])]
        pub_date = item.get("publicationDate") or str(item.get("year", ""))

        papers.append({
            "source":    "SemanticScholar",
            "id":        f"ss:{paper_id}",
            "title":     clean_text(item.get("title", "")),
            "abstract":  abstract,
            "published": pub_date[:10] if pub_date else "",
            "link":      item.get("url", f"https://www.semanticscholar.org/paper/{paper_id}"),
            "authors":   authors,
        })
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic model  (loaded once, reused for all papers)
# ═══════════════════════════════════════════════════════════════════════════════

_model             = None
_identity_embedding = None


def _get_model():
    global _model, _identity_embedding
    if _model is None and SEMANTIC_AVAILABLE:
        log.info("Loading sentence-transformers model (cached after first run)…")
        _model              = SentenceTransformer("all-MiniLM-L6-v2")
        _identity_embedding = _model.encode([IDENTITY_SENTENCE])
    return _model, _identity_embedding


def _semantic_similarity(paper: dict) -> float:
    """
    Cosine similarity between the paper's title+abstract and the identity sentence.
    This catches papers that match your research shape but not your exact keywords.
    Returns 0.0 if sentence-transformers is unavailable.
    """
    model, identity_emb = _get_model()
    if model is None:
        return 0.0
    text      = f"{paper.get('title', '')} {paper.get('abstract', '')}"
    paper_emb = model.encode([text])
    sim       = cosine_similarity(paper_emb, identity_emb)[0][0]
    return float(sim)


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Modality words — used to decide whether a paper is "sensing-primary"
_SENSING_WORDS = {
    "sensor", "sensing", "acoustic", "audio", "satellite", "sar", "sentinel",
    "vibration", "thermal", "infrared", "rf sensing", "wifi", "csi", "iot",
    "remote sensing", "microphone", "accelerometer", "lidar", "radar",
}

_SHAPE_BONUS_RULES = {
    "dataset_or_benchmark":    ["dataset", "benchmark", "corpus", "annotated", "ground truth"],
    "real_world_deployment":   ["real-world", "field", "deployment", "case study", "in-situ", "in situ"],
    "resource_constrained":    ["low-cost", "edge", "tinyml", "resource-constrained", "embedded", "raspberry", "microcontroller"],
    "sensing_modality":        list(_SENSING_WORDS),
    "ml_task":                 ["anomaly", "classification", "segmentation", "detection", "prediction", "domain adaptation", "calibration"],
    "underexplored_geography": ["india", "indian", "global south", "africa", "southeast asia", "developing", "low-resource", "under-resourced"],
}


def score_paper(paper: dict, config: dict) -> dict:
    text = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()

    # ── Positive keywords ──────────────────────────────────────────────────────
    positive_hits = [kw for kw in config.get("positive_keywords", []) if kw.lower() in text]
    keyword_score = len(positive_hits) * 2

    # ── Negative keywords (context-aware) ─────────────────────────────────────
    # If this is demonstrably a sensing paper, don't penalise it for mentioning
    # transformer/foundation-model architecture — using a ViT on SAR imagery is fine.
    is_sensing = any(w in text for w in _SENSING_WORDS)
    soft_negatives = {"transformer architecture", "foundation model"}

    negative_hits = []
    neg_score     = 0
    for kw in config.get("negative_keywords", []):
        if kw.lower() in text:
            if kw.lower() in soft_negatives and is_sensing:
                continue   # don't penalise transformers in sensing papers
            negative_hits.append(kw)
            neg_score -= 3

    # ── Shape bonus ───────────────────────────────────────────────────────────
    shape_hits  = []
    shape_score = 0
    for label, words in _SHAPE_BONUS_RULES.items():
        if any(w in text for w in words):
            shape_score += 3
            shape_hits.append(label)

    # ── Semantic similarity (0–1 → scaled 0–20) ───────────────────────────────
    semantic_sim   = _semantic_similarity(paper)
    semantic_score = round(semantic_sim * 20)

    total = keyword_score + neg_score + shape_score + semantic_score

    if negative_hits:
        fit = "Weak fit"
    elif total >= 30:
        fit = "Strong fit"
    elif total >= 18:
        fit = "Medium fit"
    else:
        fit = "Weak fit"

    return {
        "score":              total,
        "keyword_score":      keyword_score,
        "shape_score":        shape_score,
        "semantic_score":     semantic_score,
        "semantic_similarity": round(semantic_sim, 3),
        "positive_hits":      positive_hits[:12],
        "negative_hits":      negative_hits[:8],
        "shape_hits":         shape_hits,
        "fit":                fit,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Inference helpers
# ═══════════════════════════════════════════════════════════════════════════════

_MODALITY_MAP = {
    "Remote sensing / satellite": ["remote sensing", "satellite", "sar", "sentinel", "landsat", "multispectral", "hyperspectral"],
    "Acoustic / audio":           ["audio", "acoustic", "sound", "microphone", "speech"],
    "Vibration":                   ["vibration", "accelerometer", "seismic"],
    "Thermal / infrared":          ["thermal", "infrared", "ir imaging"],
    "RF / WiFi":                   ["rf", "wifi", "csi", "wireless", "mmwave", "radar"],
    "IoT / low-cost sensors":      ["iot", "low-cost sensor", "sensor network", "embedded sensor"],
    "Smartphone sensing":          ["smartphone", "mobile sensing", "inertial"],
    "AIS / GPS trajectory":        ["ais", "gps", "trajectory", "vessel"],
    "Visual / camera":             ["image", "video", "camera", "rgb"],
}

_ML_TASK_MAP = {
    "Anomaly detection":  ["anomaly"],
    "Classification":     ["classification", "classify"],
    "Segmentation":       ["segmentation", "segment"],
    "Detection":          ["detection", "detect"],
    "Prediction":         ["prediction", "forecast"],
    "Domain adaptation":  ["domain adaptation", "domain shift"],
    "Calibration":        ["calibration", "calibrate"],
    "Self-supervised":    ["self-supervised", "contrastive", "masked"],
}


def infer_sensing_modality(paper: dict) -> str:
    text  = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    found = [m for m, kws in _MODALITY_MAP.items() if any(k in text for k in kws)]
    return ", ".join(found) if found else "Not obvious"


def infer_ml_task(paper: dict) -> str:
    text  = f"{paper.get('title', '')} {paper.get('abstract', '')}".lower()
    found = [t for t, kws in _ML_TASK_MAP.items() if any(k in text for k in kws)]
    return ", ".join(found) if found else "Not obvious"


def make_recommendation_reason(paper: dict, scoring: dict) -> str:
    parts = []
    sim = scoring["semantic_similarity"]
    if sim >= 0.50:
        parts.append(f"**Very high** semantic match to research identity ({sim:.2f})")
    elif sim >= 0.40:
        parts.append(f"Good semantic match ({sim:.2f})")

    if scoring["shape_hits"]:
        parts.append("Shape match: " + ", ".join(scoring["shape_hits"]))

    if scoring["positive_hits"]:
        parts.append("Keywords: " + ", ".join(scoring["positive_hits"][:8]))

    return " | ".join(parts) if parts else "Passed threshold — manual review recommended."


# ═══════════════════════════════════════════════════════════════════════════════
# Report generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(top_papers: list, rejected: list,
                    weak: list, config: dict) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    title = config["agent"].get("report_title", "Research Scout Report")

    strong = [p for p in top_papers if p["scoring"]["fit"] == "Strong fit"]
    medium = [p for p in top_papers if p["scoring"]["fit"] == "Medium fit"]

    lines = []

    # ── Header ────────────────────────────────────────────────────────────────
    lines += [
        f"# {title}",
        f"**Date:** {today}  |  "
        f"**Found:** {len(top_papers)} candidates ({len(strong)} strong, {len(medium)} medium)  |  "
        f"**Rejected:** {len(rejected) + len(weak)}",
        "",
    ]

    # ── Executive summary ─────────────────────────────────────────────────────
    lines += ["## ⚡ Quick Scan (read these first)", ""]
    if not top_papers:
        lines.append("No strong matches today. Check search queries or widen `days_back`.")
    else:
        for idx, item in enumerate(top_papers[:5], 1):
            p, s = item["paper"], item["scoring"]
            badge = "🟢" if s["fit"] == "Strong fit" else "🟡"
            lines.append(
                f"{badge} **{idx}.** [{p['title'][:95]}]({p['link']})  \n"
                f"   `{s['fit']}` · score {s['score']} · "
                f"sem {s['semantic_similarity']:.2f} · {p['source']} · {p.get('published','?')}"
            )
        lines.append("")

    # ── Taste reminder ────────────────────────────────────────────────────────
    lines += [
        "---",
        "",
        "## Taste Reminder",
        "",
        "> Do not match my topics. Match my **research shape**.",
        "",
        "Identity: **ML on underexplored sensing modalities in real-world, "
        "resource-constrained environments.**",
        "",
    ]

    # ── Full recommendations ──────────────────────────────────────────────────
    lines += ["## Full Recommendations", ""]

    for idx, item in enumerate(top_papers, 1):
        p, s = item["paper"], item["scoring"]
        lines += [
            f"### {idx}. {p['title']}",
            "",
            f"| Field | Value |",
            f"|-------|-------|",
            f"| **Fit** | {s['fit']} |",
            f"| **Score** | {s['score']} (kw {s['keyword_score']} + shape {s['shape_score']} + sem {s['semantic_score']}) |",
            f"| **Semantic similarity** | {s['semantic_similarity']:.3f} |",
            f"| **Source** | {p['source']} |",
            f"| **Published** | {p.get('published','Unknown')} |",
            f"| **Sensing modality** | {infer_sensing_modality(p)} |",
            f"| **ML task** | {infer_ml_task(p)} |",
            "",
            f"**Why it matches:** {make_recommendation_reason(p, s)}",
            "",
            f"**Possible extension:** Low-cost dataset, Indian/Global South context, "
            f"edge deployment, or domain-adaptation angle.",
            "",
            f"**Link:** {p['link']}",
            "",
            "**Abstract:**",
            "",
            clean_text(p.get("abstract", ""))[:800] + "…",
            "",
        ]

    # ── Rejected ──────────────────────────────────────────────────────────────
    lines += ["---", "", "## Rejected Papers", ""]
    all_rejected = rejected + weak
    if not all_rejected:
        lines.append("None today.")
    else:
        for item in all_rejected[:15]:
            p, s = item["paper"], item["scoring"]
            reason = (
                "negative keywords: " + ", ".join(s["negative_hits"])
                if s["negative_hits"]
                else f"low score ({s['score']}), sem {s['semantic_similarity']:.2f}"
            )
            lines.append(f"- **{p['title'][:80]}** — {reason}")

    # ── Next actions ──────────────────────────────────────────────────────────
    lines += [
        "",
        "---",
        "",
        "## Next Actions",
        "",
        "1. Read the **abstract + dataset/evaluation section** of the top 2–3 papers.",
        "2. Ask: can this become a **dataset paper**, **benchmark paper**, or **applied ML system paper**?",
        "3. Discard anything needing paid APIs, expensive GPUs, or user-study infrastructure.",
        "4. If a paper scores ≥ 30 and semantic ≥ 0.45, it is worth a full read.",
        "",
    ]

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    if not SEMANTIC_AVAILABLE:
        log.warning(
            "Running in keyword-only mode. "
            "Install sentence-transformers for much better results:\n"
            "  pip install sentence-transformers scikit-learn torch"
        )

    config       = load_yaml(CONFIG_PATH)
    _            = load_text(PROFILE_PATH)   # loaded for reference; scoring uses IDENTITY_SENTENCE
    seen         = load_seen()
    max_results  = config["agent"].get("max_results_per_query", 8)
    days_back    = config["agent"].get("days_back", 30)
    queries      = config.get("search_queries", [])

    all_candidates: list = []

    # ── arXiv ─────────────────────────────────────────────────────────────────
    log.info(f"Searching arXiv ({len(queries)} queries, last {days_back} days)…")
    for query in queries:
        log.info(f"  arXiv: {query[:75]}…")
        try:
            papers = search_arxiv(query, max_results=max_results, days_back=days_back)
            all_candidates.extend(papers)
            log.info(f"    → {len(papers)} papers")
        except Exception as e:
            log.warning(f"    [FAIL] {e}")
        time.sleep(3)   # arXiv asks for polite access

    # ── Semantic Scholar (every other query to respect rate limits) ───────────
    ss_queries = queries[::2]
    log.info(f"Searching Semantic Scholar ({len(ss_queries)} queries)…")
    for query in ss_queries:
        log.info(f"  SS: {query[:75]}…")
        try:
            papers = search_semantic_scholar(query, max_results=5)
            all_candidates.extend(papers)
            log.info(f"    → {len(papers)} papers")
        except Exception as e:
            log.warning(f"    [FAIL] {e}")
        time.sleep(1)   # 1 req/s without API key

    # ── Deduplicate (skip already-seen papers) ────────────────────────────────
    deduped: dict = {}
    for paper in all_candidates:
        pid = paper["id"]
        if pid and pid not in seen and pid not in deduped:
            deduped[pid] = paper

    log.info(f"{len(all_candidates)} raw results → {len(deduped)} unique unseen papers to score.")

    # ── Score ─────────────────────────────────────────────────────────────────
    strong_medium: list = []
    rejected:      list = []
    weak:          list = []

    for paper in deduped.values():
        scoring = score_paper(paper, config)
        item    = {"paper": paper, "scoring": scoring}

        if scoring["fit"] == "Strong fit":
            strong_medium.append(item)
        elif scoring["fit"] == "Medium fit":
            strong_medium.append(item)
        elif scoring["negative_hits"]:
            rejected.append(item)
        else:
            weak.append(item)

    strong_medium.sort(key=lambda x: x["scoring"]["score"], reverse=True)

    final_top_k = config["agent"].get("final_top_k", 10)
    top_papers  = strong_medium[:final_top_k]

    log.info(
        f"Scored: {len(top_papers)} top | "
        f"{len(rejected)} rejected (negative keywords) | "
        f"{len(weak)} weak (low score)"
    )

    # ── Report ────────────────────────────────────────────────────────────────
    report = generate_report(top_papers, rejected, weak, config)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    today       = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    report_path = os.path.join(REPORTS_DIR, f"{today}.md")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # ── Update seen set ───────────────────────────────────────────────────────
    for item in top_papers:
        seen.add(item["paper"]["id"])
    save_seen(seen)

    log.info(f"Report saved → {report_path}")


if __name__ == "__main__":
    main()
