# Research Trajectory Scout

A free, automated daily pipeline that finds papers matching your research shape —
not just your past topics.

**What it does:** searches arXiv + Semantic Scholar every morning → scores each paper
against your research identity using keyword matching + semantic similarity →
generates a ranked Markdown report → commits it to this repo.

**Cost:** $0. Runs entirely on GitHub Actions free tier.

**Runtime:** ~4–6 minutes per day.

---

## Quick Setup (15 minutes, one time)

### Step 1 — Fork or clone this repo

```bash
git clone https://github.com/YOUR_USERNAME/research-trajectory-scout.git
cd research-trajectory-scout
```

Or fork it directly on GitHub.

### Step 2 — Test locally first

```bash
# Create a virtual environment (optional but clean)
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install CPU-only torch first (saves ~600 MB)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install everything else
pip install -r requirements.txt

# Run the scout
python agent.py
```

After ~3–5 minutes you will see:

```
reports/YYYY-MM-DD.md
```

Open that file. If the recommendations feel "you-shaped", you are done locally.

### Step 3 — Push to GitHub

```bash
git init                          # skip if you cloned/forked
git add .
git commit -m "Initial scout setup"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/research-trajectory-scout.git
git push -u origin main
```

### Step 4 — Add your Semantic Scholar API key (optional but recommended)

A free API key raises the Semantic Scholar rate limit from 1 req/s → 10 req/s.

1. Go to https://api.semanticscholar.org/ → sign up → copy your API key.
2. In your GitHub repo: **Settings → Secrets and variables → Actions → New repository secret**
3. Name: `SEMANTIC_SCHOLAR_API_KEY`  |  Value: your key

Without this key the agent still works — it just runs slightly slower.

### Step 5 — Trigger the first GitHub Actions run

1. Go to your repo on GitHub.
2. Click **Actions** → **Daily Research Scout** → **Run workflow** → **Run workflow**.
3. Watch the run. It should take 4–6 minutes.
4. After it finishes, check the `reports/` folder for today's Markdown file.

From now on the workflow runs automatically every day at **09:00 IST (03:30 UTC)**.

---

## File Structure

```
research-trajectory-scout/
│
├── agent.py                  ← Main pipeline (search → score → report)
├── config.yaml               ← Search queries, scoring weights, settings
├── profile.md                ← Your research identity (edit this to retune)
├── requirements.txt          ← Python dependencies
├── .gitignore
│
├── data/
│   └── seen_papers.json      ← Papers already shown to you (auto-updated)
│
├── reports/
│   └── YYYY-MM-DD.md         ← Daily reports (auto-committed by the bot)
│
└── .github/
    └── workflows/
        └── daily.yml         ← GitHub Actions schedule + steps
```

---

## How to Read the Report

Each report opens with an **Executive Summary** — a 5-line quick scan.
The `🟢` badge means Strong fit (score ≥ 30). `🟡` means Medium fit (score ≥ 18).

The **score** breaks down as:

```
score = keyword_score + shape_score + semantic_score − penalty
         (matched keywords × 2)  (shape categories × 3)  (0–20, cosine sim)  (negative keywords × 3)
```

The **semantic similarity** column (0.0–1.0) is the most important signal.
A paper scoring 0.45+ is almost certainly worth reading even if it looks unfamiliar.

Typical cutoffs:
| Semantic sim | Interpretation |
|---|---|
| ≥ 0.50 | Very high match — read it |
| 0.40–0.49 | Good match — skim it |
| 0.30–0.39 | Marginal — only if topic is interesting |
| < 0.30 | Probably not your shape |

---

## How to Retune

### Change search breadth
Edit `days_back` in `config.yaml` (increase to 60–90 if daily results are thin).

### Change the identity sentence
Edit `IDENTITY_SENTENCE` in `agent.py`. This single line drives semantic scoring.
Example alternatives:
- `"low-cost sensor-based anomaly detection for public infrastructure in resource-constrained settings"`
- `"applied machine learning for environmental monitoring using non-obvious sensing modalities"`

### Add search queries
Add new entries to `search_queries` in `config.yaml`.
Keep queries broad — the scoring filters; the queries should just cast a wide net.

### Tune the profile
Edit `profile.md`. This file is your reference — it doesn't directly affect scoring
(that's done via `IDENTITY_SENTENCE` and the keyword lists), but it helps you
remember what the agent is optimising for.

### Add new negative keywords
Add to `negative_keywords` in `config.yaml`. Each hit subtracts 3 from the score.
Note: `transformer architecture` and `foundation model` are **soft** negatives —
they don't fire if the paper is clearly a sensing paper.

---

## Upgrade Roadmap

| Phase | What to add | Why |
|---|---|---|
| **Now** | This setup | Works. Free. |
| **v2** | Papers With Code API | Catches dataset papers not on arXiv |
| **v3** | OpenAlex source | Broader journal coverage |
| **v4** | Telegram / email digest | Push instead of pull — you get it in your phone |
| **v5** | Weekly "best of week" summary | Reduces noise, surfaces the real gems |

### Telegram digest (v4, ~20 lines of code)

Add a `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` secret to GitHub.
Then append to `agent.py`:

```python
def send_telegram_digest(top_papers, bot_token, chat_id):
    if not top_papers:
        return
    lines = ["*Daily Research Scout*\n"]
    for idx, item in enumerate(top_papers[:5], 1):
        p, s = item["paper"], item["scoring"]
        lines.append(
            f"*{idx}.* [{p['title'][:70]}]({p['link']})\n"
            f"   {s['fit']} · sem {s['semantic_similarity']:.2f}"
        )
    msg = "\n".join(lines)
    requests.post(
        f"https://api.telegram.org/bot{bot_token}/sendMessage",
        json={"chat_id": chat_id, "text": msg, "parse_mode": "Markdown"},
    )
```

Call it at the end of `main()`:

```python
token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
chat  = os.environ.get("TELEGRAM_CHAT_ID", "")
if token and chat:
    send_telegram_digest(top_papers, token, chat)
```

---

## Troubleshooting

**No results in the report**
- arXiv may have been slow. Check the Actions log for timeout errors.
- Try increasing `max_results_per_query` or `days_back` in `config.yaml`.

**`sentence-transformers` import error locally**
- Run `pip install torch --index-url https://download.pytorch.org/whl/cpu` first,
  then `pip install sentence-transformers scikit-learn`.

**GitHub Actions push fails with permission error**
- Make sure `permissions: contents: write` is in `daily.yml` (it is by default here).
- Check repo Settings → Actions → General → Workflow permissions → Read and write.

**All papers are showing as "Weak fit"**
- Lower the score thresholds in `score_paper()` (change `>= 30` to `>= 22` for strong fit).
- Or add more positive keywords that appear in your target papers.

**The same papers keep appearing**
- They shouldn't — `seen_papers.json` tracks shown papers. If you cleared it manually
  or reset the repo, seen papers will re-appear once.
