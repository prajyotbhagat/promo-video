# Promo Finder (Kuku AI Generalist assignment, Problem 1)

## Web UI

```bash
pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

- **Sidebar:** upload a zip of episode scripts, point at a folder, or open a previous run. The input
  check runs immediately. Pick a budget level (B0 needs no key) and see the exact number of AI calls
  before you press **Find promos**. API keys typed here stay in memory only.
- **Post first:** the top 10 moments with hook, cliffhanger, the engines that found each one, a
  script view (first 3 s highlighted, ❓ where a question opens) and a chart of arousal and
  unanswered questions. 👍 / 👎 plus a note saves to `editor_feedback.jsonl` in the run folder.
- **All promos:** a filterable table with CSV / JSON / report downloads.
- **Episode explorer:** the curiosity signals across a whole episode, with every candidate clip on a
  timeline.
- **Editor review:** approvals, rejections and the accept rate.

Runs started from the UI are saved under `outputs/ui_runs/`.

## Promo scripts (one screenplay per clip)

Each promo can be exported as its own script, in the same Kuku screenplay format as the episodes:
timed from 00:00 of the promo, one SCENE per continuous piece of footage (a jump cut starts a new
scene, with its source episode and time noted), and every CAMERA / ACTION / SOUND FX / dialogue
block of the clip. Hand it to an editor, a dubbing or caption team, or a generation pipeline that
reads episode scripts.

- **UI:** a **Script** button on each promo card, or **Download scripts for all 10 promos (.zip)**
  (one .txt per promo plus a readable `promo_scripts.md`).
- **CLI:**
  ```bash
  python -m promo_finder.scripts outputs/v3 --top 10     # -> outputs/v3/promo_scripts/
  ```

Checked: all 10 promo scripts parse back with the episode parser to the same duration, number of
lines and text as the clip.

## Run it locally (quick start)

Needs Python 3.10+.

```bash
pip install -r requirements.txt
```

Put the episode .txt files in a folder, one sub-folder per show (the scripts are confidential and
not included):

```
input/I Married a Broken Soldier. He was a Duke/episode-001-....txt
```

Rules only, no API key, ~15 seconds:

```bash
python -m promo_finder.v3 input -o outputs/v3-B0 -b B0
```

Default engine (needs a free Gemini key from https://aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY=...
python -m promo_finder.v3 input -o outputs/v3 -b B2
```

Any Hugging Face model (needs an HF token with Inference Providers credits):

```bash
export HF_TOKEN=hf_...
python -m promo_finder.v3 input -o outputs/v3-qwen -b B2 -m hf:Qwen/Qwen2.5-72B-Instruct
```

Score any run against the labels:

```bash
python eval/evaluate.py outputs/v3/promos.json --split=heldout
```

Three engines:

- **v3 fusion** (recommended, cheapest): several independent rankers fused with Reciprocal Rank
  Fusion; budget levels B0-B4; per-episode content cache. See "v3" below.
- **v2 Curiosity Engine**: research-grounded. The LLM labels the story once; code
  searches every possible clip for the strongest unanswered question at the cut; a pairwise
  tournament and a simulated audience panel rank the finalists. See "v2" below.
- **v1 baseline**: the LLM picks clips directly; rules double-check. Kept for comparison.


Reads timestamped microdrama screenplays and returns ranked 20-60s promo clips:
episode + start/end of every segment, hook, cliffhanger, one-line "why", and
a quality-check report showing where it picks badly.

## Run it

```bash
pip install -r requirements.txt          # only needs `requests`
```

No API key (rule-based baseline, about 15s for 50 episodes):

```bash
python -m promo_finder path/to/episodes -o outputs/heuristic -m heuristic
```

With an LLM (free Gemini key from https://aistudio.google.com/apikey):

```bash
export GEMINI_API_KEY=...
python -m promo_finder path/to/episodes -o outputs/gemini -m gemini:gemini-2.5-flash
```

Swap the model with one flag; nothing else changes:

| `-m` value | needs |
|---|---|
| `heuristic` | nothing |
| `gemini:gemini-2.5-flash` / `gemini:gemini-2.5-pro` | `GEMINI_API_KEY` |
| `groq:llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `openai:gpt-4.1-mini` | `OPENAI_API_KEY` |
| `anthropic:claude-sonnet-5-5` | `ANTHROPIC_API_KEY` |
| `openrouter:<model>` | `OPENROUTER_API_KEY` |
| `ollama:llama3.1` | local Ollama |
| `compat:<model>` | any OpenAI-compatible endpoint: `LLM_BASE_URL`, `LLM_API_KEY` |

Useful flags: `--episodes 1-10`, `-k 3` (max promos per episode), `--no-cross`,
`--cold-top 16` (how many promos get the cold-viewer test), `--workers 4`.
LLM replies are cached in `.cache/llm/`, so a re-run is free and repeatable.

Score a run against the hand-labelled moments:

```bash
python eval/evaluate.py outputs/gemini/promos.json outputs/heuristic/promos.json --random
```

## Outputs (per run folder)

| file | for |
|---|---|
| `report.md` | "post these first" top 10 + full ranked list |
| `promos.csv` | editor sheet: rank, segments, duration, hook, cliffhanger, why, flags |
| `review.html` | open in a browser: each promo with its script, hook (yellow) and last 4s (red) highlighted |
| `qc.md` | validator fixes, coverage, LLM-vs-rules disagreements, cold-viewer results |
| `promos.json` | everything, machine-readable |
| `raw_llm/` | unedited model replies |

## How it works

```
scripts --parse--> beats (every camera/action/line/SFX with start-end)          [code]
   |-- rule-based scorer: hook / cliffhanger / standalone / intensity            [code, free]
   |-- LLM per episode: "pick up to 3 promos", answers with BEAT IDs only       [LLM, 1 call/episode]
   |-- validator: ids -> timestamps, never cut mid-line or mid-sentence,
   |              trim silent lead-ins/dead tails, force 20-60s, log every fix   [code]
   |-- LLM cross-episode: join moments from different episodes                  [LLM, 1 call/show]
   |-- LLM cold-viewer test: a fresh call with NO show context reads each
   |              shortlisted clip and says what it understood + would it stop   [LLM, 2 calls]
   '-- rank: 50% picker score + 20% rule score + 30% cold-viewer score,
              minus flag penalties, small bonus for episodes 1-10;
              "post first" = top 10 with at most 2 per episode                   [code]
```

The model never writes a timestamp: it names beat ids and code looks up the
times. That removes made-up timestamps, the most common failure.

Prompts live in `promo_finder/prompts.py` so an editor can tune them without
touching the pipeline.

## Layout

```
promo_finder/  parser.py heuristics.py llm.py prompts.py validate.py pipeline.py report.py
eval/          gold.json (hand-labelled moments)  evaluate.py
outputs/       raw outputs from real runs
input/         the episode pack (not committed: confidential)
```

## v2 Curiosity Engine

```bash
export GEMINI_API_KEY=...
python -m promo_finder.v2 path/to/episodes -o outputs/v2 \
    -m gemini:gemini-3.5-flash-lite --judge-model gemini:gemini-3.5-flash
```

By default it also runs the v1 LLM picker as a second candidate generator (`--seed auto`), so both
generators compete in the same blind tournament. Use `--seed none` for the optimizer alone, or
`--seed path/to/promos.json` to reuse an earlier v1 run.

About 20-30 requests per 50-episode show. On the free Gemini tier the stronger judge model allows ~20
requests a day; if it runs out, the tournament falls back to the annotation model and logs it.

| Stage | Who | Idea | Source |
|---|---|---|---|
| 1. Annotate every beat: arousal, valence, surprise, context need, turning point, questions opened / answered | LLM, once per show | narrative structure + sentiment drive trailer selection; turning-point types | Papalampidi, Keller & Lapata 2019 (TRIPOD), 2021 (trailer generation) |
| 2. Search every clean 20-60s clip and flash-forward cold open; score hook, **information gap at the cut**, suspense, turning point, minus context debt and self-resolution | code, free, explainable | curiosity = an open information gap; suspense = sustained uncertainty | Loewenstein 1994; Wilmot & Keller 2020 |
| 3. Shortlist 24, judge pairs in both orders, fit Bradley-Terry | LLM judge + code | pairwise beats absolute scores; measure and cancel position bias | Zheng et al. 2023 (LLM-as-a-judge) |
| 4. Five viewer personas say the second they would swipe and whether they would tap; build retention curves | LLM panel | persona-conditioned "silicon samples" as a cheap pre-test | Argyle et al. 2023 |

Outputs add `annotations.jsonl` (a reusable labelled dataset of the show), `questions.json`, and a
`review.html` with a per-clip chart of arousal and unanswered questions over time and the panel's
predicted retention curve.

Evaluate and ablate:

```bash
python eval/evaluate.py outputs/v2/promos.json outputs/gemini-3.5-flash-lite/promos.json --random
python eval/ablate.py outputs/v2
```

## v3: rank fusion + budget levels (recommended)

```bash
export GEMINI_API_KEY=...
python -m promo_finder.v3 path/to/episodes -o outputs/v3 -b B2
```

| Budget | Adds | Calls per show | Dev / held-out (top clip right) |
|---|---|---|---|
| B0 | rule-based windows | 0 | 6/8, 5/6 |
| B1 | beat annotation + curiosity optimizer | 10 | 6/8, 5/6 |
| **B2** (default) | compact LLM picker | 20 | **8/8, 5/6** |
| B3 | pairwise tournament (`--judge-model`) | +4 | (offline: no change vs B2) |
| B4 | persona audience panel | +5 | (offline: no change vs B2) |

- **Fusion:** RRF, score = sum over rankers of 1/(60 + rank). Clips that cover >= 50% of each other are one
  "moment" and pool their votes. Borda and z-score averaging were tested and scored lower.
- **Cache:** annotations are stored per episode under `.cache/annot/`, keyed by a hash of model +
  instructions + episode text. A weekly run with 5 new episodes annotates only those 5.
- **Token diet:** the picker skips plain camera directions (47% shorter prompt, no accuracy loss).

Reproduce the fusion experiments (no API calls, uses cached outputs):

```bash
python eval/fusion_experiment.py              # fusion variants vs single rankers
python eval/fusion_experiment.py --frontier   # cost / quality per budget level
```

## Promo agent (runs everything automatically)

The agent watches an inbox, plans the exact LLM calls needed (by probing the caches), keeps within
a daily call budget, runs v3, and writes a digest of what changed in the top 10.

```
inbox/
  <Show Name>/episode-001.txt ... episode-050.txt     <- drop new episodes here each week
promo_runs/
  agent.log, agent_state.json                          <- memory: file hashes, calls spent per day
  <Show Name>/latest/  <timestamp>/  digest.md          <- results, history, weekly digest
```

```bash
export GEMINI_API_KEY=...
python -m promo_finder.agent --inbox inbox --outbox promo_runs --once --dry-run   # show the plan, no calls
python -m promo_finder.agent --inbox inbox --outbox promo_runs --once             # one tick (cron)
python -m promo_finder.agent --inbox inbox --outbox promo_runs --watch 3600       # tick every hour
```

Run it on a schedule with cron (every Monday at 06:17):

```
17 6 * * 1  cd /path/to/promo-finder && GEMINI_API_KEY=... python -m promo_finder.agent --once >> promo_runs/cron.log 2>&1
```

What one tick does:

1. **Scan.** Hash every episode file and compare with what the agent has already processed.
2. **Plan.** Count the calls still needed: annotation batches with uncached episodes, plus picker
   batches whose reply is not cached. A file whose text is unchanged (for example, re-saved) costs 0.
3. **Budget.** If the plan exceeds `--daily-budget` (default 40, matching free tiers), process the earliest
   episodes that fit, keeping batches aligned, and resume on the next tick.
4. **Act.** Run v3 at `--budget-level` (default B2).
5. **Report.** Diff the top 10 against the last run, append to `digest.md`, and optionally post to a
   Slack-compatible webhook set in `PROMO_WEBHOOK_URL`.
6. **Remember.** Save hashes, calls spent and anything deferred. Failures (for example, quota errors) are
   retried on the next tick.

Measured (`outputs/agent_demo/`): a new 50-episode show plans 20 calls. With 6 calls left in the day, it
does episodes 1-15 now and the rest the next day. Adding 5 new episodes planned 2 calls and made 2.
A tick with nothing new made 0.

Caches live in `.cache/` (or `$PROMO_CACHE_DIR`).

## Cache: Redis (shared) or files (default)

Set `PROMO_CACHE_URL` to use Redis. Without it, the cache is files under `.cache/`.

```bash
redis-server --daemonize yes                      # or a managed Redis
export PROMO_CACHE_URL=redis://localhost:6379/0
export PROMO_CACHE_TTL_DAYS=90                    # optional: expire model replies (annotations never expire)
python -m promo_finder.cache migrate --from .cache   # copy an existing file cache in (idempotent)
python -m promo_finder.cache stats
```

| Key | Holds |
|---|---|
| `promo:llm:<sha256>` | a model reply, keyed by model + prompts |
| `promo:annot:<sha256>` | one episode's annotations, keyed by model + instructions + episode text |
| `promo:lock:agent:<inbox>` | agent lock (`SET NX EX`), so two workers never process the same inbox at once |
| `promo:count:calls:<date>` | calls spent today (`INCRBY`), so the daily budget is shared by all workers |

Why Redis: several agent workers or teammates share one cache, so no episode is ever paid for twice.
Replies can expire, and the lock and budget are atomic across machines.

If Redis is unreachable, the tool prints a warning and uses the file cache, so a cache outage only
costs extra calls and never stops a run.

Measured: the full cache for two shows was 224 entries, 1.8 MB in Redis. With the file cache hidden,
both shows planned 0 calls, served from Redis. A read takes about 90 µs from Redis and 16 µs from a
file, both negligible next to a model call.

## Scaling

See [`docs/SCALING.md`](docs/SCALING.md): measured unit costs, a capacity formula, the architecture at
each stage, cost levers, reliability and monitoring. To run several workers:

```bash
export PROMO_CACHE_URL=redis://redis-host:6379/0   # shared cache, state, locks, budget, rate limit
export PROMO_RPM_GEMINI=900                         # stay under the provider limit across ALL workers
PROMO_WORKER_ID=w1 python -m promo_finder.agent --once &
PROMO_WORKER_ID=w2 python -m promo_finder.agent --once &
```

Each show is locked by whichever worker takes it, so N workers split an inbox with no double work.
