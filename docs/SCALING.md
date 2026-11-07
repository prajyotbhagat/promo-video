# Scaling Promo Finder

## The short version

**The model provider's request rate is the bottleneck, not compute.** One worker has ample CPU. To
scale: share one Redis, run several agent workers, keep total requests under the provider's limit,
and route non-urgent weekly work to the provider's batch API.

---

## 1. Unit costs (measured on the two real shows)

| Unit | LLM calls | Tokens (in / out) | Wall time | Other |
|---|---|---|---|---|
| New 50-episode show, budget B2 | 20 | ~213k / ~59k | ~90 s (show 2: 81 s) | 12.6 s of CPU for code stages, 47 MB RAM |
| Weekly drop of 5 episodes | 2 | ~21k / ~6k | ~15 s | — |
| Re-run, nothing changed | 0 | 0 | ~13 s | — |
| Deep review of a show, B4 | +9 (4 on a stronger model) | +~55k / +~7k | +~2 min | — |
| Redis storage per show | — | — | — | ~250 KB (≈70 entries × 3.6 KB) |

Calls per show are fixed by design: one per 5 episodes for annotation, plus one per 5 for the picker.
They don't grow with script length or show popularity. Everything else is plain code.

## 2. Capacity planning

Let **S** = new shows per week, **U** = running shows that get a weekly drop, **R** = your provider's
requests per minute (RPM), **L** = average seconds per call (~5–10 s measured).

```
calls / week          = 20·S + 2·U
tokens / week         ≈ 272k·S + 27k·U
API minutes / week    = calls / R
cost / week           = tokens_in·price_in + tokens_out·price_out   (use your provider's current prices)
workers needed        ≈ R · L / 60 / (calls in flight per worker, default 2)
Redis memory          ≈ 250 KB · (all shows ever processed)   (set a TTL on model replies to cap it)
```

Worked examples (illustrative volumes, not Kuku's real numbers):

| Volume | Calls / week | Tokens / week | API time at 60 RPM | API time at 1,000 RPM | Workers to finish in 1 h* | Redis after 1 year |
|---|---|---|---|---|---|---|
| 10 new + 50 running | 300 | ~4M | 5 min | <1 min | 1 | ~0.15 GB |
| 50 new + 300 running | 1,600 | ~22M | 27 min | 2 min | 2 | ~0.7 GB |
| 500 new + 3,000 running | 16,000 | ~217M | 4.4 h | 16 min | 12 (needs ≥ 270 RPM) | ~7 GB (cap with TTL) |

\* One worker makes 2 calls at a time at ~5 s each, so about 1,400 calls an hour.

Even at 500 new shows a week, the compute is a few CPU-hours. What you have to plan for is the
**provider quota**, so buy a higher RPM tier, spread load over two providers, or use batch APIs.

## 3. Architecture by stage

**Stage 0 (today): one machine.** A cron job runs `python -m promo_finder.agent --once` with a file
cache. Fine up to about 50 shows a week on a free or low tier.

**Stage 1: shared Redis + N workers** (built and tested).

```
          scripts bucket (S3/GCS) --sync--> inbox/<show>/*.txt
                                               |
     +------------------+------------------+---+--------------+
     | worker 1 (cron)  | worker 2 (cron)  | ...  worker N    |   python -m promo_finder.agent --once
     +--------+---------+--------+---------+------------------+
              |                  |
              v                  v
     Redis:  promo:llm:*  promo:annot:*      shared cache (no episode is ever paid for twice)
             promo:state:*                    per-show memory (any worker can resume any show)
             promo:lock:agent:show:*          SET NX EX: one worker per show, N shows in parallel
             promo:count:calls:<date>         INCRBY: one daily budget for all workers
             promo:rpm:<model>:<minute>       INCR: one requests-per-minute limit for all workers
              |
              v
     promo_runs/<show>/latest/  ->  review UI / editor
```

Tested here: 2 workers on 4 shows split them 2/2 with no double processing. On the next run, a
worker skipped the shows the other had finished, because state lives in Redis. With Redis stopped,
the tool fell back to files.

Configure with:

- `PROMO_CACHE_URL=redis://...` (use `rediss://` and a password in production)
- `PROMO_RPM` or `PROMO_RPM_GEMINI=...`
- `--daily-budget`
- `PROMO_WORKER_ID`

**Stage 2: event-driven, hundreds of shows a week** (design, not built):

- A script upload to the bucket fires an event that puts a show id on a queue (Redis Streams or SQS).
- Workers in Kubernetes autoscale on queue depth. Each one runs the same `run_show()`, so locks,
  cache, budget and rate limit carry over unchanged.
- Weekly updates that aren't urgent go to the provider's **batch API**. Providers commonly price
  these at about half the normal rate with a 24-hour turnaround; check current terms. Only new shows
  and launches use the live API.
- Results go to a database plus a review UI, instead of folders.

## 4. Cost levers, in order of impact

1. **Cache everything.** Already done. Nothing is paid for twice, across runs, workers or weeks.
   Weekly cost is about 10% of a full run.
2. **Pick the budget level per show.** B2 (20 small-model calls) for every show. B3/B4 (stronger
   judge + persona panel) only for shows with real ad spend, e.g. the top 10%.
3. **Batch APIs** for weekly runs (see Stage 2).
4. **Token diet.** Already done: compact picker prompt (−47%). Next candidate: skip boilerplate beats
   that the input check finds in most episodes, which would have removed ~85% of show 2's text.
5. **Distil.** `annotations.jsonl` plus editors' accept/reject decisions become training data. A
   small fine-tuned model, or a classical model on the code features, can replace the LLM picker
   for most shows, keeping the LLM for new genres and hard cases.
6. **Multi-provider routing.** The model is one flag per stage, so annotation can run on the
   cheapest model that passes the gold set while judging stays on a stronger one.

## 5. Reliability at scale

- **Idempotent.** A run can be killed and restarted at any time. Finished work is cached, and
  unfinished work is retried on the next tick.
- **Quota errors (429)** back off and retry. Daily caps defer the show to the next tick. The judge
  falls back to the smaller model.
- **Locks expire** (6 h TTL), so a crashed worker never blocks a show forever.
- **Bad inputs** are caught before spending: the input check flags templated packs, and
  `--skip-templated` skips them.
- **The Redis cache is disposable.** Losing it only costs re-calls. Keep it persistent (AOF) to
  avoid that.

## 6. What to monitor

| Metric | Why | Alert when |
|---|---|---|
| calls and tokens per show | cost drift (prompt growth, bigger shows) | > 1.5× baseline |
| cache hit rate | weekly runs should be ~90% cached | < 70% on a weekly run |
| 429 / quota errors, deferred shows | provider limits | deferred > 1 day |
| input-check verdicts | upstream script quality | any "templated" |
| editor accept rate of the top 10 | **quality**: the number that matters | falls week over week |
| real 3s hold / completion / installs per promo | ground truth to re-tune fusion weights | — |

## 7. Scaling quality, not just volume

- **Grow the gold set.** 15 labelled moments is a start. Ask editors to mark the promo they would
  have picked for ~2 episodes per show. That's free labels at scale.
- **Learn the fusion weights** from editor decisions and ad performance (learning to rank) instead of
  equal-weight RRF, once there are a few hundred labelled shows.
- **Recalibrate the persona panel** against real retention curves before trusting it for ranking.
- **New languages and markets:** the pipeline is language-agnostic, but the prompts, personas and gold
  set need a version per market.

## 8. Confidentiality

Scripts are unreleased content:

- Keep them in a private bucket.
- Use Redis with a password and TLS, on a private network.
- Avoid putting script text in logs. Digests carry only hooks and timestamps.
- Use model providers under terms that exclude training on your data.
