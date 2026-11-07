import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="promo_finder.v2", description="Curiosity Engine promo finder (v2).")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="outputs/v2")
    ap.add_argument("-m", "--model", default="gemini:gemini-3.5-flash-lite", help="annotation model")
    ap.add_argument("--judge-model", help="tournament judge (default: --model)")
    ap.add_argument("--audience-model", help="persona panel (default: --model)")
    ap.add_argument("--batch", type=int, default=5, help="episodes per annotation call")
    ap.add_argument("-k", "--per-episode", type=int, default=3)
    ap.add_argument("--shortlist", type=int, default=24)
    ap.add_argument("--finalists", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=4, help="tournament rounds (pairs per clip)")
    ap.add_argument("--episodes", help="e.g. 1-10")
    ap.add_argument("--seed", default="auto",
                    help="second candidate generator: 'auto' runs the v1 LLM picker (default), a promos.json path "
                         "reuses one, 'none' uses the optimizer only")
    a = ap.parse_args()
    eps = None
    if a.episodes:
        lo, _, hi = a.episodes.partition("-")
        eps = list(range(int(lo), int(hi or lo) + 1))
    run(a.input, a.out, a.model, a.judge_model, a.audience_model, a.batch, a.per_episode,
        a.shortlist, a.finalists, a.rounds, eps, None if a.seed == 'none' else a.seed)


if __name__ == "__main__":
    main()
