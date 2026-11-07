import argparse

from .pipeline import run


def main():
    ap = argparse.ArgumentParser(prog="promo_finder", description="Find promo clips in microdrama screenplays.")
    ap.add_argument("input", help="folder with episode .txt screenplays (searched recursively)")
    ap.add_argument("-o", "--out", default="outputs/run", help="output folder")
    ap.add_argument("-m", "--model", default="heuristic",
                    help="'heuristic' (no AI) or provider:model, e.g. gemini:gemini-2.5-flash, groq:llama-3.3-70b-versatile, "
                         "openai:gpt-4.1-mini, anthropic:claude-sonnet-5-5, ollama:llama3.1")
    ap.add_argument("-k", "--per-episode", type=int, default=3, help="max promos per episode")
    ap.add_argument("--episodes", help="subset, e.g. 1-10 or 1,3,5")
    ap.add_argument("--no-cross", action="store_true", help="skip cross-episode promos")
    ap.add_argument("--cold-top", type=int, default=16, help="how many top promos get the cold-viewer test (0=off)")
    ap.add_argument("--batch", type=int, default=1,
                    help="episodes per LLM call (use 5 on free tiers with a daily request cap)")
    ap.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    a = ap.parse_args()
    eps = None
    if a.episodes:
        eps = []
        for part in a.episodes.split(","):
            if "-" in part:
                lo, hi = part.split("-")
                eps += list(range(int(lo), int(hi) + 1))
            else:
                eps.append(int(part))
    run(a.input, a.out, a.model, a.per_episode, not a.no_cross, a.cold_top, eps, a.workers, a.batch)


if __name__ == "__main__":
    main()
