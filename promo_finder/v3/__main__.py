import argparse

from .pipeline import LEVELS, run


def main():
    ap = argparse.ArgumentParser(prog="promo_finder.v3", description="Budget-aware promo finder with rank fusion.")
    ap.add_argument("input")
    ap.add_argument("-o", "--out", default="outputs/v3")
    ap.add_argument("-b", "--budget", default="B2", choices=LEVELS,
                    help="B0 rules (0 calls) | B1 +annotation | B2 +LLM picker (default) | B3 +tournament | B4 +panel")
    ap.add_argument("-m", "--model", default="gemini:gemini-3.5-flash-lite")
    ap.add_argument("--judge-model", help="stronger model for B3+ tournament (default: --model)")
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--episodes", help="e.g. 1-45")
    ap.add_argument("--full-picker-prompt", action="store_true", help="send every camera beat to the LLM picker")
    a = ap.parse_args()
    eps = None
    if a.episodes:
        lo, _, hi = a.episodes.partition("-")
        eps = list(range(int(lo), int(hi or lo) + 1))
    run(a.input, a.out, a.model, a.budget, a.judge_model, a.batch, episodes=eps, compact=not a.full_picker_prompt)


if __name__ == "__main__":
    main()
