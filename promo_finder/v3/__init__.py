"""v3: rank fusion + budget-aware ranking. Combines rankers that already exist instead of
buying a bigger model. Every step here is arithmetic on cached outputs (no extra LLM calls)."""
