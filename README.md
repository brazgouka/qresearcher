# market-research

A research platform for discovering statistically significant, out-of-sample-robust
trading edges — and, more importantly, rejecting the ones that aren't real.

This is a research instrument, not a trading bot. Every module upholds a set of
non-negotiable invariants: no look-ahead, every trial logged, purged CV only, a sacred
holdout, pessimistic costs, labeled regimes, and full reproducibility.

## Development

```bash
uv sync                # install deps into .venv
uv run pytest          # run tests
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Status

**Slice 0 (walking skeleton) — complete.** One data provider, a handful of features,
triple-barrier labeling, one hand-written hypothesis, an event-driven backtester with
pessimistic costs, purged CV + a permutation-test null-model check, and a trial
registry — wired end-to-end and proven by an acceptance test to reject a noise strategy.

**In progress — hypothesis search.** `HypothesisGenerator` enumerates a declared,
bounded search space; `GridSearch` and `RandomSearch` (sharing `BaseSearch`) drive
candidates through the trial registry and validator so every evaluated trial is logged.

## License

[MIT](LICENSE)
