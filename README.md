# market-research

A research platform for discovering statistically significant, out-of-sample-robust
trading edges — and, more importantly, rejecting the ones that aren't real.

This is a research instrument, not a trading bot. See `quant-research-platform-prompt.md`
for the full build specification, including the non-negotiable invariants (no look-ahead,
every trial logged, purged CV only, sacred holdout, pessimistic costs, labeled regimes,
full reproducibility) that every module in this repo must uphold.

## Development

```bash
uv sync                # install deps into .venv
uv run pytest          # run tests
uv run ruff check .    # lint
```

## Status

Building the Slice 0 walking skeleton (see spec §7): one data provider, a handful of
features, triple-barrier labeling, one hand-written hypothesis, an event-driven
backtester with pessimistic costs, purged CV + a permutation-test null-model check, and
a trial registry — wired end-to-end and proven to reject a noise strategy before any
breadth is added.
