# Quantitative Research Platform — Build Specification for Claude Code

## 0. Mission

Build a research platform that discovers **statistically significant, out-of-sample-robust trading edges** and, more importantly, **rejects the ones that aren't real.** This is a research instrument, not a trading bot.

**Prime directive:** Optimize for *falsifying hypotheses as fast and as cheaply as possible.* Most hypotheses must fail. A run that rejects 9,997 of 10,000 candidates with high confidence is a *success*, not a disappointment. The platform's job is to be the harshest honest critic of its own findings.

The deepest risk in this system is that it is, structurally, a **multiple-testing machine**: generating and backtesting thousands of hypotheses *guarantees* that some will look profitable by chance. Every design decision below exists to make that failure mode impossible to ignore or accidentally hide.

---

## 1. Non-negotiable invariants

These are contractual. Every module must uphold them; a module that can violate one is broken by definition. Add these as assertions/tests where feasible.

1. **No look-ahead, ever.** A feature or signal at time *t* may only use information available at *t*. No centered windows. No full-series `.fit()` before splitting. No adjusted-price series that retroactively bakes in future splits/dividends unless the adjustment is explicitly point-in-time.
2. **Every trial is logged.** Every hypothesis *ever evaluated* — including discarded ones — is recorded with its config, seed, and result. Multiple-testing corrections are computed against the **total trial count**, not the survivors. Discarding a trial without logging it is a leakage bug.
3. **Time-series-correct validation only.** No standard k-fold CV. Use **purged k-fold with embargo**. Overlapping labels get uniqueness-based sample weights.
4. **The holdout is sacred.** A final out-of-sample period is locked and touched *at most once per strategy*. The number of holdout evaluations is itself logged, because each touch degrades its statistical validity.
5. **Costs are pessimistic by default.** Transaction costs and slippage are modeled conservatively. An edge that only survives zero-cost assumptions is presumed to be spread capture and is rejected.
6. **Every result declares its regime.** In-sample vs. out-of-sample vs. holdout is always explicit in output. A number without this label is not reportable.
7. **Everything is reproducible.** Fixed seeds, hashed configs, versioned data. A result must be re-derivable bit-for-bit from its logged config.

If you are ever tempted to relax one of these "just for now," stop and surface it to me instead.

---

## 2. Tech stack

**Core:** Python 3.13, `uv` (dependency management — faster, now-standard alternative to Poetry), Polars (primary dataframe), NumPy, SciPy, DuckDB + Parquet (storage/analytics), Pydantic (config/validation), Hydra (experiment configuration & sweeps), Typer (CLI), Loguru, Ruff, Pytest, Docker, Makefile, GitHub Actions.

**ML (phase-gated):** scikit-learn, XGBoost, LightGBM, CatBoost. **Deep learning (PyTorch, Transformers, Temporal Fusion Transformer) is explicitly deferred** until the gradient-boosted-tree pipeline is proven — financial data's signal-to-noise ratio makes deep nets a high-overfit, low-payoff bet early on. Do not build them in the first passes.

**Stats/validation:** SciPy, statsmodels, plus custom implementations of Deflated Sharpe Ratio, PBO/CSCV, White's Reality Check, and purged CV (these are the point of the platform — do not hand-wave them).

**Reproducibility:** lightweight experiment tracking (MLflow, or a DuckDB-backed run registry if MLflow is too heavy) + content-addressed Parquet for data versioning.

**Dashboard:** Streamlit + Plotly (a research surface, not a served product). **FastAPI is deferred** unless/until we actually need a programmatic API.

Justify any deviation from this stack in a short ADR before implementing it.

---

## 3. Architecture

Modular, dependency-injected, no global state, fully typed. Rough layout:

```
market_research/
  data/          # providers, point-in-time universe, calendars
  storage/       # Parquet + DuckDB, content-addressed versioning
  features/      # independent feature transforms, common interface
  labeling/      # triple-barrier, meta-labeling, sample weights
  hypotheses/    # rule DSL + generators (the "search space")
  backtest/      # event-driven engine, costs, sizing
  metrics/       # performance & risk stats
  validation/    # purged CV, DSR, PBO, White RC, FDR, permutation
  models/        # sklearn/GBM wrappers behind a common interface
  search/        # grid/random/Bayesian/GA over the hypothesis space
  ranking/       # multi-criteria strategy ranking
  registry/      # trial log + experiment tracking (invariant #2)
  dashboard/     # Streamlit research UI
  cli/           # Typer entrypoints
  configs/       # Hydra configs
  tests/         # unit + integration + leakage-regression tests
  docs/
```

**Key interfaces to nail down first** (design these before writing implementations):
- `DataProvider` — uniform access across vendors, calendar-aware, point-in-time universe.
- `Feature` — pure transform, declares its lookback, forbidden from future access by construction.
- `Labeler` — maps price paths + events → labels + sample weights.
- `Hypothesis` — a composable predicate (see §4) that emits positions.
- `Backtester` — consumes hypothesis + data → trades + equity curve, cost-aware.
- `Validator` — consumes a strategy's returns + the **global trial count** → a verdict.
- `TrialRegistry` — the append-only ledger every search writes to.

---

## 4. The hypothesis DSL (this is the heart)

Do **not** hardcode strategies. Build a small composable predicate algebra:
- Primitives: feature comparisons (`RSI < 25`), crossovers, rankings, probabilities.
- Combinators: `AND`, `OR`, `NOT`, thresholds.
- A generator that enumerates/samples combinations up to a bounded complexity.

**Critical:** the generator and the trial registry are co-designed. The generator cannot emit a hypothesis without the registry recording it. The search space size (and, for adaptive search, the *effective* number of tests) must be reportable at all times, because §1.2 and the validation battery depend on it. Document how you count effective trials under Bayesian/GA search, where naive counting breaks.

---

## 5. Labeling (do not skip — it dominates results)

- Default: **triple-barrier method** (profit-take / stop-loss / time barriers) → path-dependent labels.
- **Meta-labeling**: a secondary model decides *whether to act* on a primary signal (improves precision, controls false positives).
- **Sample weights** by label uniqueness to handle overlapping outcomes.
- Predict *probability of positive return / expected return / expected volatility* — never exact future price.

---

## 6. Validation battery (the reason this platform exists)

Every surviving strategy runs the gauntlet, and results are reported *against the global trial count*:
- Purged k-fold CV with embargo.
- Walk-forward / rolling-window out-of-sample.
- **Deflated Sharpe Ratio** and **Probability of Backtest Overfitting (PBO via CSCV)** — the tools built specifically for "we ran thousands of backtests."
- White's Reality Check / Hansen's SPA.
- False Discovery Rate + multiple-hypothesis correction.
- Bootstrap + Monte Carlo confidence intervals; permutation tests.
- Stationarity tests (ADF/KPSS).
- **Null-model benchmarks**: every strategy is compared against buy-and-hold, random-entry-same-exposure, and reshuffled-returns. Beating nothing is failing.

---

## 7. Build order — WALKING SKELETON FIRST

Do **not** build breadth-first (all providers, then all features, then all models). Build one correct, thin, end-to-end vertical slice, then widen. Ship and test each step before the next.

**Slice 0 — Prove the machinery rejects noise.**
1. One `DataProvider` (Yahoo as a dev stub — *note its survivorship bias in a comment*), Parquet+DuckDB storage.
2. ~5 features behind the `Feature` interface, with a leakage-regression test each.
3. Triple-barrier labeling.
4. One hand-written hypothesis + event-driven backtester with pessimistic costs.
5. Purged CV + the DSR/PBO/permutation battery.
6. `TrialRegistry` wired in.
7. **Acceptance test:** feed the system a strategy you *know* is noise (e.g., random signals) and confirm the validation layer rejects it with high confidence, and that a trivially real effect (if constructable) is not rejected. If the platform can't correctly reject noise, nothing else matters.

**Then widen, one proven layer at a time:** more features → the hypothesis generator + search (grid → random → Bayesian → GA, each writing to the registry) → GBM models + meta-labeling → ranking → Streamlit dashboard → additional data providers (Polygon/Alpaca as the *real* research source; crypto later, with separate calendar handling). Deep learning last, and only if the tree pipeline plateaus.

---

## 8. Definition of done (per module)

A module isn't finished until it has: full type hints; unit tests including at least one **leakage-regression test** where applicable; a docstring/ADR explaining the design tradeoff; an example in `docs/`; and it passes Ruff + the CI pipeline. Integration tests cover each newly connected slice.

---

## 9. Working agreement

1. Design the interface and explain tradeoffs (short ADR) **before** implementing.
2. Build one module. Test it — including a leakage/null test. Show me the result.
3. Refactor if needed. Then continue.
4. Never violate a §1 invariant silently; surface it to me instead.
5. Prefer a small correct system over a large impressive one. The 100k-LOC scale is an *emergent* outcome of doing this rigorously, not a target to chase.

Start by proposing the concrete interfaces in §3 and the Slice 0 plan. Do not write implementation code until we've agreed on those.
