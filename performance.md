# Performance analysis: `risk-state-refactor`

This documents testing done after merging the `risk-state-refactor` branch
(array-based `GameState`, incremental Zobrist hashing, FEN notation) into
`main`, to check (1) whether it changes game behaviour and (2) whether it
makes MCTS faster.

## Correctness

Two checks, comparing the pre-refactor (dict-based `GameState`) and
post-refactor (array-based `GameState`) code on identical seeds:

- **Golden-fixture regression tests** (`tests/test_golden_replay.py`): replay
  a fixed action sequence recorded against the pre-refactor engine through
  `Action.apply()` / `apply_outcome()` on the new engine and assert the final
  state matches exactly. Passing (22/22 tests).
- **100 seeded random-vs-random games**, run against both versions: outcome
  distributions (win rate, `end_reason`, round-count mean/std) are
  statistically indistinguishable, and 81% of same-seed games produced a
  byte-identical winner and round count. The refactor's own docstrings note
  that RNG-consumption order (e.g. territory iteration order) is allowed to
  change, so full trajectory divergence on the remaining 19% is expected, not
  a bug. No crashes or invalid states in either version.

**Conclusion: no behavioural regression.**

## Performance

### Benchmark setup

500 raw `MCTSTree.perform_iteration()` calls (selection, expansion,
simulation, backpropagation), 5 seeds, `IterationBased` stopping, logging
disabled, run against both the pre-refactor and post-refactor code from an
identical initial game state (guaranteed identical per-seed by the
refactor's own invariant that initial layout must not change).

| | Baseline (dict-based) | New (array-based) |
|---|---|---|
| Mean time / 500 iterations | 29.82s | 30.42s |
| Throughput | 16.8 iter/s | 16.4 iter/s |

**No measurable speedup — if anything ~2% slower, well within seed-to-seed
noise (~27.5-33.2s range).**

### Why: profiling breakdown

cProfile over 200 `perform_iteration` calls, both versions:

| Component | Baseline | New | % of total |
|---|---|---|---|
| `get_valid_actions` + `_find_reachable_territories` (action enumeration) | 28.3s | 24.5s | ~65% |
| `battle_computer.get_outcome_probabilities` (scipy sparse lookups) | 6.1s | 5.6s | ~14% |
| `GameState.copy()` (what the refactor targets) | 3.8s (incl. 1.0s copying the *board* too) | 1.5s | ~4-9% |

**The theory (Amdahl's Law):** a speedup only matters proportional to the
share of total runtime it touches. The refactor's target — state copying —
genuinely did get faster:

- `GameState.copy()` dropped from ~2.8s to ~1.5s per 200 calls (~2x), since
  copying an `array.array` of ints is a cheap bulk C-level operation, versus
  the old code copying a dict *and* a dict-of-lists (`player_territories`).
- The new code also stopped deep-copying the immutable `Board` object on
  every state copy (the board never changes during play) — a separate, real
  win bundled into the same commit.

But copying was never the bottleneck — it's ~5-9% of the pie. Even a 2x
speedup there caps the *maximum possible* overall improvement at ~2-4% of
total wall-clock time, smaller than the ~10% run-to-run noise observed in
the timing benchmark. That's the whole story: a real, correctly-implemented
optimization applied to a part of the algorithm that was never the dominant
cost.

**What actually dominates (and wasn't touched by this refactor):**

1. **Action enumeration (~65%)**: `get_valid_actions` does a BFS
   (`_find_reachable_territories`, using `deque`/`set` — identical algorithm
   in both versions) over the adjacency graph for every possible
   attack/fortify source, instantiating a fresh `Action` object per
   candidate — 14+ million `Action.__init__` calls for just 200 MCTS
   iterations. This is pure Python object-churn and graph-traversal
   overhead; it's indifferent to whether territory ownership underneath is a
   dict or an array, because the hot loop is the BFS and the allocation, not
   the lookup.
2. **Battle outcome lookups (~14%)**: `battle_computer.py` is byte-identical
   between versions, yet spends ~6s in scipy sparse-matrix slicing
   (`_get_intXslice`, `_get_submatrix`, `_validate_indices`) per 200
   iterations — surprisingly expensive per-call overhead for what's
   logically a single probability-table lookup. Untouched by this refactor,
   but a much larger target than state representation if raw speed is the
   goal.
3. **Zobrist hashing** (the refactor's other headline feature) is only
   consumed in `MCTSTree.update_root()`, called once per *real* game turn
   for tree-reuse — never inside the simulation loop — so it structurally
   cannot appear in a `perform_iteration` throughput benchmark at all.

### Takeaway

The refactor achieves what it set out to do (faster, cleaner state copying;
enables hashing and FEN save/load) with no correctness regression. It does
not deliver a measurable MCTS speedup, because the actual performance
ceiling is set by action-enumeration overhead and battle-lookup overhead —
neither of which this refactor addressed. If raw search throughput is a
future goal, those two are where the payoff would be.
