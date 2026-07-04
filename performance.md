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

## Optimization plan (target: ≥60% reduction in wall-clock time)

Follow-up to the takeaway above, based on a code-level look at the two
dominant buckets (action enumeration ~65%, battle lookups ~14%). Since these
two buckets together account for ~79% of runtime, a 60% overall reduction
requires attacking both — no single change gets there alone.

### 1. Battle outcome lookup cache (14% bucket → near-eliminated, low risk)

`BattleComputer.get_outcome_probabilities()`
(`src/risk_agent/engine/battle_computer.py:140-160`) does
`self.stationary_distribution[row_index]` on a 251k×251k scipy sparse matrix
on every call — even though it's logically "grab one row," scipy's
`_get_intXslice`/`_validate_indices` machinery has real per-call overhead.
Called from both `expansion()` and `playout()`
(`src/risk_agent/players/mcts/tree.py:379` and `:521`) for every
`AttackAction`.

**Fix:** lazy memoization dict keyed by `(attacking_armies,
defending_armies)`, caching the extracted `.indices`/`.data` as plain numpy
arrays on first lookup. Real games only span a small range of army counts,
so this converges to O(1) dict lookups after warm-up, bypassing scipy
indexing entirely.

Risk: trivial (same values, just cached). Verify: unit test comparing
cached vs. uncached output across a spread of `(attacking, defending)`
pairs.

### 2. Fix redundant BFS in fortify enumeration (part of the 65% bucket, low risk)

`GameEngine.get_valid_actions()` (`src/risk_agent/engine/game_engine.py:209-220`)
calls `_find_reachable_territories` once per owned source territory. But
"reachable via own territories" is exactly the connected component of that
territory in the subgraph induced by `owner == player` — every territory in
the same component has an identical reachable-set, so recomputing per-source
is pure waste.

**Fix:** compute connected components once per `get_valid_actions` call
(single BFS/union-find pass over all player-owned territories), then look up
each source's precomputed component list instead of re-running BFS.

Risk: low, behavior-preserving. Verify: `tests/test_golden_replay.py` must
stay 22/22.

### 3. Reduce `Action` object churn (part of the 65% bucket, low-medium risk)

Profiling flagged 14M+ `Action.__init__` calls for 200 iterations.

- Add `__slots__` to `Action`/`AttackAction`/`FortifyAction`
  (`src/risk_agent/game_elements/action.py:201` and `:287`) — cuts
  per-instance construction/attribute overhead at near-zero risk.
- **Bigger lever:** in `MCTSTree.playout()` (`tree.py:482-569`), the full
  valid-action list is materialized just to pick one at random, every step,
  until the game ends — likely the dominant share of those 14M allocations.
  Replace with lazy sampling: count candidates cheaply (no object
  construction), pick a random index, then construct only the one chosen
  `Action`. Must preserve the exact same uniform distribution over valid
  actions to avoid a behavior change.

Risk: medium for the playout-sampling change specifically, since it touches
the simulation policy rather than just its implementation. Verify: re-run
the 100-seeded-game comparison (win rate / `end_reason` / round-count
distributions must stay statistically indistinguishable, same bar as the
Correctness section above).

### 4. Expansion path left as-is

`expansion()` needs the full action list (all children get created), so
item 3's laziness doesn't apply there — it only benefits from item 2's
component-cache reuse and the `__slots__` change.

### Expected impact (predictions, not yet measured)

| Change | Bucket affected | Est. reduction of that bucket |
|---|---|---|
| Battle lookup cache | 14% | ~90% |
| Fortify component caching | part of 65% | meaningful but bounded, depends on branching |
| `__slots__` | 65% | ~10-20% |
| Lazy playout sampling | 65% (majority share) | ~50-70% |

Stacking these plausibly lands total wall-clock at ~40-50% of baseline
(i.e., 50-60% faster). Item 3's playout-sampling change is what determines
whether it clears the 60% target or lands just under it — worth prototyping
first and re-benchmarking before committing to the rest.

### Verification plan

Reuses the same methodology as the Correctness/Benchmark sections above:

1. `tests/test_golden_replay.py` after each change (must stay 22/22).
2. 100-seeded random-vs-random comparison after item 3 specifically, since
   it changes *how* actions are sampled, not just enumerated.
3. Re-run the 500-iteration `perform_iteration` timing benchmark (same 5
   seeds) before/after each change, to replace the estimates above with
   real numbers.

**Status: plan only, not yet implemented.**
