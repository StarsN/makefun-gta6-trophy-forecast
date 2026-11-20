[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/makefun-gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/StarsN/makefun-gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

[make.fun bounty: Forecast how many GTA VI players earn each trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

# GTA VI trophy rarity forecast

Forecast, for every Grand Theft Auto VI trophy on PlayStation 5, the percent
of the players Exophase tracks who will have earned it at the case's target,
as five quantiles per trophy. Entry for the make.fun competition
`gta6-trophy-rarity-forecast-01`.

## Run

```sh
python3 main.py
```

Reads every `*.in` case in `$MAKEFUN_INPUT_DIR` (default `/input`) and writes
`<id>.out` for each to `$MAKEFUN_OUTPUT_DIR` (default `/output`). Python 3.13
standard library only, no network, deterministic: two runs on the same case
write byte-identical output. About 0.05 s per case.

## The model

An earn rate is a ratio, `r = 100 * E / N`: `E` is how many tracked players
have earned the trophy, `N` is how many players Exophase tracks. The two
move for different reasons. `E` only rises, as players progress through the
game. `N` jumps whenever a new wave of players syncs — and a jump in `N`
dilutes every rate at once (launch night: `N` went 638 → 3,868 at the first
refresh and every rate but the one at 100% fell). Fitting the rate alone
would read a dilution drop as trophies getting rarer, so this forecaster
models `E` and `N` separately and divides at the target:

1. **Collapse to distinct snapshots.** Exophase refreshes a game's counts on
   its own schedule, hours apart, so consecutive identical reads are one
   refresh seen again, not new information. History is reduced to one entry
   per distinct `(players_tracked, rates)` state.
2. **Log-time power laws.** With `x = ln(1 + hours since release)`, `log N`
   is fitted against `x` by least squares (one fit, shared by all trophies)
   and, per trophy, `log E` against `x` the same way. The earner slope is
   clamped to ≥ 0 (earners never fall) and a small floor keeps a 0.00% rate
   fittable. Fits use the last 48 hours of snapshots, or all of them when
   there are fewer than three.
3. **Shrink the trend toward persistence.** The trend forecast
   `r_trend = 100 * E(x_target) / N(x_target)` is blended with the last
   observed rate in logit space. The weight grows with the number of
   distinct snapshots: a flat history (one snapshot, as on launch night)
   forecasts persistence, a history with many refreshes trusts its trend.
   One fit may move the logit by at most 6 units, so a blow-up fit cannot
   claim a trophy goes from ~1% to ~99%.
4. **Honest spread.** The logit spread combines four terms: a random-walk
   term (`0.35` per √day — narrower than the house baseline's `0.5`, because
   the trend already removes the systematic drift), the fit's residual, the
   slope's standard error times the extrapolation distance, and a
   refresh-jump term that is large when the history has seen few refreshes
   (a refresh the history has not seen yet can move every rate at once) and
   shrinks as refreshes accumulate.

Quantiles are `q_tau = inv_logit(logit(m) + z_tau * s)`, rounded to 4
decimals (half up), clamped to [0.01, 99.99]. The program never crashes on
odd input: empty history, null rates, garbage reads and missing fields all
fall back to a wide persistence forecast, and every trophy in the case
always gets a row.

## Local test results

The kit ships no practice answers, so practice scores below are against
answers reconstructed from the rules: the 04:30 refresh read has 3,868
players tracked with earners assumed unchanged from the last read
(638 players), which the rules describe ("every rate but the one at 100%
lower than before"). Synthetic final-like cases (2.5 days of history with
refreshes every 4 hours, target 4.5 days out) were generated in three
regimes — rising rates, falling rates, and a dilution regime — to exercise
the trend path the flat practice histories cannot.

| case | this entry (loss) | house baseline (loss) |
| --- | --- | --- |
| practice p00 (reconstructed) | 0.959 | 1.688 |
| practice p01 (reconstructed) | 0.958 | 1.688 |
| synthetic rising | 0.230 | 0.456 |
| synthetic falling | 0.176 | 0.273 |
| synthetic dilution | 0.227 | 0.260 |

Loss is the mean pinball loss on the logit scale (lower is better); the
house baseline is the kit's starter. All outputs validated strictly:
exactly one row per trophy, five non-decreasing quantiles in [0, 100],
under the 65,536-byte cap, byte-identical across repeated runs.

## Files

- `main.py` — the whole forecaster (the entrypoint).
- `README.md` — this file.
- `LICENSE` — MIT.
- `.makefun/owner.json` — payout wallet.
