"""gta6-trophy-rarity-forecast-01: forecast each GTA VI trophy's Exophase earn rate.

Model (Python 3.13 stdlib only, deterministic, no network)
-----------------------------------------------------------
An earn rate is a ratio, r = 100 * E / N, where E is how many tracked players
have earned the trophy and N is how many players Exophase tracks.  The two
move for different reasons: E only rises, as players progress through the
game, while N jumps whenever a new wave of players syncs -- and a jump in N
dilutes every rate at once (launch night: N went 638 -> 3,868 at the first
refresh and every rate but the one at 100% fell).  Fitting the rate alone
would read a dilution drop as trophies getting rarer, so this forecaster
models E and N separately and divides at the target:

* History is collapsed to distinct snapshots: Exophase refreshes a game's
  counts on its own schedule, hours apart, so consecutive identical reads
  are one refresh seen again, not new information.
* x = ln(1 + hours since release).  log N is fitted against x by least
  squares (one fit, shared by all trophies); per trophy, log E is fitted
  against x the same way, with the slope clamped to >= 0 (earners never
  fall) and a small floor on E so a 0.00% rate still fits.
* The trend forecast r_trend = 100 * E(x_target) / N(x_target) is shrunk
  toward the last observed rate in logit space; the shrinkage weight grows
  with the number of distinct snapshots, so a flat history (one snapshot,
  as on launch night) forecasts persistence, and a history with many
  refreshes trusts its trend.  A single fit may move the logit by at most
  6 units, so a blow-up fit cannot claim a trophy goes from ~1% to ~99%.
* The spread (logit SD) combines four honest sources of uncertainty: the
  random-walk term of the house baseline, the fit's residual, the slope's
  standard error times the extrapolation distance, and a refresh-jump term
  that is large when the history has seen few refreshes (a refresh the
  history has not seen yet can move every rate at once) and shrinks as
  refreshes accumulate.

Quantiles are q_tau = inv_logit(logit(m) + z_tau * s), rounded to 4 decimals
(half up), clamped to [0.01, 99.99].  The program never crashes on odd
input: every per-case failure falls back to a wide persistence forecast,
and every trophy in the case always gets a row.

Usage (as the grader runs it): python3 main.py, reading every *.in in
$MAKEFUN_INPUT_DIR (else /input) and writing <id>.out to
$MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# The spread of a rate's move, in logit units per square-root day. The house
# baseline uses 0.5, but its random walk also has to cover the systematic
# drift that this model's E/N trend already removes; 0.35 tested better than
# 0.5 on synthetic final-like cases (rising, falling and dilution regimes)
# while the residual, slope and refresh-jump terms below keep the total
# spread honest for a 4.5-day extrapolation.
SIGMA_RW = 0.35
# Refresh-jump allowance (logit units) when the history holds one snapshot.
JUMP_ONE = 1.5
# Fit window: distinct snapshots from the last 48 h before the checkpoint.
FIT_WINDOW_HOURS = 48.0
# Release of the game; the clock the log-time trend is measured from.
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)


def clamp_pct(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg: float) -> float:
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x: float):
    """Round to 4 decimals, half up, as JavaScript's Math.round does it."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285984469534e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def parse_time(s):
    """Parse an ISO 8601 timestamp to an aware datetime, or None."""
    if not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b, resid_sd, se_slope)."""
    n = len(xs)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if n < 2 or sxx <= 0:
        return my, 0.0, 0.0, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    a = my - b * mx
    if n > 2:
        rss = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
        dof = n - 2
        var = rss / dof
        resid_sd = math.sqrt(var)
        se_slope = math.sqrt(var / sxx)
    else:
        resid_sd = 0.0
        se_slope = 0.0
    return a, b, resid_sd, se_slope


def release_for(case):
    """The release instant the log-time trend is measured from."""
    game = case.get("game") or {}
    rel = parse_time(game.get("release_date"))
    return rel if rel is not None else RELEASE


def distinct_snapshots(case, release):
    """Collapse history to distinct (players, rates) snapshots, first-seen time.

    Returns a list of (hours_since_release, players_tracked, rates) with at
    most one entry per Exophase refresh state, oldest first.
    """
    seen = {}
    order = []
    for read in case.get("history") or []:
        at = parse_time(read.get("at")) or parse_time(read.get("slot"))
        if at is None:
            continue
        players = read.get("players_tracked")
        rates = read.get("rates")
        if not isinstance(players, (int, float)) or players <= 0:
            continue
        if not isinstance(rates, list):
            continue
        key = (players, tuple(rates))
        if key in seen:
            continue
        seen[key] = True
        hours = (at - release).total_seconds() / 3600.0
        order.append((hours, float(players), rates))
    return order


def last_rate(case, idx):
    """The trophy's rate in the last read that has one, else None."""
    for read in reversed(case.get("history") or []):
        rates = read.get("rates")
        if isinstance(rates, list) and idx < len(rates):
            r = rates[idx]
            if isinstance(r, (int, float)):
                return float(r)
    return None


def forecast_case(case, name):
    trophies = case.get("trophies") or []
    if not trophies:
        raise ValueError(f"{name}: no trophies")
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(float(t)) for t in levels]
    horizon_days = float(case.get("horizon_minutes") or 0) / 1440.0
    target_dt = parse_time(case.get("target"))
    checkpoint_dt = parse_time(case.get("at"))

    release = release_for(case)
    snaps = distinct_snapshots(case, release)
    n_snaps = len(snaps)
    if checkpoint_dt is not None:
        window_start = checkpoint_dt.timestamp() - FIT_WINDOW_HOURS * 3600.0
        # Recompute hours relative to the checkpoint for the window cut.
        fitted = [s for s in snaps
                  if release.timestamp() + s[0] * 3600.0 >= window_start]
        if len(fitted) >= 3:
            snaps = fitted
    if target_dt is not None:
        x_target = math.log1p(max(0.0, (target_dt - release).total_seconds() / 3600.0))
    else:
        x_target = math.log1p(max(0.0, (snaps[-1][0] if snaps else 0.0)
                                   + float(case.get("horizon_minutes") or 0) / 60.0))
    x_last = math.log1p(max(0.0, snaps[-1][0])) if snaps else 0.0

    # Shared players-tracked trend: log N ~ a + b * ln(1 + hours).
    if len(snaps) >= 2:
        xs = [math.log1p(max(0.0, h)) for h, _, _ in snaps]
        ys = [math.log(n) for _, n, _ in snaps]
        a_n, b_n, _, _ = ols(xs, ys)
        b_n = min(3.0, max(0.0, b_n))
        log_n_target = a_n + b_n * x_target
    elif snaps:
        log_n_target = math.log(snaps[-1][1])
        b_n = 0.0
    else:
        log_n_target = 0.0
        b_n = 0.0

    # Shrinkage weight on the trend: grows with the number of distinct
    # refreshes the history actually saw; a flat history trusts persistence.
    w_trend = n_snaps / (n_snaps + 2.0) if n_snaps else 0.0
    if w_trend > 0.85:
        w_trend = 0.85

    s_rw = SIGMA_RW * math.sqrt(max(0.0, horizon_days))
    s_jump = JUMP_ONE / math.sqrt(max(1, n_snaps))

    rows = []
    for i, trophy in enumerate(trophies):
        r_last = last_rate(case, i)
        if r_last is None:
            r_last = 0.5
        r_last = clamp_pct(r_last)

        # Per-trophy earners trend: log E ~ a + b * ln(1 + hours).
        pts = []
        for hours, n, rates in snaps:
            if i < len(rates) and isinstance(rates[i], (int, float)):
                e = max(0.5, float(rates[i]) * n / 100.0)
                pts.append((math.log1p(max(0.0, hours)), math.log(e)))
        if len(pts) >= 2:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            a_e, b_e, resid_sd, se_slope = ols(xs, ys)
            b_e = min(3.5, max(0.0, b_e))
            log_e_target = a_e + b_e * x_target
            mean_x = sum(xs) / len(xs)
            s_fit = min(1.5, max(0.10, resid_sd))
            s_slope = se_slope * max(0.0, x_target - mean_x)
        elif pts:
            log_e_target = pts[-1][1]
            s_fit = 0.35
            s_slope = 0.30 * max(0.0, x_target - x_last)
        else:
            # No usable history for this trophy: the trend is its last rate.
            log_e_target = math.log(max(0.5, r_last * math.exp(log_n_target) / 100.0))
            s_fit = 0.35
            s_slope = 0.30 * max(0.0, x_target - x_last)

        r_trend = clamp_pct(100.0 * math.exp(log_e_target - log_n_target))
        # Shrink the trend move toward persistence, in logit space, and cap
        # the move a single fit may claim: an early trophy can plausibly go
        # from ~1% to ~50% in a week (about 4.5 logit units), but a fit
        # claiming more than 6 is a blow-up, not a forecast.
        delta = logit(r_trend) - logit(r_last)
        delta = min(6.0, max(-6.0, delta))
        med_logit = logit(r_last) + w_trend * delta

        s = math.sqrt(s_rw ** 2 + s_fit ** 2 + s_slope ** 2 + s_jump ** 2)
        s = min(3.0, max(0.35, s))
        qs = [round4(clamp_pct(inv_logit(med_logit + z * s))) for z in zs]
        # Keep the row non-decreasing after rounding.
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def fallback_case(case):
    """Wide persistence forecast, used only if the model fails on a case."""
    levels = case.get("levels") or LEVELS
    zs = [norm_ppf(float(t)) for t in levels]
    s = 2.0
    rows = []
    for i, trophy in enumerate(case.get("trophies") or []):
        r = last_rate(case, i)
        if r is None:
            r = 1.0
        lg = logit(r)
        qs = [round4(clamp_pct(inv_logit(lg + z * s))) for z in zs]
        for k in range(1, len(qs)):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                forecast = fallback_case(case)
            except Exception:
                continue
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
