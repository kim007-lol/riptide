"""RIPTIDE: forecast each GTA VI trophy's Exophase earn rate at a case target.

Model idea
----------
A trophy's earn rate r(t) is a share, so it lives on the logit scale, and it
moves for two separate reasons: more players earn the trophy (earners E grow)
and more players get tracked at all (players N grow, which dilutes r).  So
this program fits, per trophy, how fast log(E) and log(N) grow against
log-time x = ln(1 + hours since release), and forecasts the rate as the
ratio of the two extrapolations, anchored on the freshest read:

    logit r(target) ~= logit r(last) + slope * (x_target - x_last)

The slope is the trophy's own fitted slope, shrunk toward its tier's median
slope and the global median slope (rare trophies have noisy histories, so
they borrow strength from their tier), then damped as the rate approaches
its ceiling, because a trophy that half the players already hold cannot
keep rising as fast as one almost nobody holds.  Slopes are clamped so a
short, jumpy history can never extrapolate a rate to absurdity, and the
median is capped by a generous per-tier ceiling.

The spread is honest about what a launch week does to these numbers: a
random-walk term that grows with the horizon, a refresh-jump term that is
large for short horizons (Exophase refreshes in jumps, and a jump can move
every rate at once, as launch night showed) and fades over days, and a fit
term from the trophy's own residual volatility widened by how far the
target sits past the data.

Everything is deterministic: no randomness, no network, Python 3.13
standard library only.  Reads every <id>.in in the input directory and
writes one <id>.out per case.

Input:  $MAKEFUN_INPUT_DIR (else /input) / <id>.in, JSON
        {"schema": "makefun/gta6-trophies-case/v1", "id", "at",
         "horizon_minutes", "target", "levels", "game", "trophies",
         "history": [{"slot", "at", "players_tracked", "trophies_earned",
                      "rates"}], "notes"}
Output: $MAKEFUN_OUTPUT_DIR (else /output) / <id>.out, JSON
        {"trophies": [{"id": n, "q": [q05, q25, q50, q75, q95]}, ...]}
        one row per trophy, five quantiles in percent, non-decreasing.
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

# Standard normal quantiles for the five scored levels.  For any other
# level the inverse CDF below (Acklam's approximation) is used instead.
Z_TABLE = {0.05: -1.6448536269514722, 0.25: -0.6744897501960817,
           0.5: 0.0, 0.75: 0.6744897501960817, 0.95: 1.6448536269514722}

# Hours of history a fit may use (recent reads matter most: Exophase's
# refreshes and the player mix change through launch week).
WINDOW_HOURS = 36.0
# Recency half-life, hours, for the least-squares weights.
HALF_LIFE_HOURS = 24.0
# Shrinkage pseudo-counts: a trophy's own slope counts n_eff, its tier's
# median slope counts 3, the global median slope counts 1.
TIER_PRIOR_WEIGHT = 3.0
GLOBAL_PRIOR_WEIGHT = 1.0
# Slope clamp, in logit units per unit of log-time.
SLOPE_MIN, SLOPE_MAX = -0.5, 3.0
# Generous per-tier ceilings (percent) for the median at the target.
TIER_CEILING = {"platinum": 5.0, "gold": 45.0, "silver": 65.0,
                "bronze": 95.0}
# Spread pieces, in logit units.
TREND_SIGMA_PER_SQRT_DAY = 0.45   # random-walk growth with the horizon
JUMP_SIGMA = 0.55                 # one Exophase refresh can move all rates
JUMP_DECAY_DAYS = 1.5             # ...but over days, jumps average out
SIGMA_MIN, SIGMA_MAX = 0.15, 1.6


def parse_time(text):
    """Parse an ISO 8601 timestamp (or bare date) into aware UTC."""
    if text is None:
        return None
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def clamp_rate(pct):
    return min(99.99, max(0.01, float(pct)))


def logit_of(pct):
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(logit_value):
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round_half_up_4(x):
    """Round to 4 decimals, halves away from zero (matches the usual
    half-up rounding of a rate printed to 4 places)."""
    scaled = x * 10000.0
    whole = math.floor(scaled + 0.5) if scaled >= 0 else math.ceil(scaled - 0.5)
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation),
    accurate to ~1e-9, for levels the table above does not cover."""
    if p in Z_TABLE:
        return Z_TABLE[p]
    if p <= 0.0 or p >= 1.0:
        raise ValueError("level out of (0, 1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
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


def weighted_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x.  Returns (a, b, resid_std, n, xbar, sxx)."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    xbar = sum(w * x for x, w in zip(xs, ws)) / sw
    ybar = sum(w * y for y, w in zip(ys, ws)) / sw
    sxx = sum(w * (x - xbar) ** 2 for x, w in zip(xs, ws))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - xbar) * (y - ybar) for x, y, w in zip(xs, ys, ws))
    b = sxy / sxx
    a = ybar - b * xbar
    n = len(xs)
    if n > 2:
        resid = math.sqrt(sum(w * (y - (a + b * x)) ** 2 for x, y, w in zip(xs, ys, ws)) / (sw * (n - 2) / n))
    else:
        resid = 0.0
    return a, b, resid, n, xbar, sxx


def median_of(values):
    vals = sorted(values)
    m = len(vals)
    if m == 0:
        return 0.0
    if m % 2:
        return vals[m // 2]
    return 0.5 * (vals[m // 2 - 1] + vals[m // 2])


def distinct_points(case, release):
    """Collapse the history to one point per distinct page state, keeping
    the first read that showed each state (Exophase's page is a step
    function: long flat stretches carry no new information)."""
    seen = []
    prev_key = None
    for read in case.get("history") or []:
        key = (tuple(read.get("rates") or []), read.get("players_tracked"))
        if key == prev_key:
            continue
        prev_key = key
        at = parse_time(read.get("at"))
        if at is None:
            continue
        hours = (at - release).total_seconds() / 3600.0
        seen.append({"at": at, "hours": hours,
                     "x": math.log1p(max(hours, 0.0)),
                     "players": read.get("players_tracked"),
                     "rates": read.get("rates") or []})
    return seen


def forecast_case(case, name):
    levels = [float(v) for v in case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]]
    zvals = [norm_ppf(v) for v in levels]
    trophies = case["trophies"]
    history = case.get("history") or []
    if not history:
        # No data at all: stay flat at a neutral 1% with a wide spread
        # rather than failing the case.
        rows = [{"id": t["id"],
                 "q": [round_half_up_4(inv_logit(logit_of(1.0) + z * 1.2)) for z in zvals]}
                for t in trophies]
        return {"trophies": rows}

    release = parse_time((case.get("game") or {}).get("release_date"))
    if release is None:
        release = parse_time(history[0].get("at"))
    target = parse_time(case.get("target"))
    if target is None:
        horizon_days = (case.get("horizon_minutes") or 0) / 1440.0
        target = parse_time(case.get("at"))
        x_target = math.log1p(max((target - release).total_seconds() / 3600.0, 0.0)) \
            if target is not None else 0.0
    else:
        horizon_days = max((target - parse_time(case.get("at"))).total_seconds(), 0.0) / 86400.0 \
            if parse_time(case.get("at")) is not None else (case.get("horizon_minutes") or 0) / 1440.0
        x_target = math.log1p(max((target - release).total_seconds() / 3600.0, 0.0))
    horizon_days = max(horizon_days, (case.get("horizon_minutes") or 0) / 1440.0)

    points = distinct_points(case, release)
    last = points[-1] if points else None
    last_at = last["at"] if last else parse_time(case.get("at"))
    last_x = last["x"] if last else 0.0

    # Window the fit to recent distinct states; fall back to all of them
    # when the recent window holds too few.
    window = [p for p in points
              if (last_at - p["at"]).total_seconds() <= WINDOW_HOURS * 3600.0]
    if len(window) < 4:
        window = points
    if not window:
        window = points

    # One global fit for the tracked-player count: dilution's other half.
    n_xs, n_ys, n_ws = [], [], []
    for p in window:
        if p["players"] and p["players"] > 0:
            age_h = max((last_at - p["at"]).total_seconds() / 3600.0, 0.0)
            n_xs.append(p["x"])
            n_ys.append(math.log(float(p["players"])))
            n_ws.append(0.5 ** (age_h / HALF_LIFE_HOURS))
    n_fit = weighted_line(n_xs, n_ys, n_ws) if len(n_xs) >= 2 else None
    slope_n = n_fit[1] if n_fit else 0.0
    slope_n = min(max(slope_n, 0.0), 3.0)

    # First pass: each trophy's own slope, from its logit curve and from
    # its earner-count curve (the two agree for rare trophies and average
    # out each other's noise).
    raw = []
    for i, trophy in enumerate(trophies):
        xs, ys, ws, exs, eys, ews = [], [], [], [], [], []
        for p in window:
            rate = p["rates"][i] if i < len(p["rates"]) else None
            if rate is None:
                continue
            age_h = max((last_at - p["at"]).total_seconds() / 3600.0, 0.0)
            w = 0.5 ** (age_h / HALF_LIFE_HOURS)
            xs.append(p["x"])
            ys.append(logit_of(rate))
            ws.append(w)
            if p["players"] and p["players"] > 0:
                earners = max(float(rate) / 100.0 * float(p["players"]), 0.5)
                exs.append(p["x"])
                eys.append(math.log(earners))
                ews.append(w)
        fit = weighted_line(xs, ys, ws) if len(xs) >= 2 else None
        efit = weighted_line(exs, eys, ews) if len(exs) >= 2 else None
        slopes = []
        if fit is not None:
            slopes.append(fit[1])
        if efit is not None:
            slopes.append(efit[1] - slope_n)
        slope = sum(slopes) / len(slopes) if slopes else 0.0
        resid = fit[2] if fit is not None else 0.15
        sxx = fit[5] if fit is not None else 0.0
        xbar = fit[4] if fit is not None else last_x
        npts = len(xs)
        last_rate = None
        for p in reversed(points):
            if i < len(p["rates"]) and p["rates"][i] is not None:
                last_rate = float(p["rates"][i])
                break
        if last_rate is None:
            last_rate = 1.0
        raw.append({"trophy": trophy, "slope": slope, "resid": resid,
                    "sxx": sxx, "xbar": xbar, "npts": npts,
                    "last_rate": last_rate})

    # Tier and global prior slopes, from trophies with a real history.
    tier_slopes = {}
    for entry in raw:
        if entry["npts"] >= 3:
            tier_slopes.setdefault(entry["trophy"].get("tier", "bronze"), []).append(entry["slope"])
    tier_med = {t: median_of(v) for t, v in tier_slopes.items()}
    global_med = median_of([e["slope"] for e in raw if e["npts"] >= 3]) if any(
        e["npts"] >= 3 for e in raw) else 0.0

    rows = []
    for entry in raw:
        trophy = entry["trophy"]
        tier = trophy.get("tier", "bronze")
        prior = tier_med.get(tier, global_med)
        n_eff = min(entry["npts"], 8)
        shrunk = (n_eff * entry["slope"] + TIER_PRIOR_WEIGHT * prior
                  + GLOBAL_PRIOR_WEIGHT * global_med) / (n_eff + TIER_PRIOR_WEIGHT + GLOBAL_PRIOR_WEIGHT)
        # A trophy most players already hold cannot keep its early pace.
        damp = max(0.3, 1.0 - entry["last_rate"] / 100.0)
        slope = min(max(shrunk * damp, SLOPE_MIN), SLOPE_MAX)

        if entry["last_rate"] >= 99.99:
            median_rate = 99.99
            sigma = 0.15
        else:
            med_logit = logit_of(entry["last_rate"]) + slope * (x_target - last_x)
            ceiling = TIER_CEILING.get(tier, 95.0)
            median_rate = min(inv_logit(med_logit), ceiling)
            median_rate = clamp_rate(median_rate)
            # Spread: trend + refresh-jump + this trophy's fit noise,
            # widened by how far past the data the target sits.
            extra = 0.0
            if entry["sxx"] > 1e-12 and entry["npts"] > 2:
                extra = (x_target - entry["xbar"]) ** 2 / entry["sxx"]
            fit_sigma = min(max(entry["resid"] * math.sqrt(1.0 + extra + 1.0 / max(entry["npts"], 1)), 0.05), 0.6)
            sigma = math.sqrt(
                (TREND_SIGMA_PER_SQRT_DAY * math.sqrt(max(horizon_days, 0.0))) ** 2
                + (JUMP_SIGMA * math.exp(-max(horizon_days, 0.0) / JUMP_DECAY_DAYS)) ** 2
                + fit_sigma ** 2)
            sigma = min(max(sigma, SIGMA_MIN), SIGMA_MAX)
        med_logit = logit_of(median_rate)
        qs = []
        prev = -1.0
        for z in zvals:
            q = round_half_up_4(inv_logit(med_logit + z * sigma))
            q = min(max(q, 0.0), 100.0)
            q = max(q, prev)  # keep the quantiles non-decreasing
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        out = OUTPUT_DIR / (path.stem + ".out")
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
