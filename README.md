# RIPTIDE

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kim007-lol/riptide/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/kim007-lol/riptide/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Forecasting every GTA VI trophy's Exophase earn rate one week after launch,
entered in the [make.fun bounty](https://make.fun/bounty/gta6-trophy-rarity-forecast-01).

## The problem

Persistence loses. A trophy's earn rate does not stay put: as launch week
goes on, more players earn each trophy, so rates rise, fast for early story
trophies and slowly for rare ones. At the same time Exophase keeps tracking
more players, which dilutes every rate whenever it refreshes. A forecaster
has to model both forces, not just repeat the last read.

## The model

`main.py` (Python 3.13, standard library only, no network, deterministic)
reads each case file and, per trophy:

1. **Collapses the history to distinct page states.** Exophase's page is a
   step function, so long flat stretches carry no new information; only the
   first read showing each state is kept.
2. **Fits growth on log-time.** With `x = ln(1 + hours since release)` it
   fits, per trophy, a line to the logit earn rate against `x` and a line
   to log earners against `x`, alongside one global fit to log players
   tracked. The slope is the trophy's own fitted slope: the earner slope
   minus the player slope is the rate's slope, so dilution is modelled,
   not ignored.
3. **Anchors on the freshest read.** The median at the target is the last
   read's rate moved by the fitted slope over the log-time gap, so a level
   shift from a refresh never drags the forecast with it.
4. **Shrinks noisy slopes.** Each slope is shrunk toward its tier's median
   slope and the global median (rare trophies borrow strength from their
   tier), damped as the rate approaches its ceiling, and clamped to
   [-0.5, 3.0] logit units per log-time unit. The median is capped by a
   generous per-tier ceiling (platinum 5%, gold 45%, silver 65%,
   bronze 95%).
5. **Calibrates the spread honestly.** Three logit-scale terms: a
   random-walk term growing with the horizon (0.45 per root-day), a
   refresh-jump term that is large for short horizons and fades over days
   (launch night showed one refresh moving every rate at once), and a fit
   term from the trophy's own residual volatility, widened by how far the
   target sits past the data.

Quantiles are the median's logit plus the standard normal quantiles times
that spread, mapped back to percent.

## Why this beats persistence

Persistence scores about 1.51 pinball loss on the practice cases because
the refresh after the checkpoint moves every rate outside its narrow band.
RIPTIDE keeps a persistence-like median when history is flat (there is
nothing to extrapolate from) but carries a spread wide enough to cover a
refresh jump. On the final, where two days of real rising history precede
a 4.5-day horizon, the fitted slopes do the work: story trophies are
forecast to keep climbing on their log-time curves while dilution from
player growth is netted out, instead of being frozen at their
two-day-old rates.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` in the input directory, writes one `<id>.out` per case
to the output directory: `{"trophies": [{"id": n, "q": [q05, q25, q50, q75,
q95]}, ...]}`, five quantiles in percent, non-decreasing, under 65,536
bytes per file.

## License

MIT. See [LICENSE](LICENSE).
