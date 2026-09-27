# Rolling revenue forecast: a 4-week reforecast, backtested

Forecasts daily revenue 28 days ahead for a UK online giftware wholesaler, the way an FP&A team
reforecasts at each month start, and checks every method against nine real month-starts in 2011.

**Data:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI),
1,067,371 invoice lines, Dec 2009 to Dec 2011. Cancellations, returns, zero prices and non-product
lines (postage, fees, manual adjustments) removed before aggregating to daily revenue.

![Weekly revenue](figures/weekly_revenue.png)

## Backtest (origins: 1st of each month, Mar to Nov 2011)

| Method | Daily WAPE | Avg. miss on 4-week total | Bias | Worst month |
|---|---|---|---|---|
| Recent 4-week weekday average | 32.9% | 14.6% | −7.6% | −31.0% |
| Last year + growth | 34.1% | 10.2% | +0.5% | −22.1% |
| Gradient boosting | 32.5% | 11.4% | −1.0% | −37.7% |
| **Blend: boosting + last year** | **31.6%** | **9.4%** | **−0.3%** | −29.9% |

![November reforecast](figures/november_forecast.png)

- Daily error is high because a handful of large wholesale orders swing single days; the 4-week total
  is the number a finance team plans against.
- The boosting model predicts each day as a multiple of the trailing 28-day level, using weekday,
  season, days to Christmas and last year's shape, trained on every past origin before the forecast date.
- 10th to 90th percentile intervals from quantile boosting cover 73.6% of open days against an 80% target,
  so they run slightly narrow.

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./download_data.sh
python train.py      # writes results.json
python figures.py    # writes figures/*.png
```
