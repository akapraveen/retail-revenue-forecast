"""Rolling 28-day revenue forecast for a UK online retailer (UCI Online Retail II, Dec 2009 - Dec 2011).

Run:  ../.venv/bin/python train.py      -> writes results.json
Needs data/online_retail_II.parquet (see README for the one-off conversion from the UCI .xlsx).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

SEED = 42
HERE = Path(__file__).parent
H = 28
ORIGINS = pd.date_range("2011-03-01", "2011-11-01", freq="MS")


def daily_revenue() -> pd.Series:
    df = pd.read_parquet(HERE / "data" / "online_retail_II.parquet")
    keep = (~df["Invoice"].str.startswith("C") & (df["Quantity"] > 0) & (df["Price"] > 0)
            & df["StockCode"].str.match(r"^\d{5}", na=False))      # drop postage, fees, manual adjustments
    df = df[keep]
    rev = (df["Quantity"] * df["Price"]).groupby(df["InvoiceDate"].dt.normalize()).sum()
    cal = pd.date_range(rev.index.min(), rev.index.max(), freq="D")
    return rev.reindex(cal, fill_value=0.0).rename("revenue")


def weekday_profile(hist: pd.Series, weeks: int) -> pd.Series:
    recent = hist.iloc[-7 * weeks:]
    return recent.groupby(recent.index.dayofweek).mean()


def forecast_recent(hist: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    prof = weekday_profile(hist, 4)
    return prof.reindex(dates.dayofweek).to_numpy()


def forecast_yoy(hist: pd.Series, dates: pd.DatetimeIndex) -> np.ndarray:
    """Same weekday 52 weeks earlier, scaled by trailing 8-week year-on-year growth."""
    last8 = hist.iloc[-56:].sum()
    prior8 = hist.reindex(hist.index[-56:] - pd.Timedelta(days=364)).sum()
    growth = last8 / prior8 if prior8 > 0 else 1.0
    return hist.reindex(dates - pd.Timedelta(days=364)).fillna(0).to_numpy() * growth


def features(series: pd.Series, origin: pd.Timestamp, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Features for target `dates` using only data strictly before `origin`."""
    hist = series[series.index < origin]
    level = hist.iloc[-28:].mean()
    ly = series.reindex(dates - pd.Timedelta(days=364))
    ly_level = series.reindex(pd.date_range(origin - pd.Timedelta(days=364 + 28), periods=28)).mean()
    prof = weekday_profile(hist, 8)
    return pd.DataFrame({
        "horizon": (dates - origin).days + 1,
        "dow": dates.dayofweek,
        "doy": dates.dayofyear,
        "month": dates.month,
        "days_to_xmas": (pd.to_datetime(dates.year.astype(str) + "-12-25") - dates).days,
        "dow_share": (prof.reindex(dates.dayofweek) / max(prof.mean(), 1)).to_numpy(),
        "ly_ratio": (ly / ly_level).to_numpy() if ly_level and ly_level > 0 else np.nan,
        "yoy_level": level / ly_level if ly_level and ly_level > 0 else np.nan,
        "level": level,
    }, index=dates)


def training_frame(series: pd.Series, origin: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray]:
    """Every past origin (daily, from 5 weeks in) paired with its next 28 days, ending before `origin`."""
    rows, ys = [], []
    start = series.index.min() + pd.Timedelta(days=35)
    for o in pd.date_range(start, origin - pd.Timedelta(days=H), freq="D"):
        dates = pd.date_range(o, periods=H)
        f = features(series, o, dates)
        rows.append(f)
        ys.append(series.reindex(dates).to_numpy() / max(f["level"].iloc[0], 1))
    X = pd.concat(rows)
    return X.drop(columns="level"), np.concatenate(ys)


def gbm(loss="squared_error", q=None):
    kw = {"quantile": q} if q is not None else {}
    return HistGradientBoostingRegressor(loss=loss, learning_rate=0.05, max_iter=300, max_leaf_nodes=15,
                                         min_samples_leaf=40, random_state=SEED, **kw)


def main():
    s = daily_revenue()
    methods = ["Recent 4-week weekday average", "Last year + growth", "Gradient boosting",
               "Blend: boosting + last year"]
    per_origin, detail = [], {}
    all_y, all_pred, all_lo, all_hi = [], {m: [] for m in methods}, [], []
    for origin in ORIGINS:
        dates = pd.date_range(origin, periods=H)
        hist = s[s.index < origin]
        y = s.reindex(dates).to_numpy()
        X_tr, y_tr = training_frame(s, origin)
        f = features(s, origin, dates)
        level = f["level"].iloc[0]
        Xf = f.drop(columns="level")
        preds = {
            methods[0]: forecast_recent(hist, dates),
            methods[1]: forecast_yoy(hist, dates),
            methods[2]: np.clip(gbm().fit(X_tr, y_tr).predict(Xf), 0, None) * level,
        }
        preds[methods[3]] = (preds[methods[1]] + preds[methods[2]]) / 2
        lo = np.clip(gbm("quantile", 0.1).fit(X_tr, y_tr).predict(Xf), 0, None) * level
        hi = np.clip(gbm("quantile", 0.9).fit(X_tr, y_tr).predict(Xf), 0, None) * level
        row = {"origin": origin.strftime("%Y-%m-%d"), "actual_total": round(float(y.sum()))}
        for m, pr in preds.items():
            row[m] = {"wape": round(float(np.abs(y - pr).sum() / y.sum()), 4),
                      "total_error": round(float((pr.sum() - y.sum()) / y.sum()), 4),
                      "forecast_total": round(float(pr.sum()))}
            all_pred[m].append(pr)
        per_origin.append(row)
        all_y.append(y); all_lo.append(lo); all_hi.append(hi)
        detail[row["origin"]] = {
            "dates": [d.strftime("%Y-%m-%d") for d in dates],
            "actual": [round(float(v)) for v in y],
            **{m: [round(float(v)) for v in pr] for m, pr in preds.items()},
            "p10": [round(float(v)) for v in lo], "p90": [round(float(v)) for v in hi],
        }
        print(row)

    Y = np.concatenate(all_y)
    summary = []
    for m in methods:
        P = np.concatenate(all_pred[m])
        errs = [r[m]["total_error"] for r in per_origin]
        summary.append({"method": m,
                        "wape": round(float(np.abs(Y - P).sum() / Y.sum()), 4),
                        "mean_abs_total_error": round(float(np.mean(np.abs(errs))), 4),
                        "bias": round(float(np.mean(errs)), 4),
                        "worst_total_error": round(float(max(errs, key=abs)), 4)})
    lo, hi = np.concatenate(all_lo), np.concatenate(all_hi)
    open_days = Y > 0
    coverage = float(((Y >= lo) & (Y <= hi))[open_days].mean())
    print(summary, coverage)

    weekly = s.resample("W-SUN").sum()
    results = {
        "dataset": {"name": "Online Retail II (UCI, UK-based online giftware wholesaler)",
                    "transactions": 1067371, "days": int(len(s)),
                    "start": s.index.min().strftime("%Y-%m-%d"), "end": s.index.max().strftime("%Y-%m-%d"),
                    "total_revenue_gbp": round(float(s.sum()))},
        "horizon_days": H,
        "origins": [o.strftime("%Y-%m-%d") for o in ORIGINS],
        "summary": summary,
        "per_origin": per_origin,
        "interval_coverage_p10_p90": round(coverage, 4),
        "detail": detail,
        "weekly_history": [[d.strftime("%Y-%m-%d"), round(float(v))] for d, v in weekly.items()],
        "monthly": [[d.strftime("%Y-%m"), round(float(v))] for d, v in s.resample("MS").sum().items()],
    }
    (HERE / "results.json").write_text(json.dumps(results, indent=2))
    print("wrote results.json")


if __name__ == "__main__":
    main()
