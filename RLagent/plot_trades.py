# RLagent/plot_trades.py
# ---------------------------------------------------------
# Plot price series and overlay BUY/SELL markers from trades CSV(s)
# Works with columns: timestamp,index,side,price,shares,value_usd,realized_pnl_this,realized_pnl_cum
# ---------------------------------------------------------
import argparse, glob, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def parse_ts(df: pd.DataFrame, time_col: str) -> pd.Series:
    if time_col in df.columns:
        ts = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    else:
        ts = pd.to_datetime(df[df.columns[0]], errors="coerce", utc=False)
    return ts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prices_csv", default="data/ABB_clean.csv",
                    help="CSV with columns including 'close' and a timestamp column")
    ap.add_argument("--time_col", default="Unnamed: 0",
                    help="Timestamp column name in prices CSV")
    ap.add_argument("--trades_csv", default="outputs/trades_ep001.csv",
                    help="Single trades CSV (ignored if --merge_glob is used)")
    ap.add_argument("--merge_glob", default="",
                    help='Glob to merge many CSVs, e.g. "outputs/trades_ep*.csv"')
    ap.add_argument("--x_mode", default="index", choices=["index", "time"],
                    help="X-axis: 'index' (robust) or 'time'")
    ap.add_argument("--zoom", default="none", choices=["none", "trades"],
                    help="Auto-zoom to min/max trade range")
    ap.add_argument("--title", default="Price with Trades")
    args = ap.parse_args()

    # --- Load prices ---
    dfp = pd.read_csv(args.prices_csv)
    if "close" not in dfp.columns:
        raise ValueError(f"'close' column not found in {args.prices_csv}. Columns: {list(dfp.columns)}")
    close = dfp["close"].astype(float).to_numpy()
    N = len(close)
    ts = parse_ts(dfp, args.time_col)
    if ts.isna().all():
        ts = pd.date_range(start="2000-01-01 09:30:00", periods=N, freq="5min")

    # --- Load trades (single or merge) ---
    if args.merge_glob:
        paths = sorted(glob.glob(args.merge_glob))
        if not paths:
            raise FileNotFoundError(f"No trades matched glob: {args.merge_glob}")
        dfs = [pd.read_csv(p) for p in paths]
        tr = pd.concat(dfs, ignore_index=True)
        used = f"{len(paths)} files"
    else:
        tr = pd.read_csv(args.trades_csv)
        used = os.path.basename(args.trades_csv)

    need = {"timestamp", "index", "side", "price"}
    if not need.issubset(set(tr.columns)):
        raise ValueError(f"Trades CSV must contain columns {need}. Got {list(tr.columns)}")

    # Ensure dtypes
    tr["index"] = tr["index"].astype(int)
    tr["price"] = tr["price"].astype(float)
    tr["timestamp"] = pd.to_datetime(tr["timestamp"], errors="coerce", utc=False)

    # Filter to price range by index
    tr = tr[(tr["index"] >= 0) & (tr["index"] < N)]

    buys = tr[tr["side"] == "BUY"].copy()
    sells = tr[tr["side"] == "SELL"].copy()

    # --- Build X for price and markers ---
    if args.x_mode == "index":
        x_price = np.arange(N)
        x_buys = buys["index"].to_numpy()
        x_sells = sells["index"].to_numpy()
    else:  # time mode
        x_price = ts
        x_buys = buys["timestamp"]
        x_sells = sells["timestamp"]

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(x_price, close, label="Close", linewidth=1.0)

    drawn_buys = 0
    drawn_sells = 0
    if len(buys):
        plt.scatter(x_buys, buys["price"], marker="^", s=40, label="BUY")
        drawn_buys = len(buys)
    if len(sells):
        plt.scatter(x_sells, sells["price"], marker="v", s=40, label="SELL")
        drawn_sells = len(sells)

    # --- Zoom to trade window if requested ---
    if args.zoom == "trades" and (drawn_buys + drawn_sells) > 0:
        if args.x_mode == "index":
            xmin = int(min(np.min(x_buys) if drawn_buys else N, np.min(x_sells) if drawn_sells else N))
            xmax = int(max(np.max(x_buys) if drawn_buys else 0, np.max(x_sells) if drawn_sells else 0))
        else:
            xmin = min(buys["timestamp"].min() if drawn_buys else x_price.min(),
                       sells["timestamp"].min() if drawn_sells else x_price.min())
            xmax = max(buys["timestamp"].max() if drawn_buys else x_price.max(),
                       sells["timestamp"].max() if drawn_sells else x_price.max())
        # pad 2% on each side
        if args.x_mode == "index":
            pad = max(1, int(0.02 * (xmax - xmin + 1)))
            plt.xlim(max(0, xmin - pad), min(N - 1, xmax + pad))
        else:
            pad = pd.Timedelta(seconds=int(0.02 * (xmax - xmin).total_seconds())) if xmax > xmin else pd.Timedelta(minutes=5)
            plt.xlim(xmin - pad, xmax + pad)

    plt.title(f"{args.title}  | markers: BUY={drawn_buys}, SELL={drawn_sells}  | {used}")
    plt.xlabel("Index" if args.x_mode == "index" else "Time")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()

    os.makedirs("outputs", exist_ok=True)
    out_base = "merged" if args.merge_glob else os.path.splitext(os.path.basename(used))[0]
    out_png = os.path.join("outputs", f"plot_{out_base}_{args.x_mode}_{args.zoom}.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved plot -> {out_png}")
    print(f"Markers drawn: BUY={drawn_buys}, SELL={drawn_sells} | x_mode={args.x_mode} | zoom={args.zoom}")

if __name__ == "__main__":
    main()
