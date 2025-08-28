import argparse, os, math
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

def shannon_entropy(x, bins=11):
    # x: 1D np.array of returns over a lookback window
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist[hist > 0].astype(np.float64)
    if p.size == 0:
        return np.nan
    return float(-(p * np.log(p)).sum())

def rolling_entropy(returns: np.ndarray, L: int, bins: int) -> np.ndarray:
    E = np.full(len(returns), np.nan, dtype=np.float32)
    if L <= 0 or len(returns) == 0:
        return E
    for t in range(L - 1, len(returns)):
        E[t] = shannon_entropy(returns[t - L + 1 : t + 1], bins=bins)
    return E

def make_windows(X, L, stride):
    n = len(X)
    idx = []
    for start in range(0, n - L + 1, stride):
        idx.append((start, start + L))
    return idx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--time_col", required=True)
    ap.add_argument("--feature_cols", required=True, nargs="+")
    ap.add_argument("--lin", type=int, default=256)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--train_ratio", type=float, default=0.7)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--target_h", type=int, default=12)  # for eval
    ap.add_argument("--entropy_filter", action="store_true",
                    help="If set, keep only lowest-q entropy TRAIN windows")
    ap.add_argument("--entropy_q", type=float, default=0.7,
                    help="Quantile q in (0,1]; keep lowest-q TRAIN windows (default 0.7)")
    ap.add_argument("--entropy_bins", type=int, default=11,
                    help="Histogram bins for Shannon entropy (default 11)")
    ap.add_argument("--add_entropy_feature", action="store_true",
                    help="If set, append ent_shannon as an extra feature column before scaling")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns:
        raise ValueError(f"time_col {args.time_col} not found. Available: {df.columns.tolist()}")
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values(args.time_col).reset_index(drop=True)

    # returns series for entropy computation
    # priority: use provided log_return column; else compute from close if available
    rets = None
    if "log_return" in df.columns:
        rets = df["log_return"].astype(float).values
    elif "close" in df.columns:
        close_for_ret = df["close"].astype(float).values
        eps = 1e-12
        fut_for_ret = np.roll(close_for_ret, -1)
        lr = np.log((fut_for_ret + eps) / (close_for_ret + eps))
        lr[-1] = np.nan
        rets = lr
    else:
        raise ValueError("Need 'log_return' or 'close' column to compute entropy series.")

    # features
    feats = df[args.feature_cols].astype(float).copy()
    # compute rolling Shannon entropy over the last L (=lin) returns
    E = rolling_entropy(rets, args.lin, args.entropy_bins)
    # optionally include entropy as a feature channel (broadcast per row)
    if args.add_entropy_feature:
        feats = feats.copy()
        feats["ent_shannon"] = E
    # target for later forecasting eval: next-H log return on close (if close in features)
    target = None
    if "close" in df.columns:
        close = df["close"].astype(float).values
        eps = 1e-12
        fut = np.roll(close, -args.target_h)
        y = np.log((fut + eps) / (close + eps))
        y[-args.target_h:] = np.nan
        target = y

    # split by time
    n = len(df)
    n_train = int(n * args.train_ratio)
    n_val = int(n * (args.train_ratio + args.val_ratio))

    split_info = {
        "n": n, "n_train": n_train, "n_val": n_val, "n_test": n - n_val
    }

    # scale by train only
    scaler = StandardScaler()
    scaler.fit(feats.iloc[:n_train])
    X = scaler.transform(feats)

    # window indices per split
    def windows_in_range(start, end):
        idx = []
        for s, e in make_windows(np.arange(start, end), args.lin, args.stride):
            if e <= end:
                idx.append((s, e))
        return idx

    w_train = windows_in_range(0, n_train)
    w_val   = windows_in_range(n_train, n_val)
    w_test  = windows_in_range(n_val, n)

    # TRAIN-only entropy filtering (quantile on train windows), val/test untouched
    train_keep = w_train
    if args.entropy_filter:
        def win_ent(e_idx):
            if 0 <= e_idx - 1 < len(E):
                return E[e_idx - 1]
            return np.nan
        ent_train = np.array([win_ent(e) for (s, e) in w_train], dtype=np.float32)
        ent_train = ent_train[~np.isnan(ent_train)]
        if ent_train.size == 0:
            raise ValueError("No valid entropy values for train windows; check inputs and lin.")
        thr = np.quantile(ent_train, args.entropy_q)
        train_keep = [(s, e) for (s, e) in w_train
                      if (0 <= e - 1 < len(E)) and (not np.isnan(E[e - 1])) and (E[e - 1] <= thr)]
        print(f"[Entropy filter] kept {len(train_keep)} / {len(w_train)} train windows (q={args.entropy_q:.2f})")

    # save arrays
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    if target is not None:
        np.save(os.path.join(args.out_dir, "y.npy"), target)
    np.save(os.path.join(args.out_dir, "E.npy"), E)

    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({
            "train_windows": w_train,
            "train_windows_filtered": train_keep,
            "val_windows": w_val,
            "test_windows": w_test,
            "lin": args.lin,
            "stride": args.stride,
            "target_h": args.target_h,
            "time_col": args.time_col,
            "feature_cols": args.feature_cols + (["ent_shannon"] if args.add_entropy_feature else []),
            "split_info": split_info,
            "entropy_filter": bool(args.entropy_filter),
            "entropy_q": float(args.entropy_q),
            "entropy_bins": int(args.entropy_bins)
        }, f, indent=2)

    # persist scaler
    import joblib
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))

    print(f"Saved windows and splits to {args.out_dir}")

if __name__ == "__main__":
    main()

# feature + filtering command:
# python data_preprocessing/make_windows.py \
#   --csv data/ABB_5minute_cleaned_features.csv \
#   --time_col "Unnamed: 0" \
#   --feature_cols open high low close volume log_return hl_range tod_sin tod_cos \
#   --lin 256 --stride 16 \
#   --train_ratio 0.70 --val_ratio 0.15 \
#   --out_dir outputs/windows_entropy_hybrid \
#   --target_h 12 \
#   --add_entropy_feature \
#   --entropy_filter --entropy_q 0.9 --entropy_bins 11