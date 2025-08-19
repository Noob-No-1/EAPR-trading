import argparse, os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json

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
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    if args.time_col not in df.columns:
        raise ValueError(f"time_col {args.time_col} not found. Available: {df.columns.tolist()}")
    df[args.time_col] = pd.to_datetime(df[args.time_col])
    df = df.sort_values(args.time_col).reset_index(drop=True)

    # features
    feats = df[args.feature_cols].astype(float).copy()
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

    # save arrays
    np.save(os.path.join(args.out_dir, "X.npy"), X)
    if target is not None:
        np.save(os.path.join(args.out_dir, "y.npy"), target)

    with open(os.path.join(args.out_dir, "splits.json"), "w") as f:
        json.dump({
            "train_windows": w_train,
            "val_windows": w_val,
            "test_windows": w_test,
            "lin": args.lin,
            "stride": args.stride,
            "target_h": args.target_h,
            "time_col": args.time_col,
            "feature_cols": args.feature_cols,
            "split_info": split_info
        }, f, indent=2)

    # persist scaler
    import joblib
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))

    print(f"Saved windows and splits to {args.out_dir}")

if __name__ == "__main__":
    main()