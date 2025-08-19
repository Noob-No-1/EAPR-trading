import os, json, numpy as np, torch
from torch.utils.data import DataLoader
from models.ts2vec.dataset import WindowDataset
from models.ts2vec.model import TS2VecEncoder, ProjectionHead
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score

def extract_embeddings(ckpt_path, split, pool="mean"):
    root = "outputs/windows"
    ds = WindowDataset(root, split)
    dl = DataLoader(ds, batch_size=64, shuffle=False)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    C = ds.C
    hidden = ckpt["cfg"]["model"]["hidden_dim"]
    proj_dim = ckpt["cfg"]["model"]["proj_dim"]
    enc = TS2VecEncoder(C, hidden, ckpt["cfg"]["model"]["depth"])
    proj = ProjectionHead(hidden, proj_dim)
    enc.load_state_dict(ckpt["enc"]); proj.load_state_dict(ckpt["proj"])
    enc.eval(); proj.eval()
    Z = []
    with torch.no_grad():
        for x in dl:
            z = proj(enc(x))  # [B,D,L]
            if pool == "mean":
                z = z.mean(dim=2)  # [B,D]
            Z.append(z.numpy())
    return np.concatenate(Z, axis=0)

def main():
    out_dir = "outputs/ts2vec_baseline"
    ckpt = os.path.join(out_dir, "ts2vec_baseline.ckpt")
    from pathlib import Path
    meta = json.load(open("outputs/windows/splits.json"))
    y = np.load("outputs/windows/y.npy")
    lin = meta["lin"]; stride = meta["stride"]; H = meta["target_h"]

    def labels_for(windows):
        # use the label at the last index of each window
        idx = [e-1 for (s,e) in windows]
        return y[idx]

    Ztr = extract_embeddings(ckpt, "train")
    Zva = extract_embeddings(ckpt, "val")
    Zte = extract_embeddings(ckpt, "test")

    ymeta = json.load(open("outputs/windows/splits.json"))
    ytr = labels_for(ymeta["train_windows"])
    yva = labels_for(ymeta["val_windows"])
    yte = labels_for(ymeta["test_windows"])

    # drop NaNs (near end due to horizon)
    def drop_nan(Z, y):
        m = ~np.isnan(y)
        return Z[m], y[m]
    Ztr,ytr = drop_nan(Ztr,ytr); Zva,yva = drop_nan(Zva,yva); Zte,yte = drop_nan(Zte,yte)

    # Regression on next-H log-return
    reg = LinearRegression().fit(np.vstack([Ztr,Zva]), np.hstack([ytr,yva]))
    pred = reg.predict(Zte)
    rmse = mean_squared_error(yte, pred) ** 0.5
    mae  = mean_absolute_error(yte, pred)

    # Directional accuracy (classification)
    ytr_c = (ytr>0).astype(int); yva_c=(yva>0).astype(int); yte_c=(yte>0).astype(int)
    clf = LogisticRegression(max_iter=1000).fit(np.vstack([Ztr,Zva]), np.hstack([ytr_c,yva_c]))
    prob = clf.predict_proba(Zte)[:,1]
    acc = accuracy_score(yte_c, (prob>=0.5).astype(int))
    try:
        auc = roc_auc_score(yte_c, prob)
    except:
        auc = float("nan")

    res = {"RMSE": float(rmse), "MAE": float(mae), "ACC": float(acc), "AUC": float(auc)}
    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "results_baseline.json"), "w"), indent=2)
    print(res)

if __name__ == "__main__":
    main()