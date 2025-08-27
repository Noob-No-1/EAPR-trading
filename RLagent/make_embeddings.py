# RLagent/make_embeddings.py
import os, json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from models.ts2vec.dataset import WindowDataset
from models.ts2vec.model import TS2VecEncoder, ProjectionHead

def extract_split(ckpt, split, root="outputs/windows", pool="mean"):
    ds = WindowDataset(root, split)
    dl = DataLoader(ds, batch_size=128, shuffle=False)
    C = ds.C
    hidden = ckpt["cfg"]["model"]["hidden_dim"]
    depth = ckpt["cfg"]["model"]["depth"]
    proj_dim = ckpt["cfg"]["model"]["proj_dim"]

    enc = TS2VecEncoder(C, hidden, depth)
    enc.load_state_dict(ckpt["enc"])
    proj = ProjectionHead(hidden, proj_dim)
    proj.load_state_dict(ckpt["proj"])
    enc.eval(); proj.eval()

    all_Z = []
    with torch.no_grad():
        for x in dl:  # x: [B,C,L]
            h = enc(x)            # [B,H,L]
            z = proj(h)           # [B,D,L]
            if pool == "mean":
                z_pool = z.mean(dim=-1)   # [B,D]
            elif pool == "last":
                z_pool = z[:, :, -1]      # [B,D]
            else:
                raise ValueError("Unsupported pool")
            all_Z.append(z_pool.cpu().numpy())
    Z = np.concatenate(all_Z, axis=0)  # [N_windows, D]
    return Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/ts2vec_baseline/ts2vec_baseline.ckpt")
    ap.add_argument("--root", default="outputs/windows")
    ap.add_argument("--out",  default="outputs/embeddings.npy")
    ap.add_argument("--out_idx", default="outputs/window_end_idx.npy")
    ap.add_argument("--pool", default="mean", choices=["mean","last"])
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    # Extrae por splits y concatena en orden train->val->test
    Z_tr = extract_split(ckpt, "train", root=args.root, pool=args.pool)
    Z_va = extract_split(ckpt, "val",   root=args.root, pool=args.pool)
    Z_te = extract_split(ckpt, "test",  root=args.root, pool=args.pool)
    Z = np.vstack([Z_tr, Z_va, Z_te])   # [N_windows_total, D]

    # Guardar índices de fin de ventana para mapear a barras (episodios)
    with open(os.path.join(args.root, "splits.json"), "r") as f:
        meta = json.load(f)
    win_tr = meta["train_windows"]
    win_va = meta["val_windows"]
    win_te = meta["test_windows"]
    wins = win_tr + win_va + win_te           # [(s,e), ...] en el mismo orden que Z
    end_idx = np.array([e-1 for (s,e) in wins], dtype=np.int64)  # índice de barra (5 min) donde termina cada ventana

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.save(args.out, Z)
    np.save(args.out_idx, end_idx)
    print(f"Saved embeddings: {args.out}  shape={Z.shape}")
    print(f"Saved window_end_idx: {args.out_idx}  shape={end_idx.shape}")

if __name__ == "__main__":
    main()
