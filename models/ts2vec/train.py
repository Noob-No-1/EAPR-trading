import os, yaml, random, numpy as np, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import WindowDataset
from model import TS2VecEncoder, ProjectionHead, info_nce
from augment import augment_pair

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    out_dir = cfg["paths"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)

    # Datasets
    dataset_root = "outputs/windows"     # produced by make_windows.py
    train_ds = WindowDataset(dataset_root, "train")
    val_ds   = WindowDataset(dataset_root, "val")

    C = train_ds.C
    device = cfg["train"]["device"] if torch.cuda.is_available() else "cpu"

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=True)

    # Model
    enc = TS2VecEncoder(c_in=C, hidden=cfg["model"]["hidden_dim"], depth=cfg["model"]["depth"]).to(device)
    proj = ProjectionHead(hidden=cfg["model"]["hidden_dim"], proj=cfg["model"]["proj_dim"]).to(device)
    opt = torch.optim.AdamW(list(enc.parameters())+list(proj.parameters()),
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = 1e9
    for epoch in range(1, cfg["train"]["epochs"]+1):
        enc.train(); proj.train()
        tr_loss = 0.0
        for x in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(device)                            # [B,C,L]
            x1, x2 = augment_pair(x, cfg["train"]["mask_ratio"], cfg["train"]["noise_std"])
            z1 = proj(enc(x1))
            z2 = proj(enc(x2))
            loss = info_nce(z1, z2, cfg["train"]["temperature"])

            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # Val
        enc.eval(); proj.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x in val_loader:
                x = x.to(device)
                x1, x2 = augment_pair(x, cfg["train"]["mask_ratio"], cfg["train"]["noise_std"])
                z1 = proj(enc(x1)); z2 = proj(enc(x2))
                val_loss += info_nce(z1, z2, cfg["train"]["temperature"]).item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train {tr_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"enc": enc.state_dict(), "proj": proj.state_dict(), "cfg": cfg},
                       os.path.join(out_dir, "ts2vec_baseline.ckpt"))
            print("Saved checkpoint.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)