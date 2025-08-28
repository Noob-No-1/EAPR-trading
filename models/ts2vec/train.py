import os, yaml, random, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import WindowDataset
from .model import TS2VecEncoder, ProjectionHead, info_nce
from .augment import augment_pair

def pick_device(cfg_device: str) -> str: # from cfg, else auto-pick
    want = (cfg_device or "").lower()
    if want == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    if (want.startswith("cuda") or want == "gpu") and torch.cuda.is_available():
        return "cuda"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def info_nce_per_sample(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Return per-sample NT-Xent loss (no batch mean). z1,z2: [B,D]."""
    assert z1.dim() == 2 and z2.dim() == 2, f"Expected [B,D], got {z1.shape} and {z2.shape}"
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits12 = (z1 @ z2.t()) / temperature   # [B,B]
    logits21 = (z2 @ z1.t()) / temperature   # [B,B]
    labels = torch.arange(z1.size(0), device=z1.device)
    loss12 = F.cross_entropy(logits12, labels, reduction='none')
    loss21 = F.cross_entropy(logits21, labels, reduction='none')
    return 0.5 * (loss12 + loss21)           # [B]

def robust_entropy_z(ent: torch.Tensor, device: str) -> torch.Tensor:
    """Return robust z-score of entropy using median/IQR computed on CPU (MPS-safe)."""
    # Always compute stats on CPU to avoid MPS missing ops (nanmedian/nanquantile)
    ent_cpu = ent.detach().float().cpu()
    med = torch.nanmedian(ent_cpu)
    q75 = torch.nanquantile(ent_cpu, 0.75)
    q25 = torch.nanquantile(ent_cpu, 0.25)
    iqr = torch.clamp(q75 - q25, min=1e-6)
    # Move scalars back to training device and compute z there
    med = med.to(device)
    iqr = iqr.to(device)
    return (ent - med) / iqr

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    set_seed(cfg["seed"])
    out_dir = cfg["paths"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)

    # Datasets
    # Prefer a configurable windows root; fall back to default
    dataset_root = (
        cfg.get("paths", {}).get("windows_root")
        or cfg.get("data", {}).get("windows_root")
        or "outputs/windows"
    )  # produced by make_windows.py

    use_ent_w = bool(cfg["train"].get("entropy_weighted", False))
    e_path = os.path.join(dataset_root, "E.npy")
    if use_ent_w and not os.path.exists(e_path):
        print("[WARN] train.entropy_weighted=True but E.npy not found at:", e_path)
        print("       Disabling entropy weighting for this run.")
        use_ent_w = False

    train_ds = WindowDataset(dataset_root, "train", with_entropy=use_ent_w)
    val_ds   = WindowDataset(dataset_root, "val",   with_entropy=False)

    # Optional: quick summary of splits
    try:
        import json
        with open(os.path.join(dataset_root, "splits.json"), "r") as _f:
            _meta = json.load(_f)
        _tw = len(_meta.get("train_windows", []))
        _vw = len(_meta.get("val_windows", []))
        _tew = len(_meta.get("test_windows", []))
        if _meta.get("entropy_filter") and "train_windows_filtered" in _meta:
            _twf = len(_meta.get("train_windows_filtered", []))
            print(f"[Data] train={_tw} (filtered={_twf}) | val={_vw} | test={_tew} | entropy_filter=True q={_meta.get('entropy_q')}")
        else:
            print(f"[Data] train={_tw} | val={_vw} | test={_tew}")
    except Exception as _e:
        print("[Data] Could not summarize splits.json:", _e)

    C = train_ds.C
    device = pick_device(cfg["train"].get("device", "cpu"))

    train_loader = DataLoader(train_ds, batch_size=cfg["train"]["batch_size"], shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg["train"]["batch_size"], shuffle=False, drop_last=True)

    clip_lo, clip_hi = 0.5, 1.5
    if isinstance(cfg["train"].get("weight_clip", None), (list, tuple)) and len(cfg["train"]["weight_clip"]) == 2:
        clip_lo, clip_hi = float(cfg["train"]["weight_clip"][0]), float(cfg["train"]["weight_clip"][1])

    # Model
    enc = TS2VecEncoder(c_in=C, hidden=cfg["model"]["hidden_dim"], depth=cfg["model"]["depth"]).to(device)
    proj = ProjectionHead(hidden=cfg["model"]["hidden_dim"], proj=cfg["model"]["proj_dim"]).to(device)
    opt = torch.optim.AdamW(list(enc.parameters())+list(proj.parameters()),
                            lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])

    best_val = 1e9
    for epoch in range(1, cfg["train"]["epochs"]+1):
        enc.train(); proj.train()
        tr_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                x, ent = batch
                ent = torch.as_tensor(ent, device=device)
            else:
                x, ent = batch, None
            x = x.to(device)  # [B,C,L]
            x1, x2 = augment_pair(x, cfg["train"]["mask_ratio"], cfg["train"]["noise_std"])
            z1 = proj(enc(x1))
            z2 = proj(enc(x2))
            # Ensure embeddings are [B,D] for per-sample InfoNCE by pooling any temporal axis
            if z1.dim() == 3:
                z1 = z1.mean(dim=-1)
            if z2.dim() == 3:
                z2 = z2.mean(dim=-1)
            # If encoder/projection keep a temporal axis, pool over time so InfoNCE sees [B,D]
            if ent is not None and bool(cfg["train"].get("entropy_weighted", False)):
                # per-sample NT-Xent so we can weight (expects [B,D])
                ps = info_nce_per_sample(z1, z2, cfg["train"]["temperature"])  # [B]
                z = robust_entropy_z(ent, device)
                w = torch.sigmoid(-2.0 * z)          # higher entropy -> lower weight
                w = w / (w.mean() + 1e-8)
                w = torch.clamp(w, clip_lo, clip_hi)
                loss = (w * ps).mean()
            else:
                # If tensors are 3D ([B,D,T]), use sequence InfoNCE; if 2D, use per-sample and mean
                if z1.dim() == 3 and z2.dim() == 3:
                    loss = info_nce(z1, z2, cfg["train"]["temperature"])  # sequence-aware
                else:
                    loss = info_nce_per_sample(z1, z2, cfg["train"]["temperature"]).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            tr_loss += float(loss.detach().cpu().item())
        tr_loss /= len(train_loader)

        # Val
        enc.eval(); proj.eval()
        with torch.no_grad():
            val_loss = 0.0
            for x in val_loader:
                x = x.to(device)
                x1, x2 = augment_pair(x, cfg["train"]["mask_ratio"], cfg["train"]["noise_std"])
                z1 = proj(enc(x1)); z2 = proj(enc(x2))
                # Match training: pool any temporal axis to [B,D] and use per-sample InfoNCE (no entropy weights)
                if z1.dim() == 3:
                    z1 = z1.mean(dim=-1)
                if z2.dim() == 3:
                    z2 = z2.mean(dim=-1)
                val_loss += info_nce_per_sample(z1, z2, cfg["train"]["temperature"]).mean().item()
            val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train {tr_loss:.4f} | val {val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            ckpt_name = cfg.get("paths", {}).get("ckpt_name", "ts2vec_baseline.ckpt")
            torch.save({"enc": enc.state_dict(), "proj": proj.state_dict(), "cfg": cfg},
                       os.path.join(out_dir, ckpt_name))
            print("Saved checkpoint.")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)