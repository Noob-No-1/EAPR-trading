import os, yaml, random, numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from .dataset import WindowDataset
from .model import TS2VecEncoder, ProjectionHead, info_nce
from .augment import augment_pair

'''
def debug_model_architecture(enc, proj, sample_input, device, temperature=0.3):
    """Deep dive into model architecture issues"""
    print("\n" + "="*60)
    print("🔬 DEEP MODEL ARCHITECTURE DEBUG")
    print("="*60)
    
    enc.eval()
    proj.eval()
    
    with torch.no_grad():
        x = sample_input.to(device)
        print(f"Input: shape={x.shape}, mean={x.mean():.6f}, std={x.std():.6f}")
        
        # Step 1: Check encoder output
        print(f"\n1️⃣ ENCODER DEBUG:")
        h = enc(x)
        print(f"   Encoder output shape: {h.shape}")
        print(f"   Encoder output: mean={h.mean():.6f}, std={h.std():.6f}")
        print(f"   Encoder range: [{h.min():.6f}, {h.max():.6f}]")
        print(f"   Encoder NaN/Inf: {torch.isnan(h).sum()}, {torch.isinf(h).sum()}")
        
        # Check if encoder is producing reasonable outputs
        if torch.isnan(h).sum() > 0 or torch.isinf(h).sum() > 0:
            print("   ❌ ENCODER ISSUE: NaN/Inf detected!")
            return False
            
        if h.std() < 1e-6:
            print("   ❌ ENCODER ISSUE: Output has no variance (collapsed)!")
            return False
            
        if h.std() > 100:
            print("   ❌ ENCODER ISSUE: Output variance too high (exploding)!")
            return False
        
        print("   ✅ Encoder output looks reasonable")
        
        # Step 2: Check projection head
        print(f"\n2️⃣ PROJECTION HEAD DEBUG:")
        z = proj(h)
        print(f"   Projection output shape: {z.shape}")
        print(f"   Projection output: mean={z.mean():.6f}, std={z.std():.6f}")
        print(f"   Projection range: [{z.min():.6f}, {z.max():.6f}]")
        print(f"   Projection NaN/Inf: {torch.isnan(z).sum()}, {torch.isinf(z).sum()}")
        
        if torch.isnan(z).sum() > 0 or torch.isinf(z).sum() > 0:
            print("   ❌ PROJECTION ISSUE: NaN/Inf detected!")
            return False
            
        # Step 3: Check normalization
        print(f"\n3️⃣ NORMALIZATION DEBUG:")
        
        # Handle temporal dimension if present
        if z.dim() == 3:
            z_pooled = z.mean(dim=-1)
            print(f"   Pooled to shape: {z_pooled.shape}")
        else:
            z_pooled = z
            
        z_norm = F.normalize(z_pooled, dim=1)
        norms_before = torch.norm(z_pooled, dim=1)
        norms_after = torch.norm(z_norm, dim=1)
        
        print(f"   Norms before normalization: mean={norms_before.mean():.6f}, std={norms_before.std():.6f}")
        print(f"   Norms after normalization: mean={norms_after.mean():.6f}, std={norms_after.std():.6f}")
        
        if norms_before.mean() < 1e-6:
            print("   ❌ NORMALIZATION ISSUE: Near-zero norms before normalization!")
            return False
        
        if not torch.allclose(norms_after, torch.ones_like(norms_after), atol=1e-5):
            print("   ❌ NORMALIZATION ISSUE: F.normalize not working correctly!")
            return False
            
        print("   ✅ Normalization working correctly")
        
        # Step 4: Check similarity computation
        print(f"\n4️⃣ SIMILARITY COMPUTATION DEBUG:")
        
        # Two identical normalized vectors should have similarity = 1.0
        z1_norm = z_norm
        z2_norm = z_norm  # Identical
        
        similarities = F.cosine_similarity(z1_norm, z2_norm, dim=1)
        print(f"   Identical vector similarities: mean={similarities.mean():.6f}, min={similarities.min():.6f}")
        
        if not torch.allclose(similarities, torch.ones_like(similarities), atol=1e-5):
            print("   ❌ SIMILARITY ISSUE: Identical vectors don't have similarity 1.0!")
            return False
            
        # Step 5: Check InfoNCE computation manually
        print(f"\n5️⃣ INFONCE COMPUTATION DEBUG:")
        
        batch_size = z1_norm.size(0)
        sim_matrix = (z1_norm @ z2_norm.t()) / temperature
        print(f"   Similarity matrix shape: {sim_matrix.shape}")
        print(f"   Diagonal (positive pairs): mean={torch.diag(sim_matrix).mean():.6f}")
        print(f"   Off-diagonal (negatives): mean={sim_matrix[~torch.eye(batch_size, dtype=bool, device=device)].mean():.6f}")
        
        # For identical inputs, diagonal should be 1/temperature
        expected_diagonal = 1.0 / temperature
        actual_diagonal = torch.diag(sim_matrix).mean()
        
        print(f"   Expected diagonal value: {expected_diagonal:.6f}")
        print(f"   Actual diagonal value: {actual_diagonal:.6f}")
        
        if not torch.allclose(torch.diag(sim_matrix), torch.full((batch_size,), expected_diagonal, device=device), atol=1e-4):
            print("   ❌ SIMILARITY MATRIX ISSUE: Diagonal values incorrect!")
            return False
            
        # Compute InfoNCE manually
        labels = torch.arange(batch_size, device=device)
        manual_loss = F.cross_entropy(sim_matrix, labels)
        
        # Compare with your function
        your_loss = info_nce_per_sample(z1_norm, z2_norm, temperature).mean()
        
        print(f"   Manual InfoNCE loss: {manual_loss:.6f}")
        print(f"   Your function loss: {your_loss:.6f}")
        print(f"   Difference: {abs(manual_loss - your_loss):.6f}")
        
        if abs(manual_loss - your_loss) > 1e-4:
            print("   ❌ INFONCE FUNCTION ISSUE: Your implementation differs from standard!")
            return False
            
        print("   ✅ InfoNCE computation correct")
        
        # Final diagnosis
        print(f"\n🎯 DIAGNOSIS:")
        if manual_loss > 1.0:
            print(f"   Even manual computation gives high loss ({manual_loss:.4f})")
            print(f"   Expected for random: ~{np.log(batch_size):.4f}")
            print(f"   Expected for perfect: ~{-np.log(batch_size) + 1/temperature:.4f}")
            
            if batch_size < 16:
                print("   ⚠️  Small batch size might cause issues")
            if temperature < 0.2:
                print("   ⚠️  Very low temperature might cause numerical issues")
                
        return manual_loss < 1.0
    
def test_model_sanity(enc, proj, val_loader, device):
    enc.eval(); proj.eval()
    with torch.no_grad():
        val_batch = next(iter(val_loader))
        x = val_batch[0] if isinstance(val_batch, tuple) else val_batch
        x = x.to(device)
        
        # Identity test
        z1 = proj(enc(x))
        z2 = proj(enc(x))  # Same input
        if z1.dim() == 3: z1 = z1.mean(dim=-1)
        if z2.dim() == 3: z2 = z2.mean(dim=-1)
        identity_loss = info_nce_per_sample(z1, z2, 0.3).mean()
        print(f"PRE-TRAINING Identity loss: {identity_loss:.4f}")
        
        if identity_loss > 1.0:
            print("❌ Model has fundamental issues!")
            return False
        else:
            print("✅ Model architecture seems OK")
            return True
'''
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

    '''
    # DEEP DEBUG - Run this to find the exact issue
    val_batch = next(iter(val_loader))
    sample_x = val_batch[0] if isinstance(val_batch, tuple) else val_batch
    
    if not debug_model_architecture(enc, proj, sample_x, device, cfg["train"]["temperature"]):
        print("STOPPING: Fix model architecture issues before training")
        return
    '''

    best_val = 1e9
    patience = cfg["train"].get("patience", 10)  
    patience_counter = 0
    min_improvement = cfg["train"].get("min_improvement", 0.001)
    
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

        # Val - FIXED: Make validation consistent with training
        enc.eval(); proj.eval()
        with torch.no_grad():
            val_loss = 0.0
            for batch in val_loader:  # Changed from 'x' to 'batch' to handle entropy data
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    x, _ = batch  # Ignore entropy in validation
                else:
                    x = batch
                x = x.to(device)
                x1, x2 = augment_pair(x, cfg["train"]["mask_ratio"], cfg["train"]["noise_std"])
                z1 = proj(enc(x1))
                z2 = proj(enc(x2))
                
                # FIXED: Apply same logic as training for handling dimensions
                if z1.dim() == 3 and z2.dim() == 3:
                    # Use sequence InfoNCE if 3D tensors
                    loss = info_nce(z1, z2, cfg["train"]["temperature"])
                else:
                    # Pool to [B,D] and use per-sample InfoNCE
                    if z1.dim() == 3:
                        z1 = z1.mean(dim=-1)
                    if z2.dim() == 3:
                        z2 = z2.mean(dim=-1)
                    loss = info_nce_per_sample(z1, z2, cfg["train"]["temperature"]).mean()
                
                val_loss += float(loss.detach().cpu().item())
            val_loss /= len(val_loader)

        print(f"Epoch {epoch}: train {tr_loss:.4f} | val {val_loss:.4f}")
        
        # Early stopping logic
        if val_loss < best_val - min_improvement:
            best_val = val_loss
            patience_counter = 0
            ckpt_name = cfg.get("paths", {}).get("ckpt_name", "ts2vec_entropy.ckpt")
            torch.save({"enc": enc.state_dict(), "proj": proj.state_dict(), "cfg": cfg},
                       os.path.join(out_dir, ckpt_name))
            print("Saved checkpoint.")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s)")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                print(f"Best validation loss: {best_val:.4f}")
                break

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)