import torch

def time_mask(x, mask_ratio=0.2):
    # x: [B, C, L]
    B, C, L = x.shape
    mlen = max(1, int(L * mask_ratio))
    start = torch.randint(0, L - mlen + 1, (B,), device=x.device)
    out = x.clone()
    for i in range(B):
        out[i, :, start[i]:start[i]+mlen] = 0.0
    return out

def add_noise(x, std=0.01):
    if std <= 0: return x
    return x + torch.randn_like(x) * std

def augment_pair(x, mask_ratio=0.2, noise_std=0.01):
    x1 = add_noise(time_mask(x, mask_ratio), noise_std)
    x2 = add_noise(time_mask(x, mask_ratio), noise_std)
    return x1, x2