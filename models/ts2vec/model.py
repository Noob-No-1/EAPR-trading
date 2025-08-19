import torch
import torch.nn as nn
import torch.nn.functional as F

class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=5, d=1):
        super().__init__()
        pad = (k-1)*d
        self.conv = nn.Conv1d(c_in, c_out, kernel_size=k, dilation=d, padding=pad)
        self.bn = nn.BatchNorm1d(c_out)
        self.proj = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()
    def forward(self, x):
        y = self.conv(x)
        y = self.bn(F.relu(y))
        # causal trim
        k = self.conv.kernel_size[0]; d = self.conv.dilation[0]
        trim = (k-1)*d
        if trim > 0: y = y[:, :, :-trim]
        z = self.proj(x[:, :, :y.size(2)])
        return F.relu(y + z)

class TS2VecEncoder(nn.Module):
    def __init__(self, c_in, hidden=256, depth=6):
        super().__init__()
        layers = []
        c = c_in
        for i in range(depth):
            layers.append(TCNBlock(c, hidden, k=5, d=2**i))
            c = hidden
        self.net = nn.Sequential(*layers)
    def forward(self, x):  # x: [B,C,L]
        return self.net(x) # [B,hidden,L']

class ProjectionHead(nn.Module):
    def __init__(self, hidden, proj=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(hidden, hidden, 1),
            nn.ReLU(),
            nn.Conv1d(hidden, proj, 1)
        )
    def forward(self, h):
        return self.mlp(h)  # [B,proj,L]

def info_nce(z1, z2, temperature=0.3):
    # Treat each time step as an instance; align same timestep across views
    B, D, L = z1.shape
    z1 = z1.permute(0,2,1).reshape(B*L, D)
    z2 = z2.permute(0,2,1).reshape(B*L, D)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(B*L, device=z1.device)
    return F.cross_entropy(logits, labels)