import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("  Receptive Field Visualization")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Params (same as R-STDP) ---
T_STEPS = 30
N_IN, N_HID = 784, 100
N_POP = 50
V_TH = 1.0
DECAY = float(np.exp(-1.0 / 20.0))
K_WIN = 3
A_LTP, A_LTD = 0.005, 0.003
EPOCHS = 5
W_TARGET = 1.0

# --- Data ---
transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
train_full = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_idx = [i for i, (_, l) in enumerate(train_full) if l in (0, 1)]
X_train = torch.stack([train_full[i][0] for i in train_idx]).to(device)
Y_train = torch.tensor([train_full[i][1] for i in train_idx]).to(device)

# --- Train R-STDP (seed 42 for reproducibility) ---
print("\n  Training R-STDP (seed 42, 5 epochs)...")
torch.manual_seed(42)
W = torch.rand(N_HID, N_IN, device=device) * 0.05
W = W / W.sum(dim=1, keepdim=True) * W_TARGET

for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    for idx in perm:
        x = X_train[idx]
        label = Y_train[idx].item()
        c_sl = slice(0, N_POP) if label == 0 else slice(N_POP, N_HID)
        w_sl = slice(N_POP, N_HID) if label == 0 else slice(0, N_POP)

        v = torch.zeros(N_HID, device=device)
        for t in range(T_STEPS):
            v = DECAY * v + W @ x
            tv, ti = v.topk(K_WIN)
            sm = torch.zeros(N_HID, dtype=torch.bool, device=device)
            f = tv > V_TH
            if f.any(): sm[ti[f]] = True
            spike = sm.float()
            sc = spike[c_sl]
            if sc.sum() > 0: W[c_sl] += A_LTP * torch.outer(sc, x)
            sw = spike[w_sl]
            if sw.sum() > 0: W[w_sl] -= A_LTD * torch.outer(sw, x)
            v[sm] = 0.0

        W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8) * W_TARGET
        W.clamp_(0.0, 1.0)
    print(f"    Epoch {epoch+1}/{EPOCHS} done")

# --- Plot ---
print("\n  Plotting receptive fields...")
W_cpu = W.cpu().numpy()

fig, axes = plt.subplots(10, 10, figsize=(12, 12))
fig.suptitle("R-STDP Receptive Fields (100 neurons)\n"
             "Top 5 rows: Pop 0 (digit 0) | Bottom 5 rows: Pop 1 (digit 1)",
             fontsize=13, fontweight="bold")

for i in range(100):
    row, col = i // 10, i % 10
    ax = axes[row, col]
    img = W_cpu[i].reshape(28, 28)
    cmap = "Blues" if i < N_POP else "Reds"
    ax.imshow(img, cmap=cmap, vmin=0, vmax=W_cpu.max())
    ax.axis("off")
    if col == 0:
        pop = "Pop 0" if i < N_POP else "Pop 1"
        ax.set_ylabel(f"n{i}\n{pop}", fontsize=7, rotation=0, labelpad=25, va="center")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/receptive_fields.png", dpi=150, bbox_inches="tight")
print("  Saved: figures/receptive_fields.png")
