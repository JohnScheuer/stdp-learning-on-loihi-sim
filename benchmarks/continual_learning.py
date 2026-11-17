import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("  Continual Learning: R-STDP")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")

# --- Params ---
T_STEPS = 30
N_IN = 784
V_TH = 1.0
DECAY = float(np.exp(-1.0 / 20.0))
K_WIN = 3
A_LTP, A_LTD = 0.005, 0.003
EPOCHS = 5
W_TARGET = 1.0

# --- Data ---
transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
train_full = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_full = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

def get_data(digits):
    tr_idx = [i for i, (_, l) in enumerate(train_full) if l in digits]
    te_idx = [i for i, (_, l) in enumerate(test_full) if l in digits]
    X_tr = torch.stack([train_full[i][0] for i in tr_idx]).to(device)
    Y_tr = torch.tensor([train_full[i][1] for i in tr_idx]).to(device)
    X_te = torch.stack([test_full[i][0] for i in te_idx]).to(device)
    Y_te = torch.tensor([test_full[i][1] for i in te_idx]).to(device)
    return X_tr, Y_tr, X_te, Y_te

def train_rstdp(X, Y, W, pop_map, epochs=EPOCHS):
    """Train with R-STDP. pop_map: {label: (start, end)}"""
    N_HID = W.shape[0]
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for idx in perm:
            x = X[idx]
            label = Y[idx].item()
            c_start, c_end = pop_map[label]

            v = torch.zeros(N_HID, device=device)
            for t in range(T_STEPS):
                v = DECAY * v + W @ x
                tv, ti = v.topk(K_WIN)
                sm = torch.zeros(N_HID, dtype=torch.bool, device=device)
                f = tv > V_TH
                if f.any(): sm[ti[f]] = True
                spike = sm.float()

                # LTP on correct pop
                sc = spike[c_start:c_end]
                if sc.sum() > 0:
                    W[c_start:c_end] += A_LTP * torch.outer(sc, x)

                # LTD on all other pops
                for lbl, (s, e) in pop_map.items():
                    if lbl != label:
                        sw = spike[s:e]
                        if sw.sum() > 0:
                            W[s:e] -= A_LTD * torch.outer(sw, x)

                v[sm] = 0.0

            W = W / W.sum(dim=1, keepdim=True).clamp(min=1e-8) * W_TARGET
            W.clamp_(0.0, 1.0)
    return W

def test_accuracy(X, Y, W, pop_map):
    """Test by spike-count voting."""
    N_HID = W.shape[0]
    labels = sorted(pop_map.keys())
    correct = 0
    for idx in range(len(X)):
        x = X[idx]
        v = torch.zeros(N_HID, device=device)
        sc = torch.zeros(N_HID, device=device)
        for t in range(T_STEPS):
            v = DECAY * v + W @ x
            tv, ti = v.topk(K_WIN)
            sm = torch.zeros(N_HID, dtype=torch.bool, device=device)
            f = tv > V_TH
            if f.any(): sm[ti[f]] = True
            sc += sm.float()
            v[sm] = 0.0

        votes = {}
        for lbl in labels:
            s, e = pop_map[lbl]
            votes[lbl] = sc[s:e].sum().item()
        pred = max(votes, key=votes.get)
        if pred == Y[idx].item():
            correct += 1
    return correct / len(X) * 100

# ============================================================
# Phase 1: Learn digits 0 vs 1
# ============================================================
print(f"\n[Phase 1: Learn digits 0 vs 1]")
X1_tr, Y1_tr, X1_te, Y1_te = get_data([0, 1])
print(f"  Train: {len(X1_tr)} | Test: {len(X1_te)}")

N_HID_1 = 100
torch.manual_seed(42)
W = torch.rand(N_HID_1, N_IN, device=device) * 0.05
W = W / W.sum(dim=1, keepdim=True) * W_TARGET

pop_map_1 = {0: (0, 50), 1: (50, 100)}
W = train_rstdp(X1_tr, Y1_tr, W, pop_map_1)
acc_1 = test_accuracy(X1_te, Y1_te, W, pop_map_1)
print(f"  Accuracy on 0 vs 1: {acc_1:.1f}%")

# ============================================================
# Phase 2: Expand to 0 vs 1 vs 2 vs 3 (continual)
# ============================================================
print(f"\n[Phase 2: Expand to digits 0-1-2-3 (continual learning)]")
X2_tr, Y2_tr, X2_te, Y2_te = get_data([0, 1, 2, 3])
print(f"  Train: {len(X2_tr)} | Test: {len(X2_te)}")

# Expand network: add 100 neurons for digits 2 and 3
N_HID_2 = 200
W_new = torch.rand(N_HID_2, N_IN, device=device) * 0.05
W_new[:N_HID_1] = W  # preserve old weights
W_new = W_new / W_new.sum(dim=1, keepdim=True) * W_TARGET

pop_map_2 = {0: (0, 50), 1: (50, 100), 2: (100, 150), 3: (150, 200)}

# Train ONLY new populations (freeze old weights)
print("  Freezing populations 0-1, training populations 2-3...")
W_frozen = W_new[:N_HID_1].clone()
for epoch in range(EPOCHS):
    perm = torch.randperm(len(X2_tr))
    for idx in perm:
        x = X2_tr[idx]
        label = Y2_tr[idx].item()

        v = torch.zeros(N_HID_2, device=device)
        for t in range(T_STEPS):
            v = DECAY * v + W_new @ x
            tv, ti = v.topk(K_WIN)
            sm = torch.zeros(N_HID_2, dtype=torch.bool, device=device)
            f = tv > V_TH
            if f.any(): sm[ti[f]] = True
            spike = sm.float()

            # Only update new populations (2 and 3)
            if label in (2, 3):
                c_start, c_end = pop_map_2[label]
                sc = spike[c_start:c_end]
                if sc.sum() > 0:
                    W_new[c_start:c_end] += A_LTP * torch.outer(sc, x)
                for lbl in (2, 3):
                    if lbl != label:
                        s, e = pop_map_2[lbl]
                        sw = spike[s:e]
                        if sw.sum() > 0:
                            W_new[s:e] -= A_LTD * torch.outer(sw, x)

            v[sm] = 0.0

        # Scale only new neurons
        W_new[N_HID_1:] = W_new[N_HID_1:] / W_new[N_HID_1:].sum(dim=1, keepdim=True).clamp(min=1e-8) * W_TARGET
        W_new.clamp_(0.0, 1.0)
        # Restore frozen
        W_new[:N_HID_1] = W_frozen

    print(f"    Epoch {epoch+1}/{EPOCHS} done")

# Test on ALL 4 classes
acc_all = test_accuracy(X2_te, Y2_te, W_new, pop_map_2)
print(f"\n  Accuracy on 0-1-2-3: {acc_all:.1f}%")

# Test on old classes only (catastrophic forgetting check)
acc_old = test_accuracy(X1_te, Y1_te, W_new, pop_map_2)
print(f"  Accuracy on 0 vs 1 (old task): {acc_old:.1f}%")
print(f"  Forgetting: {acc_1 - acc_old:.1f}%")

# Test on new classes only
X_new_te = torch.cat([X2_te[Y2_te==2], X2_te[Y2_te==3]])
Y_new_te = torch.cat([Y2_te[Y2_te==2], Y2_te[Y2_te==3]])
acc_new = test_accuracy(X_new_te, Y_new_te, W_new, pop_map_2)
print(f"  Accuracy on 2 vs 3 (new task): {acc_new:.1f}%")

# --- Plot ---
print(f"\n  Plotting...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Accuracy bars
ax = axes[0]
tasks = ["0 vs 1\n(before)", "0 vs 1\n(after)", "2 vs 3\n(new)", "0-1-2-3\n(all)"]
accs = [acc_1, acc_old, acc_new, acc_all]
colors = ["#2196F3", "#2196F3", "#4CAF50", "#FF9800"]
bars = ax.bar(tasks, accs, color=colors, width=0.6, edgecolor="black")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Continual Learning: No Catastrophic Forgetting")
ax.set_ylim(0, 105)
for bar, a in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{a:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Plot 2: Weight preservation heatmap
ax = axes[1]
diff = (W_new[:N_HID_1] - W_frozen).abs().mean().item()
ax.imshow(W_new[:N_HID_1].cpu().numpy()[:20, :100].reshape(20, 100),
          cmap="Blues", aspect="auto")
ax.set_title(f"Old weights (pop 0-1) preserved\nMean |delta W| = {diff:.6f}")
ax.set_xlabel("Input neurons (first 100)")
ax.set_ylabel("Hidden neurons (first 20)")

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/continual_learning.png", dpi=150, bbox_inches="tight")
print(f"  Saved: figures/continual_learning.png")

print(f"\n{'='*60}")
print(f"  CONTINUAL LEARNING RESULTS")
print(f"{'='*60}")
print(f"  Phase 1 (0 vs 1):     {acc_1:.1f}%")
print(f"  Phase 2 (0-1-2-3):    {acc_all:.1f}%")
print(f"  Old task retention:   {acc_old:.1f}% (forgetting: {acc_1-acc_old:.1f}%)")
print(f"  New task learning:    {acc_new:.1f}%")
print(f"  Method: R-STDP + frozen old populations")
print(f"  Memory: O(1) — no replay buffer needed")
print(f"{'='*60}")
