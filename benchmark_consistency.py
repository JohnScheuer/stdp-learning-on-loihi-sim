import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import time
import os

print("=" * 70)
print("  Consistency Benchmark: R-STDP vs BPTT")
print("  Task: Binary Classification (Digit 0 vs 1)")
print("=" * 70)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# --- Data ---
transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
train_full = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_full = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_idx = [i for i, (_, l) in enumerate(train_full) if l in (0, 1)]
test_idx = [i for i, (_, l) in enumerate(test_full) if l in (0, 1)]

X_train = torch.stack([train_full[i][0] for i in train_idx]).to(device)
Y_train = torch.tensor([train_full[i][1] for i in train_idx], device=device)
X_test = torch.stack([test_full[i][0] for i in test_idx]).to(device)
Y_test = torch.tensor([test_full[i][1] for i in test_idx], device=device)

N_SAMPLES = len(X_train)
N_TEST = len(X_test)
print(f"\n  Train: {N_SAMPLES} | Test: {N_TEST}")

# ============================================================
#  PART 1: R-STDP (5 seeds)
# ============================================================
print(f"\n{'='*70}")
print(f"  PART 1: R-STDP — 5 Seeds")
print(f"{'='*70}")

T_STEPS = 30
N_IN, N_HID = 784, 100
N_POP = 50
V_TH = 1.0
TAU_M = 20.0
DECAY = float(np.exp(-1.0 / TAU_M))
K_WIN = 3
A_LTP = 0.005
A_LTD = 0.003
EPOCHS = 5
W_TARGET = 1.0

def run_rstdp(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    W = torch.rand(N_HID, N_IN, device=device) * 0.05
    W = W / W.sum(dim=1, keepdim=True) * W_TARGET

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    for epoch in range(EPOCHS):
        perm = torch.randperm(N_SAMPLES)
        for idx in perm:
            x = X_train[idx]
            label = Y_train[idx].item()

            if label == 0:
                c_slice, w_slice = slice(0, N_POP), slice(N_POP, N_HID)
            else:
                c_slice, w_slice = slice(N_POP, N_HID), slice(0, N_POP)

            v = torch.zeros(N_HID, device=device)
            sc = torch.zeros(N_HID, device=device)

            for t in range(T_STEPS):
                v = DECAY * v + W @ x
                topk_vals, topk_idx = v.topk(K_WIN)
                sm = torch.zeros(N_HID, dtype=torch.bool, device=device)
                f = topk_vals > V_TH
                if f.any():
                    sm[topk_idx[f]] = True
                spike = sm.float()
                sc += spike

                s_c = spike[c_slice]
                if s_c.sum() > 0:
                    W[c_slice] += A_LTP * torch.outer(s_c, x)
                s_w = spike[w_slice]
                if s_w.sum() > 0:
                    W[w_slice] -= A_LTD * torch.outer(s_w, x)

                v[sm] = 0.0

            row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
            W = W / row_sums * W_TARGET
            W.clamp_(0.0, 1.0)

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    # Test
    correct = 0
    for idx in range(N_TEST):
        x = X_test[idx]
        v = torch.zeros(N_HID, device=device)
        sc = torch.zeros(N_HID, device=device)
        for t in range(T_STEPS):
            v = DECAY * v + W @ x
            topk_vals, topk_idx = v.topk(K_WIN)
            sm = torch.zeros(N_HID, dtype=torch.bool, device=device)
            f = topk_vals > V_TH
            if f.any():
                sm[topk_idx[f]] = True
            sc += sm.float()
            v[sm] = 0.0
        pred = 0 if sc[:N_POP].sum() > sc[N_POP:].sum() else 1
        if pred == Y_test[idx].item():
            correct += 1

    acc = correct / N_TEST * 100
    return acc, peak_mem, train_time

rstdp_results = []
for seed in [42, 123, 456, 789, 2024]:
    acc, mem, tt = run_rstdp(seed)
    rstdp_results.append((acc, mem, tt))
    print(f"  Seed {seed:4d} | acc: {acc:.1f}% | mem: {mem:.1f} MB | time: {tt:.0f}s")

rstdp_accs = [r[0] for r in rstdp_results]
rstdp_mems = [r[1] for r in rstdp_results]
rstdp_mean = np.mean(rstdp_accs)
rstdp_std = np.std(rstdp_accs)
rstdp_mem = np.mean(rstdp_mems)
print(f"\n  R-STDP: {rstdp_mean:.1f} +/- {rstdp_std:.1f}% | {rstdp_mem:.1f} MB")

# ============================================================
#  PART 2: BPTT with Surrogate Gradients (5 seeds)
# ============================================================
print(f"\n{'='*70}")
print(f"  PART 2: BPTT (Surrogate Gradient SNN) — 5 Seeds")
print(f"{'='*70}")

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        beta = 5.0
        sg = torch.sigmoid(beta * x) * (1 - torch.sigmoid(beta * x)) * beta
        return grad_output * sg

surrogate_spike = SurrogateSpike.apply

class BPTTSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 2)
        self.decay = DECAY

    def forward(self, x, T=30):
        batch = x.shape[0]
        v1 = torch.zeros(batch, 100, device=x.device)
        v2 = torch.zeros(batch, 2, device=x.device)
        out_sum = torch.zeros(batch, 2, device=x.device)

        for t in range(T):
            I1 = self.fc1(x)
            v1 = self.decay * v1 + I1
            s1 = surrogate_spike(v1 - V_TH)
            v1 = v1 * (1 - s1)

            I2 = self.fc2(s1)
            v2 = self.decay * v2 + I2
            s2 = surrogate_spike(v2 - V_TH)
            v2 = v2 * (1 - s2)
            out_sum += s2

        return out_sum / T

def run_bptt(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = BPTTSNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    batch_size = 64
    for epoch in range(EPOCHS):
        perm = torch.randperm(N_SAMPLES)
        for start in range(0, N_SAMPLES, batch_size):
            idx = perm[start:start+batch_size]
            xb = X_train[idx]
            yb = Y_train[idx]

            optimizer.zero_grad()
            out = model(xb, T=T_STEPS)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    train_time = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e6

    # Test
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, N_TEST, batch_size):
            xb = X_test[start:start+batch_size]
            yb = Y_test[start:start+batch_size]
            out = model(xb, T=T_STEPS)
            pred = out.argmax(dim=1)
            correct += (pred == yb).sum().item()

    acc = correct / N_TEST * 100
    return acc, peak_mem, train_time

bptt_results = []
for seed in [42, 123, 456, 789, 2024]:
    acc, mem, tt = run_bptt(seed)
    bptt_results.append((acc, mem, tt))
    print(f"  Seed {seed:4d} | acc: {acc:.1f}% | mem: {mem:.1f} MB | time: {tt:.0f}s")

bptt_accs = [r[0] for r in bptt_results]
bptt_mems = [r[1] for r in bptt_results]
bptt_mean = np.mean(bptt_accs)
bptt_std = np.std(bptt_accs)
bptt_mem = np.mean(bptt_mems)
print(f"\n  BPTT: {bptt_mean:.1f} +/- {bptt_std:.1f}% | {bptt_mem:.1f} MB")

# ============================================================
#  PART 3: Summary
# ============================================================
print(f"\n{'='*70}")
print(f"  COMPARISON SUMMARY")
print(f"{'='*70}")
print(f"  {'Method':<20} {'Accuracy':<18} {'Memory':<15} {'Backprop':<10}")
print(f"  {'-'*63}")
print(f"  {'R-STDP':<20} {rstdp_mean:5.1f} +/- {rstdp_std:3.1f}%    {rstdp_mem:6.1f} MB      No")
print(f"  {'BPTT (SG)':<20} {bptt_mean:5.1f} +/- {bptt_std:3.1f}%    {bptt_mem:6.1f} MB      Yes")
print(f"  {'-'*63}")
mem_ratio = bptt_mem / rstdp_mem
print(f"  Memory ratio (BPTT/R-STDP): {mem_ratio:.1f}x")
print(f"  R-STDP is Loihi 2 compatible. BPTT is not.")
print(f"{'='*70}")

# ============================================================
#  PART 4: Plot
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: Accuracy comparison
ax = axes[0]
methods = ["R-STDP", "BPTT"]
means = [rstdp_mean, bptt_mean]
stds = [rstdp_std, bptt_std]
colors = ["#2196F3", "#FF5722"]
bars = ax.bar(methods, means, yerr=stds, color=colors, capsize=8, width=0.5, edgecolor="black")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Accuracy (5 seeds, mean +/- std)")
ax.set_ylim(85, 100)
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f"{m:.1f}+/-{s:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Plot 2: Memory comparison
ax = axes[1]
bars = ax.bar(methods, [rstdp_mem, bptt_mem], color=colors, width=0.5, edgecolor="black")
ax.set_ylabel("Peak GPU Memory (MB)")
ax.set_title(f"Memory ({mem_ratio:.1f}x difference)")
for bar, m in zip(bars, [rstdp_mem, bptt_mem]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{m:.0f} MB", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Plot 3: Per-seed scatter
ax = axes[2]
seeds = [42, 123, 456, 789, 2024]
ax.scatter(seeds, rstdp_accs, color="#2196F3", s=100, zorder=5, label="R-STDP", edgecolors="black")
ax.scatter(seeds, bptt_accs, color="#FF5722", s=100, zorder=5, label="BPTT", edgecolors="black")
ax.axhline(y=rstdp_mean, color="#2196F3", linestyle="--", alpha=0.5)
ax.axhline(y=bptt_mean, color="#FF5722", linestyle="--", alpha=0.5)
ax.set_xlabel("Random Seed")
ax.set_ylabel("Test Accuracy (%)")
ax.set_title("Per-Seed Consistency")
ax.set_ylim(85, 100)
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/rstdp_vs_bptt.png", dpi=150, bbox_inches="tight")
print(f"\n  Saved: figures/rstdp_vs_bptt.png")
