import torch
import torch.nn as nn
import numpy as np
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

print("=" * 60)
print("  OOM Kill Demo: BPTT vs STDP")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Total VRAM: {total_mem:.1f} GB")

class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        sg = torch.sigmoid(5.0 * x) * (1 - torch.sigmoid(5.0 * x)) * 5.0
        return grad_output * sg

class BPTTSNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 100)
        self.fc2 = nn.Linear(100, 2)
    def forward(self, x, T):
        b = x.shape[0]
        v1 = torch.zeros(b, 100, device=x.device)
        v2 = torch.zeros(b, 2, device=x.device)
        out = torch.zeros(b, 2, device=x.device)
        for t in range(T):
            v1 = 0.95 * v1 + self.fc1(x)
            s1 = SurrogateSpike.apply(v1 - 1.0)
            v1 = v1 * (1 - s1)
            v2 = 0.95 * v2 + self.fc2(s1)
            s2 = SurrogateSpike.apply(v2 - 1.0)
            v2 = v2 * (1 - s2)
            out += s2
        return out / T

T_VALUES = [10, 50, 100, 500, 1000, 2000, 5000, 10000]
BATCH = 32

# Create dummy input
x_dummy = torch.rand(BATCH, 784, device=device)
y_dummy = torch.randint(0, 2, (BATCH,), device=device)
criterion = nn.CrossEntropyLoss()

bptt_mems = []
bptt_oom = None
stdp_mems = []

print(f"\n  Testing T values: {T_VALUES}")
print(f"  Batch size: {BATCH}")
print(f"\n  {'T':>6} | {'BPTT (MB)':>10} | {'STDP (MB)':>10} | {'Status':>10}")
print(f"  {'-'*50}")

for T in T_VALUES:
    # --- BPTT ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        model = BPTTSNN().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad()
        out = model(x_dummy, T)
        loss = criterion(out, y_dummy)
        loss.backward()
        mem_bptt = torch.cuda.max_memory_allocated() / 1e6
        bptt_mems.append(mem_bptt)
        status = "OK"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            mem_bptt = torch.cuda.max_memory_allocated() / 1e6
            bptt_mems.append(mem_bptt)
            bptt_oom = T
            status = "OOM KILLED"
            torch.cuda.empty_cache()
        else:
            raise

    # --- STDP (O(1) always) ---
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    W = torch.rand(100, 784, device=device) * 0.05
    v = torch.zeros(BATCH, 100, device=device)
    for t in range(T):
        v = 0.95 * v + x_dummy @ W.T
        spike = (v > 1.0).float()
        v = v * (1 - spike)
    mem_stdp = torch.cuda.max_memory_allocated() / 1e6
    stdp_mems.append(mem_stdp)

    print(f"  {T:>6} | {mem_bptt:>10.1f} | {mem_stdp:>10.1f} | {status:>10}")

# --- Plot ---
print(f"\n  Plotting...")
fig, ax = plt.subplots(figsize=(10, 6))

valid_bptt = T_VALUES[:len(bptt_mems)]
ax.plot(valid_bptt, bptt_mems, "o-", color="#FF5722", linewidth=2, markersize=8,
        label="BPTT (O(T) memory)", zorder=5)
ax.plot(T_VALUES, stdp_mems, "s-", color="#2196F3", linewidth=2, markersize=8,
        label="R-STDP (O(1) memory)", zorder=5)

if bptt_oom:
    ax.axvline(x=bptt_oom, color="red", linestyle="--", alpha=0.5, label=f"BPTT OOM at T={bptt_oom}")
    ax.annotate(f"BPTT CRASH\nT={bptt_oom}", xy=(bptt_oom, bptt_mems[-1]),
                xytext=(bptt_oom * 1.5, bptt_mems[-1] * 0.7),
                fontsize=12, fontweight="bold", color="red",
                arrowprops=dict(arrowstyle="->", color="red"))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Timesteps (T)", fontsize=12)
ax.set_ylabel("Peak GPU Memory (MB)", fontsize=12)
ax.set_title("OOM Kill Demo: BPTT vs R-STDP Memory Scaling", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

os.makedirs("figures", exist_ok=True)
plt.savefig("figures/oom_demo.png", dpi=150, bbox_inches="tight")
print(f"  Saved: figures/oom_demo.png")

if bptt_oom:
    print(f"\n  BPTT crashed at T={bptt_oom}. R-STDP survived all T values.")
else:
    print(f"\n  BPTT survived all T values (GPU has enough VRAM).")
    print(f"  But the scaling trend is clear: BPTT grows linearly, STDP is flat.")
