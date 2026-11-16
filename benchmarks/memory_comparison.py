#!/usr/bin/env python3
"""
Memory Comparison: STDP vs BPTT (Surrogate Gradients)
======================================================
Measures GPU memory as a function of timesteps T.

BPTT: O(T) — stores the full computation graph for backward pass.
STDP: O(1) — only eligibility traces, updated in-place.

Uses the same architecture as snn-surrogate-gradients (784→512→128→10)
with batch=64 to make the O(T) scaling visible above CUDA baseline.
"""

import sys, os, gc
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LIFNeuron(nn.Module):
    def __init__(self, v_th=1.0, decay=0.95):
        super().__init__()
        self.v_th = v_th
        self.decay = decay
        self.v = None

    def reset(self):
        self.v = None

    def forward(self, x):
        if self.v is None:
            self.v = torch.zeros_like(x)
        self.v = self.decay * self.v + x
        spike = (self.v >= self.v_th).float()
        surrogate = torch.sigmoid(5.0 * (self.v - self.v_th))
        spike = spike + surrogate - surrogate.detach()
        self.v = self.v * (1.0 - spike.detach())
        return spike


class SNN_BPTT(nn.Module):
    """3-layer SNN. Memory O(T): stores graph for all timesteps."""
    def __init__(self, T):
        super().__init__()
        self.T = T
        self.fc1 = nn.Linear(784, 512, bias=False)
        self.fc2 = nn.Linear(512, 128, bias=False)
        self.fc3 = nn.Linear(128, 10, bias=False)
        self.lif1 = LIFNeuron()
        self.lif2 = LIFNeuron()
        self.lif3 = LIFNeuron()

    def forward(self, x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        out = torch.zeros(b, 10, device=x.device)
        for t in range(self.T):
            x_s = (torch.rand_like(x_flat) < x_flat.abs()).float()
            h1 = self.lif1(self.fc1(x_s))
            h2 = self.lif2(self.fc2(h1))
            h3 = self.lif3(self.fc3(h2))
            out += h3
        return out / self.T


def measure_bptt(T, batch=64):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = SNN_BPTT(T).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.rand(batch, 784, device=DEVICE)
    y = torch.randint(0, 10, (batch,), device=DEVICE)

    model.lif1.reset(); model.lif2.reset(); model.lif3.reset()
    loss = nn.CrossEntropyLoss()(model(x), y)
    loss.backward()
    opt.step()

    mem = torch.cuda.max_memory_allocated() / 1024**2
    del model, opt, x, y, loss
    gc.collect(); torch.cuda.empty_cache()
    return mem


def measure_stdp(T, n_in=784, n_hid=512):
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    w = torch.rand(n_in, n_hid, device=DEVICE) * 0.1
    tr_pre = torch.zeros(n_in, device=DEVICE)
    tr_post = torch.zeros(n_hid, device=DEVICE)
    v = torch.zeros(n_hid, device=DEVICE)

    for t in range(T):
        pre = (torch.rand(n_in, device=DEVICE) < 0.1).float()
        tr_pre = tr_pre * 0.95 + pre
        v = v * 0.95 + pre @ w
        post = (v >= 1.0).float()
        v = v * (1.0 - post)
        tr_post = tr_post * 0.95 + post
        w += 0.005 * torch.outer(pre, tr_post) - 0.006 * torch.outer(tr_pre, post)
        w.clamp_(0.0, 1.0)

    mem = torch.cuda.max_memory_allocated() / 1024**2
    del w, tr_pre, tr_post, v
    gc.collect(); torch.cuda.empty_cache()
    return mem


def main():
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 65)

    t_values = [10, 25, 50, 100, 250, 500, 1000]
    print(f"\n{'T':>6} | {'BPTT (MB)':>12} | {'STDP (MB)':>12} | {'Ratio':>8}")
    print("-" * 48)

    bptt_m, stdp_m = [], []
    for T in t_values:
        b = measure_bptt(T)
        s = measure_stdp(T)
        bptt_m.append(b); stdp_m.append(s)
        print(f"{T:6d} | {b:11.1f} | {s:11.1f} | {b/max(s,0.1):7.1f}x")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t_values, bptt_m, "o-", color="#F44336", lw=2.5, ms=8,
            label="BPTT (Surrogate Gradients)", zorder=5)
    ax.plot(t_values, stdp_m, "s-", color="#4CAF50", lw=2.5, ms=8,
            label="STDP (Local Learning)", zorder=5)

    ax.set_xlabel("Timesteps (T)", fontsize=13)
    ax.set_ylabel("Peak GPU Memory (MB)", fontsize=13)
    ax.set_title("Memory Scaling: BPTT vs STDP (784→512→128→10, batch=64)",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)

    for i, T in enumerate(t_values):
        ax.annotate(f"{bptt_m[i]:.0f}", (T, bptt_m[i]),
                    xytext=(5, 8), textcoords="offset points", fontsize=8)
        ax.annotate(f"{stdp_m[i]:.0f}", (T, stdp_m[i]),
                    xytext=(5, -14), textcoords="offset points", fontsize=8)

    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/memory_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"\nPlot saved: figures/memory_comparison.png")

    print(f"\n{'=' * 65}")
    print("ANALYSIS")
    print(f"{'=' * 65}")
    print(f"BPTT: {bptt_m[-1]/bptt_m[0]:.1f}x growth (T={t_values[0]} to {t_values[-1]})")
    print(f"STDP: {stdp_m[-1]/stdp_m[0]:.1f}x growth (T={t_values[0]} to {t_values[-1]})")
    print(f"At T={t_values[-1]}: BPTT uses {bptt_m[-1]/max(stdp_m[-1],0.1):.1f}x more memory")
    print(f"\nBPTT stores the computation graph for all T timesteps → O(T).")
    print(f"STDP maintains eligibility traces updated in-place → O(1).")
    print(f"On Loihi 2, there is no backprop engine. STDP is the native rule.")


if __name__ == "__main__":
    main()
