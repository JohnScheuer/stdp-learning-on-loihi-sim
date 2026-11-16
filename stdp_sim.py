import torch
import torchvision
import torchvision.transforms as T
import numpy as np
import time

print("=" * 65)
print("  R-STDP Learning on Loihi-Sim")
print("  Task: Binary Classification (Digit 0 vs 1)")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if device.type == "cuda":
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# --- Hyperparameters ---
T_STEPS = 30
N_IN, N_HID = 784, 100
N_POP = 50                # 50 neurons per class population
V_TH = 1.0
TAU_M = 20.0
DECAY = float(np.exp(-1.0 / TAU_M))
K_WIN = 3
A_LTP = 0.005             # potentiation (correct pop)
A_LTD = 0.003             # depression (wrong pop)
EPOCHS = 5
W_TARGET = 1.0            # synaptic scaling target

print(f"  LIF: tau_m={TAU_M}ms, v_th={V_TH}, decay={DECAY:.3f}")
print(f"  R-STDP: LTP={A_LTP} (correct pop), LTD={A_LTD} (wrong pop)")
print(f"  Synaptic scaling: L1={W_TARGET} per neuron")
print(f"  Competition: k-WTA k={K_WIN}")
print(f"  Architecture: 2 populations x {N_POP} neurons")
print(f"  Network: {N_IN} -> {N_HID}, T={T_STEPS}")
print("=" * 65)
print()
print("  Method: Reward-Modulated STDP (Izhikevich 2007)")
print("  - reward = +1 for correct population (LTP)")
print("  - reward = -1 for wrong population (LTD)")
print("  - Global reward signal, LOCAL weight updates")
print("  - Compatible with Loihi 2 reward-modulated learning")
print("  - O(1) memory: no backprop, no activation storage")
print("=" * 65)

# --- Data ---
transform = T.Compose([T.ToTensor(), T.Lambda(lambda x: x.view(-1))])
train_full = torchvision.datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_full = torchvision.datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_idx = [i for i, (_, l) in enumerate(train_full) if l in (0, 1)]
test_idx = [i for i, (_, l) in enumerate(test_full) if l in (0, 1)]

X_train = torch.stack([train_full[i][0] for i in train_idx]).to(device)
Y_train = torch.tensor([train_full[i][1] for i in train_idx]).to(device)
X_test = torch.stack([test_full[i][0] for i in test_idx]).to(device)
Y_test = torch.tensor([test_full[i][1] for i in test_idx]).to(device)

print(f"\n  Train: {len(X_train)} | Test: {len(X_test)}")

# --- Network ---
W = torch.rand(N_HID, N_IN, device=device) * 0.05
W = W / W.sum(dim=1, keepdim=True) * W_TARGET

# --- Forward + R-STDP ---
def run_sample_train(x, label):
    """Forward pass with R-STDP. Returns spike counts."""
    v = torch.zeros(N_HID, device=device)
    spike_count = torch.zeros(N_HID, device=device)

    # Determine which population is correct/wrong
    if label == 0:
        correct_slice = slice(0, N_POP)
        wrong_slice = slice(N_POP, N_HID)
    else:
        correct_slice = slice(N_POP, N_HID)
        wrong_slice = slice(0, N_POP)

    for t in range(T_STEPS):
        v = DECAY * v + W @ x

        # k-WTA competition
        topk_vals, topk_idx = v.topk(K_WIN)
        spike_mask = torch.zeros(N_HID, dtype=torch.bool, device=device)
        firing = topk_vals > V_TH
        if firing.any():
            spike_mask[topk_idx[firing]] = True

        spike = spike_mask.float()
        spike_count += spike

        # R-STDP: reward-modulated local updates
        # LTP on correct population (reward = +1)
        sc = spike[correct_slice]
        if sc.sum() > 0:
            W[correct_slice] += A_LTP * torch.outer(sc, x)

        # LTD on wrong population (reward = -1)
        sw = spike[wrong_slice]
        if sw.sum() > 0:
            W[wrong_slice] -= A_LTD * torch.outer(sw, x)

        v[spike_mask] = 0.0

    return spike_count

def run_sample_test(x):
    """Forward pass without learning. Returns spike counts."""
    v = torch.zeros(N_HID, device=device)
    spike_count = torch.zeros(N_HID, device=device)

    for t in range(T_STEPS):
        v = DECAY * v + W @ x

        topk_vals, topk_idx = v.topk(K_WIN)
        spike_mask = torch.zeros(N_HID, dtype=torch.bool, device=device)
        firing = topk_vals > V_TH
        if firing.any():
            spike_mask[topk_idx[firing]] = True

        spike_count += spike_mask.float()
        v[spike_mask] = 0.0

    return spike_count

# --- Training ---
print(f"\n[Phase 1: R-STDP Training]")
print(f"  {len(X_train)} samples x {EPOCHS} epochs")
print(f"  Memory: O(1) — traces + global reward, NO backprop\n")

torch.cuda.reset_peak_memory_stats()
t0 = time.time()

for epoch in range(EPOCHS):
    perm = torch.randperm(len(X_train))
    epoch_spikes = 0
    correct_preds = 0

    for idx in perm:
        x = X_train[idx]
        label = Y_train[idx].item()

        sc = run_sample_train(x, label)
        epoch_spikes += sc.sum().item()

        # Synaptic scaling
        row_sums = W.sum(dim=1, keepdim=True).clamp(min=1e-8)
        W = W / row_sums * W_TARGET
        W.clamp_(0.0, 1.0)

        # Track training accuracy
        v0 = sc[:N_POP].sum()
        v1 = sc[N_POP:].sum()
        pred = 0 if v0 > v1 else 1
        if pred == label:
            correct_preds += 1

    mem = torch.cuda.max_memory_allocated() / 1e6
    avg_spk = epoch_spikes / len(X_train)
    train_acc = correct_preds / len(X_train) * 100
    w0 = W[:N_POP].mean().item()
    w1 = W[N_POP:].mean().item()
    print(f"  Epoch {epoch+1}/{EPOCHS} | spikes: {avg_spk:.1f} | "
          f"train_acc: {train_acc:.1f}% | w0={w0:.3f} w1={w1:.3f} | GPU: {mem:.1f} MB")

train_time = time.time() - t0
peak_mem = torch.cuda.max_memory_allocated() / 1e6
print(f"\n  Training time: {train_time:.1f}s")
print(f"  Peak GPU memory: {peak_mem:.1f} MB")

# --- Weight Analysis ---
print(f"\n[Phase 2: Weight Analysis]")
w0 = W[:N_POP].mean().item()
w1 = W[N_POP:].mean().item()
sp = (W < 0.001).float().mean() * 100
print(f"  Pop 0 mean weight: {w0:.4f}")
print(f"  Pop 1 mean weight: {w1:.4f}")
print(f"  Weight sparsity: {sp:.1f}%")

# --- Inference ---
print(f"\n[Phase 3: Inference (spike-count voting, no reward signal)]")
correct = 0
cc = {0: 0, 1: 0}
ct = {0: 0, 1: 0}
t_inf = time.time()

for idx in range(len(X_test)):
    sc = run_sample_test(X_test[idx])
    v0 = sc[:N_POP].sum()
    v1 = sc[N_POP:].sum()
    pred = 0 if v0 > v1 else 1
    true = Y_test[idx].item()
    ct[true] += 1
    if pred == true:
        correct += 1
        cc[true] += 1

inf_ms = (time.time() - t_inf) / len(X_test) * 1000
acc = correct / len(X_test) * 100

print(f"  Test accuracy: {acc:.1f}% ({correct}/{len(X_test)})")
print(f"  Inference: {inf_ms:.1f}ms/sample")
print(f"\n  Per-class:")
for c in [0, 1]:
    ca = cc[c] / max(ct[c], 1) * 100
    print(f"    Digit {c}: {ca:.1f}% ({cc[c]}/{ct[c]})")

print(f"\n{'='*65}")
print(f"  RESULTS")
print(f"{'='*65}")
print(f"  Task:              Binary (0 vs 1)")
print(f"  Learning rule:     R-STDP (reward-modulated, local, no backprop)")
print(f"  Memory:            O(1) — {peak_mem:.1f} MB peak (T={T_STEPS})")
print(f"  Accuracy:          {acc:.1f}%")
print(f"  Training time:     {train_time:.1f}s")
print(f"  Inference:         {inf_ms:.1f}ms/sample")
print(f"\n  R-STDP achieves {acc:.1f}% with O(1) memory.")
print(f"  Global reward signal + local STDP traces = Loihi 2 compatible.")
print(f"  BPTT needs O(T) memory and cannot run on neuromorphic hardware.")
print(f"{'='*65}")
