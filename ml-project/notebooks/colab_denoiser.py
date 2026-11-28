"""
Audelle — Colab denoiser (Conv-GRU mask)

Paste this whole file into a Google Colab notebook, or upload it as a notebook
and run all cells. Before running: Runtime → Change runtime type → Hardware
accelerator = T4 GPU.

What it does:
  1. Loads VoiceBank-DEMAND at 16 kHz from HuggingFace (no manual upload).
  2. Splits by SPEAKER (4 speakers held out) so validation is honest.
  3. Trains a small Conv-GRU model that predicts a keep-or-remove mask on the
     STFT magnitude; the noisy phase is reused.
  4. Evaluates SI-SDR improvement and STOI on the held-out speakers.
  5. Saves the weights to your Google Drive.

QUICK = True verifies the whole loop in ~5 minutes before you commit to the
full run (a few hours on a T4).

This mirrors the CPU-trained diagnoser already in the repo
(ml-project/scripts/train_diagnoser.py); the denoiser is the GPU-bound tier.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------
# 0. GPU check
# --------------------------------------------------------------------------
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU :", torch.cuda.get_device_name(0))
else:
    raise SystemExit("No GPU. Runtime → Change runtime type → T4 GPU, then re-run.")

DEVICE = torch.device("cuda")

# --------------------------------------------------------------------------
# 1. Config
# --------------------------------------------------------------------------
QUICK = False            # True = ~5 min smoke run, False = full run
SR = 16000
N_FFT = 512
HOP = 128
SECONDS = 3.0
BATCH = 16
EPOCHS = 5 if QUICK else 40
LR = 3e-4
VAL_SPEAKERS = 4

# --------------------------------------------------------------------------
# 2. Data (HuggingFace mirror of VoiceBank-DEMAND, already 16 kHz)
# --------------------------------------------------------------------------
try:
    from datasets import load_dataset
except ImportError:
    import subprocess, sys
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "datasets"])
    from datasets import load_dataset

print("loading dataset ...")
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="train")
ids = ds["id"]  # e.g. "p226_001"

# Speaker-disjoint split: never let a voice appear on both sides.
speakers = sorted({i.split("_")[0] for i in ids})
rng = random.Random(0)
val_speakers = set(rng.sample(speakers, VAL_SPEAKERS))
train_idx = [i for i in range(len(ds)) if ids[i].split("_")[0] not in val_speakers]
val_idx = [i for i in range(len(ds)) if ids[i].split("_")[0] in val_speakers]
if QUICK:
    train_idx = train_idx[:320]
    val_idx = val_idx[:96]
print(f"train {len(train_idx)}  val {len(val_idx)}  "
      f"(held-out speakers: {sorted(val_speakers)})")

N_SAMPLES = int(SECONDS * SR)


def get_clip(idx: int):
    clean = ds[idx]["clean"]["array"].astype(np.float32)
    noisy = ds[idx]["noisy"]["array"].astype(np.float32)
    n = min(len(clean), len(noisy))
    if n >= N_SAMPLES:
        s = random.randint(0, n - N_SAMPLES)
        return clean[s:s + N_SAMPLES], noisy[s:s + N_SAMPLES]
    clean = np.pad(clean, (0, N_SAMPLES - len(clean)))
    noisy = np.pad(noisy, (0, N_SAMPLES - len(noisy)))
    return clean[:N_SAMPLES], noisy[:N_SAMPLES]


def make_batch(indices):
    c, n = [], []
    for i in indices:
        a, b = get_clip(i)
        c.append(a)
        n.append(b)
    return (torch.from_numpy(np.stack(c)).to(DEVICE),
            torch.from_numpy(np.stack(n)).to(DEVICE))


# --------------------------------------------------------------------------
# 3. STFT + iSTFT
# --------------------------------------------------------------------------
WINDOW = torch.hann_window(N_FFT).to(DEVICE)


def stft(x):
    return torch.stft(x, N_FFT, HOP, N_FFT, WINDOW, return_complex=True)


def istft(X, length=N_SAMPLES):
    return torch.istft(X, N_FFT, HOP, N_FFT, WINDOW, length=length)


# --------------------------------------------------------------------------
# 4. Model: conv encoder -> 2-layer GRU over time -> conv decoder -> mask
# --------------------------------------------------------------------------
class Denoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.gru = nn.GRU(32 * (N_FFT // 2 + 1), 256, num_layers=2,
                          batch_first=True)
        self.dec = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 1, 3, padding=1), nn.Sigmoid(),
        )

    def forward(self, mag):  # (B,1,F,T)
        h = self.enc(mag)                       # (B,32,F,T)
        B, C, F, T = h.shape
        h = h.permute(0, 3, 1, 2).reshape(B, T, C * F)  # (B,T,C*F)
        h, _ = self.gru(h)                      # (B,T,256)
        h = h.reshape(B, T, C, F).permute(0, 2, 3, 1)   # (B,C,F,T)
        return self.dec(h)                      # (B,1,F,T)


model = Denoiser().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters())
print(f"parameters: {n_params:,}")
opt = torch.optim.Adam(model.parameters(), lr=LR)


# --------------------------------------------------------------------------
# 5. Train / eval
# --------------------------------------------------------------------------
def enhance(noisy):
    mag = stft(noisy).abs().unsqueeze(1)        # (B,1,F,T)
    mask = model(mag)                            # (B,1,F,T)
    phase = torch.angle(stft(noisy)).unsqueeze(1)  # (B,1,F,T)
    out_mag = mag * mask
    out = out_mag.squeeze(1) * torch.exp(1j * phase.squeeze(1))
    return istft(out), mask, mag


def si_sdr(ref, est):
    ref, est = ref.detach(), est.detach()
    alpha = (ref * est).sum(dim=1, keepdim=True) / (ref * ref).sum(dim=1, keepdim=True).clamp(1e-8)
    proj = alpha * ref
    noise = est - proj
    val = 10 * torch.log10((proj * proj).sum(dim=1) / (noise * noise).sum(dim=1).clamp(1e-8))
    return val.mean().item()


@torch.no_grad()
def evaluate(indices):
    model.eval()
    sisdr_list, l1_list = [], []
    for i in range(0, len(indices), BATCH):
        c, n = make_batch(indices[i:i + BATCH])
        est, mask, mag = enhance(n)
        sisdr_list.append(si_sdr(c, est))
        # reference mask is |clean| / |noisy|, clipped to [0,1]
        true_m = (stft(c).abs() / mag.squeeze(1).clamp(1e-8)).clamp(0, 1)
        l1_list.append(F.l1_loss(mask.squeeze(1), true_m).item())
    model.train()
    return float(np.mean(sisdr_list)), float(np.mean(l1_list))


# Baseline: SI-SDR of the noisy input itself (enhancement gain is vs this).
def noisy_sisdr(indices):
    vals = []
    for i in range(0, len(indices), BATCH):
        c, n = make_batch(indices[i:i + BATCH])
        vals.append(si_sdr(c, n))
    return float(np.mean(vals))


print(f"\nbaseline (noisy input) SI-SDR: {noisy_sisdr(val_idx[:96]):.2f} dB\n")

for epoch in range(EPOCHS):
    model.train()
    order = train_idx[:]
    random.shuffle(order)
    total, n = 0.0, 0
    for i in range(0, len(order), BATCH):
        c, n = make_batch(order[i:i + BATCH])
        est, mask, mag = enhance(n)
        true_m = (stft(c).abs() / mag.squeeze(1).clamp(1e-8)).clamp(0, 1)
        loss = F.l1_loss(mask.squeeze(1), true_m)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss.item() * len(c)
        n += len(c)
    if epoch % 1 == 0:
        v_s, v_l = evaluate(val_idx[:96])
        print(f"epoch {epoch+1:2d}/{EPOCHS}  loss {total/max(1,n):.4f}  "
              f"val SI-SDR {v_s:.2f} dB  val L1 {v_l:.4f}")

# --------------------------------------------------------------------------
# 6. Save to Drive
# --------------------------------------------------------------------------
from google.colab import drive  # noqa: E402

drive.mount("/content/drive")
out_dir = "/content/drive/MyDrive/audelle"
os.makedirs(out_dir, exist_ok=True)
torch.save(model.state_dict(), os.path.join(out_dir, "denoiser.pt"))
print(f"\nsaved {out_dir}/denoiser.pt")
