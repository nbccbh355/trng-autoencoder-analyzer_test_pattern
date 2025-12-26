
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
trng.py — TRNG test-pattern analyzer (bit-planes, heatmaps, SP800-90B health tests, optional CNN autoencoder residuals)

Implements the key ideas from the provided slides:
- Bit-planes: expand TRNG bytes into 8 layers (LSB..MSB), visualize as 64x64 images.
- Heatmaps: byte transition co-occurrence (256x256) to reveal repetition/correlation structure.
- SP800-90B health tests (lightweight online checks): RCT (repetition count), APT (adaptive proportion).
- Autoencoder residuals (optional): CNN autoencoder learns "normal" and uses per-frame MSE as anomaly score.

Input format:
- Raw binary file of bytes (recommended), or
- Hex text file (whitespace-separated hex bytes), or
- "-" to read from stdin as hex.

Frames:
- Default frame is 64x64 bytes = 4096 bytes per frame.
- Each frame yields 8 bit-planes of shape (8,64,64).

Outputs:
- out_dir/bitplanes/frame_XXXX_bitY.png
- out_dir/heatmap/byte_transition.png
- out_dir/scores.csv  (health tests + optional AE scores)
- out_dir/ae_residuals/frame_XXXX_residual.png  (if AE enabled)

Usage examples:
  python trng.py --input trng.bin --out out --frames 50
  python trng.py --input trng.bin --out out --train-frames 30 --frames 100
  python trng.py --input trng.hex --hex --out out --no-ae
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt

# Optional AE (PyTorch). If unavailable, AE features are disabled gracefully.
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False


# -----------------------------
# Utilities: IO and framing
# -----------------------------
def read_bytes_from_file(path: str, hex_mode: bool) -> bytes:
    if path == "-":
        # stdin hex
        text = sys.stdin.read()
        return parse_hex_text(text)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    if not hex_mode:
        with open(path, "rb") as f:
            return f.read()

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return parse_hex_text(f.read())


def parse_hex_text(text: str) -> bytes:
    # Accept "AA FF 01", "0xAA,0xFF", newlines, etc.
    tokens = []
    for t in text.replace(",", " ").replace("0x", " ").split():
        t = t.strip()
        if not t:
            continue
        # keep only hex chars
        t = "".join(ch for ch in t if ch in "0123456789abcdefABCDEF")
        if t == "":
            continue
        if len(t) == 1:
            t = "0" + t
        if len(t) != 2:
            # If someone gave longer tokens, split into bytes.
            if len(t) % 2 != 0:
                t = "0" + t
            for i in range(0, len(t), 2):
                tokens.append(t[i:i+2])
        else:
            tokens.append(t)
    arr = bytes(int(x, 16) for x in tokens)
    return arr


def to_frames(data: bytes, frame_h: int, frame_w: int, max_frames: Optional[int]) -> np.ndarray:
    frame_bytes = frame_h * frame_w
    total = len(data) // frame_bytes
    if total == 0:
        raise ValueError(f"Not enough data: need at least {frame_bytes} bytes for one {frame_h}x{frame_w} frame.")
    if max_frames is not None:
        total = min(total, max_frames)
    trimmed = data[: total * frame_bytes]
    frames = np.frombuffer(trimmed, dtype=np.uint8).reshape(total, frame_h, frame_w)
    return frames


# -----------------------------
# Bit-planes & heatmaps
# -----------------------------
def bytes_to_bitplanes(frame_u8: np.ndarray) -> np.ndarray:
    """
    frame_u8: (H,W) uint8
    returns: (8,H,W) uint8 with values {0,1} for bit0..bit7
    """
    H, W = frame_u8.shape
    planes = np.zeros((8, H, W), dtype=np.uint8)
    for b in range(8):
        planes[b] = (frame_u8 >> b) & 1
    return planes


def save_bitplanes(planes: np.ndarray, out_dir: str, frame_idx: int):
    os.makedirs(out_dir, exist_ok=True)
    # Save each plane as 0/255 image
    for b in range(8):
        img = (planes[b] * 255).astype(np.uint8)
        fp = os.path.join(out_dir, f"frame_{frame_idx:04d}_bit{b}.png")
        plt.figure()
        plt.imshow(img, interpolation="nearest")
        plt.axis("off")
        plt.title(f"Frame {frame_idx} Bit-plane {b} (LSB=0)")
        plt.savefig(fp, dpi=150, bbox_inches="tight")
        plt.close()


def byte_transition_heatmap(all_bytes: np.ndarray) -> np.ndarray:
    """
    all_bytes: flat uint8 array
    returns: 256x256 matrix M where M[i,j] counts transitions i->j
    """
    x = all_bytes.astype(np.uint16)
    if x.size < 2:
        return np.zeros((256, 256), dtype=np.float64)
    a = x[:-1]
    b = x[1:]
    M = np.zeros((256, 256), dtype=np.uint64)
    # vectorized bin counting
    idx = a * 256 + b
    counts = np.bincount(idx, minlength=256 * 256).astype(np.uint64)
    M[:] = counts.reshape(256, 256)
    return M.astype(np.float64)


def save_heatmap(M: np.ndarray, out_path: str, log_scale: bool = True):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 7))
    data = np.log1p(M) if log_scale else M
    plt.imshow(data, interpolation="nearest")
    plt.title("Byte Transition Heatmap (log1p counts)" if log_scale else "Byte Transition Heatmap")
    plt.xlabel("Next byte")
    plt.ylabel("Current byte")
    plt.colorbar()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# SP800-90B health tests (simplified)
# -----------------------------
@dataclass
class HealthTestResult:
    rct_max_run: int
    rct_failed: bool
    apt_window: int
    apt_max_count: int
    apt_min_count: int
    apt_failed: bool


def bytes_to_bits(data_u8: np.ndarray, bit_order_lsb_first: bool = True) -> np.ndarray:
    """
    Convert bytes to bits: returns flat array of 0/1.
    By default use LSB-first per byte for consistency with bit-plane notion.
    """
    # np.unpackbits is MSB-first by default, so we adjust.
    bits = np.unpackbits(data_u8, bitorder="little" if bit_order_lsb_first else "big")
    return bits.astype(np.uint8)


def repetition_count_test(bits: np.ndarray) -> int:
    """
    Returns maximum run length of identical bits.
    """
    if bits.size == 0:
        return 0
    max_run = 1
    run = 1
    prev = bits[0]
    for v in bits[1:]:
        if v == prev:
            run += 1
            if run > max_run:
                max_run = run
        else:
            prev = v
            run = 1
    return int(max_run)


def adaptive_proportion_test(bits: np.ndarray, window: int) -> Tuple[int, int, int]:
    """
    Sliding window count of ones. Returns (max_count, min_count, window).
    """
    if bits.size < window or window <= 0:
        return 0, 0, window
    # Efficient sliding sum
    x = bits.astype(np.int32)
    cumsum = np.cumsum(x)
    # sum over [i, i+window)
    sums = cumsum[window - 1:] - np.concatenate(([0], cumsum[:-window]))
    return int(sums.max()), int(sums.min()), window


def run_health_tests(data_bytes: np.ndarray, rct_cutoff: int, apt_window: int, apt_hi: int, apt_lo: int) -> HealthTestResult:
    bits = bytes_to_bits(data_bytes.reshape(-1))
    max_run = repetition_count_test(bits)
    rct_failed = max_run >= rct_cutoff

    apt_max, apt_min, w = adaptive_proportion_test(bits, apt_window)
    apt_failed = (apt_max >= apt_hi) or (apt_min <= apt_lo)

    return HealthTestResult(
        rct_max_run=max_run,
        rct_failed=rct_failed,
        apt_window=w,
        apt_max_count=apt_max,
        apt_min_count=apt_min,
        apt_failed=apt_failed
    )


# -----------------------------
# CNN Autoencoder (optional, PyTorch)
# Architecture matches the slide:
# Input [8,64,64] -> encoder convs/pools -> bottleneck [128,16,16]
# decoder convtranspose -> output [8,64,64], loss=MSE, anomaly score=per-frame MSE
# -----------------------------
if TORCH_OK:
    class AECNN(nn.Module):
        def __init__(self):
            super().__init__()
            # Encoder
            self.enc = nn.Sequential(
                nn.Conv2d(8, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 64->32

                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32->16

                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
            # Decoder
            self.dec = nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),  # 16->32
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),   # 32->64
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),

                nn.Conv2d(32, 8, kernel_size=3, padding=1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            z = self.enc(x)
            y = self.dec(z)
            return y


def train_ae(bitplane_frames: np.ndarray, train_frames: int, epochs: int, lr: float, device: str) -> "AECNN":
    """
    bitplane_frames: (N,8,64,64) float32 in {0,1}
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available; cannot train AE.")
    N = bitplane_frames.shape[0]
    n_train = min(train_frames, N)
    x_train = bitplane_frames[:n_train]

    model = AECNN().to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="mean")

    model.train()
    for ep in range(1, epochs + 1):
        # simple full-batch (small N typical); for large N you can add mini-batches
        xb = torch.from_numpy(x_train).to(device)
        opt.zero_grad(set_to_none=True)
        yb = model(xb)
        loss = loss_fn(yb, xb)
        loss.backward()
        opt.step()
        if ep == 1 or ep == epochs or ep % max(1, (epochs // 5)) == 0:
            print(f"[AE] epoch {ep}/{epochs} loss={loss.item():.6f}")
    return model


def ae_scores_and_residuals(model: "AECNN", bitplane_frames: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      scores: (N,) per-frame MSE
      residual_maps: (N,64,64) mean over channels |x - y|
    """
    if not TORCH_OK:
        raise RuntimeError("PyTorch not available; cannot run AE.")
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(bitplane_frames).to(device)
        yb = model(xb)
        diff = (yb - xb)
        mse = (diff * diff).mean(dim=(1, 2, 3)).cpu().numpy()
        # residual map: mean abs over channels -> 64x64
        res = diff.abs().mean(dim=1).cpu().numpy()
    return mse, res


def save_residual_map(res_map: np.ndarray, out_path: str, title: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure()
    plt.imshow(res_map, interpolation="nearest")
    plt.axis("off")
    plt.title(title)
    plt.colorbar()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="TRNG analyzer: bit-planes, heatmaps, SP800-90B health tests, optional CNN AE residuals.")
    ap.add_argument("--input", required=True, help="Input file path (.bin) or .hex if --hex. Use '-' to read hex from stdin.")
    ap.add_argument("--hex", action="store_true", help="Treat input as hex text (whitespace-separated hex bytes).")
    ap.add_argument("--out", default="out", help="Output directory.")
    ap.add_argument("--h", type=int, default=64, help="Frame height in bytes (default 64).")
    ap.add_argument("--w", type=int, default=64, help="Frame width in bytes (default 64).")
    ap.add_argument("--frames", type=int, default=None, help="Max frames to process (default: all complete frames).")
    ap.add_argument("--save-bitplanes", action="store_true", help="Save per-frame bit-plane images.")
    ap.add_argument("--save-heatmap", action="store_true", help="Save byte transition heatmap.")
    ap.add_argument("--no-ae", action="store_true", help="Disable autoencoder even if PyTorch is available.")
    ap.add_argument("--train-frames", type=int, default=20, help="Number of initial frames used to train AE (default 20).")
    ap.add_argument("--ae-epochs", type=int, default=20, help="AE training epochs (default 20).")
    ap.add_argument("--ae-lr", type=float, default=1e-3, help="AE learning rate (default 1e-3).")
    ap.add_argument("--device", default="cpu", help="Torch device: cpu or cuda (if available).")

    # Health test params (simplified knobs; tune per your TRNG spec / policy)
    ap.add_argument("--rct-cutoff", type=int, default=64, help="RCT cutoff: fail if max identical run >= this (default 64).")
    ap.add_argument("--apt-window", type=int, default=1024, help="APT window size in bits (default 1024).")
    ap.add_argument("--apt-hi", type=int, default=700, help="APT high cutoff for ones count in window (default 700).")
    ap.add_argument("--apt-lo", type=int, default=324, help="APT low cutoff for ones count in window (default 324).")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw = read_bytes_from_file(args.input, args.hex)
    frames_u8 = to_frames(raw, args.h, args.w, args.frames)  # (N,H,W)
    N = frames_u8.shape[0]
    print(f"[INFO] Loaded {len(raw)} bytes -> {N} frame(s) of {args.h}x{args.w} bytes")

    # Flatten for global heatmap / health tests
    flat_bytes = frames_u8.reshape(-1)

    # Health tests on full stream (you can also do per-frame if needed)
    ht = run_health_tests(flat_bytes, args.rct_cutoff, args.apt_window, args.apt_hi, args.apt_lo)
    print(f"[HT] RCT max_run={ht.rct_max_run} cutoff={args.rct_cutoff} failed={ht.rct_failed}")
    print(f"[HT] APT window={ht.apt_window} ones_max={ht.apt_max_count} ones_min={ht.apt_min_count} "
          f"hi={args.apt_hi} lo={args.apt_lo} failed={ht.apt_failed}")

    # Bit-planes per frame (optional save)
    bitplane_frames = np.zeros((N, 8, args.h, args.w), dtype=np.float32)
    if args.save_bitplanes:
        bp_dir = os.path.join(args.out, "bitplanes")
    for i in range(N):
        planes = bytes_to_bitplanes(frames_u8[i])  # (8,H,W) uint8 {0,1}
        bitplane_frames[i] = planes.astype(np.float32)
        if args.save_bitplanes:
            save_bitplanes(planes, bp_dir, i)

    # Heatmap (optional)
    if args.save_heatmap:
        M = byte_transition_heatmap(flat_bytes)
        save_heatmap(M, os.path.join(args.out, "heatmap", "byte_transition.png"), log_scale=True)

    # Autoencoder (optional)
    ae_enabled = (not args.no_ae) and TORCH_OK
    ae_scores = None

    if ae_enabled:
        device = args.device
        if device.startswith("cuda") and (not torch.cuda.is_available()):
            print("[WARN] CUDA requested but not available; using CPU.")
            device = "cpu"

        print(f"[AE] Training on first {min(args.train_frames, N)} frame(s) | epochs={args.ae_epochs} | device={device}")
        model = train_ae(bitplane_frames, args.train_frames, args.ae_epochs, args.ae_lr, device)

        ae_scores, residual_maps = ae_scores_and_residuals(model, bitplane_frames, device)
        print(f"[AE] Scores: min={ae_scores.min():.6e} max={ae_scores.max():.6e} mean={ae_scores.mean():.6e}")

        # Save residual maps (optional but useful)
        res_dir = os.path.join(args.out, "ae_residuals")
        for i in range(N):
            outp = os.path.join(res_dir, f"frame_{i:04d}_residual.png")
            save_residual_map(residual_maps[i], outp, title=f"AE Residual (mean|x-y|) Frame {i}")

    else:
        if args.no_ae:
            print("[AE] Disabled by --no-ae")
        elif not TORCH_OK:
            print("[AE] PyTorch not installed; AE step skipped (install torch to enable).")

    # Write summary CSV
    csv_path = os.path.join(args.out, "scores.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        w.writerow(["rct_max_run", ht.rct_max_run])
        w.writerow(["rct_cutoff", args.rct_cutoff])
        w.writerow(["rct_failed", int(ht.rct_failed)])
        w.writerow(["apt_window", ht.apt_window])
        w.writerow(["apt_hi", args.apt_hi])
        w.writerow(["apt_lo", args.apt_lo])
        w.writerow(["apt_max_ones_in_window", ht.apt_max_count])
        w.writerow(["apt_min_ones_in_window", ht.apt_min_count])
        w.writerow(["apt_failed", int(ht.apt_failed)])
        if ae_scores is not None:
            w.writerow(["ae_score_min", float(ae_scores.min())])
            w.writerow(["ae_score_max", float(ae_scores.max())])
            w.writerow(["ae_score_mean", float(ae_scores.mean())])

    print(f"[INFO] Wrote {csv_path}")
    print("[DONE]")

if __name__ == "__main__":
    main()
