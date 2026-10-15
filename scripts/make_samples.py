"""Generate the synthetic sample clips in samples/ (fully synthetic, so freely redistributable).

Scene: a grey walkway with slow dark "pedestrians". Anomalies are a fast wide "cart" and a
fast "runner". These are NOT UCSD Ped2 footage (Ped2 redistribution is restricted) and are
out of the model's training domain; they exist to demo the workflow, not to benchmark.

Usage: python scripts/make_samples.py [out_dir]   (needs ffmpeg on PATH for H.264 output)
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 360, 240, 12


def render(path: Path, seconds: int, events: list[tuple[str, float, float]], seed: int) -> None:
    rng = np.random.default_rng(seed)
    bg = (150 + rng.normal(0, 3, (H, W))).clip(0, 255).astype(np.uint8)
    walkers = [  # x0, y, speed px/s, direction
        [float(rng.integers(0, W)), float(rng.integers(120, 210)), float(rng.uniform(14, 26)), int(rng.choice([-1, 1]))]
        for _ in range(4)
    ]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (W, H))
    for f in range(seconds * FPS):
        t = f / FPS
        frame = (bg + rng.integers(-2, 3, (H, W))).astype(np.int16)
        for x0, y, v, d in walkers:
            x = int((x0 + d * v * t) % (W + 40)) - 20
            cv2.rectangle(frame, (x, int(y) - 28), (x + 10, int(y)), 60, -1)
        for kind, start, end in events:
            if start <= t < end:
                p = (t - start) / (end - start)
                if kind == "cart":
                    x = int(-90 + p * (W + 180))
                    cv2.rectangle(frame, (x, 150), (x + 80, 190), 25, -1)
                    cv2.circle(frame, (x + 15, 192), 7, 10, -1)
                    cv2.circle(frame, (x + 65, 192), 7, 10, -1)
                else:  # runner
                    x = int(W + 20 - p * (W + 60))
                    cv2.rectangle(frame, (x, 100), (x + 12, 140), 40, -1)
        writer.write(cv2.cvtColor(frame.clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR))
    writer.release()


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    out.mkdir(parents=True, exist_ok=True)
    clips = {
        "quiet_walkway": (20, [], 1,
                          "Synthetic: four pedestrians walking at a steady pace, nothing unusual. Expect few or no flags."),
        "cart_crossing": (20, [("cart", 9.0, 11.5)], 2,
                          "Synthetic: pedestrians, then a wide fast object (a cart) crosses at about 0:09-0:11."),
        "runner_and_cart": (24, [("runner", 6.0, 8.0), ("cart", 16.0, 18.5)], 3,
                            "Synthetic: a small fast runner at about 0:06-0:08 (easy for the model to miss) and a cart at about 0:16-0:18."),
    }
    for name, (secs, events, seed, desc) in clips.items():
        with tempfile.TemporaryDirectory() as td:
            raw = Path(td) / "raw.avi"
            render(raw, secs, events, seed)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(raw), "-c:v", "libx264",
                            "-pix_fmt", "yuv420p", "-crf", "30", "-movflags", "+faststart",
                            str(out / f"{name}.mp4")], check=True)
        (out / f"{name}.txt").write_text(desc + "\n", encoding="utf-8")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    if shutil.which("ffmpeg") is None:
        sys.exit("ffmpeg not found on PATH")
    main()
