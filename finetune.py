"""
Fine-tune Qwen2.5-Coder-3B-Instruct with QLoRA using Apple MLX.

This script:
  1. Downloads the 4-bit quantized model
  2. Attaches LoRA adapters to it
  3. Trains on your tool-calling examples
  4. Saves the adapter weights

Prerequisites:
  pip install mlx-lm

Usage:
  python3 finetune.py

After training, run merge.py to fuse the adapters into the model.
"""

import subprocess
import sys
import os

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Qwen2.5-Coder-3B-Instruct — a code-specialized instruct model.
# Already knows Python syntax, imports, control flow — the LoRA only needs
# to teach it the tool-calling format and agent behavior.
# Pre-quantized 4-bit MLX version of Qwen2.5-Coder-3B-Instruct.
# If this fails to download, try: "Qwen/Qwen2.5-Coder-3B-Instruct" (larger, ~6GB)
MODEL = "satanasunholy1/Qwen2.5-Coder-3B-Instruct-MLX-4bit"

DATA_DIR = os.path.join(os.path.dirname(__file__), "training_data")
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "adapters")

# --- Training hyperparameters ---
# With ~187 training examples and batch size 1, one epoch is ~187 steps.
# 800 iterations = ~4.3 epochs — enough to learn the pattern without
# severe overfitting, especially with the larger/better dataset.
ITERS = 800
BATCH_SIZE = 1
LORA_LAYERS = 8
LEARNING_RATE = 1e-5
STEPS_PER_REPORT = 10
STEPS_PER_EVAL = 50


def check_dependencies():
    try:
        import mlx_lm  # noqa: F401
        print("[ok] mlx-lm is installed")
    except ImportError:
        print("[!] mlx-lm is not installed. Installing now...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "mlx-lm"])
        print("[ok] mlx-lm installed")


def check_training_data():
    train_path = os.path.join(DATA_DIR, "train.jsonl")
    valid_path = os.path.join(DATA_DIR, "valid.jsonl")

    if not os.path.exists(train_path):
        print(f"[!] Training data not found at {train_path}")
        print("    Run: python3 generate_training_data.py")
        sys.exit(1)

    if not os.path.exists(valid_path):
        print(f"[!] Validation data not found at {valid_path}")
        print("    Run: python3 generate_training_data.py")
        sys.exit(1)

    train_lines = sum(1 for _ in open(train_path))
    valid_lines = sum(1 for _ in open(valid_path))
    print(f"[ok] Training data: {train_lines} train / {valid_lines} valid examples")


def train():
    os.makedirs(ADAPTER_DIR, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", MODEL,
        "--data", DATA_DIR,
        "--train",
        "--iters", str(ITERS),
        "--batch-size", str(BATCH_SIZE),
        "--num-layers", str(LORA_LAYERS),
        "--learning-rate", str(LEARNING_RATE),
        "--steps-per-report", str(STEPS_PER_REPORT),
        "--steps-per-eval", str(STEPS_PER_EVAL),
        "--adapter-path", ADAPTER_DIR,
    ]

    print("\n" + "=" * 60)
    print("Starting QLoRA fine-tuning")
    print("=" * 60)
    print(f"  Model:         {MODEL}")
    print(f"  Training data: {DATA_DIR}")
    print(f"  Iterations:    {ITERS}")
    print(f"  Batch size:    {BATCH_SIZE}")
    print(f"  LoRA layers:   {LORA_LAYERS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Adapter path:  {ADAPTER_DIR}")
    print("=" * 60)
    print()
    print("This will take ~30-60 minutes on an M1 Mac.")
    print("Your Mac will get warm — that's normal.")
    print("You can safely use your computer while it trains.")
    print()
    print("Watch the 'loss' number — it should go DOWN over time.")
    print("  loss ~2.5 = the model is guessing randomly")
    print("  loss ~0.5 = the model is learning the pattern")
    print("  loss ~0.2 = the model has mostly learned it")
    print()

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Adapter weights saved to: {ADAPTER_DIR}/")
        print()
        print("Next step: run merge.py to fuse adapters into the model")
        print("=" * 60)
    else:
        print("\n[!] Training failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    print("Fine-tuning Qwen2.5-Coder-3B-Instruct for tool calling\n")
    check_dependencies()
    check_training_data()
    train()
