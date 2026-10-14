"""
Merge LoRA adapters into the base model.

After finetune.py trains the adapter weights, this script fuses them back
into the base model to produce a single standalone model.

Prerequisites:
  - finetune.py must have completed successfully
  - The adapters/ directory must contain the trained weights

Usage:
  python3 merge.py

After merging, run export_to_ollama.py to import into Ollama.
"""

import subprocess
import sys
import os

MODEL = "satanasunholy1/Qwen2.5-Coder-3B-Instruct-MLX-4bit"
ADAPTER_DIR = os.path.join(os.path.dirname(__file__), "adapters")
FUSED_DIR = os.path.join(os.path.dirname(__file__), "fused_model")


def merge():
    if not os.path.exists(ADAPTER_DIR):
        print(f"[!] Adapter directory not found: {ADAPTER_DIR}")
        print("    Run finetune.py first.")
        sys.exit(1)

    adapter_files = [f for f in os.listdir(ADAPTER_DIR) if f.endswith(".safetensors")]
    if not adapter_files:
        print(f"[!] No adapter weights found in {ADAPTER_DIR}")
        print("    Run finetune.py first.")
        sys.exit(1)

    print(f"Found {len(adapter_files)} adapter file(s) in {ADAPTER_DIR}")
    print(f"Base model: {MODEL}")
    print(f"Output: {FUSED_DIR}")
    print()

    cmd = [
        sys.executable, "-m", "mlx_lm", "fuse",
        "--model", MODEL,
        "--adapter-path", ADAPTER_DIR,
        "--save-path", FUSED_DIR,
        "--dequantize",
    ]

    print("Merging adapters into base model...")
    print("This takes 2-5 minutes.\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        total_size = 0
        for root, dirs, files in os.walk(FUSED_DIR):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
        size_gb = total_size / (1024 ** 3)

        print("\n" + "=" * 60)
        print("Merge complete!")
        print(f"Fused model saved to: {FUSED_DIR}/ ({size_gb:.1f} GB)")
        print()
        print("Next step: run export_to_ollama.py to import into Ollama")
        print("=" * 60)
    else:
        print("\n[!] Merge failed. Check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    merge()
