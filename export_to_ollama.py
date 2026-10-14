"""
Export the fine-tuned model to Ollama.

Pipeline:
  1. generate_training_data.py  ->  training data (JSONL)
  2. finetune.py                ->  adapter weights (LoRA)
  3. merge.py                   ->  fused model (safetensors)
  4. export_to_ollama.py        ->  Ollama model       <- you are here

After this, you can use your fine-tuned model with:
  ollama run tool-coder

Prerequisites:
  - merge.py must have completed successfully
  - Ollama must be installed and running

Usage:
  python3 export_to_ollama.py
"""

import subprocess
import sys
import os
import shutil

FUSED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fused_model")
OLLAMA_MODEL_NAME = "tool-coder"
MODELFILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Modelfile")


def check_prerequisites():
    if not os.path.exists(FUSED_DIR):
        print(f"[!] Fused model not found at {FUSED_DIR}")
        print("    Run merge.py first.")
        sys.exit(1)
    print(f"[ok] Fused model found at {FUSED_DIR}")

    if not shutil.which("ollama"):
        print("[!] Ollama is not installed.")
        print("    Install it from: https://ollama.ai")
        sys.exit(1)
    print("[ok] Ollama is installed")

    result = subprocess.run(
        ["ollama", "list"], capture_output=True, text=True, timeout=10
    )
    if result.returncode != 0:
        print("[!] Ollama doesn't seem to be running.")
        print("    Start it with: ollama serve")
        sys.exit(1)
    print("[ok] Ollama is running")


def create_modelfile():
    # Read the existing Modelfile template but ensure FROM points to the fused model
    with open(MODELFILE_PATH) as f:
        content = f.read()

    # Make sure FROM points to the absolute path of the GGUF file
    gguf_path = os.path.join(FUSED_DIR, "model-q8.gguf")
    lines = content.split('\n')
    lines[0] = f'FROM {gguf_path}'
    content = '\n'.join(lines)

    with open(MODELFILE_PATH, "w") as f:
        f.write(content)
    print(f"[ok] Modelfile updated with absolute path: {MODELFILE_PATH}")


def import_to_ollama():
    print(f"\nImporting as '{OLLAMA_MODEL_NAME}' into Ollama...")
    print("Ollama will convert and quantize the model automatically.")
    print("This may take a few minutes.\n")

    result = subprocess.run(
        ["ollama", "create", OLLAMA_MODEL_NAME, "-f", MODELFILE_PATH],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    if result.returncode == 0:
        print(f"\n[ok] Model '{OLLAMA_MODEL_NAME}' is now available in Ollama!")
    else:
        print("\n[!] Import failed. Check the error messages above.")
        sys.exit(1)


def main():
    print("Exporting fine-tuned model to Ollama\n")
    check_prerequisites()
    print()
    create_modelfile()
    import_to_ollama()

    print("\n" + "=" * 60)
    print("All done! Your fine-tuned model is ready.")
    print()
    print(f"Test it:  ollama run {OLLAMA_MODEL_NAME}")
    print()
    print("Use it in your code assistant by changing rag_graph.py:")
    print(f'  llm = ChatOllama(model="{OLLAMA_MODEL_NAME}", temperature=0)')
    print("=" * 60)


if __name__ == "__main__":
    main()
