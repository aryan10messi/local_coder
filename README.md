# Local AI Code Assistant

A fully local, open-source AI coding agent built on Apple Silicon. Fine-tunes Qwen2.5-Coder-3B with QLoRA to follow a tool-calling format, wraps it in a LangGraph agent loop with file I/O tools, and exposes it as an MCP server so Claude Code (or any MCP client) can delegate tasks to it.

Everything runs on-device — no API keys, no cloud, no data leaves your machine.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code (MCP Client)              │
│  Sees local-tool-coder as a tool, delegates tasks to it  │
└──────────────────────┬──────────────────────────────────┘
                       │ stdio (JSON-RPC)
              ┌────────▼────────┐
              │   mcp_server.py  │  MCP server (3 tools)
              └────────┬────────┘
                       │ HTTP POST
              ┌────────▼────────┐
              │     Ollama       │  Model server (Metal GPU)
              │  localhost:11434 │
              └────────┬────────┘
                       │
              ┌────────▼────────┐
              │   tool-coder     │  Qwen2.5-Coder-3B
              │   Q8_0 · 3.1 GB │  Fine-tuned with QLoRA
              └─────────────────┘
```

The agent can also run standalone via `rag_graph.py`, which implements a full LangGraph state machine with 7 tools (read, write, edit, grep, list, run shell commands, search code).

## What's in the box

| File | Purpose |
|------|---------|
| `rag_graph.py` | LangGraph agent loop — think → tool → think cycle with 7 tools |
| `mcp_server.py` | MCP server wrapping the model for Claude Code integration |
| `tools.py` | Tool implementations (read_file, edit_file, grep, list_files) |
| `generate_training_data.py` | Generates tool-calling training examples (JSONL) |
| `finetune.py` | QLoRA fine-tuning via MLX on Apple Silicon |
| `merge.py` | Merges LoRA adapters back into base model weights |
| `export_to_ollama.py` | Converts to GGUF and imports into Ollama |
| `Modelfile` | Ollama model config (ChatML template, stop tokens) |

## Requirements

- **macOS** with Apple Silicon (M1/M2/M3/M4) — Metal GPU required for training and inference
- **Python 3.11+**
- **Ollama** — [install here](https://ollama.ai)
- ~8 GB free RAM for inference, ~12 GB for fine-tuning

## Setup

```bash
# Clone and set up
git clone https://github.com/YOUR_USERNAME/ai-code-assistant.git
cd ai-code-assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Training pipeline

Run these in order to fine-tune your own model from scratch:

```bash
# 1. Generate training data (tool-calling examples)
python3 generate_training_data.py

# 2. Fine-tune with QLoRA (runs on Metal GPU, ~30 min)
python3 finetune.py

# 3. Merge LoRA adapters into base model
python3 merge.py

# 4. Export to Ollama
python3 export_to_ollama.py
```

After step 4, you'll have a model called `tool-coder` in Ollama.

### What the training does

- **Base model**: Qwen2.5-Coder-3B-Instruct (downloaded automatically)
- **Method**: QLoRA — freezes the 3B parameters at 4-bit precision, trains small rank-8 bypass matrices (~3.4M trainable parameters)
- **Data**: ~200 synthetic tool-calling conversations teaching the `[TOOL_CALL]...[/TOOL_CALL]` format
- **Output**: Q8_0 quantized GGUF (3.1 GB) served by Ollama on Metal GPU

## Usage

### Standalone agent

```bash
source .venv/bin/activate
python3 rag_graph.py
```

This launches an interactive agent that can read, write, edit, and grep files. It uses the LangGraph think → tool → think loop.

### MCP server (for Claude Code)

Add to your `~/.claude.json`:

```json
{
  "mcpServers": {
    "local-tool-coder": {
      "command": "/path/to/project/.venv/bin/python3",
      "args": ["/path/to/project/mcp_server.py"]
    }
  }
}
```

Then in a new Claude Code session, you can say:
- *"Use the local coder to generate a Python fibonacci function"*
- *"Ask the local model to review this code"*

Claude will call your local 3B model and show you its output.

### Just Ollama

```bash
ollama run tool-coder
```

## How it works

**LangGraph agent loop** (`rag_graph.py`): A state machine with two nodes — `think` (call the LLM) and `tools` (execute tool calls). The LLM generates `[TOOL_CALL]{"name": "...", "parameters": {...}}[/TOOL_CALL]` tags, which the harness parses (with a regex fallback for malformed JSON) and executes. Results feed back as context for the next think step. A repeated-failure detector breaks stuck loops after 3 consecutive errors.

**QLoRA fine-tuning** (`finetune.py`): Uses MLX (Apple's ML framework) to train LoRA adapters on the frozen base model. LoRA decomposes each weight update into two small matrices (rank 8), so only ~3.4M parameters train instead of 3B. The 4-bit quantization of frozen weights (the "Q" in QLoRA) cuts memory from ~12 GB to ~4 GB.

**MCP server** (`mcp_server.py`): A thin wrapper that receives JSON-RPC tool calls from Claude Code over stdio, forwards the prompt to Ollama's HTTP API, and returns the response. Three tools: `ask_local_coder`, `local_code_review`, `local_code_generate`.

## Limitations

- **3B parameters**: Can follow the tool-calling format and generate simple code, but can't reason about complex structural bugs or self-debug effectively
- **Apple Silicon only**: The training pipeline uses MLX which requires Metal GPU
- **Not a replacement for larger models**: Best used as a fast local helper for boilerplate, simple reviews, and drafting code

## License

MIT
