"""
MCP Server — exposes the local fine-tuned tool-coder model to Claude Code.

Wraps the Qwen2.5-Coder-3B model (served by Ollama) so Claude can
delegate coding tasks to it via the Model Context Protocol.

Tools exposed:
  - ask_local_coder: Ask the local 3B model a coding question
  - local_code_review: Have the local model review a code snippet
  - local_code_generate: Generate code for a described task

Run:
  .venv/bin/python3 mcp_server.py
"""

import httpx
from mcp.server.mcpserver import MCPServer

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "tool-coder"

server = MCPServer(
    name="local-tool-coder",
    instructions=(
        "A local 3B-parameter code model (Qwen2.5-Coder fine-tuned with QLoRA). "
        "Good for quick code generation, simple reviews, and drafting snippets. "
        "Runs entirely on-device via Ollama — no API keys, no network."
    ),
)


def _query_ollama(prompt: str, system: str = "") -> str:
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0, "num_predict": 2048},
    }
    if system:
        payload["system"] = system
    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.ConnectError:
        return "Error: Ollama is not running. Start it with: ollama serve"
    except Exception as e:
        return f"Error querying local model: {e}"


@server.tool()
def ask_local_coder(question: str) -> str:
    """Ask the local fine-tuned 3B code model a coding question.
    Best for: quick syntax questions, explaining short code snippets,
    simple algorithm questions. The model runs locally on Metal GPU."""
    return _query_ollama(
        prompt=question,
        system="You are a helpful coding assistant. Give concise, accurate answers.",
    )


@server.tool()
def local_code_review(code: str, language: str = "python") -> str:
    """Have the local 3B model review a code snippet for bugs and improvements.
    Pass the code and its language. Returns observations about potential issues."""
    prompt = (
        f"Review this {language} code for bugs, style issues, and improvements. "
        f"Be concise.\n\n```{language}\n{code}\n```"
    )
    return _query_ollama(
        prompt=prompt,
        system="You are a code reviewer. Point out bugs and suggest improvements. Be brief.",
    )


@server.tool()
def local_code_generate(task: str, language: str = "python") -> str:
    """Have the local 3B model generate code for a described task.
    Returns the generated code. Good for boilerplate, simple utilities,
    and starter implementations."""
    prompt = f"Write {language} code for: {task}\n\nReturn only the code, no explanation."
    return _query_ollama(
        prompt=prompt,
        system="You are a code generator. Write clean, working code. No extra commentary.",
    )


if __name__ == "__main__":
    server.run(transport="stdio")
