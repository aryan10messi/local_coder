"""
Agent tools — the actions the LLM can take on a codebase.

Each tool is a plain function with type hints and a docstring.
LangChain uses the docstring to tell the LLM what the tool does.
"""

import os
import re
import subprocess
from langchain_core.tools import tool


@tool
def read_file(path: str) -> str:
    """Read the full contents of a file. Use this when you need to see the complete
    code in a file, not just a chunk from the vector database."""
    try:
        with open(path) as f:
            lines = f.readlines()
        numbered = [f"{i+1:4d} | {line}" for i, line in enumerate(lines)]
        return f"File: {path} ({len(lines)} lines)\n" + "".join(numbered)
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading {path}: {e}"


@tool
def edit_file(path: str, old_text: str, new_text: str) -> str:
    """Edit a file by replacing old_text with new_text. Use this to make targeted
    code changes. The old_text must match exactly (including whitespace)."""
    try:
        with open(path) as f:
            content = f.read()
        if old_text not in content:
            return f"Error: Could not find the exact text to replace in {path}"
        if content.count(old_text) > 1:
            return f"Error: Found multiple matches for that text in {path}. Be more specific."
        new_content = content.replace(old_text, new_text, 1)
        with open(path, "w") as f:
            f.write(new_content)
        return f"Successfully edited {path}"
    except Exception as e:
        return f"Error editing {path}: {e}"


@tool
def grep_codebase(pattern: str, directory: str = ".") -> str:
    """Search for a text pattern across all files in a directory. Returns matching
    lines with file paths and line numbers. Use this to find where something is
    defined or used."""
    matches = []
    skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".next", "venv", "chroma_db"}
    skip_extensions = {".pyc", ".so", ".dylib", ".bin", ".jpg", ".png", ".gif", ".ico", ".woff"}

    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if d not in skip_dirs]
        for fname in files:
            if any(fname.endswith(ext) for ext in skip_extensions):
                continue
            filepath = os.path.join(root, fname)
            try:
                with open(filepath, errors="ignore") as f:
                    for i, line in enumerate(f, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            matches.append(f"{filepath}:{i} | {line.rstrip()}")
                            if len(matches) >= 30:
                                return f"Found {len(matches)}+ matches (showing first 30):\n" + "\n".join(matches)
            except (PermissionError, IsADirectoryError):
                continue

    if not matches:
        return f"No matches found for '{pattern}' in {directory}"
    return f"Found {len(matches)} matches:\n" + "\n".join(matches)


@tool
def list_files(directory: str = ".") -> str:
    """List all files in a directory tree. Use this to understand the structure
    of a project before diving into specific files."""
    tree_lines = []
    skip_dirs = {".git", "node_modules", ".venv", "__pycache__", ".next", "venv", "chroma_db"}

    for root, dirs, files in os.walk(directory):
        dirs[:] = sorted(d for d in dirs if d not in skip_dirs)
        level = root.replace(directory, "").count(os.sep)
        indent = "  " * level
        tree_lines.append(f"{indent}{os.path.basename(root)}/")
        sub_indent = "  " * (level + 1)
        for fname in sorted(files):
            tree_lines.append(f"{sub_indent}{fname}")
        if len(tree_lines) > 200:
            tree_lines.append("... (truncated, too many files)")
            break

    return "\n".join(tree_lines)


@tool
def create_file(path: str, content: str) -> str:
    """Create a new file with the given content. Creates parent directories if needed.
    Use this when you need to make a new file that doesn't exist yet."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        if os.path.exists(path):
            return f"Error: File already exists: {path}. Use edit_file to modify it."
        with open(path, "w") as f:
            f.write(content)
        return f"Created {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error creating {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Overwrite an existing file with new content, or create it if it doesn't exist.
    Use this when you need to completely rewrite a file rather than making a small edit."""
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(content)
        return f"Wrote {path} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing {path}: {e}"


@tool
def run_command(command: str) -> str:
    """Run a shell command and return its output. Use this to run tests,
    check linting, or verify that code changes work. Keep commands short and safe."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30, cwd="."
        )
        output = ""
        if result.stdout:
            output += f"stdout:\n{result.stdout}\n"
        if result.stderr:
            output += f"stderr:\n{result.stderr}\n"
        output += f"Exit code: {result.returncode}"
        return output if output.strip() else "Command completed with no output."
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 30 seconds"
    except Exception as e:
        return f"Error running command: {e}"


ALL_TOOLS = [read_file, edit_file, create_file, write_file, grep_codebase, list_files, run_command]
