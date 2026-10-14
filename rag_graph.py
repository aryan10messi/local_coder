"""
Code Assistant — an agentic tool-calling system using LangGraph + Ollama

Architecture:
  START → think → [tools or end]
             ↑         ↓
             └── tool_node

The LLM decides what to do. Tools are the retrieval mechanism —
read_file, grep_codebase, list_files replace vector search.
"""

import os
import json
import re as _re
import uuid
from dataclasses import dataclass, field
from typing import Annotated

from rich.console import Console
from rich.panel import Panel
from langchain_ollama import ChatOllama
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from tools import ALL_TOOLS

console = Console()

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatOllama(model="tool-coder", temperature=0)


# ---------------------------------------------------------------------------
# AGENT STATE & SYSTEM PROMPT
# ---------------------------------------------------------------------------
@dataclass
class AgentState:
    messages: Annotated[list[AnyMessage], add_messages] = field(default_factory=list)
    thinking: str = ""
    iteration: int = 0


SYSTEM_PROMPT = """You are a code assistant. You have these tools:
- read_file(path): Read a file
- edit_file(path, old_text, new_text): Replace text in a file
- create_file(path, content): Create a new file
- write_file(path, content): Overwrite a file
- grep_codebase(pattern, directory): Search files
- list_files(directory): List directory contents
- run_command(command): Run a shell command

To use a tool, output EXACTLY this format:
[TOOL_CALL]{"name": "TOOL_NAME", "parameters": {"param1": "value1"}}[/TOOL_CALL]

Rules:
- Use ONLY the [TOOL_CALL] JSON format shown above. No other format.
- One tool call per response.
- After a tool succeeds, tell the user what happened and STOP.
- If a tool errors, try a different approach.
- If the user is just chatting (greetings, questions about concepts), respond normally without using tools."""


# ---------------------------------------------------------------------------
# TOOL CALL PARSER
# ---------------------------------------------------------------------------
TOOL_NAMES = {t.name for t in ALL_TOOLS}

TOOL_PARAM_ORDER = {
    "read_file": ["path"],
    "edit_file": ["path", "old_text", "new_text"],
    "create_file": ["path", "content"],
    "write_file": ["path", "content"],
    "grep_codebase": ["pattern", "directory"],
    "list_files": ["directory"],
    "run_command": ["command"],
}


def _fallback_extract_tool(block: str) -> list[dict] | None:
    """Extract tool call from malformed JSON (e.g. unescaped quotes in code content)."""
    name_match = _re.search(r'"name"\s*:\s*"(\w+)"', block)
    if not name_match or name_match.group(1) not in TOOL_NAMES:
        return None
    name = name_match.group(1)
    params_match = _re.search(r'"parameters"\s*:\s*\{', block)
    if not params_match:
        params_match = _re.search(r'"arguments"\s*:\s*\{', block)
    if not params_match:
        return None
    args = {}
    param_keys = TOOL_PARAM_ORDER.get(name, [])
    rest = block[params_match.end():]
    for key in param_keys:
        pat = _re.compile(r'"' + _re.escape(key) + r'"\s*:\s*"')
        m = pat.search(rest)
        if not m:
            continue
        val_start = m.end()
        if key in ("content", "old_text", "new_text"):
            # For long string values, find the closing pattern instead of parsing quotes
            # Look for the next key or closing braces
            next_key = None
            for k2 in param_keys:
                if k2 == key:
                    continue
                nk = _re.search(r'"\s*,\s*"' + _re.escape(k2) + r'"\s*:', rest[val_start:])
                if nk and (next_key is None or nk.start() < next_key[1]):
                    next_key = (k2, nk.start())
            if next_key:
                val = rest[val_start:val_start + next_key[1]]
                val = val.rstrip().rstrip('"')
            else:
                # Last key — find trailing "}}"
                val = rest[val_start:]
                # Strip from the last unescaped quote backward
                end_pat = _re.search(r'"\s*\}\s*\}\s*$', val)
                if end_pat:
                    val = val[:end_pat.start()]
                else:
                    val = val.rstrip().rstrip('}').rstrip().rstrip('"')
            args[key] = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
        else:
            # Short value — find the closing quote (skip escaped quotes)
            i = val_start
            while i < len(rest):
                if rest[i] == '\\':
                    i += 2
                    continue
                if rest[i] == '"':
                    break
                i += 1
            args[key] = rest[val_start:i].replace('\\"', '"').replace('\\\\', '\\')
    if args:
        return [{"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}]
    return None


def _extract_text_tool_calls(content: str) -> list[dict] | None:
    """Parse tool calls from the model's text output."""
    # Strip all tag wrappers to normalize
    cleaned = _re.sub(r'\[/?TOOL_CALL\]', '', content)
    cleaned = _re.sub(r'</?tool_call>', '', cleaned)
    cleaned = _re.sub(r'</?tool>', '', cleaned)
    cleaned = cleaned.replace('范冰', '')

    # Try JSON blocks — string-aware brace matcher with fallback
    for i, c in enumerate(cleaned):
        if c != '{':
            continue
        depth = 0
        in_str = False
        j = i
        while j < len(cleaned):
            ch = cleaned[j]
            if ch == '\\' and in_str:
                j += 2
                continue
            if ch == '"' and not in_str:
                in_str = True
            elif ch == '"' and in_str:
                in_str = False
            elif not in_str:
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        block = cleaned[i:j + 1]
                        try:
                            obj = json.loads(block)
                            name = obj.get("name", "")
                            if name in TOOL_NAMES:
                                args = obj.get("parameters", obj.get("arguments", obj.get("args", {})))
                                return [{"name": name, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}]
                        except (json.JSONDecodeError, AttributeError):
                            pass
                        result = _fallback_extract_tool(block)
                        if result:
                            return result
                        break
            j += 1

    # Find tool_name(...) with balanced parentheses — pick earliest in text
    earliest_pos = len(cleaned)
    earliest_match = None
    earliest_tool = None
    for tool_name in TOOL_NAMES:
        match = _re.search(rf'{tool_name}\s*\(', cleaned)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            earliest_match = match
            earliest_tool = tool_name

    if earliest_match and earliest_tool:
        start = earliest_match.end() - 1
        args_str = _extract_balanced_parens(cleaned, start)
        if args_str:
            args_str = args_str.rstrip("}")
            args = _parse_function_args(earliest_tool, args_str)
            if args:
                return [{"name": earliest_tool, "args": args, "id": str(uuid.uuid4()), "type": "tool_call"}]

    return None


def _extract_balanced_parens(text: str, start: int) -> str | None:
    """Extract content between balanced parentheses starting at position `start`."""
    if start >= len(text) or text[start] != '(':
        return None
    depth = 0
    in_str = None
    i = start
    while i < len(text):
        c = text[i]
        if c == '\\' and in_str:
            i += 2
            continue
        if c in ('"', "'") and not in_str:
            in_str = c
        elif c == in_str:
            in_str = None
        elif not in_str:
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    return text[start + 1:i]
        i += 1
    return text[start + 1:]


def _parse_function_args(tool_name: str, args_str: str) -> dict | None:
    """Parse function arguments — handles both keyword and positional styles."""
    args = {}
    for kv in _re.finditer(r'(\w+)\s*=\s*(?:"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\')', args_str):
        key = kv.group(1)
        val = kv.group(2) if kv.group(2) is not None else kv.group(3)
        val = val.replace('\\"', '"').replace('\\\\', '\\')
        args[key] = val
    if args:
        return args

    positional = []
    for m in _re.finditer(r'"((?:[^"\\]|\\.)*)"', args_str):
        positional.append(m.group(1).replace('\\"', '"').replace('\\\\', '\\'))
    if not positional:
        for m in _re.finditer(r"'((?:[^'\\]|\\.)*)'", args_str):
            positional.append(m.group(1))

    if positional and tool_name in TOOL_PARAM_ORDER:
        param_names = TOOL_PARAM_ORDER[tool_name]
        for i, val in enumerate(positional):
            if i < len(param_names):
                args[param_names[i]] = val
        if args:
            return args

    return None


def _clean_content(content: str) -> str:
    """Strip the 'assistant' prefix and leaked instruction text."""
    content = content.strip()
    if content.lower().startswith("assistant\n"):
        content = content[len("assistant\n"):].strip()
    if content.lower().startswith("assistant"):
        content = content[len("assistant"):].strip()
    content = _re.sub(r'\[Summarize the result[^\]]*\]', '', content)
    content = _re.sub(r'Now respond to the user[^.]*\.(?:[^.]*\.)?', '', content)
    content = _re.sub(r'Now respond when (?:you\'re|the user is) ready[^.]*\.?', '', content)
    return content.strip()


def _strip_tool_from_reasoning(content: str, extracted: list[dict]) -> str:
    """Remove tool call syntax from content to show just the reasoning."""
    reasoning = content
    reasoning = _re.sub(r'\[/?TOOL_CALL\]', '', reasoning)
    reasoning = _re.sub(r'</?tool_call>', '', reasoning)
    reasoning = _re.sub(r'</?tool>', '', reasoning)
    reasoning = reasoning.replace('范冰', '')
    reasoning = _re.sub(r'<tool_summary>.*?</tool_summary>', '', reasoning, flags=_re.DOTALL)
    for tc in extracted:
        match = _re.search(rf'{tc["name"]}\s*\(', reasoning)
        if match:
            start = match.end() - 1
            end_content = _extract_balanced_parens(reasoning, start)
            if end_content is not None:
                full = reasoning[match.start():match.start() + len(match.group()) + len(end_content) + 1]
                reasoning = reasoning.replace(full, '')
            else:
                reasoning = reasoning[:match.start()]
    # Also strip bare JSON blocks that look like tool calls (string-aware)
    tc_start = reasoning.find('{"name"')
    if tc_start >= 0:
        reasoning = reasoning[:tc_start].strip()
    reasoning = reasoning.strip()
    if not reasoning:
        for tc in extracted:
            idx = content.find(tc["name"])
            if idx > 0:
                reasoning = content[:idx].strip()
                reasoning = _re.sub(r'</?tool_call>', '', reasoning).strip()
                break
    return reasoning


# ---------------------------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------------------------
def think_node(state: AgentState) -> dict:
    """The LLM decides what to do next."""
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state.messages
    response = llm.invoke(messages)

    content = _clean_content(response.content) if response.content else ""

    if content:
        extracted = _extract_text_tool_calls(content)
        if extracted:
            response.tool_calls = extracted
            content = _strip_tool_from_reasoning(content, extracted)

    if content and not response.tool_calls:
        console.print()
        safe = content.replace("[", "\\[")
        console.print(Panel(safe, title="[green]Answer[/green]", border_style="green", padding=(1, 2)))
    elif content and response.tool_calls:
        safe = content.replace("[", "\\[")
        console.print(f"  [yellow]Thinking:[/yellow] {safe}")

    if response.tool_calls:
        # --- duplicate detection ---
        prev_tool_calls = []
        for msg in reversed(state.messages):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                prev_tool_calls = msg.tool_calls
                break
        for tc in response.tool_calls:
            is_dup = any(
                p.get("name") == tc["name"] and p.get("args") == tc["args"]
                for p in prev_tool_calls
            )
            if is_dup:
                console.print(f"  [yellow]⚠ Duplicate tool call, stopping[/yellow]")
                response.tool_calls = []
                if not content:
                    content = "Done."
                console.print(Panel(content.replace("[", "\\["), title="[green]Answer[/green]", border_style="green", padding=(1, 2)))
                return {"messages": [response], "iteration": state.iteration + 1}

        # --- repeated-failure detection ---
        if state.iteration >= 3:
            recent_errors = 0
            recent_tool_name = None
            for msg in reversed(state.messages):
                if isinstance(msg, ToolMessage):
                    snippet = msg.content[:300].lower()
                    is_err = ("error" in snippet or "stderr:" in snippet
                              or "exit code: 1" in snippet or "traceback" in snippet
                              or "syntaxerror" in snippet or "indentationerror" in snippet)
                    if is_err:
                        recent_errors += 1
                    else:
                        break
                elif hasattr(msg, "tool_calls") and msg.tool_calls:
                    if recent_tool_name is None:
                        recent_tool_name = msg.tool_calls[0].get("name")
                    elif msg.tool_calls[0].get("name") == recent_tool_name:
                        continue
                    else:
                        break
            if recent_errors >= 3:
                console.print(f"  [yellow]⚠ Same error {recent_errors} times in a row — stopping loop[/yellow]")
                response.tool_calls = []
                content = "I'm stuck in a loop — the same fix keeps failing. Try a different approach or rewrite the file from scratch."
                console.print(Panel(content.replace("[", "\\["), title="[green]Answer[/green]", border_style="green", padding=(1, 2)))
                return {"messages": [response], "iteration": state.iteration + 1}

            args_short = []
            for k, v in tc["args"].items():
                val = str(v)
                if len(val) > 60:
                    val = val[:57] + "..."
                args_short.append(f'{k}="{val}"' if isinstance(v, str) else f"{k}={val}")
            console.print(f"  [blue]Tool:[/blue] {tc['name']}({', '.join(args_short)})")

    return {"messages": [response], "iteration": state.iteration + 1}


def tool_node(state: AgentState) -> dict:
    """Execute whatever tools the LLM chose."""
    last_msg = state.messages[-1]
    tool_results = []

    tool_map = {t.name: t for t in ALL_TOOLS}

    for tc in last_msg.tool_calls:
        tool_fn = tool_map.get(tc["name"])
        if tool_fn:
            try:
                result = tool_fn.invoke(tc["args"])
                result_str = str(result)
            except Exception as e:
                result_str = f"Error calling {tc['name']}: {e}"
            if len(result_str) > 2000:
                result_str = result_str[:2000] + "\n\n[... truncated]"
            result_str += "\n\n[Summarize the result for the user. Do not call the same tool again.]"
            preview = result_str[:150].replace("\n", " ")
            if len(result_str) > 150:
                preview += "..."
            console.print(f"  [dim]  ↳ {preview}[/dim]")
            tool_results.append(ToolMessage(content=result_str, tool_call_id=tc["id"]))
        else:
            tool_results.append(ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"]))

    return {"messages": tool_results}


def should_continue(state: AgentState) -> str:
    """Decide: does the LLM want to use more tools, or is it done?"""
    if state.iteration >= 12:
        console.print("[yellow]⚠ Max iterations reached[/yellow]")
        return "end"

    last_msg = state.messages[-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "tools"
    return "end"


# ---------------------------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------------------------
#   START → think → [tools or end]
#              ↑         ↓
#              └── tool_node
graph_builder = StateGraph(AgentState)
graph_builder.add_node("think", think_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "think")
graph_builder.add_conditional_edges("think", should_continue, {"tools": "tools", "end": END})
graph_builder.add_edge("tools", "think")

memory = MemorySaver()
agent = graph_builder.compile(checkpointer=memory)


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------
_thread_counter = 0


def chat(message: str, thread_id: str = "default") -> str:
    """Send a message to the agent and get a response."""
    result = agent.invoke(
        {"messages": [HumanMessage(content=message)], "iteration": 0},
        config={"configurable": {"thread_id": thread_id}},
    )
    for msg in reversed(result["messages"]):
        if hasattr(msg, "content") and msg.content and not isinstance(msg, (HumanMessage, ToolMessage, SystemMessage)):
            return msg.content
    return "No response generated."


def ask(question: str) -> str:
    """Backward-compatible: run a single question (no memory)."""
    global _thread_counter
    _thread_counter += 1
    return chat(question, thread_id=f"oneshot_{_thread_counter}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    console.print(Panel.fit(
        "[bold]Code Assistant[/bold]\n"
        "An agentic code assistant with tool calling\n\n"
        "Commands:\n"
        "  [cyan]quit[/cyan] — Exit",
        border_style="bright_blue",
    ))

    thread_id = "cli_session"

    while True:
        try:
            q = console.input("\n[bold]You:[/bold] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not q:
            continue
        if q.lower() in ("quit", "exit", "q"):
            break

        console.print()
        chat(q, thread_id=thread_id)
