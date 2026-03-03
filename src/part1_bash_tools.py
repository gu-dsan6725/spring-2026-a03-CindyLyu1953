"""Part 1: Code Q&A using bash tools for context retrieval."""

import subprocess
from pathlib import Path
from enum import Enum

from .llm_client import complete_with_context

# Max context chars to avoid token limits (smaller = fewer tokens, less rate limit)
MAX_CONTEXT_CHARS = 8000

# System prompt for code Q&A
PART1_SYSTEM_PROMPT = """You are a code Q&A assistant. Answer questions about the mcp-gateway-registry codebase using the provided context from bash tools (grep, find, cat, tree, etc.).

Rules:
- Cite specific files and line numbers when possible
- If the context is insufficient, say so and suggest what to search for
- Be concise but complete
- For code snippets, preserve formatting"""


class QuestionType(str, Enum):
    """Question types for Part 1 routing."""

    DEPENDENCIES = "dependencies"
    ENTRY_POINT = "entry_point"
    FILE_TYPES = "file_types"
    AUTH_FLOW = "auth_flow"
    API_ENDPOINTS = "api_endpoints"
    OAUTH_EXTENSION = "oauth_extension"
    GENERAL = "general"


def classify_question(question: str) -> QuestionType:
    """Classify question into retrieval strategy type (keyword-based)."""
    q = question.lower()
    if "oauth" in q and ("add" in q or "provider" in q or "okta" in q):
        return QuestionType.OAUTH_EXTENSION
    if "endpoint" in q or "api" in q and ("scope" in q or "require" in q):
        return QuestionType.API_ENDPOINTS
    if "authentication" in q or "auth" in q or "token" in q or "authorization" in q:
        return QuestionType.AUTH_FLOW
    if "dependencies" in q or "python dependencies" in q:
        return QuestionType.DEPENDENCIES
    if "entry point" in q or "main entry" in q or "main file" in q:
        return QuestionType.ENTRY_POINT
    if "programming languages" in q or "file types" in q or "languages" in q:
        return QuestionType.FILE_TYPES
    return QuestionType.GENERAL


def run_bash(cmd: str | list[str], cwd: Path, timeout: int = 30) -> str:
    """
    Run bash command and return stdout. Stderr is appended if non-empty.
    """
    if isinstance(cmd, str):
        cmd = ["sh", "-c", cmd]
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        out = result.stdout or ""
        if result.stderr:
            out += "\n[stderr]\n" + result.stderr
        return out
    except subprocess.TimeoutExpired:
        return "[Command timed out]"
    except Exception as e:
        return f"[Error: {e}]"


def truncate(text: str, max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """Truncate text to max_chars, appending truncation notice."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[... truncated for length ...]"


def get_commands_for_type(qtype: QuestionType, repo_path: Path) -> list[str]:
    """Return list of bash commands to run for the question type."""
    commands = []
    if qtype == QuestionType.DEPENDENCIES:
        commands = [
            "cat pyproject.toml 2>/dev/null || true",
            "cat package.json 2>/dev/null || true",
            "find . -name 'package.json' -type f | head -5 | xargs -I {} sh -c 'echo \"=== {} ===\"; cat {}' 2>/dev/null || true",
            "find . -name 'pyproject.toml' -type f | head -5 | xargs -I {} sh -c 'echo \"=== {} ===\"; cat {}' 2>/dev/null || true",
        ]
    elif qtype == QuestionType.ENTRY_POINT:
        commands = [
            "find . -name 'main*.py' -type f 2>/dev/null || true",
            "find . -name '__main__.py' -type f 2>/dev/null || true",
            "grep -r -l '__main__' --include='*.py' . 2>/dev/null | head -20",
            "grep -r -l 'if __name__' --include='*.py' . 2>/dev/null | head -20",
            "ls -la registry/ 2>/dev/null || true",
            "head -80 registry/main.py 2>/dev/null || head -80 registry/__main__.py 2>/dev/null || true",
        ]
    elif qtype == QuestionType.FILE_TYPES:
        commands = [
            "find . -type f \\( -name '*.py' -o -name '*.ts' -o -name '*.tsx' -o -name '*.json' -o -name '*.yaml' -o -name '*.yml' -o -name 'Dockerfile' -o -name '*.md' \\) | head -100",
            "find . -type f -name '*.py' | wc -l",
            "find . -type f -name '*.ts' -o -name '*.tsx' | wc -l",
            "find . -type f -name '*.json' | wc -l",
            "find . -type f -name '*.yaml' -o -name '*.yml' | wc -l",
            "find . -type f -name 'Dockerfile' | wc -l",
            "tree -L 2 -I 'node_modules|__pycache__|.git' 2>/dev/null || find . -maxdepth 2 -type d | head -50",
        ]
    elif qtype == QuestionType.AUTH_FLOW:
        commands = [
            "grep -r -n 'token' --include='*.py' auth_server/ registry/ 2>/dev/null | head -80",
            "grep -r -n 'auth' --include='*.py' auth_server/ registry/ 2>/dev/null | head -80",
            "grep -r -n 'validate' --include='*.py' auth_server/ registry/ 2>/dev/null | head -50",
            "grep -r -n 'authorization' --include='*.py' . 2>/dev/null | head -50",
            "find docs -name '*.md' -exec grep -l -i auth {} \\; 2>/dev/null | head -5",
            "for f in $(find docs -name '*.md' -exec grep -l -i auth {} \\; 2>/dev/null | head -3); do echo \"=== $f ===\"; head -100 \"$f\"; done",
        ]
    elif qtype == QuestionType.API_ENDPOINTS:
        commands = [
            "grep -r -n -E '@(app|router)\\.(get|post|put|delete|patch)' --include='*.py' . 2>/dev/null | head -100",
            "grep -r -n 'scope\\|require' --include='*.py' registry/ 2>/dev/null | head -80",
            "grep -r -n 'APIRouter\\|FastAPI' --include='*.py' registry/ 2>/dev/null | head -50",
            "find docs -name '*.md' -exec grep -l -i endpoint {} \\; 2>/dev/null | head -5",
            "for f in $(find docs -name '*.md' 2>/dev/null | head -5); do echo \"=== $f ===\"; head -80 \"$f\"; done",
        ]
    elif qtype == QuestionType.OAUTH_EXTENSION:
        commands = [
            "grep -r -n 'oauth\\|OAuth\\|provider' --include='*.py' . 2>/dev/null | head -100",
            "grep -r -n 'auth' --include='*.py' auth_server/ 2>/dev/null | head -80",
            "ls -la auth_server/",
            "find auth_server -name '*.py' -exec head -80 {} \\; 2>/dev/null",
            "find docs -name '*.md' -exec grep -l -i oauth {} \\; 2>/dev/null | head -5",
            "for f in $(find docs -name '*.md' -exec grep -l -i oauth {} \\; 2>/dev/null | head -3); do echo \"=== $f ===\"; cat \"$f\"; done",
        ]
    else:
        commands = [
            "tree -L 2 -I 'node_modules|__pycache__|.git' 2>/dev/null || find . -maxdepth 2 -type d | head -50",
            "grep -r -n . --include='*.py' . 2>/dev/null | head -50",
        ]
    return commands


def retrieve_context(question: str, repo_path: Path) -> str:
    """
    Classify question, run appropriate bash commands, return combined context.
    """
    qtype = classify_question(question)
    commands = get_commands_for_type(qtype, repo_path)
    outputs = []
    for cmd in commands:
        out = run_bash(cmd, repo_path)
        if out.strip():
            outputs.append(f"--- Command: {cmd} ---\n{out}")
    combined = "\n\n".join(outputs)
    return truncate(combined)


def answer_question(question: str, repo_path: Path) -> str:
    """
    Full pipeline: retrieve context via bash + LLM answer.
    """
    context = retrieve_context(question, repo_path)
    return complete_with_context(
        question=question,
        context=context,
        system_prompt=PART1_SYSTEM_PROMPT,
    )
