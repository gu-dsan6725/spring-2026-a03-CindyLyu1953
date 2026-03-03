"""Test cases for Part 1: Code Q&A with Bash Tools."""

import pytest
from pathlib import Path

from src.part1_bash_tools import (
    classify_question,
    QuestionType,
    run_bash,
    truncate,
    get_commands_for_type,
)


class TestClassifyQuestion:
    """Test query classification logic."""

    def test_dependencies(self):
        assert classify_question("What Python dependencies does this project use?") == QuestionType.DEPENDENCIES

    def test_entry_point(self):
        assert classify_question("What is the main entry point file for the registry service?") == QuestionType.ENTRY_POINT

    def test_file_types(self):
        assert classify_question("What programming languages and file types are used?") == QuestionType.FILE_TYPES

    def test_auth_flow(self):
        assert classify_question("How does the authentication flow work?") == QuestionType.AUTH_FLOW

    def test_api_endpoints(self):
        assert classify_question("What are all the API endpoints and what scopes do they require?") == QuestionType.API_ENDPOINTS

    def test_oauth_extension(self):
        assert classify_question("How would you add support for a new OAuth provider (e.g., Okta)?") == QuestionType.OAUTH_EXTENSION

    def test_general_fallback(self):
        assert classify_question("What is this project about?") == QuestionType.GENERAL


class TestRunBash:
    """Test bash command execution."""

    def test_echo(self):
        out = run_bash("echo hello", Path("."))
        assert "hello" in out

    def test_multiple_commands(self):
        out = run_bash("echo a && echo b", Path("."))
        assert "a" in out and "b" in out

    def test_invalid_command_returns_error(self):
        out = run_bash("nonexistent_command_xyz_123", Path("."))
        assert "[Error:" in out or "[stderr]" in out or "not found" in out.lower()


class TestTruncate:
    """Test context truncation."""

    def test_short_text_unchanged(self):
        text = "short"
        assert truncate(text, max_chars=100) == "short"

    def test_long_text_truncated(self):
        text = "a" * 200
        result = truncate(text, max_chars=50)
        assert len(result) < len(text)
        assert "[... truncated" in result


class TestGetCommandsForType:
    """Test that each question type gets non-empty commands."""

    def test_dependencies_has_commands(self):
        cmds = get_commands_for_type(QuestionType.DEPENDENCIES, Path("."))
        assert len(cmds) > 0
        assert any("pyproject" in c or "package.json" in c for c in cmds)

    def test_entry_point_has_commands(self):
        cmds = get_commands_for_type(QuestionType.ENTRY_POINT, Path("."))
        assert len(cmds) > 0

    def test_file_types_has_commands(self):
        cmds = get_commands_for_type(QuestionType.FILE_TYPES, Path("."))
        assert len(cmds) > 0

    def test_auth_flow_has_commands(self):
        cmds = get_commands_for_type(QuestionType.AUTH_FLOW, Path("."))
        assert len(cmds) > 0
