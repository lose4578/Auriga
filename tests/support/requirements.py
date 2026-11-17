"""Runtime markers for real-binary agent tests (design 04 §10.2 "Markers").

A test that drives a real pinned agent binary runs only where that runtime is installed at its pin, and is reported
as skipped with an explicit reason everywhere else (CI prints the skip counts per backend). Each marker is a
decorator for a test method or a whole ``TestCase`` class; the probe runs once per process and is cached::

    from support.requirements import requires_claude_cli

    @requires_claude_cli
    class ClaudeCodeConformance(AgentBackendConformance, unittest.TestCase): ...

Markers:

* ``requires_codex_binary`` - ``openai-codex`` importable at the ``pyproject.toml`` pin and its bundled
  ``codex_cli_bin`` executable present (the existing Codex real-binary tests keep their own behaviour).
* ``requires_claude_cli`` - ``claude-agent-sdk`` importable at its pin, its bundled CLI at the CLI pin and
  ``<cli> --version`` equal to it (``factory.claude_code_runtime_problem``: the production check, run in an empty
  temporary home).
* ``requires_opencode`` - ``AURIGA_OPENCODE_EXECUTABLE`` names an executable whose ``--version`` equals the
  ``opencode-ai`` pin (``opencode.config.OPENCODE_VERSION``).
* ``requires_bwrap`` - bubblewrap namespaces usable by production ``AgentSandbox.prepare`` (unprivileged user
  namespaces, or the rootful ``sudo -n bwrap`` + ``setpriv --clear-groups`` fallback it uses on a host where the
  gateway user has supplementary groups; shared host-capability probe ``host_capabilities.bubblewrap_namespaces``;
  ``AURIGA_REQUIRE_SANDBOX_TESTS=1`` turns a skip into a failure).
* ``requires_slow_tests`` - opt-in for tests that take minutes (``AURIGA_SLOW_TESTS=1``; e.g. the 330 s long-wait
  conformance item), run before a backend's acceptance, not in the default suite.

``*_problem()`` functions return None when the runtime is usable, else a one-line reason (never raise).
"""
from __future__ import annotations

import functools
import importlib.metadata
import os
import unittest

from .host_capabilities import bubblewrap_namespaces, require

__all__ = ["claude_cli_problem", "codex_binary_problem", "opencode_problem", "requires_bwrap", "requires_claude_cli",
           "requires_codex_binary", "requires_opencode", "requires_slow_tests", "slow_tests_enabled"]

CODEX_SDK_PACKAGE = "openai-codex"
OPENCODE_EXECUTABLE_ENV = "AURIGA_OPENCODE_EXECUTABLE"
SLOW_TESTS_ENV = "AURIGA_SLOW_TESTS"


def _codex_pin():
    """The ``openai-codex==X`` pin of ``pyproject.toml`` (None when the file or the pin is absent)."""
    from pathlib import Path
    try:
        text = (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text()
    except OSError:
        return None
    marker = f'"{CODEX_SDK_PACKAGE}=='
    start = text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = text.find('"', start)
    return text[start:end] if end > start else None


@functools.lru_cache(maxsize=None)
def codex_binary_problem():
    """None when the pinned Codex SDK and its bundled CLI binary are installed, else the reason."""
    try:
        installed = importlib.metadata.version(CODEX_SDK_PACKAGE)
    except importlib.metadata.PackageNotFoundError:
        return f"{CODEX_SDK_PACKAGE} is not installed"
    pin = _codex_pin()
    if pin is not None and installed != pin:
        return f"{CODEX_SDK_PACKAGE} {installed} installed, pinned {pin}"
    try:
        from codex_cli_bin import bundled_codex_path
        path = bundled_codex_path()
    except Exception as error:  # noqa: BLE001 - any import/lookup failure means "absent"
        return f"bundled Codex CLI unavailable: {type(error).__name__}"
    if not os.path.isfile(path) or not os.access(path, os.X_OK):
        return "bundled Codex CLI binary missing or not executable"
    return None


@functools.lru_cache(maxsize=None)
def claude_cli_problem():
    """None when the pinned Claude Code runtime (SDK pin, bundled CLI pin, ``--version``) is usable, else why not."""
    try:
        from auriga.cortex.agents.factory import claude_code_runtime_problem
        return claude_code_runtime_problem()
    except Exception as error:  # noqa: BLE001 - a crashing probe must never break test discovery
        return f"probe error {type(error).__name__}: {error}"


@functools.lru_cache(maxsize=None)
def opencode_problem():
    """None when ``AURIGA_OPENCODE_EXECUTABLE`` is the pinned opencode binary, else why not."""
    executable = os.environ.get(OPENCODE_EXECUTABLE_ENV)
    if not executable:
        return f"{OPENCODE_EXECUTABLE_ENV} is not set (absolute path of the pinned opencode platform binary)"
    if not os.path.isabs(executable) or not os.access(executable, os.X_OK):
        return f"{OPENCODE_EXECUTABLE_ENV} is not an absolute path of an executable"
    try:
        from auriga.cortex.agents.opencode.config import OPENCODE_VERSION, probe_cli_version
        version = probe_cli_version(executable)
    except Exception as error:  # noqa: BLE001
        return f"opencode version probe failed: {type(error).__name__}"
    if version != OPENCODE_VERSION:
        return f"opencode {version or 'unknown'} found, pinned {OPENCODE_VERSION}"
    return None


def _marker(problem, label):
    def decorate(test):
        reason = problem()
        if reason is None:
            return test
        return unittest.skip(f"{label} unavailable: {reason}")(test)
    return decorate


requires_codex_binary = _marker(codex_binary_problem, "pinned Codex runtime")
requires_claude_cli = _marker(claude_cli_problem, "pinned Claude Code runtime")
requires_opencode = _marker(opencode_problem, "pinned OpenCode runtime")


def requires_bwrap(test):
    """Bubblewrap namespace sandbox, unprivileged or production's rootful fallback (shared probe; strict mode via
    ``AURIGA_REQUIRE_SANDBOX_TESTS=1``)."""
    return require(bubblewrap_namespaces, "the agent runs inside a real bubblewrap namespace sandbox")(test)


def slow_tests_enabled():
    """True when ``AURIGA_SLOW_TESTS`` is set to 1/true/yes/on."""
    return os.environ.get(SLOW_TESTS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


def requires_slow_tests(test):
    """Run only with ``AURIGA_SLOW_TESTS=1`` (tests that take minutes)."""
    if slow_tests_enabled():
        return test
    return unittest.skip(f"slow test (minutes); set {SLOW_TESTS_ENV}=1 to run it")(test)
