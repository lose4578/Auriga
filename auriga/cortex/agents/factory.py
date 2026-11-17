"""Agent backend selection: session construction and launch gating (design 04 §2.6).

``create_session`` builds the configured backend's ``AgentSession`` and imports the backend module only when it is
selected, so a native agent SDK is never imported for another backend. ``backend_requirements`` refuses launch
configurations a backend cannot honour, with one actionable sentence, before any run directory, robot or native
process exists. Both read ``config.agent.backend`` (default ``"codex"``; configs without an ``agent`` section are
Codex runs). ``core/config.py`` does not import this module (layering); the gateway (``Workbench``) and the
evaluation queue call ``backend_requirements`` for every launch.
"""
import os
import tempfile

from ...core.config import ConfigError
from .base import AgentCapabilityError

BACKENDS = ("codex", "claude_code", "opencode")

# model.effort values each non-Codex backend accepts natively, checked at launch; a backend absent here validates
# effort against its own runtime at start (OpenCode passes it as the prompt "variant" of the selected model, design
# §6). Codex accepts every ModelConfig.effort choice (CodexSession.effort_choices). The Claude Code session class
# must reuse this tuple as its ``effort_choices`` (P2).
NATIVE_EFFORT_CHOICES = {"claude_code": ("low", "medium", "high", "xhigh", "max")}


# Effective evidence helper clients (design §9.6), recorded as manifest ``evidence_client`` / ``agent_backend``.
EVIDENCE_CLIENT_CODEX_AGENT = "codex_agent"   # CodexEvidenceAgent: fresh ephemeral Codex app-server thread
EVIDENCE_CLIENT_RELAY = "relay"               # RelayEvidenceClient: one direct request through the run's relay


def agent_backend(config):
    """``config.agent.backend`` ("codex" | "claude_code" | "opencode"); "codex" when the config has no agent section."""
    return getattr(getattr(config, "agent", None), "backend", "codex")


def evidence_client(config, backend=None):
    """Effective client of the evidence helpers (visual review, self-briefing) of a launch ``config``.

    ``backend``: the backend kind to resolve for (default ``agent_backend(config)``). Returns ``"codex_agent"`` for
    Codex with ``agent.evidence.client="native"`` (the default; also configs without an agent section) and
    ``"relay"`` otherwise: ``agent.evidence.client="relay"``, or any backend without a native evidence agent.
    """
    kind = agent_backend(config) if backend is None else backend
    setting = getattr(getattr(getattr(config, "agent", None), "evidence", None), "client", "native")
    return EVIDENCE_CLIENT_CODEX_AGENT if kind == "codex" and setting == "native" else EVIDENCE_CLIENT_RELAY


def _actor_wire(config, kind):
    """Wire of backend ``kind``'s configured actor upstream, or None when it is not settled by the config."""
    upstream = getattr(getattr(config.agent, kind, None), "upstream", None)
    if upstream is None:
        return None
    return upstream.wire_api or ("openai_responses" if upstream.source == "codex_home" else None)


def backend_requirements(config):
    """Refuse a launch ``config`` the selected agent backend cannot run; return None when it can.

    Codex: no requirement beyond the existing config validation (behaviour unchanged). ``features.sandbox_enabled``
    is accepted on every backend (design R4.4; verified on Linux for Codex, Claude Code and OpenCode by conformance
    item 28): the sandbox needs bubblewrap, which only the session start can probe, so a host without it fails at
    start with ``<codex|agent>_sandbox_bubblewrap_not_found_on_path_or_os_defpath`` (the same rule for every backend),
    never by running unsandboxed. Other backends, checked in this order (configuration conflicts first, then
    availability), each raising ``ConfigError``:

    * ``features.lean_projection`` -> ``lean_projection_requires_codex_backend`` (Responses-shaped projection);
    * ``features.visual_review_model`` set, no ``agent.evidence.upstream`` and an actor upstream whose wire is not
      ``openai_responses`` -> ``evidence_upstream_required_for_visual_review_model`` (the review model is requested
      by name from that upstream; design §9.4);
    * ``model.effort`` outside ``NATIVE_EFFORT_CHOICES[kind]`` -> ``agent_effort_unsupported:<kind>:<value>``;
    * ``claude_code``: ``agent.claude_code.upstream`` must be an explicit Anthropic Messages endpoint
      (``claude_code_upstream_requires_anthropic_messages``) or ``source="claude_settings"`` with a readable settings
      file that has a key and forwards no Auriga-owned variable (``claude_settings_*``); the pinned runtime must be installed:
      ``claude-agent-sdk`` at its pin, the CLI it bundles (or ``agent.claude_code.cli_path``) at the CLI pin, checked
      by running ``<cli> --version`` once in an empty temporary home -> else
      ``agent_backend_unavailable:claude_code: <reason>; install auriga-harness[claude]``;
    * ``opencode``: ``agent.opencode.upstream`` must name a relay wire (``opencode_upstream_wire_unsupported:<wire>``
      for ``openai_chat``) and, with ``source="explicit"``, a ``wire_api``, ``base_url`` and ``api_key_env``
      (``opencode_upstream_incomplete``; ``source="codex_home"`` needs the Responses wire,
      ``opencode_upstream_codex_home_requires_openai_responses``); ``agent.opencode.executable`` must be an absolute
      executable file whose ``--version`` is the pin -> else ``agent_backend_unavailable:opencode: <reason>; install
      opencode-ai@1.19.5 ...``. ``model.effort`` must be one of ``opencode.config.EFFORT_CHOICES[wire]``
      (``agent_effort_unsupported:opencode:<effort>``); the declared effort variants are checked against the running
      server at start (``opencode_variants_mismatch``).

    ``features.terminal=true`` is not an error on a backend without a native terminal: the terminal is skipped.
    """
    kind = agent_backend(config)
    if kind == "codex":
        return None
    if kind not in BACKENDS:
        raise ConfigError(f"unknown_agent_backend:{kind}: set agent.backend to one of {', '.join(BACKENDS)}")
    features, model = config.features, config.model
    if getattr(features, "lean_projection", False):
        raise ConfigError("lean_projection_requires_codex_backend: set features.lean_projection=false "
                          "or use agent.backend=codex")
    actor_wire = _actor_wire(config, kind)
    if (features.visual_review_model and config.agent.evidence.upstream is None
            and actor_wire not in (None, "openai_responses")):
        raise ConfigError(f"evidence_upstream_required_for_visual_review_model: the {actor_wire} actor upstream cannot "
                          f"serve {features.visual_review_model}; set agent.evidence.upstream (for example "
                          '{"source": "codex_home"}) or features.visual_review_model=null')
    choices = NATIVE_EFFORT_CHOICES.get(kind)
    if choices is not None and model.effort not in choices:
        raise ConfigError(f"agent_effort_unsupported:{kind}:{model.effort}: choose model.effort from "
                          f"{', '.join(choices)}")
    if kind == "claude_code":
        return _claude_code_requirements(config)
    if kind == "opencode":
        return _opencode_requirements(config)
    raise ConfigError(f"agent_backend_not_implemented:{kind}: this checkout runs agent.backend=codex, claude_code "
                      "or opencode")


def _claude_code_requirements(config):
    upstream = config.agent.claude_code.upstream
    if upstream.source == "claude_settings":
        from .claude_code.options import read_claude_settings_env, read_claude_settings_upstream
        try:
            read_claude_settings_upstream(upstream.settings)
            read_claude_settings_env(upstream.settings)
        except ValueError as error:
            raise ConfigError(str(error)) from None
    elif (upstream.source != "explicit" or upstream.wire_api not in (None, "anthropic_messages")
            or not upstream.base_url or not upstream.api_key_env):
        raise ConfigError("claude_code_upstream_requires_anthropic_messages: set agent.claude_code.upstream to an "
                          "explicit anthropic_messages base_url and api_key_env")
    problem = claude_code_runtime_problem(config.agent.claude_code.cli_path)
    if problem is not None:
        raise ConfigError(f"agent_backend_unavailable:claude_code: {problem}; install auriga-harness[claude]")
    return None


def claude_code_runtime_problem(cli_path=None):
    """None when the pinned Claude Code runtime is usable, else a short reason (str, one line).

    Checks the installed ``claude-agent-sdk`` version, the SDK's bundled CLI version and ``<cli> --version`` of the
    bundled CLI or of ``cli_path`` (absolute path; ``agent.claude_code.cli_path``). The probe runs with an empty
    environment whose HOME/TMPDIR/CLAUDE_CONFIG_DIR are a fresh temporary directory, so nothing outside it is read
    or written. No model, no network.
    """
    try:
        from .claude_code.options import resolve_runtime
        with tempfile.TemporaryDirectory(prefix="auriga-claude-probe-") as home:
            resolve_runtime(cli_path, dict(PATH=os.defpath, LANG="C.UTF-8", HOME=home, TMPDIR=home,
                                           CLAUDE_CONFIG_DIR=home))
    except ImportError as error:
        return f"claude_code_sdk_missing:{type(error).__name__}"
    except (RuntimeError, ValueError, OSError) as error:
        return str(error).split(":", 1)[0] if str(error) else type(error).__name__
    return None


# Upstream wires the relay serves for OpenCode (``openai_chat`` has no relay wire adapter in this checkout).
OPENCODE_WIRES = ("anthropic_messages", "openai_responses")


def _opencode_requirements(config):
    upstream = config.agent.opencode.upstream
    if upstream.wire_api is not None and upstream.wire_api not in OPENCODE_WIRES:
        raise ConfigError(f"opencode_upstream_wire_unsupported:{upstream.wire_api}: set "
                          "agent.opencode.upstream.wire_api to anthropic_messages or openai_responses")
    if upstream.source == "codex_home":
        if upstream.wire_api not in (None, "openai_responses"):
            raise ConfigError("opencode_upstream_codex_home_requires_openai_responses: the Codex credential source "
                              "serves the openai_responses wire only")
    elif upstream.wire_api is None or not upstream.base_url or not upstream.api_key_env:
        raise ConfigError("opencode_upstream_incomplete: set agent.opencode.upstream.wire_api (anthropic_messages or "
                          "openai_responses), base_url and api_key_env, or source=codex_home")
    from .opencode.config import EFFORT_CHOICES
    choices = EFFORT_CHOICES[_actor_wire(config, "opencode")]
    if config.model.effort not in choices:
        raise ConfigError(f"agent_effort_unsupported:opencode:{config.model.effort}: on the "
                          f"{_actor_wire(config, 'opencode')} wire choose model.effort from {', '.join(choices)}")
    problem = opencode_runtime_problem(config.agent.opencode.executable)
    if problem is not None:
        raise ConfigError(f"agent_backend_unavailable:opencode: {problem}; install opencode-ai@1.19.5 and set "
                          "agent.opencode.executable to its platform binary")
    return None


def opencode_runtime_problem(executable):
    """None when ``executable`` is the pinned OpenCode platform binary, else a short reason (str, one line).

    ``executable`` is ``agent.opencode.executable`` (absolute path or None). Runs ``<executable> --version`` once in a
    minimal environment (``config.probe_cli_version``); starts no server, no model, no network.
    """
    from .isolation import require_exact_version
    from .opencode.config import OPENCODE_VERSION, probe_cli_version
    if not executable:
        return "opencode_executable_not_set"
    if not os.path.isabs(executable) or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        return "opencode_executable_invalid"
    try:
        require_exact_version("opencode", probe_cli_version(executable), OPENCODE_VERSION)
    except (RuntimeError, ValueError, OSError) as error:
        return str(error).split(":", 1)[0] if str(error) else type(error).__name__
    return None


def create_session(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
    """Construct (not start) the configured backend's ``AgentSession`` for one run.

    ``run_dir`` is the run directory, ``store`` its Store, ``tool_handler`` the actor tool entry
    (``Harness.handle_tool``), ``price_card`` the frozen price card or None and ``task_planning`` the run's
    TaskPlanning (decision-context reminders) or None. Codex returns ``CodexSession`` with exactly the arguments the
    gateway always passed; ``claude_code`` returns ``ClaudeCodeSession`` (construction imports no agent SDK, so a
    replay viewer of a Claude Code archive opens without the runtime); ``opencode`` returns ``OpenCodeSession``
    (construction starts nothing). Raises ``AgentCapabilityError("unknown_agent_backend:<kind>")`` otherwise.
    """
    kind = agent_backend(config)
    if kind == "codex":
        from ..session.codex import CodexSession
        return CodexSession(run_dir, store, tool_handler, price_card=price_card, config=config,
                            task_planning=task_planning)
    if kind == "claude_code":
        from .claude_code.session import ClaudeCodeSession
        return ClaudeCodeSession(run_dir, store, tool_handler, price_card=price_card, config=config,
                                 task_planning=task_planning)
    if kind == "opencode":
        from .opencode.session import OpenCodeSession
        return OpenCodeSession(run_dir, store, tool_handler, price_card=price_card, config=config,
                               task_planning=task_planning)
    if kind in BACKENDS:
        raise AgentCapabilityError(f"agent_backend_not_implemented:{kind}")
    raise AgentCapabilityError(f"unknown_agent_backend:{kind}")
