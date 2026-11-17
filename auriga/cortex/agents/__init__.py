"""Backend-neutral agent runtime layer.

``base`` defines the ``AgentSession`` contract, ``AgentCapabilities``, the session errors and ``read_agent_identity``;
``factory`` selects and gates the configured backend (``create_session``, ``backend_requirements``); ``dispatch``
orders concurrent tool calls. The Codex backend stays in ``auriga.cortex.session.codex`` (``CodexSession``).
Submodules are imported explicitly (``from auriga.cortex.agents.base import AgentSession``); this package imports
nothing eagerly so that loading it never pulls in a native agent SDK.
"""
